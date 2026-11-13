import argparse
import csv
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .config import ExperimentConfig
from .data import loaders
from .experiments import EXPERIMENTS, trainable_experiments
from .losses import loss_for_experiment
from .model import build_model
from .utils import choose_device, load_json, save_json, set_seed


def run_name(experiment: str, seed: int, default_seed: int) -> str:
    return experiment if seed == default_seed else f"{experiment}_seed{seed}"


def run_epoch(model, loader, criterion, device, optimizer, epoch):
    training = optimizer is not None
    model.train(training)
    dataset = loader.dataset
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    total_loss = 0.0
    total_items = 0
    true_positive = false_positive = false_negative = 0.0
    for images, masks, _ in tqdm(loader, leave=False):
        images = images.to(device)
        masks = masks.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.shape[0]
        total_items += images.shape[0]
        if not training:
            predictions = torch.sigmoid(logits) >= 0.5
            truth = masks >= 0.5
            true_positive += torch.logical_and(predictions, truth).sum().item()
            false_positive += torch.logical_and(predictions, ~truth).sum().item()
            false_negative += torch.logical_and(~predictions, truth).sum().item()
    iou = true_positive / max(true_positive + false_positive + false_negative, 1)
    return total_loss / max(total_items, 1), iou


def train(
    config: ExperimentConfig,
    experiment: str,
    seed: int | None = None,
    epochs: int | None = None,
) -> None:
    spec = EXPERIMENTS[experiment]
    if spec.zero_shot:
        raise ValueError(f"{experiment} is zero-shot and has nothing to train")
    started_at = time.perf_counter()
    seed = config.seed if seed is None else seed
    set_seed(seed)
    device = choose_device(config.device)
    split = load_json(config.split_file)
    data_loaders = loaders(
        config.data_root,
        split,
        config.image_size,
        config.batch_size,
        config.num_workers,
        seed=seed,
        aug=spec.aug,
        resize=config.resize,
        distractor_probability=config.distractor_probability if spec.distractors else 0.0,
        aug_config=config.aug,
    )
    model = build_model(spec.model, pretrained=config.pretrained).to(device)
    criterion = loss_for_experiment(experiment)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.lr or config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_epochs = epochs or config.epochs
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        if config.scheduler == "cosine"
        else None
    )
    name = run_name(experiment, seed, config.seed)
    run_dir = config.checkpoint_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.json"
    history: list[dict[str, float]] = []
    best_iou = -1.0
    best_loss = float("inf")
    best_epoch = 0
    progress = {
        "run": name,
        "experiment": experiment,
        "seed": seed,
        "status": "running",
        "epoch": 0,
        "total_epochs": total_epochs,
        "best_epoch": best_epoch,
        "best_validation_iou": None,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(progress_path, progress)
    for epoch in range(total_epochs):
        train_loss, _ = run_epoch(model, data_loaders["train"], criterion, device, optimizer, epoch)
        validation_loss, validation_iou = run_epoch(
            model, data_loaders["validation"], criterion, device, None, epoch
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_iou": validation_iou,
            }
        )
        if scheduler is not None:
            scheduler.step()
        print(
            f"{name} epoch {epoch + 1}/{total_epochs} train_loss={train_loss:.5f} "
            f"validation_loss={validation_loss:.5f} validation_iou={validation_iou:.4f}"
        )
        if validation_iou > best_iou:
            best_iou, best_loss, best_epoch = validation_iou, validation_loss, epoch + 1
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": spec.model,
                    "epoch": best_epoch,
                    "validation_iou": best_iou,
                    "validation_loss": best_loss,
                    "experiment": experiment,
                    "seed": seed,
                },
                run_dir / "best.pt",
            )
        progress.update(
            {
                "epoch": epoch + 1,
                "best_epoch": best_epoch,
                "best_validation_iou": best_iou,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_iou": validation_iou,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        save_json(progress_path, progress)
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    save_json(
        run_dir / "summary.json",
        {
            "run": name,
            "experiment": experiment,
            "seed": seed,
            "model": spec.model,
            "augmentation": spec.aug,
            "distractors": spec.distractors,
            "epochs": total_epochs,
            "best_epoch": best_epoch,
            "best_validation_iou": best_iou,
            "best_validation_loss": best_loss,
            "device": str(device),
            "torch_version": torch.__version__,
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "training_seconds": time.perf_counter() - started_at,
            "checkpoint_bytes": (run_dir / "best.pt").stat().st_size,
        },
    )
    progress.update(
        {
            "status": "completed",
            "epoch": total_epochs,
            "best_epoch": best_epoch,
            "best_validation_iou": best_iou,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    save_json(progress_path, progress)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--experiment", required=True, choices=trainable_experiments())
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    train(ExperimentConfig.from_yaml(args.config), args.experiment, args.seed, args.epochs)


if __name__ == "__main__":
    main()
