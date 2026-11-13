import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .config import ExperimentConfig
from .data import loaders, source_of
from .experiments import EXPERIMENTS
from .inference import collect_logits
from .metrics import binary_metrics, bootstrap_mean_ci, per_image_scores
from .model import build_model, load_trained
from .utils import choose_device, load_json, save_json, set_seed

THRESHOLDS = [round(value, 2) for value in np.arange(0.20, 0.81, 0.05).tolist()]


def discover_runs(checkpoint_dir: Path) -> list[Path]:
    # A checkpoint may be written before training completes. The summary is
    # written only after all epochs finish, so require it before evaluation.
    return sorted(
        path.parent
        for path in checkpoint_dir.glob("*/best.pt")
        if (path.parent / "summary.json").is_file()
    )


def tune_threshold(logits: torch.Tensor, masks: torch.Tensor) -> float:
    scores = {value: binary_metrics(logits, masks, value)["iou"] for value in THRESHOLDS}
    return max(scores, key=scores.get)


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(logits, names, threshold, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names):
        prediction = torch.sigmoid(logits[index]).squeeze(0).numpy() >= threshold
        Image.fromarray(prediction.astype(np.uint8) * 255).save(
            directory / f"{Path(name).stem}_prediction.png"
        )


def evaluate_run(run: str, model, summary: dict, data_loaders, config, device, output_dir: Path) -> dict:
    validation_logits, validation_masks, _ = collect_logits(model, data_loaders["validation"], device, config.tta)
    test_logits, test_masks, names = collect_logits(model, data_loaders["test"], device, config.tta)
    pooled = binary_metrics(test_logits, test_masks, config.threshold)
    validation_pooled = binary_metrics(validation_logits, validation_masks, config.threshold)
    tuned = tune_threshold(validation_logits, validation_masks)
    tuned_pooled = binary_metrics(test_logits, test_masks, tuned)
    scores = per_image_scores(test_logits, test_masks, config.threshold)
    image_iou = [score["iou"] for score in scores]
    low, high = bootstrap_mean_ci(image_iou)
    per_image = [
        {
            "image": name,
            "source": source_of(name),
            "iou": score["iou"],
            "precision": score["precision"],
            "recall": score["recall"],
            "f1": score["f1"],
            "false_positive": score["false_positive"],
            "false_negative": score["false_negative"],
        }
        for name, score in zip(names, scores)
    ]
    (output_dir / "per_image").mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "per_image" / f"{run}.csv", sorted(per_image, key=lambda row: row["iou"]))
    save_predictions(test_logits, names, config.threshold, output_dir / "predictions" / run)
    by_source = {
        source: float(np.mean([row["iou"] for row in per_image if row["source"] == source]))
        for source in ("fudan", "penn")
        if any(row["source"] == source for row in per_image)
    }
    return {
        "run": run,
        "experiment": summary.get("experiment", run),
        "seed": summary.get("seed", config.seed),
        "model": summary.get("model", "unet"),
        "parameters": summary.get("model_parameters"),
        "training_seconds": summary.get("training_seconds"),
        "checkpoint_mb": (summary["checkpoint_bytes"] / 1e6) if summary.get("checkpoint_bytes") else None,
        "best_epoch": summary.get("best_epoch"),
        "validation_iou": validation_pooled["iou"],
        "test_precision": pooled["precision"],
        "test_recall": pooled["recall"],
        "test_f1": pooled["f1"],
        "test_iou": pooled["iou"],
        "test_mean_image_iou": float(np.mean(image_iou)),
        "test_mean_image_iou_ci_low": low,
        "test_mean_image_iou_ci_high": high,
        "fudan_mean_iou": by_source.get("fudan"),
        "penn_mean_iou": by_source.get("penn"),
        "validation_tuned_threshold": tuned,
        "test_iou_at_tuned_threshold": tuned_pooled["iou"],
        "true_positive": pooled["true_positive"],
        "false_positive": pooled["false_positive"],
        "false_negative": pooled["false_negative"],
    }


def plot_metrics(rows: list[dict], path: Path) -> None:
    names = [row["run"] for row in rows]
    positions = np.arange(len(names))
    metric_names = ["test_precision", "test_recall", "test_f1", "test_iou"]
    width = 0.18
    figure, axis = plt.subplots(figsize=(max(9, len(names) * 1.2), 5))
    for index, metric in enumerate(metric_names):
        axis.bar(
            positions + (index - 1.5) * width,
            [row[metric] for row in rows],
            width,
            label=metric.replace("test_", ""),
        )
    axis.set_xticks(positions, names, rotation=30, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Penn-Fudan person segmentation test metrics")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def evaluate(config: ExperimentConfig, zero_shot: bool) -> None:
    set_seed(config.seed)
    device = choose_device(config.device)
    split = load_json(config.split_file)
    data_loaders = loaders(
        config.data_root, split, config.image_size, config.batch_size,
        config.num_workers, seed=config.seed, resize=config.resize,
    )
    rows = []
    for run_dir in discover_runs(config.checkpoint_dir):
        model, _ = load_trained(run_dir / "best.pt", device)
        summary_path = run_dir / "summary.json"
        summary = load_json(summary_path) if summary_path.exists() else {}
        print(f"evaluating {run_dir.name}")
        rows.append(evaluate_run(run_dir.name, model, summary, data_loaders, config, device, config.checkpoint_dir))
    if zero_shot:
        for name, spec in EXPERIMENTS.items():
            if spec.zero_shot:
                model = build_model(spec.model, pretrained=True).to(device).eval()
                print(f"evaluating {name}")
                rows.append(
                    evaluate_run(name, model, {"experiment": name, "model": spec.model}, data_loaders, config, device, config.checkpoint_dir)
                )
    if not rows:
        raise FileNotFoundError(f"No checkpoints found under {config.checkpoint_dir}")
    write_rows(config.checkpoint_dir / "results_summary.csv", rows)
    save_json(config.checkpoint_dir / "results_summary.json", rows)
    plot_metrics(rows, config.checkpoint_dir / "metric_comparison.png")
    print(f"Saved {config.checkpoint_dir / 'results_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--zero-shot", action="store_true")
    args = parser.parse_args()
    evaluate(ExperimentConfig.from_yaml(args.config), args.zero_shot)


if __name__ == "__main__":
    main()
