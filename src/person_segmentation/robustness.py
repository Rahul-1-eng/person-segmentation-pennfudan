import argparse
import random
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .data import PennFudanDataset, to_tensors
from .distract import KINDS, add_distractor
from .evaluate import discover_runs, write_rows
from .experiments import EXPERIMENTS
from .metrics import binary_metrics
from .model import build_model, load_trained
from .utils import choose_device, load_json, set_seed


def evaluate_condition(model, dataset, names, kind, seed, device, threshold):
    logits, masks, regions = [], [], []
    for index, name in enumerate(names):
        image, mask = dataset.load_pair(name)
        region = np.zeros((dataset.image_size, dataset.image_size), dtype=bool)
        if kind != "clean":
            rng = random.Random(seed * 1000 + index * 17 + KINDS.index(kind))
            image, region = add_distractor(image, mask, kind, rng, hard=True)
        image_tensor, mask_tensor = to_tensors(image, mask)
        with torch.no_grad():
            logits.append(model(image_tensor.unsqueeze(0).to(device)).cpu())
        masks.append(mask_tensor.unsqueeze(0))
        regions.append(torch.from_numpy(region))
    logits = torch.cat(logits)
    masks = torch.cat(masks)
    regions = torch.stack(regions).unsqueeze(1)
    metrics = binary_metrics(logits, masks, threshold)
    predicted = torch.sigmoid(logits) >= threshold
    region_pixels = regions.sum().item()
    leaked = torch.logical_and(predicted, regions).sum().item()
    per_image = [
        torch.logical_and(predicted[i], regions[i]).sum().item() / regions[i].sum().item()
        for i in range(len(names))
        if regions[i].sum().item() > 0
    ]
    return {
        "iou": metrics["iou"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "distractor_pixels": region_pixels,
        "distractor_fp_rate": leaked / region_pixels if region_pixels else float("nan"),
        "distractor_fp_rate_mean_image": float(np.mean(per_image)) if per_image else float("nan"),
    }


def run(config: ExperimentConfig, zero_shot: bool) -> None:
    set_seed(config.seed)
    device = choose_device(config.device)
    split = load_json(config.split_file)
    dataset = PennFudanDataset(config.data_root, split["test"], config.image_size, config.seed, resize=config.resize)
    targets = [(path.name, load_trained(path / "best.pt", device)[0]) for path in discover_runs(config.checkpoint_dir)]
    if zero_shot:
        for name, spec in EXPERIMENTS.items():
            if spec.zero_shot:
                targets.append((name, build_model(spec.model, pretrained=True).to(device).eval()))
    rows = []
    for name, model in targets:
        for kind in ("clean", *KINDS):
            print(f"{name}: {kind}")
            rows.append(
                {"run": name, "condition": kind, **evaluate_condition(model, dataset, dataset.names, kind, config.seed, device, config.threshold)}
            )
    write_rows(config.checkpoint_dir / "robustness_metrics.csv", rows)
    print(f"Saved {config.checkpoint_dir / 'robustness_metrics.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--zero-shot", action="store_true")
    args = parser.parse_args()
    run(ExperimentConfig.from_yaml(args.config), args.zero_shot)


if __name__ == "__main__":
    main()
