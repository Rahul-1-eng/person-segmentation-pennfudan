import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from person_segmentation.config import ExperimentConfig
from person_segmentation.data import PennFudanDataset
from person_segmentation.model import load_trained
from person_segmentation.utils import choose_device, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--experiment", default="augmented")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    device = choose_device(config.device)
    split = load_json(config.split_file)
    dataset = PennFudanDataset(config.data_root, split["test"], config.image_size, config.seed, resize=config.resize)
    model, _ = load_trained(config.checkpoint_dir / args.experiment / "best.pt", device)
    rows = []
    with torch.no_grad():
        for name in dataset.names:
            image, mask = dataset.load_pair(name)
            tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
            probability = torch.sigmoid(model(tensor.unsqueeze(0).to(device))).squeeze().cpu().numpy()
            prediction = probability >= config.threshold
            truth = np.asarray(mask) > 0
            union = np.logical_or(prediction, truth).sum()
            rows.append((np.logical_and(prediction, truth).sum() / max(union, 1), name, image, truth, prediction))
    rows.sort(key=lambda item: item[0])
    selected = rows[: args.limit]
    output_dir = config.checkpoint_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(selected), 3, figsize=(10, 3.2 * len(selected)))
    axes = np.atleast_2d(axes)
    for row_axes, (iou, name, image, truth, prediction) in zip(axes, selected):
        base = np.asarray(image, dtype=np.float32) / 255.0
        overlay = base.copy()
        overlay[np.logical_and(prediction, ~truth)] = [1.0, 0.0, 0.0]
        overlay[np.logical_and(~prediction, truth)] = [0.0, 0.3, 1.0]
        overlay[np.logical_and(prediction, truth)] = overlay[np.logical_and(prediction, truth)] * 0.5 + np.array([0.0, 0.5, 0.0])
        row_axes[0].imshow(base)
        row_axes[0].set_title(name, fontsize=8)
        row_axes[1].imshow(truth, cmap="gray")
        row_axes[1].set_title("ground truth", fontsize=8)
        row_axes[2].imshow(overlay)
        row_axes[2].set_title(f"IoU {iou:.3f}: red=false positive, blue=missed", fontsize=8)
        for axis in row_axes:
            axis.axis("off")
    figure.tight_layout()
    target = output_dir / f"{args.experiment}_worst.png"
    figure.savefig(target, dpi=150)
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
