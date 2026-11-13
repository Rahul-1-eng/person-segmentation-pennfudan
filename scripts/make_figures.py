import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from person_segmentation.config import ExperimentConfig
from person_segmentation.distract import KINDS, add_distractor


def source_of(name):
    return "fudan" if name.lower().startswith("fudan") else "penn"


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_split(config):
    return json.loads(Path(config.split_file).read_text(encoding="utf-8"))


def mask_path(config, name):
    directory = config.data_root / "PedMasks"
    if not directory.exists():
        directory = config.data_root / "PNGMasks"
    stem = Path(name).stem
    for candidate in (directory / name, directory / f"{stem}_mask.png"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(name)


def figure_eda(config, output):
    split = load_split(config)
    figure, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    labels = list(split)
    fudan = [sum(source_of(n) == "fudan" for n in split[k]) for k in labels]
    penn = [sum(source_of(n) == "penn" for n in split[k]) for k in labels]
    axes[0].bar(labels, fudan, label="Fudan")
    axes[0].bar(labels, penn, bottom=fudan, label="Penn")
    axes[0].set_ylabel("Images")
    axes[0].set_title("Split sizes by source")
    axes[0].legend()
    fractions, ratios, instances = [], [], []
    for name in split["train"]:
        image = Image.open(config.data_root / "PNGImages" / name)
        mask = np.asarray(Image.open(mask_path(config, name)))
        fractions.append(float((mask > 0).mean()))
        ratios.append(image.width / image.height)
        instances.append(len(np.unique(mask[mask > 0])))
    axes[1].hist(fractions, bins=20)
    axes[1].set_xlabel("Foreground fraction of image")
    axes[1].set_ylabel("Training images")
    axes[1].set_title("Class balance (training set)")
    axes[2].hist(ratios, bins=20)
    axes[2].set_xlabel("Width / height")
    axes[2].set_title("Aspect ratio (training set)")
    axes[3].hist(instances, bins=range(1, max(instances) + 2), align="left")
    axes[3].set_xlabel("People per image")
    axes[3].set_title("Instances (training set)")
    figure.tight_layout()
    figure.savefig(output / "eda.png", dpi=160)
    plt.close(figure)


def box(axis, x, y, width, height, text, color):
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02", fc=color, ec="#14365C", lw=1.2)
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=8.5)


def arrow(axis, start, end):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, color="#14365C", lw=1.4))


def figure_architecture(output):
    figure, axis = plt.subplots(figsize=(13, 4.6))
    axis.set_xlim(0, 13)
    axis.set_ylim(0, 4.6)
    axis.axis("off")
    train, infer = "#DCE9F7", "#FBE7C6"
    box(axis, 0.1, 3.0, 1.7, 1.0, "Penn-Fudan\n170 images\n+ instance masks", "#EEEEEE")
    box(axis, 2.3, 3.0, 1.9, 1.0, "Frozen split\n102 / 34 / 34\nsplits.json", "#EEEEEE")
    box(axis, 4.7, 3.0, 2.0, 1.0, "Augmentation\n(train only)\n+ 30% distractors", train)
    box(axis, 7.2, 3.0, 2.0, 1.0, "Model fit\nU-Net / ResNet34-U-Net\ncosine LR, AdamW", train)
    box(axis, 9.7, 3.0, 1.6, 1.0, "Best epoch\nby validation IoU", train)
    box(axis, 11.7, 3.0, 1.2, 1.0, "best.pt", train)
    box(axis, 2.3, 0.7, 2.2, 1.0, "Validation set\nthreshold tuning\n(no test use)", infer)
    box(axis, 5.1, 0.7, 2.2, 1.0, "Held-out test\n(used once)", infer)
    box(axis, 7.9, 0.7, 2.4, 1.0, "IoU, precision, recall, F1\nper-image IoU + bootstrap CI", infer)
    box(axis, 10.8, 0.7, 2.1, 1.0, "Robustness:\nclean vs reflection\nand shadow images", infer)
    for start, end in [((1.8, 3.5), (2.3, 3.5)), ((4.2, 3.5), (4.7, 3.5)), ((6.7, 3.5), (7.2, 3.5)),
                       ((9.2, 3.5), (9.7, 3.5)), ((11.3, 3.5), (11.7, 3.5)), ((12.3, 3.0), (6.2, 1.7)),
                       ((4.5, 1.2), (5.1, 1.2)), ((7.3, 1.2), (7.9, 1.2)), ((10.3, 1.2), (10.8, 1.2))]:
        arrow(axis, start, end)
    axis.text(0.1, 4.35, "Blue: training phase    Orange: inference and evaluation phase", fontsize=9)
    figure.savefig(output / "architecture.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def figure_distractors(config, output):
    split = load_split(config)
    from PIL import ImageDraw
    names = [n for n in split["train"] if source_of(n) == "penn"][:1] or split["train"][:1]
    name = names[0]
    image = Image.open(config.data_root / "PNGImages" / name).convert("RGB").resize((256, 256))
    mask = Image.open(mask_path(config, name)).convert("L").resize((256, 256), Image.Resampling.NEAREST)
    figure, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    axes = axes.ravel()
    axes[0].imshow(image)
    axes[0].set_title("original")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("ground-truth mask (unchanged)")
    for axis, kind in zip(axes[2:], KINDS):
        result, region = add_distractor(image, mask, kind, random.Random(5), hard=True)
        axis.imshow(result)
        axis.contour(region, levels=[0.5], colors="yellow", linewidths=0.8)
        axis.set_title(kind.replace("_", " "))
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / "distractors.png", dpi=150)
    plt.close(figure)


def figure_training_curves(config, output):
    runs = sorted(path for path in config.checkpoint_dir.glob("*/history.csv"))
    if not runs:
        return False
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for path in runs:
        rows = read_csv(path)
        epochs = [float(r["epoch"]) for r in rows]
        axes[0].plot(epochs, [float(r["validation_loss"]) for r in rows], label=path.parent.name)
        if "validation_iou" in rows[0]:
            axes[1].plot(epochs, [float(r["validation_iou"]) for r in rows], label=path.parent.name)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation loss")
    axes[0].set_title("Validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation IoU")
    axes[1].set_title("Validation IoU")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=160)
    plt.close(figure)
    return True


def figure_qualitative(config, output, experiments):
    reference = read_csv(config.checkpoint_dir / "per_image" / f"{experiments[-1]}.csv")
    if not reference:
        return False
    from person_segmentation.data import PennFudanDataset

    split = load_split(config)
    dataset = PennFudanDataset(config.data_root, split["test"], config.image_size, config.seed, resize=config.resize)
    ordered = [row["image"] for row in reference]
    picks = [ordered[0], ordered[len(ordered) // 4], ordered[len(ordered) // 2], ordered[-1]]
    figure, axes = plt.subplots(len(picks), 2 + len(experiments), figsize=(3 * (2 + len(experiments)), 3 * len(picks)))
    for row_axes, name in zip(axes, picks):
        image, mask = dataset.load_pair(name)
        row_axes[0].imshow(image)
        row_axes[0].set_title(name, fontsize=8)
        row_axes[1].imshow(mask, cmap="gray")
        row_axes[1].set_title("ground truth", fontsize=8)
        for axis, experiment in zip(row_axes[2:], experiments):
            prediction = config.checkpoint_dir / "predictions" / experiment / f"{Path(name).stem}_prediction.png"
            if prediction.exists():
                axis.imshow(Image.open(prediction), cmap="gray")
            axis.set_title(experiment, fontsize=8)
        for axis in row_axes:
            axis.axis("off")
    figure.suptitle("Worst, lower-quartile, median and best test images (ranked by " + experiments[-1] + ")")
    figure.tight_layout()
    figure.savefig(output / "qualitative.png", dpi=140)
    plt.close(figure)
    return True


def figure_robustness(config, output):
    rows = read_csv(config.checkpoint_dir / "robustness_metrics.csv")
    if not rows:
        return False
    runs = list(dict.fromkeys(r["run"] for r in rows))
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    width = 0.8 / len(runs)
    positions = np.arange(len(conditions))
    for index, run in enumerate(runs):
        values = {r["condition"]: r for r in rows if r["run"] == run}
        axes[0].bar(positions + index * width, [float(values[c]["iou"]) for c in conditions], width, label=run)
        axes[1].bar(
            positions[1:] + index * width,
            [float(values[c]["distractor_fp_rate"]) for c in conditions[1:]],
            width,
            label=run,
        )
    for axis, title, ylabel in ((axes[0], "Test IoU under each condition", "IoU"),
                                (axes[1], "Distractor false-positive rate", "Predicted person pixels in region D / |D|")):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions + width * (len(runs) - 1) / 2, conditions, rotation=25, ha="right")
    axes[1].set_xlim(0.5, len(conditions))
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "robustness.png", dpi=160)
    plt.close(figure)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--qualitative", nargs="+", default=["baseline", "augmented"])
    args = parser.parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    output = config.checkpoint_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure_architecture(output)
    print("architecture.png")
    if config.split_file.exists() and (config.data_root / "PNGImages").exists():
        figure_eda(config, output)
        figure_distractors(config, output)
        print("eda.png, distractors.png")
    for name, function in (
        ("training_curves.png", lambda: figure_training_curves(config, output)),
        ("qualitative.png", lambda: figure_qualitative(config, output, args.qualitative)),
        ("robustness.png", lambda: figure_robustness(config, output)),
    ):
        print(name if function() else f"skipped {name}: inputs not ready")


if __name__ == "__main__":
    main()
