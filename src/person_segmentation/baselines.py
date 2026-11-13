import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from .config import ExperimentConfig
from .data import PennFudanDataset, source_of
from .evaluate import write_rows
from .metrics import binary_metrics, bootstrap_mean_ci, per_image_scores
from .utils import load_json, set_seed

SMALL = 64


def stack(dataset: PennFudanDataset) -> tuple[np.ndarray, np.ndarray, list[str]]:
    images, masks = [], []
    for index in range(len(dataset)):
        image, mask, _ = dataset[index]
        images.append(image.numpy())
        masks.append(mask.numpy())
    return np.stack(images), np.stack(masks), list(dataset.names)


def pixel_features(images: np.ndarray, size: int) -> np.ndarray:
    count, _, height, width = images.shape
    small = torch.nn.functional.interpolate(torch.from_numpy(images), size=(size, size), mode="area").numpy()
    ys, xs = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing="ij")
    position = np.broadcast_to(np.stack([ys, xs]), (count, 2, size, size))
    features = np.concatenate([small, position], axis=1)
    return features.transpose(0, 2, 3, 1).reshape(-1, 5)


def to_logits(probability: np.ndarray) -> torch.Tensor:
    clipped = np.clip(probability, 1e-4, 1 - 1e-4)
    return torch.from_numpy(np.log(clipped / (1 - clipped))).float()


def score(name, logits, masks, names, threshold):
    scores = per_image_scores(logits, masks, threshold)
    image_iou = [item["iou"] for item in scores]
    pooled = binary_metrics(logits, masks, threshold)
    low, high = bootstrap_mean_ci(image_iou)
    return {
        "run": name,
        "experiment": name,
        "model": "baseline",
        "test_precision": pooled["precision"],
        "test_recall": pooled["recall"],
        "test_f1": pooled["f1"],
        "test_iou": pooled["iou"],
        "test_mean_image_iou": float(np.mean(image_iou)),
        "test_mean_image_iou_ci_low": low,
        "test_mean_image_iou_ci_high": high,
        "true_positive": pooled["true_positive"],
        "false_positive": pooled["false_positive"],
        "false_negative": pooled["false_negative"],
    }


def run(config: ExperimentConfig) -> None:
    set_seed(config.seed)
    split = load_json(config.split_file)
    datasets = {
        key: PennFudanDataset(config.data_root, split[key], config.image_size, config.seed, resize=config.resize)
        for key in ("train", "validation", "test")
    }
    train_images, train_masks, _ = stack(datasets["train"])
    validation_images, validation_masks, _ = stack(datasets["validation"])
    test_images, test_masks, test_names = stack(datasets["test"])
    test_masks_tensor = torch.from_numpy(test_masks)
    validation_masks_tensor = torch.from_numpy(validation_masks)
    rows = []

    def add(name, test_probability, validation_probability):
        row = score(name, to_logits(test_probability), test_masks_tensor, test_names, 0.5)
        row["validation_iou"] = binary_metrics(to_logits(validation_probability), validation_masks_tensor, 0.5)["iou"]
        rows.append(row)

    add("all_background", np.zeros_like(test_masks), np.zeros_like(validation_masks))
    mean_mask = train_masks.mean(axis=0, keepdims=True)
    add("mean_training_mask", np.repeat(mean_mask, len(test_masks), 0), np.repeat(mean_mask, len(validation_masks), 0))

    small_masks = torch.nn.functional.interpolate(torch.from_numpy(train_masks), size=(SMALL, SMALL), mode="area").numpy()
    labels = (small_masks.reshape(-1) >= 0.5).astype(np.int64)
    features = pixel_features(train_images, SMALL)
    forest = RandomForestClassifier(n_estimators=60, max_depth=14, min_samples_leaf=20, n_jobs=-1, random_state=config.seed)
    forest.fit(features, labels)

    def forest_probability(images):
        probability = forest.predict_proba(pixel_features(images, SMALL))[:, 1].reshape(len(images), 1, SMALL, SMALL)
        return torch.nn.functional.interpolate(torch.from_numpy(probability).float(), size=(config.image_size, config.image_size), mode="nearest").numpy()

    add("pixel_random_forest", forest_probability(test_images), forest_probability(validation_images))
    write_rows(config.checkpoint_dir / "baselines_summary.csv", rows)
    for row in rows:
        print(row["run"], f"test_iou={row['test_iou']:.4f}", f"validation_iou={row['validation_iou']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(ExperimentConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
