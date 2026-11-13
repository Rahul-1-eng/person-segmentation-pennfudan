import numpy as np
import torch


def binary_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    predictions = torch.sigmoid(logits) >= threshold
    truth = targets >= 0.5
    true_positive = torch.logical_and(predictions, truth).sum().item()
    true_negative = torch.logical_and(~predictions, ~truth).sum().item()
    false_positive = torch.logical_and(predictions, ~truth).sum().item()
    false_negative = torch.logical_and(~predictions, truth).sum().item()
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = true_positive / max(true_positive + false_positive + false_negative, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "true_positive": float(true_positive),
        "true_negative": float(true_negative),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
    }


def merge_metric_totals(total: dict[str, float], current: dict[str, float]) -> None:
    for key in ("true_positive", "true_negative", "false_positive", "false_negative"):
        total[key] += current[key]


def metrics_from_totals(total: dict[str, float]) -> dict[str, float]:
    true_positive = total["true_positive"]
    true_negative = total["true_negative"]
    false_positive = total["false_positive"]
    false_negative = total["false_negative"]
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = true_positive / max(true_positive + false_positive + false_negative, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def per_image_scores(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> list[dict[str, float]]:
    return [
        binary_metrics(logits[index : index + 1], targets[index : index + 1], threshold)
        for index in range(logits.shape[0])
    ]


def pooled_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    return binary_metrics(logits, targets, threshold)


def bootstrap_mean_ci(
    values: list[float], resamples: int = 5000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(data), size=(resamples, len(data)))
    means = data[draws].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_ci(
    first: list[float],
    second: list[float],
    resamples: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    difference = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(difference), size=(resamples, len(difference)))
    means = difference[draws].mean(axis=1)
    return (
        float(difference.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )
