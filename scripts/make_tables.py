import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from person_segmentation.config import ExperimentConfig


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def number(row: dict, key: str) -> float | None:
    value = row.get(key)
    return float(value) if value not in (None, "", "None") else None


def summarize(values: list[float | None], digits: int = 4) -> str:
    present = [value for value in values if value is not None]
    if not present:
        return "-"
    if len(present) == 1:
        return f"{present[0]:.{digits}f}"
    return f"{np.mean(present):.{digits}f} ± {np.std(present, ddof=1):.{digits}f}"


def paired_seed_bootstrap(first_runs, second_runs, resamples: int = 5000, seed: int = 0):
    """Resample matched seeds, then matched test images within each seed."""
    common_runs = sorted(set(first_runs) & set(second_runs))
    differences = []
    for run in common_runs:
        images = sorted(set(first_runs[run]) & set(second_runs[run]))
        if images:
            differences.append(np.asarray([second_runs[run][i] - first_runs[run][i] for i in images]))
    if not differences:
        raise ValueError("No matching seeds and test images for paired comparison")
    observed = float(np.mean([values.mean() for values in differences]))
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        sampled_runs = generator.integers(0, len(differences), size=len(differences))
        means = []
        for run_index in sampled_runs:
            values = differences[run_index]
            sampled_images = generator.integers(0, len(values), size=len(values))
            means.append(float(values[sampled_images].mean()))
        draws[draw] = np.mean(means)
    return observed, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), len(differences), min(map(len, differences))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pair", nargs=2, action="append", metavar=("FIRST", "SECOND"))
    args = parser.parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    directory = config.checkpoint_dir
    rows = read_rows(directory / "baselines_summary.csv") + read_rows(directory / "results_summary.csv")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["experiment"]].append(row)
    table = []
    for experiment, items in groups.items():
        table.append(
            {
                "Model": experiment,
                "Runs": len(items),
                "Validation IoU": summarize([number(r, "validation_iou") for r in items]),
                "Test IoU (pooled)": summarize([number(r, "test_iou") for r in items]),
                "Test mean image IoU": summarize([number(r, "test_mean_image_iou") for r in items]),
                "Fudan IoU": summarize([number(r, "fudan_mean_iou") for r in items]),
                "Penn IoU": summarize([number(r, "penn_mean_iou") for r in items]),
                "Train time (s)": summarize([number(r, "training_seconds") for r in items], 0),
                "Parameters (M)": summarize([(number(r, "parameters") or 0) / 1e6 or None for r in items], 2),
                "Checkpoint (MB)": summarize([number(r, "checkpoint_mb") for r in items], 1),
            }
        )
    if not table:
        raise SystemExit("No result files found. Run baselines and evaluate first.")
    with (directory / "results_table.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)
    header = list(table[0].keys())
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(row[key]) for key in header) + " |" for row in table]
    (directory / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    pairs = args.pair or [["baseline", "augmented"]]
    paired = []
    for first, second in pairs:
        per_image_dir = directory / "per_image"
        first_paths = sorted(per_image_dir.glob(f"{first}*.csv"))
        second_paths = sorted(per_image_dir.glob(f"{second}*.csv"))
        first_runs = {
            path.stem[len(first):].lstrip("_"): {r["image"]: float(r["iou"]) for r in read_rows(path)}
            for path in first_paths if path.stem == first or path.stem.startswith(first + "_seed")
        }
        second_runs = {
            path.stem[len(second):].lstrip("_"): {r["image"]: float(r["iou"]) for r in read_rows(path)}
            for path in second_paths if path.stem == second or path.stem.startswith(second + "_seed")
        }
        if not first_runs or not second_runs:
            continue
        difference, low, high, n_seeds, n_images = paired_seed_bootstrap(first_runs, second_runs)
        paired.append(
            {"first": first, "second": second, "mean_iou_difference": difference, "ci_low": low, "ci_high": high,
             "matched_seeds": n_seeds, "images_per_seed": n_images,
             "ci_excludes_zero": not (low <= 0.0 <= high)}
        )
    if paired:
        with (directory / "paired_comparison.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=paired[0].keys())
            writer.writeheader()
            writer.writerows(paired)
        keys = list(paired[0].keys())
        md = ["| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
        for row in paired:
            md.append("| " + " | ".join(f"{row[k]:.4f}" if isinstance(row[k], float) else str(row[k]) for k in keys) + " |")
        (directory / "paired_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        for row in paired:
            print(row)


if __name__ == "__main__":
    main()
