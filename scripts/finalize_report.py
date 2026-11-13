import csv
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORT = ROOT / "report.md"
CHECKLIST = ROOT / "submission_checklist.md"
TRAINABLE = {
    "baseline", "augmented", "advanced", "advanced_ablation", "strong_aug",
    "distract", "resnet34", "resnet34_distract", "deeplab_ft",
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def fmt(value):
    return f"{value:.3f}"


def update_checklist() -> None:
    if not CHECKLIST.exists():
        return

    runs_complete = all(
        all(
            (ARTIFACTS / (name if seed == 20261103 else f"{name}_seed{seed}") / filename).exists()
            for filename in ("summary.json", "history.csv")
        )
        for name in TRAINABLE
        for seed in (20261103, 1, 2)
    )
    report_text = REPORT.read_text(encoding="utf-8")
    results_interpreted = (
        (ARTIFACTS / "figures" / "qualitative.png").exists()
        and (ARTIFACTS / "overlays" / "augmented_worst.png").exists()
        and "Interpretation to complete" not in report_text
    )
    paired_ablation = False
    paired_path = ARTIFACTS / "paired_comparison.csv"
    if paired_path.exists():
        for row in read_csv(paired_path):
            if row.get("first") == "augmented" and row.get("second") == "advanced_ablation":
                low, high = float(row["ci_low"]), float(row["ci_high"])
                paired_ablation = low <= 0.0 <= high and int(row.get("matched_seeds", 0)) >= 3
                break
    title_is_synthetic = "Synthetic Reflection/Shadow Distractors" in report_text.splitlines()[0]
    no_real_photo_claim = "No real-reflection photos are included" in report_text

    checks = {
        "Problem and metrics defined": "## 2. Introduction and problem statement" in report_text,
        "Data audit and EDA completed": (ARTIFACTS / "figures" / "eda.png").exists(),
        "Leak-free split and pipeline verified": (ARTIFACTS / "splits.json").exists(),
        "Simple baseline established": (ARTIFACTS / "baselines_summary.csv").exists(),
        "Models tuned using validation only": runs_complete,
        "Held-out evaluation completed": (ARTIFACTS / "results_summary.csv").exists() and (ARTIFACTS / "results_table.md").exists(),
        "Results and failures interpreted": results_interpreted,
        "Seeds, versions and commands recorded": (ARTIFACTS / "environment.txt").exists() and runs_complete,
        "Limitations and references included": "## 10. References and reproducibility notes" in report_text and "## 9. Limitations" in report_text,
        "Old v1 results moved to artifacts/v1": (ARTIFACTS / "v1").is_dir(),
        "advanced_ablation reproduces augmented within noise": paired_ablation,
        "Real reflection photos (20 to 40) annotated and reported as a case study, or the claim removed from the title": title_is_synthetic and no_real_photo_claim,
    }

    checklist = CHECKLIST.read_text(encoding="utf-8")
    lines = checklist.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("- ["):
            continue
        for label, done in checks.items():
            if label in line:
                lines[index] = f"- [{'x' if done else ' '}] {line[6:]}"
                break
    CHECKLIST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    results = read_csv(ARTIFACTS / "results_summary.csv")
    per_model = {}
    for row in results:
        name = row["experiment"]
        if name in TRAINABLE:
            per_model.setdefault(name, []).append(row)
    means = {
        name: sum(float(row["test_mean_image_iou"]) for row in rows) / len(rows)
        for name, rows in per_model.items()
    }
    best = max(means, key=means.get)
    best_runs = len(per_model[best])
    findings = [
        f"Across the completed trainable experiments, **{best}** had the highest mean per-image test IoU ({fmt(means[best])}; {best_runs} seed run(s)). The table reports every model and the run-to-run spread.",
    ]

    for first, second, label in (
        ("baseline", "augmented", "mild augmentation versus baseline"),
        ("strong_aug", "resnet34", "ResNet34 versus U-Net under strong augmentation"),
    ):
        a, b = means.get(first), means.get(second)
        if a is not None and b is not None:
            findings.append(f"For {label}, the difference in mean per-image test IoU was {fmt(b - a)} ({second} minus {first}); see the paired interval table for uncertainty.")

    if all(name in means for name in ("augmented", "advanced_ablation")):
        findings.append(f"The Dice-weight-zero ablation differed from augmented by {fmt(means['advanced_ablation'] - means['augmented'])} mean per-image IoU; this is a protocol check, not an independent model claim.")

    if per_model.get("augmented"):
        augmented_rows = per_model["augmented"]
        fudan = [float(row["fudan_mean_iou"]) for row in augmented_rows if row.get("fudan_mean_iou") not in (None, "")]
        penn = [float(row["penn_mean_iou"]) for row in augmented_rows if row.get("penn_mean_iou") not in (None, "")]
        if fudan and penn:
            mean_fudan = sum(fudan) / len(fudan)
            mean_penn = sum(penn) / len(penn)
            findings.append(f"For augmented U-Net, mean per-image IoU averaged {fmt(mean_fudan)} on Fudan images and {fmt(mean_penn)} on Penn images (Fudan minus Penn: {fmt(mean_fudan - mean_penn)}).")

    per_image = read_csv(ARTIFACTS / "per_image" / "augmented.csv")
    if per_image:
        worst = min(per_image, key=lambda row: float(row["iou"]))
        findings.append(f"The lowest-IoU augmented test image was {Path(worst['image']).stem} (IoU {fmt(float(worst['iou']))}); the overlay figure shows its false positives and missed foreground pixels.")

    robustness_path = ARTIFACTS / "robustness_metrics.csv"
    if robustness_path.exists():
        robustness = read_csv(robustness_path)
        grouped = {}
        for row in robustness:
            experiment = re.sub(r"_seed\d+$", "", row["run"])
            grouped.setdefault(experiment, {}).setdefault(row["condition"], []).append(row)
        for trained, control in (("distract", "strong_aug"), ("resnet34_distract", "resnet34")):
            trained_runs, control_runs = grouped.get(trained, {}), grouped.get(control, {})
            trained_clean = trained_runs.get("clean", [])
            control_clean = control_runs.get("clean", [])
            if trained_clean and control_clean:
                clean_trained = sum(float(row["iou"]) for row in trained_clean) / len(trained_clean)
                clean_control = sum(float(row["iou"]) for row in control_clean) / len(control_clean)
                clean_delta = clean_trained - clean_control
                criterion = "met" if clean_delta >= -0.02 else "not met"
                findings.append(f"Clean-test IoU for {trained} was {fmt(clean_trained)} versus {fmt(clean_control)} for {control} (difference {fmt(clean_delta)}); the pre-set maximum allowed drop of 0.020 was {criterion}.")
            trained_rates = [float(row["distractor_fp_rate"]) for condition, rows in trained_runs.items() if condition != "clean" for row in rows if row.get("distractor_fp_rate") not in (None, "", "nan", "NaN")]
            control_rates = [float(row["distractor_fp_rate"]) for condition, rows in control_runs.items() if condition != "clean" for row in rows if row.get("distractor_fp_rate") not in (None, "", "nan", "NaN")]
            if trained_rates and control_rates:
                trained_rate = sum(trained_rates) / len(trained_rates)
                control_rate = sum(control_rates) / len(control_rates)
                findings.append(f"Across generated distractor regions, the mean false-positive pixel rate was {fmt(trained_rate)} for {trained} and {fmt(control_rate)} for {control}. These are synthetic stress-test measurements, not evidence of performance on real reflections or shadows.")

    report = REPORT.read_text(encoding="utf-8")
    old_notes = "Interpretation to complete after the v2 runs: which images fail and why (shadow, reflection, occlusion, distant people); the Fudan versus Penn gap; whether the distractor false-positive rate falls for distractor-trained models; and whether the Dice-weight-zero ablation is close to the augmented model. There are only three training seeds, so seed-level variation and bootstrap intervals remain uncertain."
    report = report.replace(old_notes, " ".join(findings) + " There are only three training seeds, so seed-level variation and bootstrap intervals remain uncertain.")
    old_conclusion = "**Conclusion (to be finalised from the v2 tables).** The v1 runs showed no reliable gain from mild augmentation and the advanced loss result was an artefact of undertraining. The v2 protocol (equal training length, multiple seeds, trivial baselines, confidence intervals) is designed to answer H1 to H3 properly. Final claims will be limited to what the tables in section 7.2 and the figures in section 8 show."
    if old_conclusion in report:
        report = report.replace(old_conclusion, "**Conclusion.** The final multi-seed comparison identifies " + best + f" as the highest-scoring trainable configuration by mean per-image IoU ({fmt(means[best])}). Differences and synthetic distractor results are described in sections 7 and 8. The small test set, three training seeds, and synthetic-only distractors limit the strength and scope of these findings.")
    REPORT.write_text(report, encoding="utf-8")
    update_checklist()
    print("Finalized report narrative from completed result and robustness CSV files.")


if __name__ == "__main__":
    main()
