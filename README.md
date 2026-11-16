# Binary Person Segmentation on Penn-Fudan

**Live project website:** https://segment-people--3wgr3i3g.preview.emergentagent.com/ — a hosted showcase site with a live segmentation demo (`backend/`, `frontend/`).

Machine Learning course project, IIT Patna (Rahul Kumar Sahoo, 2401CS09). Full overview, the 7 Ws and careful findings: `docs/PROJECT_OVERVIEW.md`.

## Key results

| Result | Value |
|---|---|
| Best mean per-image IoU (fine-tuned DeepLabV3) | 0.7973 |
| Gain from ImageNet-pretrained encoder over from-scratch U-Net | +0.0736 |
| Distractor false-positive rate, before to after distractor training | 0.1314 to 0.0273 |
| Training runs | 27 runs (9 configurations x 3 seeds) |
| Data | 170 images, frozen 102 / 34 / 34 split |

The live demo runs the zero-shot DeepLabV3-ResNet50 reference model (COCO person class), scored 0.7567 mean per-image IoU on Penn-Fudan.

## Demo video

<!-- Paste the GitHub video link here, on its own line, after uploading the MP4 -->
Short video walkthrough: coming soon.

## Screenshots

Website screenshots are in `docs/screenshots/`.

**Final deliverables** are under `deliverables/`:

- `report/` - camera-ready project report (PDF + editable DOCX), Appendix A holds the public code link
- `poster/` - poster in three formats: editable PPTX, LaTeX source (+ compiled PDF) and high-resolution PNG
- `explainer/` - a plain-language walkthrough of the whole project (PDF + DOCX)
- `generators/` - the Python scripts that regenerate the report, explainer and poster from `artifacts/`

Penn-Fudan Pedestrian dataset, 170 images. All person instances are merged into one foreground mask. Reflections, mirror images and shadows are treated as background (assumption, stated in the problem definition).

## Experiments

| Name | Model | Augmentation | Loss | Distractors |
|---|---|---|---|---|
| baseline | U-Net | none | BCE | no |
| augmented | U-Net | mild (flip, zoom up to 10%, brightness) | BCE | no |
| advanced | U-Net | mild | BCE + 0.5 Dice | no |
| advanced_ablation | U-Net | mild | BCE + 0.0 Dice (must match augmented) | no |
| strong_aug | U-Net | strong (scale 0.6-1.4, rotation, colour, blur, noise) | BCE | no |
| distract | U-Net | strong | BCE | 30% of training samples |
| resnet34 | ResNet34-encoder U-Net (ImageNet) | strong | BCE | no |
| resnet34_distract | ResNet34-encoder U-Net | strong | BCE | 30% |
| deeplab_ft | DeepLabV3-ResNet50 (COCO/VOC) | strong | BCE | no |
| deeplab_zeroshot | DeepLabV3-ResNet50, person class, no training | none | none | no |

Plus three trivial baselines: all_background, mean_training_mask, pixel_random_forest.

## Commands (PowerShell or bash)

```
pip install -r requirements.txt
python scripts/download_dataset.py
python scripts/make_split.py --config config.yaml      # once only, never regenerate
.\scripts\run_all.ps1          # or: bash scripts/run_all.sh
```

Single run: `python -m person_segmentation.train --config config.yaml --experiment resnet34 --seed 1` (set `PYTHONPATH=src`).

## Outputs (artifacts/)

- `<run>/best.pt, history.csv, summary.json` per training run (best epoch chosen by validation IoU)
- `baselines_summary.csv`, `results_summary.csv`, `results_table.md/csv` (mean ± std over seeds)
- `per_image/<run>.csv` per-image IoU with Fudan/Penn source
- `paired_comparison.csv` paired bootstrap difference between two models
- `robustness_metrics.csv` clean vs each distractor, with distractor false-positive rate
- `overlays/<run>_worst.png` worst test images (red = false positive, blue = missed)

## Metrics

Pooled IoU (all test pixels together), mean per-image IoU with 95% bootstrap interval, precision, recall, F1. Distractor false-positive rate = person-predicted pixels inside the pasted distractor region divided by the region size. Distractors at test time use different parameter ranges from training.

## Rules

Test set is used once for reporting. Threshold is chosen on validation only. Do not regenerate `artifacts/splits.json`.

## Full walkthrough notebook

Open `Person_Segmentation_Full_Walkthrough.ipynb` in Jupyter or VS Code and select the `Python (Person Segmentation)` kernel. Install the project dependencies with `python -m pip install -r requirements.txt` if needed. The notebook explains the dataset, architectures, every experiment, metrics, and evaluation workflow. Its result cells read the current files under `artifacts/`; rerun them after the experiment queue and evaluation finish to display the final tables and figures. The notebook does not launch training.
