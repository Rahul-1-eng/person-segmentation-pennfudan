# Submission checklist (mirrors the 12-item list in the project guidelines)

Tick a box only after the file or number exists in the folder.

## Guideline checklist
- [x] Problem and metrics defined (report.md section 2: statement, hypotheses, success criteria)
- [ ] Literature and gap documented (table written; verify each citation before submission)
- [ ] Dataset source and license recorded (data/README.md written; licence is unconfirmed, say so)
- [x] Data audit and EDA completed (run make_figures.py, check eda.png)
- [x] Leak-free split and pipeline verified (artifacts/splits.json frozen)
- [x] Simple baseline established (run baselines.py)
- [x] Models tuned using validation only (3 seeds, 100 epochs, results_summary.csv)
- [x] Held-out evaluation completed (evaluate.py, results_table.md)
- [x] Results and failures interpreted (overlay.py, qualitative.png, fill report section 8)
- [x] Seeds, versions and commands recorded (artifacts/environment.txt from run_all)
- [ ] Code, report and slides organised (rebuild PDF and PPTX after the final run)
- [x] Limitations and references included

## Before submission
- [x] Full explanatory notebook created and wired to the project kernel (`Person_Segmentation_Full_Walkthrough.ipynb`)
- [x] Old v1 results moved to artifacts/v1
- [x] advanced_ablation reproduces augmented within noise
- [x] Real reflection photos (20 to 40) annotated and reported as a case study, or the claim removed from the title
- [ ] Title reverted to "Binary Person Segmentation on Penn-Fudan Pedestrians Using U-Net and Data Augmentation" if the robustness study is dropped
- [ ] Fresh clone run reproduces the tables
- [ ] Spare checkpoints deleted from the submission zip
