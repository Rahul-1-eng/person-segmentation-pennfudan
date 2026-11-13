# Study guide: what is in this project (plain words)

**What the project does.** A computer looks at a street photo and colours every pixel white if it belongs to a person and black if not. We train several models, measure them fairly on photos they never saw, and test whether they get fooled by reflections and shadows.

## Read the files in this order (the diya order)

**Oil: the things everything else burns on**
1. `config.yaml`: all the settings (image size, epochs, augmentation strength).
2. `artifacts/splits.json`: which images are train, validation or test. Never regenerate it.
3. `src/person_segmentation/data.py`: loads images, resizes, augments.
4. `src/person_segmentation/metrics.py`: IoU, precision, recall, bootstrap intervals.

**Wick: the part that learns**
5. `model.py`: the three models (plain U-Net, ResNet34 U-Net, DeepLab).
6. `losses.py`: how mistakes are scored during training.
7. `experiments.py`: the table of all experiments in one place.
8. `train.py`: the training loop; keeps the best epoch by validation IoU.
9. `baselines.py`: dumb models (all background, average mask, pixel forest) to beat.

**Flame: what people see**
10. `evaluate.py`: scores every trained model on the test set.
11. `distract.py` and `robustness.py`: paste fake reflections and shadows and check whether the model calls them people.
12. `scripts/make_tables.py`, `overlay.py`, `make_figures.py`: tables and pictures.
13. `report.md` and `slides.md`, built into the PDF and PPTX by `create_report_pdf.py` and `create_presentation.py`.

## Words
- **Augmentation:** training photos are randomly flipped, zoomed, rotated, recoloured so the model sees more variety.
- **IoU:** overlap divided by union of predicted and true person area. 1 is perfect.
- **Validation vs test:** validation is for choosing settings; test is the final exam, used once.
- **Seed:** a number that makes randomness repeatable. Three seeds show how much results wobble.
- **Distractor:** a fake reflection or shadow pasted onto background. The correct answer stays "background".
- **Distractor false-positive rate:** how much of the pasted fake the model wrongly calls person.

## What each v1 file was, and was it necessary
Necessary: config, data, model, losses, metrics, train, evaluate, splits.json. Replaced: shadow.py, tune_threshold.py, error_analysis.py (evaluate.py now does their jobs). Removed: pyproject.toml (duplicated requirements.txt), the dated PDF copy, viva_guide.md and LEARN_FROM_ZERO.md (merged into this file).
