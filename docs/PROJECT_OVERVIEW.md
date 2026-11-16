# Project overview: person segmentation on Penn-Fudan

Machine Learning course project, IIT Patna (Rahul Kumar Sahoo, 2401CS09).

## Key results

| Result | Value |
|---|---|
| Best mean per-image IoU (fine-tuned DeepLabV3) | 0.7973 |
| Gain from ImageNet-pretrained encoder over from-scratch U-Net | +0.0736 |
| Distractor false-positive rate, before to after distractor training | 0.1314 to 0.0273 |
| Training runs | 27 runs (9 configurations x 3 seeds) |
| Data | 170 images, frozen 102 / 34 / 34 split |

The live demo runs the zero-shot DeepLabV3-ResNet50 reference model (COCO person class), scored 0.7567 mean per-image IoU on Penn-Fudan.

## The 7 Ws

- Who: Rahul Kumar Sahoo (2401CS09), machine learning course, IIT Patna.
- What: models that mark every pixel of a photo as person or not-person, compared under one fixed, leak-free protocol.
- Why: to find what helps on a small dataset (pretraining, augmentation, loss) and whether models call reflections and shadows a person.
- Where: Penn-Fudan Pedestrian dataset, 170 images. Code, results and website are in this repository.
- Which: plain U-Net, U-Net variants, ResNet34-encoder U-Net, COCO-pretrained DeepLabV3 (zero-shot and fine-tuned).
- How: nine configurations, three seeds each, IoU with paired bootstrap intervals, then six synthetic reflection and shadow distractors.
- When to read what: explainer first, then the report, then the poster.

## Findings, stated carefully

- Pretraining beats recipe tweaks: +0.0736 IoU from the ImageNet-pretrained ResNet34 encoder.
- Distractor training works, cheaply: false-positive rate fell from 0.1314 to 0.0273.
- Some errors belong to the dataset: the worst test image hides a partially visible pedestrian, and several false positives are people missing from the masks.
- Not claimed: 34 test images and 3 seeds keep intervals wide, and the distractor study is a synthetic stress test, not a solution to real reflections.
