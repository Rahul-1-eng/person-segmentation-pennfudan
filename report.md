# Binary Person Segmentation on Penn-Fudan with U-Net, Augmentation and Synthetic Reflection/Shadow Distractors

**Student:** Rahul Kumar Sahoo - 2401CS09
**Course:** Machine Learning, IIT Patna
**Dataset:** Penn-Fudan Pedestrian (170 images)
**Task:** Merge all person instances into one foreground mask

## 1. Abstract and contributions

This project builds a reproducible pipeline for binary person segmentation on a small dataset. It compares trivial baselines, a plain U-Net, stronger augmentation, a pretrained-encoder U-Net and a pretrained DeepLabV3 reference under one frozen split and one metric definition. It also adds a robustness study in which reflections and shadows of people (and animal-shaped shadows) are pasted into test images while the ground-truth mask stays unchanged.

Contributions:

- A frozen, leak-free 102/34/34 split and a one-command pipeline that regenerates every table and figure.
- Per-image IoU, a Fudan versus Penn breakdown, multi-seed mean and standard deviation, and paired intervals that resample matched seeds and test images.
- A distractor generator and a distractor false-positive rate that measure whether a model labels reflections and shadows as people.

## 2. Introduction and problem statement

Person segmentation assigns every pixel to background or person. In this dataset only the physical pedestrian is foreground. Reflections in mirrors, windows and puddles, and cast shadows, are treated as background (an assumption made before experiments).

**Problem statement.** Given an RGB street photograph, predict a binary person mask for pedestrians in Penn-Fudan, trained on 102 images, achieving test IoU above the trivial baselines and a distractor false-positive rate that does not rise when reflections or shadows are added.

**Hypotheses and success criteria (fixed before the v2 runs).**

- H1: mild augmentation raises mean per-image IoU over the plain U-Net. The comparison uses a paired hierarchical bootstrap over matched seeds and test images; with three seeds, its interval is exploratory.
- H2: a pretrained ResNet34 encoder raises mean per-image IoU over the U-Net trained with the same strong augmentation. The same paired interval is reported and interpreted cautiously because there are three seeds.
- H3: training with distractors lowers the distractor false-positive rate without reducing clean IoU by more than 0.02.

Primary metric: IoU. Pooled IoU counts all test pixels together and favours images with more foreground, so mean per-image IoU is reported next to it.

## 3. Related work and gap

| Study | Dataset | Method | Evaluation | Gap / limitation |
|---|---|---|---|---|
| Wang et al., ACCV 2007 | Penn-Fudan (source of this dataset) | Detection combined with segmentation | Pedestrian detection and segmentation | Pre-deep-learning; no reflection handling |
| Ronneberger et al., MICCAI 2015 | Biomedical microscopy | U-Net encoder-decoder with skip connections | Segmentation overlap | Trained from scratch on small data; no pretrained encoder |
| Chen et al., 2017 | PASCAL VOC and others | DeepLabV3, atrous convolution | mIoU | Person is one of many classes; reflection behaviour not reported |
| Xie et al., NeurIPS 2021 | ADE20K, Cityscapes | SegFormer, transformer encoder, MLP decoder | mIoU | Heavier; not evaluated on mirror distractors |
| Yang et al., ICCV 2019 | MSD, 4,018 images with mirrors | MirrorNet, mirror segmentation | Mirror IoU | Segments the mirror, not people seen in it |

**Gap.** Standard person segmenters are trained on images where reflections are rare. Penn-Fudan has none. This project tests, using synthetic distractors, whether models label reflection- and shadow-like regions as foreground. It does not claim to solve real-world reflections.

## 4. Dataset and quality audit

Source: Penn-Fudan Pedestrian database (Wang et al., ACCV 2007). 170 images, 345 labelled pedestrians, from Fudan University (Fudan files) and University of Pennsylvania (Penn files). Every image contains at least one person, so the model never sees a person-free scene. No licence statement was found on the source page, so images are not redistributed.

Masks hold instance IDs; `mask > 0` merges them. Images are resized to 256 x 256 (bilinear for images, nearest for masks). 26 of 170 images are wider than 1.5:1, so squashing distorts them; a letterbox option is evaluated.

![Dataset overview: split sizes by source, foreground fraction, aspect ratio and people per image (training set only).](artifacts/figures/eda.png)

Interpretation: foreground is a small fraction of each image, so accuracy would be misleading and IoU is used. Validation is about 65% Penn while training is about 55% Penn, so the split is not stratified by source; the Fudan versus Penn breakdown below controls for this.

## 5. Split protocol and methodology

The split is created once with seed 20261103, saved in `artifacts/splits.json` and never regenerated: 102 train, 34 validation, 34 test. Augmentation and distractors are applied to training images only. The decision threshold is chosen on validation and then applied once to test.

![Pipeline architecture. Blue blocks are training, orange blocks are inference and evaluation.](artifacts/figures/architecture.png)

**Algorithm 1: Person segmentation study**

```
Input: images and masks D, frozen split (Dtr, Dval, Dte), model families M, seeds S
Output: per-model test metrics with uncertainty, robustness table
1. Load the saved split; merge instance masks into binary masks.
2. For each model m in M and seed s in S:
3.    Train on Dtr with augmentation (and distractors for the distractor models).
4.    After each epoch compute validation IoU; keep the checkpoint with the highest value.
5. For each trained model choose threshold t in {0.20,...,0.80} that maximises validation IoU.
6. Evaluate each model once on Dte at t = 0.5 and at the validation-chosen t.
7. Compute pooled IoU, mean per-image IoU, bootstrap 95% interval, Fudan and Penn means.
8. Evaluate clean Dte and Dte with each distractor type; report IoU and
   distractor false-positive rate = predicted person pixels in D / |D|.
9. Aggregate seeds as mean and standard deviation; compare matched model seeds with a hierarchical paired bootstrap that resamples seeds, then test images within each seed.
```

**Models.** Plain U-Net (4 down, 4 up blocks, batch norm). ResNet34-encoder U-Net with ImageNet weights and ImageNet normalisation inside the model. DeepLabV3-ResNet50 whose COCO subset weights use the 20 Pascal VOC categories (person is one); its person logit is used zero-shot as a pretrained reference, and also fine-tuned.

**Loss.** Binary cross-entropy; the BCE + Dice experiment uses `L = (1-w) BCE + w (1 - Dice)`.

**Augmentation.** Mild: flip, zoom 0.9 to 1.0, brightness. Strong: flip, scale 0.6 to 1.4 with padding, rotation up to 10 degrees, translation, colour jitter, blur, noise. Geometric changes apply to image and mask together; colour changes apply to the image only.

**Distractors.** Six generators paste content only onto background pixels, so the mask never changes: floor reflection, framed mirror panel, cast shadow, detached wall shadow, crude animal-shaped shadow, and a darkness patch. Training uses them on 30% of samples. Test uses stronger and different parameter ranges so the model cannot simply learn the generator.

![Synthetic distractors on one image. The yellow outline marks the region that must stay background.](artifacts/figures/distractors.png)

## 6. Experimental setup and baselines

| Run | Model | Augmentation | Loss | Distractors |
|---|---|---|---|---|
| all_background, mean_training_mask, pixel_random_forest | trivial / classical | none | n/a | no |
| baseline | U-Net | none | BCE | no |
| augmented | U-Net | mild | BCE | no |
| strong_aug | U-Net | strong | BCE | no |
| distract | U-Net | strong | BCE | 30% |
| resnet34, resnet34_distract | ResNet34 U-Net | strong | BCE | no / 30% |
| deeplab_zeroshot, deeplab_ft | DeepLabV3-ResNet50 | none / strong | n/a / BCE | no |
| advanced, advanced_ablation | U-Net | mild | BCE+Dice 0.5 / 0.0 | no |

All learned models share: 256 x 256 input, AdamW (learning rate 3e-4, weight decay 1e-5, DeepLab fine-tune 1e-4), 100 epochs with cosine schedule, batch size 8, seeds 20261103, 1 and 2. The checkpoint is selected by validation IoU. Software versions are in `artifacts/environment.txt`; each run's device, time and parameter count are in its `summary.json`.

## 7. Results

### 7.1 Preliminary v1 results (40 epochs, one seed, kept for the record)

| Model | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| Baseline U-Net | 0.8320 | 0.7397 | 0.7831 | 0.6436 |
| Augmented U-Net | 0.8274 | 0.7591 | 0.7918 | 0.6553 |
| Advanced (BCE + Dice) | 0.7739 | 0.6720 | 0.7194 | 0.5617 |
| Advanced ablation (Dice 0) | 0.6245 | 0.7369 | 0.6761 | 0.5106 |

These runs are not conclusive. The baseline and augmented models reached their best validation epoch at 36 and 37 of 40, so they were still improving. The advanced runs peaked at epochs 8 and 5; the ablation was trained for only 10 epochs. The augmentation gain of 0.012 IoU is smaller than the width of the per-image confidence interval (about 0.65 to 0.75 around a mean of 0.70). v1 therefore shows no reliable difference.

### 7.2 Final results (v2 protocol)

{{TABLE:artifacts/results_table.md}}

![Test metrics for all runs.](artifacts/metric_comparison.png)

![Validation loss and validation IoU per epoch.](artifacts/figures/training_curves.png)

Paired differences (mean per-image IoU, hierarchical 95% bootstrap interval; an interval excluding zero is descriptive, not a corrected significance test across all comparisons):

{{TABLE:artifacts/paired_comparison.md}}

## 8. Error analysis, robustness and ablations

![Representative test images from worst to best.](artifacts/figures/qualitative.png)

![Worst test images with false positives in red and missed person pixels in blue.](artifacts/overlays/augmented_worst.png)

![Clean IoU and distractor false-positive rate for each model and distractor type.](artifacts/figures/robustness.png)

Across the completed trainable experiments, **deeplab_ft** had the highest mean per-image test IoU (0.797; 3 seed run(s)). The table reports every model and the run-to-run spread. For mild augmentation versus baseline, the difference in mean per-image test IoU was 0.027 (augmented minus baseline); see the paired interval table for uncertainty. For ResNet34 versus U-Net under strong augmentation, the difference in mean per-image test IoU was 0.074 (resnet34 minus strong_aug); see the paired interval table for uncertainty. The Dice-weight-zero ablation differed from augmented by -0.002 mean per-image IoU; this is a protocol check, not an independent model claim. For augmented U-Net, mean per-image IoU averaged 0.749 on Fudan images and 0.688 on Penn images (Fudan minus Penn: 0.061). The lowest-IoU augmented test image was PennPed00042 (IoU 0.298); the overlay figure shows its false positives and missed foreground pixels. Clean-test IoU for distract was 0.683 versus 0.689 for strong_aug (difference -0.006); the pre-set maximum allowed drop of 0.020 was met. Across generated distractor regions, the mean false-positive pixel rate was 0.143 for distract and 0.179 for strong_aug. These are synthetic stress-test measurements, not evidence of performance on real reflections or shadows. Clean-test IoU for resnet34_distract was 0.767 versus 0.770 for resnet34 (difference -0.003); the pre-set maximum allowed drop of 0.020 was met. Across generated distractor regions, the mean false-positive pixel rate was 0.027 for resnet34_distract and 0.131 for resnet34. These are synthetic stress-test measurements, not evidence of performance on real reflections or shadows. There are only three training seeds, so seed-level variation and bootstrap intervals remain uncertain.

## 9. Limitations, ethics and conclusion

- 34 test images: one image is about 3% of the test set, so intervals are wide.
- Three training seeds provide limited information about run-to-run variation; the paired bootstrap intervals are descriptive and are not adjusted for the multiple model comparisons.
- No person-free images and only upright pedestrians; results do not transfer to other scenes or poses.
- A perfect mirror copy of a person is identical to a person at pixel level; the model can only use context such as frame edges and geometry. The distractor study is a stress test, not a solution.
- No real-reflection photos are included in this study. The robustness findings apply to the synthetic distractors and need validation on real reflections and shadows.
- Person segmentation can be misused for surveillance. No claim is made about demographic fairness; the data come from two campuses only.

**Conclusion.** The final multi-seed comparison identifies deeplab_ft as the highest-scoring trainable configuration by mean per-image IoU (0.797). Differences and synthetic distractor results are described in sections 7 and 8. The small test set, three training seeds, and synthetic-only distractors limit the strength and scope of these findings.

## 10. References and reproducibility notes

1. L. Wang, J. Shi, G. Song, I. Shen. Object Detection Combining Recognition and Segmentation. ACCV, 2007.
2. O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI, 2015.
3. K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. CVPR, 2016.
4. L.-C. Chen, G. Papandreou, F. Schroff, H. Adam. Rethinking Atrous Convolution for Semantic Image Segmentation. arXiv:1706.05587, 2017.
5. E. Xie et al. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS, 2021.
6. X. Yang et al. Where Is My Mirror? ICCV, 2019.
7. C. Shorten, T. Khoshgoftaar. A survey on Image Data Augmentation for Deep Learning. Journal of Big Data, 2019.
8. F. Milletari, N. Navab, S. Ahmadi. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 3DV, 2016.

Reproduce: `pip install -r requirements.txt`, `python scripts/download_dataset.py`, `python scripts/make_split.py --config config.yaml` (once), then `scripts/run_all.ps1` or `bash scripts/run_all.sh`. All tables and figures are regenerated into `artifacts/`.
