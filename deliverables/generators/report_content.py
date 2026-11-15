# -*- coding: utf-8 -*-
# Single source of truth for the project report (PDF + DOCX renderers read this).

TITLE = "Binary Person Segmentation on Penn-Fudan with U-Net, Augmentation and Synthetic Reflection/Shadow Distractors"
AUTHOR = "Rahul Kumar Sahoo"
ROLL = "Roll No. 2401CS09"
DEPT = "Department of Computer Science and Engineering"
INST = "Indian Institute of Technology Patna"
SUBTITLE = "Machine Learning Course Project Report"
GITHUB = "https://github.com/Rahul-1-eng/person-segmentation-pennfudan"

ABSTRACT = (
    "This report describes a binary person segmentation study on the Penn-Fudan Pedestrian dataset, "
    "which contains 170 street photographs with 345 labelled pedestrians. All person instances in an image "
    "are merged into a single foreground mask, so the task is to label every pixel as person or background. "
    "I train and compare three trivial baselines, a plain U-Net, U-Net variants with different augmentation "
    "and loss settings, a U-Net with an ImageNet-pretrained ResNet34 encoder, and a COCO-pretrained "
    "DeepLabV3-ResNet50 used both zero-shot and fine-tuned. Everything is run under one frozen 102/34/34 "
    "split, validation-only checkpoint and threshold selection, and a single test evaluation, repeated over "
    "three seeds. The pretrained encoders give the clearest gain: mean per-image test IoU rises from 0.6893 "
    "(plain U-Net) to 0.7895 (ResNet34 encoder) and 0.7973 (fine-tuned DeepLabV3), while mild augmentation "
    "adds a smaller but measurable 0.0271. I also paste six types of synthetic reflection- and shadow-like "
    "distractors into test backgrounds and measure how often a model calls those regions person. Distractor "
    "training cuts the false-positive rate of the ResNet34 U-Net from 0.1314 to 0.0273 while costing almost "
    "nothing on clean images. Code, splits and result files are public."
)

CHAPTERS = [
    {
        "title": "Introduction",
        "sections": [
            {
                "title": "Background and Motivation",
                "text": [
                    "Person segmentation is the task of assigning every pixel of an image to one of two "
                    "classes: person or background. It is a stricter request than object detection, which only "
                    "draws boxes, and it is the first step in applications like pedestrian analysis, image "
                    "editing, and privacy blurring, where we need the exact outline of a person rather than a "
                    "rectangle around them.",
                    "Most segmentation datasets that are easy to download are large. Penn-Fudan is not. It has "
                    "170 images, which is close to the regime where a model can memorise the training set and "
                    "still look good on a lucky test draw. That constraint shaped most of the decisions in this "
                    "project: a frozen split, model selection on validation only, several seeds instead of one, "
                    "and simple models that cannot hide behind billions of parameters.",
                    "A second motivation came from looking at failure cases of segmentation models in the "
                    "wild. A person standing near a glass door produces a reflection that looks almost exactly "
                    "like a person. A person walking in the sun drags a person-shaped shadow behind them. "
                    "Pixel-wise, a perfect mirror copy of a person is a person as far as any convolution can "
                    "tell; only context can separate the two. Penn-Fudan contains no reflections and no shadow "
                    "annotations, so the standard benchmark cannot tell us whether a model confuses the two. "
                    "Since no real data exists, I built synthetic distractors instead and measured the "
                    "confusion directly.",
                ],
            },
            {
                "title": "Problem Statement",
                "text": [
                    "Given an RGB street photograph, predict a binary mask of all pedestrians, training on the "
                    "102 images of the Penn-Fudan training split, such that test IoU clearly beats trivial "
                    "baselines; and measure, using synthetic distractors pasted into test backgrounds, how "
                    "often the model wrongly labels reflection- and shadow-like regions as person. Reflections "
                    "and shadows are treated as background by definition, an assumption I fixed before any "
                    "experiment."
                ],
            },
            {
                "title": "Objectives",
                "text": [
                    "1. Build a leak-free evaluation pipeline for the 170-image Penn-Fudan dataset with a "
                    "frozen 102/34/34 split that is created once and never regenerated.",
                    "2. Train and compare baselines and segmentation models under one protocol, over three "
                    "seeds, reporting mean and spread rather than a single lucky run.",
                    "3. Quantify how much a pretrained encoder helps on this small dataset compared to better "
                    "augmentation of a from-scratch model.",
                    "4. Build six synthetic distractor generators and a false-positive rate FP_D that measures "
                    "whether reflections and shadows are labelled as people.",
                    "5. Test whether distractor training reduces FP_D without hurting clean IoU.",
                ],
            },
            {
                "title": "Scope of the Work",
                "text": [
                    "The project covers binary segmentation only; instance identities are merged. Images are "
                    "resized to 256 x 256. The distractor study is synthetic: distractor regions are pasted by "
                    "a generator with fixed parameter ranges, and the test-time ranges differ from the "
                    "training-time ones so the model cannot simply learn the generator. No claim is made about "
                    "real photographs of reflections. All training ran on CPU, which limited model size and "
                    "the number of runs; 27 trainable runs (9 configurations x 3 seeds) plus 3 trivial "
                    "baselines and 1 zero-shot model were completed."
                ],
            },
            {
                "title": "Our Contribution",
                "text": [
                    "1. A frozen, leak-free 102/34/34 split of Penn-Fudan with a one-command pipeline that "
                    "regenerates every table and figure from raw images.",
                    "2. A single evaluation protocol over 27 multi-seed runs of 9 trainable configurations "
                    "plus 3 trivial baselines and 1 zero-shot reference, with per-image IoU and Fudan-versus-"
                    "Penn breakdowns.",
                    "3. Six synthetic distractor generators (floor reflection, framed mirror panel, cast "
                    "shadow, detached wall shadow, animal-shaped shadow, darkness patch) that modify only "
                    "background pixels, and a distractor false-positive rate FP_D computed inside those "
                    "regions.",
                    "4. Uncertainty reporting through per-image bootstrap intervals and a paired hierarchical "
                    "bootstrap over matched seeds and images for every model comparison.",
                    "5. A public repository with code, the frozen split, and all result files so every number "
                    "in this report can be traced to a CSV.",
                ],
            },
        ],
    },
    {
        "title": "Literature Review",
        "sections": [
            {
                "title": "Existing Approaches",
                "text": [
                    "The line of work relevant to this project starts with the dataset itself. Wang et al. "
                    "[1] introduced Penn-Fudan for combined pedestrian detection and segmentation, with "
                    "instance-level masks; pre-deep-learning, their pipeline is a good reference for how "
                    "little data used to be considered workable. U-Net [2] remains the default small-data "
                    "segmentation architecture: an encoder-decoder with skip connections that was originally "
                    "designed for biomedical microscopy and is usually trained from scratch. ResNet [3] "
                    "supplied the pretrained encoders that modern segmentation heads bolt on, and DeepLabV3 "
                    "[4] showed that atrous convolution with a small output stride captures large context, "
                    "which matters for full-body outlines. SegFormer [5] pushes the same idea with "
                    "transformers but is heavier than what a CPU budget allows here. On the augmentation side, "
                    "Shorten and Khoshgoftaar [7] survey how geometric and colour transforms behave for image "
                    "data, and V-Net [8] introduced the Dice loss term I use in one ablation. Closest in "
                    "spirit to the distractor study, MirrorNet [6] segments mirror regions themselves in "
                    "photos where mirrors are annotated, which is the opposite question: I ask whether a "
                    "person model calls the content inside a mirror a person. Table 1 summarises the studies "
                    "this project leans on.",
                ],
            },
            {
                "title": "Research Gaps",
                "text": [
                    "Three gaps stand out from this literature. First, standard person segmenters are trained "
                    "on corpora where reflections are rare and unlabelled, and Penn-Fudan contains none, so "
                    "reflection behaviour is untested by the usual benchmarks. Second, small-data studies "
                    "often report a single training run; without seeds and intervals it is impossible to tell "
                    "a real improvement from a lucky one. Third, papers that do study reflections segment the "
                    "mirror rather than the people seen in it, so there is no ready-made protocol for the "
                    "confusion I care about. This project fills these gaps in a deliberately modest way: a "
                    "synthetic stress test, a frozen protocol, and multi-seed reporting.",
                ],
            },
        ],
    },
    {
        "title": "Proposed Methodology",
        "sections": [
            {
                "title": "Dataset Description",
                "text": [
                    "Penn-Fudan Pedestrian [1] contains 170 RGB photographs, 96 from the University of "
                    "Pennsylvania and 74 from Fudan University, with 345 labelled pedestrians in total. Every "
                    "image contains at least one person, so the model never sees a person-free scene during "
                    "training or testing. Masks store instance identifiers; taking mask > 0 merges all "
                    "instances into one binary foreground mask. No licence statement was found on the source "
                    "page, so the dataset is not redistributed; the download script fetches it directly.",
                ],
            },
            {
                "title": "Data Preprocessing",
                "text": [
                    "Masks are binarised with mask > 0. Images and masks are resized to 256 x 256, bilinear "
                    "interpolation for images and nearest-neighbour for masks so label values stay binary. I "
                    "use plain squashing rather than letterboxing; 26 of the 170 images are wider than 1.5:1 "
                    "and get visibly distorted, which I discuss as a limitation. Models with pretrained "
                    "encoders apply their original ImageNet normalisation inside the model forward pass so "
                    "that the statistics the encoder was trained with are preserved."
                ],
            },
            {
                "title": "Exploratory Data Analysis (EDA)",
                "text": [
                    "Figure 1 shows the dataset overview. Foreground occupies a small fraction of pixels in a "
                    "typical image, so overall pixel accuracy would be misleading (predicting all background "
                    "scores high) and IoU is used as the primary metric. People per image ranges from 1 to "
                    "about 6, and most training images are Penn images but not by a large margin. The "
                    "validation set is about 65% Penn while training is about 55% Penn, because the split is "
                    "not stratified by source; the Fudan-versus-Penn breakdown in Section 4.6 controls for "
                    "this instead of pretending the imbalance away.",
                ],
                "figure": ("figures/eda.png", "Dataset overview: split sizes by source, foreground fraction, aspect ratios and people per image (training set)."),
            },
            {
                "title": "Proposed Model Architecture",
                "text": [
                    "Three families are compared. The plain U-Net has four down- and four up-sampling blocks "
                    "with batch normalisation and skip connections, 7.76 M parameters, trained from scratch. "
                    "The second family replaces the encoder with an ImageNet-pretrained ResNet34, keeping a "
                    "similar decoder, 24.52 M parameters. The third is DeepLabV3 with a ResNet50 backbone "
                    "pretrained on COCO (the 20 Pascal VOC categories include person), 42.00 M parameters. "
                    "The COCO head produces a person logit, which makes DeepLabV3 usable zero-shot, and I "
                    "also fine-tune it. Figure 2 shows the study pipeline: everything blue happens on the "
                    "102 training images, everything orange is evaluation, and the test set is touched once.",
                ],
                "figure": ("figures/architecture.png", "End-to-end pipeline. Blue blocks fit models on training data; peach blocks evaluate on validation and test."),
            },
            {
                "title": "Algorithm",
                "text": [
                    "Algorithm 1 states the full procedure. The two properties I care about are that no "
                    "decision that can leak the test set (threshold, checkpoint, epoch) is taken on test, and "
                    "that every comparison is computed on matched seeds so paired intervals are valid.",
                ],
                "algorithm": (
                    "Algorithm 1: Leak-free multi-seed person segmentation study\n"
                    "Input: images D; frozen split D_tr / D_val / D_te; configs C; seeds S = {20261103, 1, 2}\n"
                    "Output: per-model test metrics with uncertainty; robustness table\n"
                    "1. Binarse masks (mask > 0); resize images and masks to 256 x 256.\n"
                    "2. for c in C, s in S: train 100 epochs on augmented D_tr (+ distractors, p = 0.3);\n"
                    "   after each epoch compute validation IoU at t = 0.5; keep best.pt.\n"
                    "3. Choose threshold t* on D_val only; test once on D_te: IoU, P, R, F1, bootstrap 95% CI.\n"
                    "4. Paste 6 distractor types on D_te backgrounds (different parameter ranges) and report FP_D.\n"
                    "5. Report mean +/- std over seeds; paired hierarchical bootstrap differences between models.\n"
                ),
            },
            {
                "title": "Experimental Setup",
                "text": [
                    "Table 2 lists the runs. All learned models share 256 x 256 inputs, AdamW (learning rate "
                    "3e-4, weight decay 1e-5, fine-tuned DeepLab 1e-4), batch size 8, 100 epochs with a "
                    "cosine schedule, and seeds 20261103, 1 and 2. The loss is BCE everywhere except the "
                    "advanced configuration, L = (1 - w) BCE + w (1 - Dice) with w = 0.5, whose ablation sets "
                    "w = 0 to isolate implementation differences. Mild augmentation is horizontal flip, zoom "
                    "up to 10% and brightness; strong augmentation adds scale 0.6-1.4 with padding, rotation "
                    "up to 10 degrees, translation, colour jitter, blur and noise, with geometric transforms "
                    "applied to image and mask together and photometric ones to the image only. Distractors "
                    "are applied to 30% of training samples. Training ran on CPU; per-run device and training "
                    "time are recorded in each summary.json, and package versions in environment.txt.",
                ],
                "table": "configs",
            },
        ],
    },
    {
        "title": "Results and Discussion",
        "sections": [
            {
                "title": "Performance Evaluation Metrics",
                "text": [
                    "The primary metric is IoU. Pooled IoU concatenates all test pixels and favours images "
                    "with many people, so mean per-image IoU (averaging IoU over the 34 test images) is "
                    "reported next to it and used for all model comparisons. Precision, recall and F1 "
                    "complete the picture. Uncertainty comes from a 95% bootstrap interval over per-image "
                    "IoU, resampling images within each seed. For differences between two models I use a "
                    "paired hierarchical bootstrap that resamples matched seeds and then matched test images "
                    "within each seed, which respects the correlation between models evaluated on the same "
                    "images. FP_D is the fraction of pasted distractor region pixels predicted as person, "
                    "averaged over test images."
                ],
            },
            {
                "title": "Experimental Results",
                "text": [
                    "Table 3 is the main result. The trivial pixel-wise random forest reaches 0.2713 mean "
                    "per-image IoU, and both all-background and mean-mask baselines score 0, confirming the "
                    "task is non-trivial. Among from-scratch U-Nets, mild augmentation lifts mean per-image "
                    "IoU from 0.6893 to 0.7164; strong augmentation, BCE+Dice, and distractor training all "
                    "land in the same 0.71-0.73 band. The jump comes from pretraining: 0.7895 with a "
                    "ResNet34 encoder and 0.7973 for fine-tuned DeepLabV3, against 0.7567 for the same "
                    "DeepLabV3 used zero-shot. Pooled IoU ranks models the same way.",
                ],
                "table": "results",
            },
            {
                "title": "Comparison with Baselines and Existing Models",
                "text": [
                    "Table 4 reports paired differences with hierarchical bootstrap intervals. Two intervals "
                    "exclude zero: mild augmentation over the plain U-Net (+0.0271, [0.0086, 0.0446]) and the "
                    "pretrained ResNet34 encoder over the identically-augmented from-scratch U-Net (+0.0736, "
                    "[0.0500, 0.0982]). Everything else, including DeepLabV3 fine-tuning over the ResNet34 "
                    "U-Net (+0.0078, [-0.0076, 0.0261]), is not separable at three seeds. With only 34 test "
                    "images, one image is about 3% of the test set, so I read these intervals as descriptive "
                    "rather than as confirmed significance. Figure 3 compares all runs side by side.",
                ],
                "figure": ("figures/metric_comparison.png", "Test metrics for all runs: pooled IoU, mean per-image IoU, F1 and FP_D."),
                "table": "paired",
            },
            {
                "title": "Graphical Analysis",
                "text": [
                    "Figure 4 shows validation loss and validation IoU per epoch. The DeepLabV3 runs are "
                    "visibly the most stable across seeds and converge early (best epochs around 49-83 of "
                    "100), which matches their tiny seed spread of 0.0006 in validation IoU. The from-scratch "
                    "U-Nets are noisier: the baseline's best epoch lands anywhere between 36 and 48 across "
                    "seeds, and one baseline seed trains for almost three hours while another finishes in "
                    "82 minutes on CPU, which is itself a sign of unstable late training. Figure 5 places "
                    "representative test images from worst to best; the model handles well-separated "
                    "pedestrians cleanly and struggles on small or partially cut people.",
                ],
                "figure": ("figures/training_curves.png", "Validation loss (left) and validation IoU (right) per epoch across seeds."),
                "figure2": ("figures/qualitative.png", "Test images ranked from worst to best IoU under the augmented U-Net."),
            },
            {
                "title": "Ablation and Robustness",
                "text": [
                    "The Dice-weight ablation behaves as a protocol check should: advanced with w = 0.5 and "
                    "its w = 0 ablation differ by -0.0022 mean per-image IoU from the augmented run, i.e. "
                    "removing the Dice term changes nothing meaningful, and the small remaining gap is "
                    "implementation noise. Strong augmentation neither helps nor hurts the U-Net "
                    "(-0.0005 versus mild). Distractor training is the interesting ablation: on the ResNet34 "
                    "encoder it costs only 0.003 clean IoU (0.767 versus 0.770) and cuts mean FP_D from "
                    "0.1314 to 0.0273. Table 5 breaks FP_D down by distractor type. The mirror panel is the "
                    "hardest case for every model trained without distractors (FP_D 0.44-0.78, since a "
                    "framed mirror genuinely looks like a doorway), and distractor training reduces exactly "
                    "that case the most, from 0.6209 to 0.0714. Figure 6 shows both clean IoU and FP_D per "
                    "condition.",
                ],
                "table": "robustness",
                "figure": ("figures/robustness.png", "Clean IoU (top) and distractor false-positive rate FP_D (bottom) per model and distractor type."),
            },
            {
                "title": "Error Analysis",
                "text": [
                    "For the augmented U-Net, mean per-image IoU is 0.7485 on Fudan images and 0.6878 on Penn "
                    "images (difference 0.061). The worst test image is PennPed00042 with IoU 0.298; its "
                    "overlay in Figure 7 shows dense false positives around a partially visible pedestrian. "
                    "Inspecting the worst images revealed something the synthetic study could not have "
                    "shown: several of the biggest false-positive regions are people that exist in the photo "
                    "but are missing from the instance masks. In other words some errors are annotation "
                    "gaps, not model hallucinations, which is worth knowing before tuning anything.",
                ],
                "figure": ("figures/overlays_augmented_worst.png", "Worst test images for the augmented U-Net: red = false positive, blue = missed person pixels."),
            },
            {
                "title": "Discussion of Findings",
                "text": [
                    "Putting the results together: on a 170-image dataset, where the encoder was pretrained "
                    "matters far more than any augmentation recipe. The augmentation gain is real but small, "
                    "and the seed spread says three seeds are the minimum, not a luxury. Distractor training "
                    "works for its stated purpose at almost no clean cost, but the mechanism is contextual: "
                    "the model learns that mirror-like rectangles between frames are background, not that "
                    "reflections are unreal. A perfect pixel copy of a person in a mirror remains "
                    "indistinguishable from a person by construction, and no real-reflection photographs "
                    "were tested. The study measures a stress test honestly; it does not solve reflections."
                ],
            },
        ],
    },
    {
        "title": "Conclusion and Future Scope",
        "sections": [
            {
                "title": "Key Findings",
                "text": [
                    "1. Under one frozen, leak-free protocol, the pretrained encoders dominate: 0.7895 mean "
                    "per-image IoU for the ResNet34 U-Net and 0.7973 for fine-tuned DeepLabV3 versus 0.6893 "
                    "for the from-scratch U-Net, with the encoder gain (+0.0736) the largest and most robust "
                    "effect measured.",
                    "2. Mild augmentation gives a small but interval-supported gain of +0.0271; stronger "
                    "augmentation, BCE+Dice loss and distractor training do not change clean IoU meaningfully.",
                    "3. Distractor training on background-only synthetic distractors cuts the distractor "
                    "false-positive rate from 0.1314 to 0.0273, with the largest reduction exactly on the "
                    "hardest distractor (framed mirror panel, 0.6209 to 0.0714).",
                    "4. Some apparent false positives are unlabelled people in the ground truth; error "
                    "analysis before model tuning saved wasted effort.",
                    "5. All numbers, splits and plots regenerate from one public repository.",
                ],
            },
            {
                "title": "Limitations",
                "text": [
                    "The test set has 34 images, so single images move means by about 3% and all intervals "
                    "are wide. Three seeds give limited information about run-to-run spread, and the paired "
                    "intervals are not corrected for multiple comparisons. Every image contains at least one "
                    "person and all pedestrians are upright, so nothing transfers to person-free scenes or "
                    "unusual poses. Squash resizing distorts the 26 wide images; a letterbox variant was "
                    "planned but not completed. The distractor study is synthetic by construction: it does "
                    "not measure behaviour on real reflections, and a pixel-perfect mirror image is "
                    "indistinguishable from a person at the pixel level. Finally, person segmentation has "
                    "obvious surveillance misuse potential, and the data come from two campuses only, so no "
                    "fairness claim is made or implied."
                ],
            },
            {
                "title": "Future Research Directions",
                "text": [
                    "Natural next steps are: letterbox resizing evaluated against squash; a small real "
                    "reflection/shadow test set photographed on campus, to replace synthetic validation; "
                    "lighter pretrained backbones (MobileNet, EfficientNet) to see how much of the gain is "
                    "pretraining versus capacity; test-time augmentation, currently implemented but disabled; "
                    "and distractor training schedules that anneal the distractor probability instead of "
                    "fixing it at 0.3. On the evaluation side, stratifying the split by source campus and "
                    "correcting for multiple comparisons would tighten the conclusions."
                ],
            },
        ],
    },
]

REFERENCES = [
    "L. Wang, J. Shi, G. Song, I.-F. Shen. Object Detection Combining Recognition and Segmentation. ACCV, 2007.",
    "O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI, 2015.",
    "K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. CVPR, 2016.",
    "L.-C. Chen, G. Papandreou, F. Schroff, H. Adam. Rethinking Atrous Convolution for Semantic Image Segmentation. arXiv:1706.05587, 2017.",
    "E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, P. Luo. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS, 2021.",
    "X. Yang, Z. Lin, Z. Zhang, X. He, H. Zhang. Where Is My Mirror? IEEE ICCV, 2019.",
    "C. Shorten, T. M. Khoshgoftaar. A Survey on Image Data Augmentation for Deep Learning. Journal of Big Data, 2019.",
    "F. Milletari, N. Navab, S.-A. Ahmadi. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. IEEE 3DV, 2016.",
]

APPENDIX = {
    "title": "Appendix A: Implementation and Reproduction",
    "intro": (
        "All implementation code, the frozen split file and every result table behind this report are "
        "publicly available. The repository is public and can be accessed without an account:"
    ),
    "code_link": GITHUB,
    "layout": (
        "Repository layout. src/person_segmentation/ holds the pipeline modules: data.py (dataset and "
        "augmentation), model.py (U-Net, ResNet34-encoder U-Net), losses.py (BCE and BCE+Dice), "
        "distract.py (the six distractor generators), metrics.py (IoU, bootstrap intervals), train.py, "
        "evaluate.py, robustness.py and baselines.py. scripts/ contains download_dataset.py, make_split.py, "
        "run_all.sh / run_all.ps1 and figure/table builders. config.yaml fixes every hyperparameter."
    ),
    "commands": (
        "Commands to reproduce (Linux shell or PowerShell):\n"
        "  pip install -r requirements.txt\n"
        "  python scripts/download_dataset.py\n"
        "  python scripts/make_split.py --config config.yaml      # once only, never regenerate\n"
        "  bash scripts/run_all.sh                                # or .\\scripts\\run_all.ps1\n"
        "  python -m person_segmentation.train --config config.yaml --experiment resnet34 --seed 1   # single run\n"
    ),
    "mapping": (
        "Where each result comes from. Table 3 aggregates artifacts/results_summary.csv over seeds; "
        "Table 4 is artifacts/paired_comparison.csv; Table 5 and Figure 6 come from "
        "artifacts/robustness_metrics.csv; Figures 1, 2 and 4-7 are regenerated by scripts/make_figures.py "
        "into artifacts/figures/. The frozen split lives in artifacts/splits.json."
    ),
}

# ---------------- tables ----------------

TABLES = {
    "related": {
        "caption": "Table 1: Studies this project leans on, and the gap each one leaves open.",
        "header": ["Study", "Setting", "What it shows", "Gap addressed here"],
        "rows": [
            ["Wang et al. [1], ACCV 2007", "Penn-Fudan dataset", "Pedestrian detection with instance masks", "Pre-deep-learning; no reflection handling"],
            ["Ronneberger et al. [2], 2015", "Biomedical microscopy", "U-Net encoder-decoder with skips", "Trained from scratch; no pretrained encoder"],
            ["He et al. [3], CVPR 2016", "ImageNet", "Residual encoders", "Classification backbone; segmentation use untested on small data"],
            ["Chen et al. [4], 2017", "PASCAL VOC etc.", "DeepLabV3 atrous segmentation", "Person is one of many classes; reflection behaviour unreported"],
            ["Yang et al. [6], ICCV 2019", "MSD mirror dataset", "MirrorNet segments mirrors", "Segments the mirror, not people seen in it"],
            ["Shorten & Khoshgoftaar [7], 2019", "Survey", "Image augmentation techniques", "No guidance for 170-image person segmentation"],
        ],
    },
    "configs": {
        "caption": "Table 2: Run configurations under the shared training protocol.",
        "header": ["Run", "Model", "Augmentation", "Loss", "Distractors"],
        "rows": [
            ["all_background, mean_training_mask, pixel_random_forest", "trivial / classical", "none", "n/a", "no"],
            ["baseline", "U-Net", "none", "BCE", "no"],
            ["augmented", "U-Net", "mild", "BCE", "no"],
            ["strong_aug", "U-Net", "strong", "BCE", "no"],
            ["distract", "U-Net", "strong", "BCE", "30% of samples"],
            ["resnet34, resnet34_distract", "ResNet34-encoder U-Net", "strong", "BCE", "no / 30%"],
            ["deeplab_zeroshot, deeplab_ft", "DeepLabV3-ResNet50", "none / strong", "n/a / BCE", "no"],
            ["advanced, advanced_ablation", "U-Net", "mild", "BCE+Dice 0.5 / 0.0", "no"],
        ],
    },
    "results": {
        "caption": "Table 3: Test results at the validation-chosen threshold, mean +/- std over 3 seeds (best model rows shaded).",
        "header": ["Model", "Val IoU", "Test IoU (pooled)", "Test mean per-image IoU", "FP_D", "Params (M)"],
        "rows": [
            ["pixel random forest", "0.2326", "0.2645", "0.2713", "-", "-"],
            ["U-Net baseline", "0.6974 +/- 0.0158", "0.6564 +/- 0.0167", "0.6893 +/- 0.0146", "0.1465", "7.76"],
            ["U-Net + mild aug.", "0.7618 +/- 0.0139", "0.6831 +/- 0.0038", "0.7164 +/- 0.0015", "0.1470", "7.76"],
            ["U-Net + strong aug.", "0.7544 +/- 0.0168", "0.6888 +/- 0.0045", "0.7159 +/- 0.0050", "0.1787", "7.76"],
            ["U-Net + BCE+Dice (advanced)", "0.7641 +/- 0.0121", "0.6965 +/- 0.0076", "0.7252 +/- 0.0047", "0.1408", "7.76"],
            ["U-Net + distractors (distract)", "0.7539 +/- 0.0096", "0.6828 +/- 0.0067", "0.7138 +/- 0.0038", "0.1426", "7.76"],
            ["ResNet34-encoder U-Net", "0.8447 +/- 0.0028", "0.7695 +/- 0.0102", "0.7895 +/- 0.0094", "0.1314", "24.52"],
            ["ResNet34-encoder U-Net + distractors", "0.8456 +/- 0.0019", "0.7668 +/- 0.0097", "0.7903 +/- 0.0052", "0.0273", "24.52"],
            ["DeepLabV3 zero-shot (person class)", "0.7565", "0.7343", "0.7567", "0.1666", "-"],
            ["DeepLabV3 fine-tuned", "0.8342 +/- 0.0006", "0.7801 +/- 0.0054", "0.7973 +/- 0.0017", "0.1306", "42.00"],
        ],
    },
    "paired": {
        "caption": "Table 4: Paired differences in mean per-image IoU with hierarchical bootstrap 95% intervals (3 matched seeds, 34 images).",
        "header": ["Comparison", "Difference", "95% interval", "Interval excludes zero"],
        "rows": [
            ["augmented - baseline", "+0.0271", "[+0.0086, +0.0446]", "yes"],
            ["advanced - augmented", "+0.0088", "[-0.0023, +0.0199]", "no"],
            ["advanced ablation - augmented", "-0.0022", "[-0.0098, +0.0017]", "no"],
            ["strong_aug - augmented", "-0.0005", "[-0.0148, +0.0135]", "no"],
            ["distract - strong_aug", "-0.0021", "[-0.0138, +0.0096]", "no"],
            ["resnet34 - strong_aug", "+0.0736", "[+0.0500, +0.0982]", "yes"],
            ["resnet34_distract - resnet34", "+0.0008", "[-0.0109, +0.0137]", "no"],
            ["deeplab_ft - resnet34", "-0.0078", "[-0.0261, +0.0076]", "no"],
        ],
    },
    "robustness": {
        "caption": "Table 5: Distractor false-positive rate FP_D by distractor type and condition (mean over 3 seeds where applicable).",
        "header": ["Distractor type", "ResNet34 U-Net", "+ distractor training", "DeepLabV3 fine-tuned"],
        "rows": [
            ["floor reflection", "0.0520", "0.0025", "0.0226"],
            ["framed mirror panel", "0.6209", "0.0714", "0.6338"],
            ["cast shadow", "0.0183", "0.0143", "0.0140"],
            ["detached wall shadow", "0.0633", "0.0485", "0.0806"],
            ["animal-shaped shadow", "0.0063", "0.0049", "0.0011"],
            ["darkness patch", "0.0275", "0.0220", "0.0314"],
            ["mean over types", "0.1314", "0.0273", "0.1306"],
        ],
    },
}
