# -*- coding: utf-8 -*-
"""Layman explanation document content."""

E_TITLE = "Person Segmentation, Explained From Zero"
E_SUB = "A plain-language companion to the project report - what was done, why, and what every number means"
E_AUTHOR = "Companion to the report by Rahul Kumar Sahoo (2401CS09), IIT Patna"
E_GITHUB = "https://github.com/Rahul-1-eng/person-segmentation-pennfudan"

INTRO = (
    "This document explains the whole project the way I would explain it to a friend who has never "
    "taken a machine learning course. No formula is needed to read it. If you read it top to bottom, "
    "you will understand what the project does, why each step exists, what the results mean, and you "
    "will be able to answer questions about it in a viva or an interview. Where a term is unavoidable, "
    "it is explained the first time it appears, and there is a glossary at the end."
)

SECTIONS = [
    {
        "title": "The one-paragraph version",
        "text": [
            "The project teaches a computer to look at a street photo and paint every person in it, "
            "pixel by pixel. The dataset is tiny (170 photos), so the interesting question is not just "
            "'how well can we do it' but 'how do we make sure a good score is real and not luck'. To "
            "answer that, everything was trained three times with different random starts, on a split of "
            "the data that never changes, and the test photos were looked at exactly once at the end. A "
            "second question the project asks is fun: if a photo contains a person's reflection in a "
            "mirror or their shadow on the ground, does the model call that a person too? To test it, "
            "fake reflections and shadows were pasted into photos and the mistakes were counted.",
        ],
    },
    {
        "title": "What the data looks like",
        "text": [
            "The Penn-Fudan dataset is 170 photos of people walking on two university campuses (one in "
            "the USA, one in China). Every person in every photo has a 'mask': a second image, the same "
            "size, that is white where a person stands and black everywhere else. Think of it as the "
            "answer sheet. The model's job is to produce that answer sheet on its own.",
            "There are 345 people across the photos, and every photo has at least one person. The photos "
            "were split into three piles: 102 for training (the model learns from these), 34 for "
            "validation (used while choosing settings), and 34 for testing (touched only once, at the "
            "very end). Why three piles? Because if you grade a student on the same questions they "
            "practised, the grade is meaningless. Validation is the practice test, test is the real exam.",
        ],
    },
    {
        "title": "What the model actually is",
        "text": [
            "The main model is called U-Net. Picture a funnel: the left half squeezes the photo down, "
            "learning 'this patch is an edge', 'this region is a leg', and the right half expands it back "
            "to full size, deciding person/background for every pixel. The clever part of U-Net is the "
            "'skip connections': while squeezing, the funnel keeps clean copies of early details and "
            "hands them to the expanding side, so sharp edges survive. Without them, answers come out "
            "blurry.",
            "Three versions were compared. A plain U-Net whose funnel is trained from random weights. A "
            "U-Net whose funnel was already trained on a huge image collection (ImageNet) and only "
            "needed adapting - like hiring a cook who already knows 200 dishes and teaching them one "
            "more. And DeepLabV3, a bigger, well-known model that was already trained to find people in "
            "COCO photos, used both as-is and after extra training on our 102 photos.",
        ],
    },
    {
        "title": "The two experiments, in plain words",
        "text": [
            "Experiment 1 - what helps most on small data? Candidates: doing nothing (baseline), showing "
            "the model slightly edited copies of photos during training (flips, small zooms, brightness "
            "changes - this is augmentation), heavier editing, a different scoring rule during learning "
            "(BCE versus BCE+Dice), and pretrained funnels. Each candidate was trained 3 times with "
            "different random seeds. A seed is just the dice roll that starts the training; changing it "
            "changes the final score a little, so one run tells you almost nothing about which method is "
            "better. That is why 3 runs and an average.",
            "Experiment 2 - do models see ghosts? Six kinds of fake 'person-like' things were pasted "
            "into test photos, always on background areas so the true answer sheet stays the same: floor "
            "reflections, framed mirror panels, a person's cast shadow, a detached shadow on a wall, an "
            "animal-shaped shadow, and a dark patch. The fraction of the pasted area that the model "
            "wrongly calls 'person' is the distractor false-positive rate, FP_D. One model was also "
            "trained while being shown such fakes (on training photos only). It learned to stop calling "
            "mirrors 'people' - FP_D dropped from 0.1314 to 0.0273 - while barely changing its score on "
            "clean photos.",
        ],
    },
    {
        "title": "How to read the scoreboard (the metrics)",
        "text": [
            "IoU (Intersection over Union) is the main score. Lay it out as two paper cut-outs of the "
            "person - the model's and the truth. IoU asks: how much do they overlap, divided by how much "
            "space they cover together? Identical = 1.0, nothing in common = 0. Everything else is in "
            "between. Our numbers sit around 0.69-0.80, which is respectable for such a small dataset.",
            "Two flavours are reported. Pooled IoU stacks all test photos into one giant pile of pixels "
            "and computes one IoU - big crowds dominate. Mean per-image IoU scores each photo "
            "separately and averages - every photo counts the same. Comparisons use the second flavour. "
            "Precision = of the pixels I called person, how many really were. Recall = of the real "
            "person pixels, how many I caught. F1 balances the two.",
            "The +/- numbers after a score (like 0.7895 +/- 0.0094) are just the spread across the three "
            "runs: small spread means stable, large spread means the dice roll mattered. The "
            "[low, high] intervals in the comparison table are bootstrap intervals - a way of asking 'if "
            "we had slightly different test photos, would the conclusion change?' by repeatedly "
            "resampling the photos we have. If the interval for a difference crosses zero, the honest "
            "answer is 'we cannot tell these two apart with the data we have'.",
        ],
    },
    {
        "title": "The results, translated",
        "text": [
            "First, the task is real: a simple pixel-by-pixel classical method scores 0.27, and "
            "everything neural scores far higher. Second, on 170 photos, pretraining is the king: the "
            "ImageNet funnel added +0.074 over an identically-trained from-scratch U-Net - about ten "
            "times the gain from mild augmentation (+0.027). Third, the big fancy model (fine-tuned "
            "DeepLabV3, 0.797) is only slightly ahead of the ResNet34 U-Net (0.790), and the interval "
            "says we cannot honestly claim it is better. Fourth, augmentation and loss tweaks land in "
            "the same band; only the augmentation gain is distinguishable from noise, and it is small.",
            "Fifth - the distractor story. Models that never saw fake reflections call most of a framed "
            "mirror 'person' (FP_D up to 0.62 for the ResNet34 U-Net). After distractor training, 0.07. "
            "The price on clean photos: -0.003 IoU, essentially nothing. But note the boundary: a "
            "pixel-perfect mirror image of a person is a person at the pixel level; no model can reject "
            "it from pixels alone, only from context. The experiment measures context-learning, not "
            "magic.",
        ],
    },
    {
        "title": "Mistakes worth understanding",
        "text": [
            "The worst test photo for the augmented U-Net (IoU 0.298) hides a partially visible person. "
            "When the biggest errors were inspected, several 'false positives' turned out to be people "
            "who exist in the photo but are missing from the answer sheet - annotation gaps. Knowing "
            "this changed nothing about the model and everything about the interpretation: some errors "
            "belong to the dataset, not the algorithm. Always look at mistakes before tuning.",
            "Also, Fudan photos score about 0.06 higher than Penn photos across models. The validation "
            "pile has more Penn photos than training does, which makes the split slightly unfair in a "
            "known direction; reporting the two campuses separately keeps that honest.",
        ],
    },
    {
        "title": "What this project does NOT claim",
        "text": [
            "It does not claim the models handle real reflections - only the six synthetic fakes were "
            "tested. It does not claim DeepLabV3 beats the ResNet34 U-Net. It does not transfer to "
            "photos with no people (the training set never had one) or to people in unusual poses. "
            "Thirty-four test photos and three seeds are small; all intervals should be read with that "
            "in mind. And person segmentation has real surveillance uses, which is worth saying out "
            "loud.",
        ],
    },
    {
        "title": "Reproduce it yourself",
        "text": [
            "Everything is one public repository: " + E_GITHUB,
            "Three commands set it up: install the requirements, download the dataset, and make the "
            "frozen split once. Then one script trains every configuration and writes every table and "
            "figure into an artifacts folder. Single model run, if you just want to try one: train with "
            "--experiment resnet34 --seed 1. The website built alongside this project lets you upload "
            "any photo and watch the segmentation happen live.",
        ],
    },
    {
        "title": "Viva survival pack: twelve questions you will probably be asked",
        "text": [
            "Q1. Why binary and not instance masks? A: The task definition fixes reflections and shadows "
            "as background and merges all people; instance separation is out of scope.",
            "Q2. Why is the split frozen? A: If the split changes, scores across runs are no longer "
            "comparable - and a lucky split is a form of leakage.",
            "Q3. Why 3 seeds? A: Training is random. Three runs give a mean and a spread; one run gives "
            "an anecdote.",
            "Q4. Why is IoU the metric and not accuracy? A: People cover a small part of each photo, so "
            "predicting 'all background' already scores high accuracy. IoU ignores that trick.",
            "Q5. Pooled vs mean per-image IoU? A: Pooled weights crowded photos more; mean treats photos "
            "equally. Comparisons use mean.",
            "Q6. What is the threshold and where is it chosen? A: The probability cut-off that turns "
            "soft predictions into a mask. Chosen on validation only, test is evaluated once.",
            "Q7. Why does pretraining help so much? A: The encoder already knows edges, textures and "
            "shapes from millions of images; 102 photos only need to teach 'person outlines here'.",
            "Q8. Why did BCE+Dice not help? A: The ablation that removes Dice scores the same within "
            "noise; on this data the loss tweak is not the bottleneck.",
            "Q9. What is a paired bootstrap? A: Resample matched seeds and matched photos, recompute the "
            "difference between two models each time, and look at the interval of those differences. "
            "'Paired' because both models see the same photos.",
            "Q10. What is FP_D? A: Of the pixels we artificially pasted (fake reflection/shadow), the "
            "fraction predicted as person.",
            "Q11. Why can a model never be perfect on mirror images? A: A perfect mirror copy has "
            "identical pixels; only context (frame, floor, geometry) can betray it.",
            "Q12. Biggest limitation? A: 34 test images and 3 seeds - every interval is wide, and no "
            "real-reflection photos were tested.",
        ],
    },
    {
        "title": "Glossary",
        "text": [
            "Mask - the answer sheet: white where person, black elsewhere. "
            "Epoch - one full pass over the training photos. "
            "Batch - how many photos the model looks at before updating itself (here 8). "
            "Seed - the starting dice roll of a training run. "
            "Augmentation - random safe edits (flip, zoom, brightness) that multiply the training data "
            "for free. "
            "Distractor - a synthetic reflection- or shadow-like patch pasted on background areas. "
            "Checkpoint - a saved snapshot of the model; 'best.pt' is the best validation snapshot. "
            "Zero-shot - using a pretrained model with no extra training. "
            "Fine-tuning - continuing to train a pretrained model on new data. "
            "Bootstrap CI - an uncertainty interval built by resampling the data you have.",
        ],
    },
]
