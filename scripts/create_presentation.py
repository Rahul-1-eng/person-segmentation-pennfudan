import csv
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from PIL import Image

ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "Person_Segmentation_Viva_Presentation.pptx"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
STUDENT = "Rahul Kumar Sahoo"
ROLL = "2401CS09"
TITLE = "Binary Person Segmentation on Penn-Fudan with U-Net, Augmentation and Synthetic Reflection/Shadow Distractors"
NAVY = RGBColor(20, 54, 92)


def add_title(slide, title):
    box = slide.shapes.add_textbox(Inches(0.55), Inches(0.25), Inches(12.2), Inches(0.7))
    frame = box.text_frame
    frame.text = title
    frame.paragraphs[0].font.size = Pt(28)
    frame.paragraphs[0].font.bold = True
    frame.paragraphs[0].font.color.rgb = NAVY


def add_bullets(slide, bullets, left=0.8, top=1.2, width=11.8, height=5.6, size=20):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    for index, item in enumerate(bullets):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = "\u2022 " + item
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = RGBColor(35, 35, 35)
        paragraph.space_after = Pt(8)


def add_footer(slide, number):
    box = slide.shapes.add_textbox(Inches(0.6), Inches(7.05), Inches(12), Inches(0.25))
    frame = box.text_frame
    frame.text = f"Person Segmentation | Penn-Fudan | {number}"
    frame.paragraphs[0].font.size = Pt(9)
    frame.paragraphs[0].font.color.rgb = RGBColor(120, 120, 120)
    frame.paragraphs[0].alignment = PP_ALIGN.RIGHT


def add_image(slide, path, left, top, max_width, max_height, caption):
    if not path.exists():
        box = slide.shapes.add_textbox(Inches(left), Inches(top + 1), Inches(max_width), Inches(1))
        box.text_frame.text = f"[Figure pending: run the pipeline to create {path.name}]"
        box.text_frame.paragraphs[0].font.size = Pt(16)
        box.text_frame.paragraphs[0].font.italic = True
        return
    with Image.open(path) as image:
        ratio = image.width / image.height
    width = max_width
    height = width / ratio
    if height > max_height:
        height = max_height
        width = height * ratio
    slide.shapes.add_picture(str(path), Inches(left + (max_width - width) / 2), Inches(top), Inches(width), Inches(height))
    caption_box = slide.shapes.add_textbox(Inches(left), Inches(top + height + 0.05), Inches(max_width), Inches(0.5))
    caption_box.text_frame.word_wrap = True
    caption_box.text_frame.text = caption
    caption_box.text_frame.paragraphs[0].font.size = Pt(12)
    caption_box.text_frame.paragraphs[0].font.italic = True


def add_table(slide, rows, left, top, width, height, font=12, column_widths=None):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(left), Inches(top), Inches(width), Inches(height))
    table = shape.table
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(value)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(font)
                paragraph.font.bold = r == 0
    if column_widths:
        for index, value in enumerate(column_widths):
            table.columns[index].width = Inches(value)


def new_slide(prs, title):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, title)
    add_footer(slide, len(prs.slides))
    return slide


def bullet_slide(prs, title, bullets, size=20):
    slide = new_slide(prs, title)
    add_bullets(slide, bullets, size=size)
    return slide


def results_rows():
    path = ARTIFACTS / "results_table.csv"
    if path.exists():
        with path.open(encoding="utf-8") as file:
            data = list(csv.DictReader(file))
        keep = ["Model", "Runs", "Validation IoU", "Test IoU (pooled)", "Test mean image IoU", "Fudan IoU", "Penn IoU"]
        return [keep] + [[row[k] for k in keep] for row in data], True
    return (
        [
            ["Model (v1, 40 epochs, 1 seed)", "Precision", "Recall", "F1", "IoU"],
            ["Baseline U-Net", "0.8320", "0.7397", "0.7831", "0.6436"],
            ["Augmented U-Net", "0.8274", "0.7591", "0.7918", "0.6553"],
            ["Advanced (BCE+Dice), undertrained", "0.7739", "0.6720", "0.7194", "0.5617"],
            ["Ablation (Dice 0), 10 epochs", "0.6245", "0.7369", "0.6761", "0.5106"],
        ],
        False,
    )


def paired_summary_bullets():
    path = ARTIFACTS / "paired_comparison.csv"
    if not path.exists():
        return ["Paired multi-seed intervals are generated after training completes."]
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    labels = {
        ("baseline", "augmented"): "Mild augmentation vs baseline",
        ("strong_aug", "resnet34"): "ResNet34 vs U-Net with strong augmentation",
    }
    bullets = []
    for row in rows:
        label = labels.get((row["first"], row["second"]))
        if label:
            bullets.append(
                f"{label}: mean IoU difference {float(row['mean_iou_difference']):+.3f}, "
                f"95% interval [{float(row['ci_low']):+.3f}, {float(row['ci_high']):+.3f}] "
                f"across {row['matched_seeds']} matched seeds."
            )
    return bullets or ["Paired intervals are available in artifacts/paired_comparison.csv."]


def error_analysis_bullets():
    bullets = ["Red marks false-positive pixels; blue marks missed person pixels. See the per-image CSV for all test images."]
    path = ARTIFACTS / "per_image" / "augmented.csv"
    if path.exists():
        with path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if rows:
            worst = min(rows, key=lambda row: float(row["iou"]))
            bullets.append(f"Lowest-IoU augmented test image: {Path(worst['image']).stem}, IoU {float(worst['iou']):.3f}.")
    path = ARTIFACTS / "results_table.csv"
    if path.exists():
        with path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        augmented = next((row for row in rows if row.get("Model") == "augmented"), None)
        if augmented and augmented.get("Fudan IoU") and augmented.get("Penn IoU"):
            bullets.append(f"Augmented U-Net source means: Fudan {augmented['Fudan IoU']}; Penn {augmented['Penn IoU']}.")
    bullets.append("Ablations compare augmentation strength, encoder, distractors and Dice loss weight.")
    return bullets


def conclusion_bullets():
    path = ARTIFACTS / "results_table.csv"
    if not path.exists():
        return ["Final claims will be drawn from the completed multi-seed tables.", "Next: validate on real reflections and broader pedestrian scenes."]
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    learned = {"baseline", "augmented", "advanced", "advanced_ablation", "strong_aug", "distract", "resnet34", "resnet34_distract", "deeplab_ft"}
    candidates = [row for row in rows if row.get("Model") in learned and row.get("Test mean image IoU") not in (None, "", "-")]
    if not candidates:
        return ["Final claims will be drawn from the completed multi-seed tables.", "Next: validate on real reflections and broader pedestrian scenes."]
    def mean_iou(row):
        match = re.search(r"-?\d+(?:\.\d+)?", row["Test mean image IoU"])
        return float(match.group())
    best = max(candidates, key=mean_iou)
    return [
        f"Highest mean per-image IoU among trainable configurations: {best['Model']} ({best['Test mean image IoU']}).",
        "Three seeds and 34 test images limit the precision of the paired intervals.",
        "Next: evaluate on real reflections and broader pedestrian scenes.",
    ]


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(11.8), Inches(2.0))
    box.text_frame.word_wrap = True
    box.text_frame.text = TITLE
    box.text_frame.paragraphs[0].font.size = Pt(32)
    box.text_frame.paragraphs[0].font.bold = True
    box.text_frame.paragraphs[0].font.color.rgb = NAVY
    sub = slide.shapes.add_textbox(Inches(0.85), Inches(3.8), Inches(11), Inches(1.4))
    sub.text_frame.text = f"{STUDENT}   Roll No.: {ROLL}\nMachine Learning Project, IIT Patna"
    for paragraph in sub.text_frame.paragraphs:
        paragraph.font.size = Pt(20)
        paragraph.font.color.rgb = RGBColor(65, 65, 65)
    add_footer(slide, 1)

    bullet_slide(prs, "Problem, motivation and hypotheses", [
        "Input: RGB street photo. Output: one binary mask of every pedestrian.",
        "Only the physical person is foreground; reflections and shadows are background.",
        "Small dataset (102 training images): models can memorise scale and viewpoint.",
        "H1: mild augmentation raises mean per-image IoU over the plain U-Net.",
        "H2: a pretrained ResNet34 encoder improves on the strong-augmentation U-Net.",
        "H3: distractor training lowers false positives on reflections and shadows without losing 0.02 clean IoU.",
        "Matched-seed paired intervals are descriptive; three seeds limit precision.",
    ], 19)

    slide = new_slide(prs, "Literature and gap")
    add_table(slide, [
        ["Study", "Method", "Gap"],
        ["Wang et al. 2007", "Penn-Fudan detection + segmentation", "Pre-deep-learning, no reflections"],
        ["Ronneberger et al. 2015", "U-Net", "Trained from scratch on small data"],
        ["Chen et al. 2017", "DeepLabV3", "Reflection behaviour not reported"],
        ["Xie et al. 2021", "SegFormer", "Not tested on mirror distractors"],
        ["Yang et al. 2019", "MirrorNet, 4,018-image mirror set", "Segments the mirror, not the person in it"],
    ], 0.7, 1.3, 12, 3.5, 15)
    add_bullets(slide, ["Gap: Penn-Fudan has no reflection or shadow cases. This project measures stress-test behaviour with synthetic distractors; it does not include a real-reflection dataset."], top=5.2, height=1.5, size=18)

    slide = new_slide(prs, "Dataset overview and audit")
    add_image(slide, FIGURES / "eda.png", 0.5, 1.2, 12.3, 3.6, "Training-set statistics: foreground is a small fraction, aspect ratios vary, 26 of 170 images are wider than 1.5:1.")
    add_bullets(slide, ["170 images, 345 pedestrians, two sources (Fudan, Penn); every image contains a person.", "No licence statement found; images are not redistributed."], top=5.4, height=1.5, size=17)

    bullet_slide(prs, "Split and leakage controls", [
        "One frozen split: 102 train / 34 validation / 34 test (seed 20261103, splits.json).",
        "Augmentation and distractors touch training images only; masks are never altered by distractors.",
        "Best epoch and decision threshold are chosen on validation only.",
        "Test set is evaluated once per model.",
        "Not stratified by source: validation is about 65% Penn, train about 55%, so a Fudan vs Penn breakdown is reported.",
    ])

    slide = new_slide(prs, "Workflow and architecture")
    add_image(slide, FIGURES / "architecture.png", 0.5, 1.2, 12.3, 4.6, "Blue: training. Orange: inference and evaluation.")
    add_bullets(slide, ["Models: plain U-Net, ResNet34-encoder U-Net (ImageNet), DeepLabV3-ResNet50 (COCO/VOC) as pretrained reference."], top=6.2, height=0.8, size=16)

    bullet_slide(prs, "Baselines and experiment protocol", [
        "Trivial baselines: all-background, mean training mask, pixel random forest (RGB + position).",
        "Learned models: baseline, augmented, advanced loss and ablation, strong_aug, distract, ResNet34 U-Net, and fine-tuned DeepLabV3; zero-shot DeepLabV3 is a pretrained reference.",
        "Identical for all: 256 px input, AdamW, 100 epochs, cosine schedule, batch 8, same split and metrics.",
        "Three seeds per model; report mean ± std. Checkpoint chosen by validation IoU.",
        "Sanity check: compare the Dice-weight-zero ablation with augmented training.",
    ])

    rows, final = results_rows()
    slide = new_slide(prs, "Main results" if final else "Results so far (v1 preliminary)")
    add_table(slide, rows, 0.6, 1.3, 12.1, 3.2, 13)
    result_bullets = [
        "Pooled IoU counts all pixels together; mean image IoU weights each image equally.",
        "v1 values are preliminary and the advanced runs were undertrained." if not final else "Three seeds make paired intervals descriptive; they are not adjusted across model comparisons.",
    ]
    if final:
        result_bullets.extend(paired_summary_bullets())
    add_bullets(slide, result_bullets, top=5.0, height=1.8, size=14)

    slide = new_slide(prs, "Training curves and metric comparison")
    add_image(slide, FIGURES / "training_curves.png", 0.4, 1.2, 6.3, 4.5, "Validation loss and IoU per epoch.")
    add_image(slide, ARTIFACTS / "metric_comparison.png", 6.8, 1.2, 6.1, 4.5, "Test metrics per run.")

    slide = new_slide(prs, "Qualitative results")
    add_image(slide, FIGURES / "qualitative.png", 0.5, 1.1, 12.3, 5.6, "Worst, lower-quartile, median and best test images; failures are shown, not only successes.")

    slide = new_slide(prs, "Robustness: reflections and shadows")
    add_image(slide, FIGURES / "distractors.png", 0.3, 1.1, 6.4, 4.3, "Distractors; yellow outline = region that must stay background.")
    add_image(slide, FIGURES / "robustness.png", 6.8, 1.1, 6.2, 4.3, "IoU and distractor false-positive rate.")
    add_bullets(slide, ["A perfect mirror copy equals a person at pixel level; only context can separate them. This is a stress test, not a solution."], top=5.9, height=1.0, size=16)

    slide = new_slide(prs, "Error analysis and ablations")
    add_image(slide, ARTIFACTS / "overlays" / "augmented_worst.png", 0.5, 1.1, 5.5, 5.6, "Red = false positive, blue = missed person.")
    add_bullets(slide, error_analysis_bullets(), left=6.3, top=1.3, width=6.6, height=5, size=18)

    bullet_slide(prs, "Limitations and implications", [
        "34 test images: one image is about 3% of the test set; intervals are wide.",
        "No person-free images; upright pedestrians from two campuses only.",
        "Synthetic distractors are not real reflections; no real-reflection photos are included.",
        "Segmentation can be misused for surveillance; no fairness claim is made.",
    ])

    bullet_slide(prs, "Conclusion and future work", conclusion_bullets())

    bullet_slide(prs, "References", [
        "Wang, Shi, Song, Shen. ACCV 2007 (Penn-Fudan).   Ronneberger et al. MICCAI 2015 (U-Net).",
        "He et al. CVPR 2016 (ResNet).   Chen et al. 2017, arXiv:1706.05587 (DeepLabV3).",
        "Xie et al. NeurIPS 2021 (SegFormer).   Yang et al. ICCV 2019 (MirrorNet).",
        "Shorten and Khoshgoftaar 2019 (augmentation survey).   Milletari et al. 2016 (Dice loss).",
    ], 17)

    prs.save(OUTPUT)
    print(OUTPUT, len(prs.slides), "slides")


if __name__ == "__main__":
    build()
