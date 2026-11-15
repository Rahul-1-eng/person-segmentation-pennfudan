# -*- coding: utf-8 -*-
"""Editable PPTX poster (13.333 x 7.5 in), same layout as the LaTeX/PNG version."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
CREAM = RGBColor(0xFB, 0xE9, 0xDC)
BLUE = RGBColor(0xEA, 0xF3, 0xFB)
PEACH = RGBColor(0xFD, 0xEF, 0xE3)
GREEN = RGBColor(0xEA, 0xF7, 0xEE)
INK = RGBColor(0x20, 0x30, 0x4A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

ART = "/app/work/artifacts"
OUT = "/app/outputs/poster/poster.pptx"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])
SW, SH = 13.333, 7.5

def box(x, y, w, h, fill, line=NAVY, line_w=1.75, radius=0.06):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(line_w)
    try:
        sh.adjustments[0] = radius
    except Exception:
        pass
    sh.shadow.inherit = False
    return sh

def text(x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.0):
    """runs: list of (text, size, bold, color, font, space_after) or list of paragraphs of those tuples"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    first = True
    for para in runs:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.line_spacing = spacing
        if isinstance(para, tuple):
            para = [para]
        for (t, size, bold, color, font, after) in para:
            r = p.add_run(); r.text = t
            r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = font
            p.space_after = Pt(after)
    return tb

H = "Georgia"
B = "Calibri"

# ---------------- header ----------------
box(0.12, 0.08, SW - 0.24, 1.12, CREAM)
slide.shapes.add_picture("/app/outputs/poster/logo_iitpatna.png", Inches(0.3), Inches(0.12), height=Inches(0.95))
slide.shapes.add_picture("/app/outputs/poster/logo_impact.png", Inches(11.9), Inches(0.22), height=Inches(0.78))
text(1.7, 0.10, 9.95, 0.62, [
    [("Binary Person Segmentation on Penn-Fudan with U-Net, Augmentation and Synthetic Reflection/Shadow Distractors", 19, True, NAVY, H, 0)],
], align=PP_ALIGN.CENTER)
text(1.7, 0.68, 9.95, 0.5, [
    [("Rahul Kumar Sahoo, Roll No. 2401CS09", 11.5, False, INK, B, 2)],
    [("Department of Computer Science and Engineering, Indian Institute of Technology Patna", 11, False, INK, B, 0)],
], align=PP_ALIGN.CENTER)

COL_Y, COL_H = 1.32, 5.32
LX, LW = 0.15, 3.75
MX, MW = 4.02, 5.85
RX, RW = 9.99, 3.19

# ---------------- left column ----------------
box(LX, COL_Y, LW, COL_H, BLUE)
text(LX + 0.14, COL_Y + 0.1, LW - 0.28, COL_H - 0.2, [
    [("Abstract", 14, True, NAVY, H, 4)],
    [("We segment pedestrians in the 170-image Penn-Fudan dataset, merging all people into one binary mask. With one frozen 102/34/34 split, validation-only model selection and a single test evaluation, we compare baselines, U-Net variants, a ResNet34-encoder U-Net and DeepLabV3 over three seeds. The pretrained encoder gives the largest gain (+0.0736 mean per-image IoU); fine-tuned DeepLabV3 scores best (0.7973). Training with synthetic reflection and shadow distractors cuts the ResNet34 false-positive rate on them from 0.1314 to 0.0273.", 8.6, False, INK, B, 8)],
    [("Introduction", 14, True, NAVY, H, 4)],
    [("\u2022 Person segmentation labels every pixel as person or background. It supports pedestrian analysis, editing and privacy blurring.", 8.6, False, INK, B, 3)],
    [("\u2022 Small datasets make gains hard to trust: one lucky seed can look like progress.", 8.6, False, INK, B, 3)],
    [("\u2022 Segmenters may also call mirror images or person-shaped shadows people, but Penn-Fudan has no such cases to test this.", 8.6, False, INK, B, 8)],
    [("Problem Statement", 14, True, NAVY, H, 4)],
    [("Given an RGB street photo, predict a binary mask of all pedestrians, training on 102 images, so that test IoU clearly beats trivial baselines; and measure how often synthetic reflection- and shadow-like regions pasted on the background are wrongly called person.", 8.6, False, INK, B, 0)],
], spacing=1.04)

# ---------------- middle column ----------------
box(MX, COL_Y, MW, COL_H, PEACH)
text(MX + 0.14, COL_Y + 0.08, MW - 0.28, 1.15, [
    [("Key Contributions", 14, True, NAVY, H, 4)],
    [("\u2022 27 runs (9 configs \u00d7 3 seeds) and 3 baselines under one frozen, leak-free protocol.", 8.6, False, INK, B, 3)],
    [("\u2022 Seed spread and paired bootstrap CIs; mild augmentation: +0.0271 [0.0086, 0.0446].", 8.6, False, INK, B, 3)],
    [("\u2022 Six synthetic distractors on background only, scored by a false-positive rate FP_D.", 8.6, False, INK, B, 3)],
    [("\u2022 Error analysis: several of the worst false positives are unlabelled people.", 8.6, False, INK, B, 0)],
])
text(MX + 0.14, COL_Y + 1.28, MW - 0.28, 0.95, [
    [("Methodology", 14, True, NAVY, H, 4)],
    [("U-Net (7.76 M), ResNet34-encoder U-Net (24.52 M) and DeepLabV3-ResNet50 (42.00 M), trained with AdamW on BCE or BCE + Dice; only training images are augmented.", 8.6, False, INK, B, 3)],
    [("Training phase (blue): all fitting uses the 102 training images  |  Evaluation phase (peach): no fitting, test set used once", 8.2, True, NAVY, B, 0)],
])
# architecture figure
arch_w = MW - 0.5
im = Image.open(f"{ART}/figures/architecture.png")
arch_h = arch_w * im.height / im.width
slide.shapes.add_picture(f"{ART}/figures/architecture.png", Inches(MX + 0.25), Inches(COL_Y + 2.28), width=Inches(arch_w))
text(MX + 0.25, COL_Y + 2.28 + arch_h + 0.02, arch_w, 0.24, [
    [("Figure 1: End-to-end architecture of the study pipeline", 8.2, False, INK, B, 0)],
], align=PP_ALIGN.CENTER)
# algorithm box
algo_y = COL_Y + 2.28 + arch_h + 0.3
box(MX + 0.2, algo_y, MW - 0.4, COL_H - (algo_y - COL_Y) - 0.15, WHITE, line_w=1.2)
text(MX + 0.34, algo_y + 0.08, MW - 0.68, COL_H - (algo_y - COL_Y) - 0.3, [
    [("Algorithm 1: Leak-free multi-seed study", 8.8, True, NAVY, B, 2)],
    [("Input: images D; frozen split D_tr/D_val/D_te; configs C; seeds S = {20261103, 1, 2}", 7.6, False, INK, "Consolas", 1)],
    [("1. Binarise masks (mask > 0); resize to 256\u00d7256.", 7.6, False, INK, "Consolas", 1)],
    [("2. for c in C, s in S: train 100 epochs on augmented D_tr (+ distractors, p = 0.3); keep best D_val-IoU epoch.", 7.6, False, INK, "Consolas", 1)],
    [("3. Choose t* on D_val only; test once on D_te: IoU, P, R, F1, bootstrap CI.", 7.6, False, INK, "Consolas", 1)],
    [("4. Paste 6 distractor types on D_te backgrounds; report FP_D.", 7.6, False, INK, "Consolas", 1)],
    [("5. Report mean \u00b1 std over seeds; paired bootstrap differences.", 7.6, False, INK, "Consolas", 0)],
], spacing=1.05)

# ---------------- right column ----------------
box(RX, COL_Y, RW, COL_H, GREEN)
text(RX + 0.12, COL_Y + 0.08, RW - 0.24, 0.32, [
    [("Result and Discussion", 14, True, NAVY, H, 0)],
])
text(RX + 0.12, COL_Y + 0.42, RW - 0.24, 0.22, [
    [("Table 1: Test results at t = 0.5 (mean \u00b1 std, 3 seeds)", 8.4, True, INK, B, 0)],
])
rows_data = [
    ("Model", "Mean image IoU", "Pooled IoU", "FP_D"),
    ("pixel RF", "0.2713", "0.2645", "\u2013"),
    ("U-Net baseline", "0.6893 \u00b1 0.0146", "0.6564", "0.1465"),
    ("U-Net + mild aug.", "0.7164 \u00b1 0.0015", "0.6831", "0.1470"),
    ("ResNet34-U-Net", "0.7895 \u00b1 0.0094", "0.7695", "0.1314"),
    ("ResNet34 + distr.", "0.7903 \u00b1 0.0052", "0.7668", "0.0273"),
    ("DeepLab zero-shot", "0.7567", "0.7343", "0.1666"),
    ("DeepLab fine-tuned", "0.7973 \u00b1 0.0017", "0.7801", "0.1306"),
]
tbl_shape = slide.shapes.add_table(len(rows_data), 4, Inches(RX + 0.1), Inches(COL_Y + 0.66), Inches(RW - 0.2), Inches(1.62))
tbl = tbl_shape.table
tbl.columns[0].width = Inches(1.06)
tbl.columns[1].width = Inches(0.86)
tbl.columns[2].width = Inches(0.62)
tbl.columns[3].width = Inches(0.45)
for i, row in enumerate(rows_data):
    for j, val in enumerate(row):
        cell = tbl.cell(i, j)
        cell.text = val
        cell.margin_left = Inches(0.03); cell.margin_right = Inches(0.02)
        cell.margin_top = Inches(0.01); cell.margin_bottom = Inches(0.01)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(6.8)
                r.font.name = B
                if i == 0:
                    r.font.bold = True; r.font.color.rgb = WHITE
                elif i == 7:
                    r.font.bold = True; r.font.color.rgb = NAVY
                else:
                    r.font.color.rgb = INK
        if i == 0:
            cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        elif i == 7:
            cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0xCF, 0xE3, 0xF5)
        else:
            cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if i % 2 else RGBColor(0xF3, 0xF8, 0xFD)

# charts
im1 = Image.open("/app/outputs/poster/poster_iou.png")
cw = RW - 0.24
ch = cw * im1.height / im1.width
cy1 = COL_Y + 2.42
slide.shapes.add_picture("/app/outputs/poster/poster_iou.png", Inches(RX + 0.12), Inches(cy1), width=Inches(cw))
slide.shapes.add_picture("/app/outputs/poster/poster_fpd.png", Inches(RX + 0.12), Inches(cy1 + ch + 0.3), width=Inches(cw))
text(RX + 0.12, cy1 + ch + 0.04, cw, 0.22, [[("(a) Test mean per-image IoU", 8.2, False, INK, B, 0)]], align=PP_ALIGN.CENTER)
text(RX + 0.12, cy1 + 2 * ch + 0.34, cw, 0.22, [[("(b) Distractor false-positive rate FP_D", 8.2, False, INK, B, 0)]], align=PP_ALIGN.CENTER)
text(RX + 0.12, cy1 + 2 * ch + 0.56, cw, 0.22, [[("Figure 2: Test IoU by model and distractor robustness", 8.2, False, INK, B, 0)]], align=PP_ALIGN.CENTER)
cy_conc = cy1 + 2 * ch + 0.82
text(RX + 0.12, cy_conc, RW - 0.24, COL_H - (cy_conc - COL_Y) - 0.1, [
    [("Conclusion", 14, True, NAVY, H, 4)],
    [("Pretrained encoders matter most; distractor training helps ResNet34 (FP_D 0.1314 \u2192 0.0273) at almost no clean cost. Real reflections remain untested.", 8.6, False, INK, B, 0)],
], spacing=1.04)

# ---------------- footer ----------------
FY = COL_Y + COL_H + 0.1
box(0.15, FY, SW - 0.3, SH - FY - 0.12, CREAM)
text(0.32, FY + 0.08, 0.85, 0.5, [[("References", 11, True, NAVY, H, 0)]])
text(1.25, FY + 0.06, 8.0, 0.56, [
    [("[1] L. Wang, J. Shi, G. Song, I.-F. Shen, \u201cObject detection combining recognition and segmentation,\u201d ACCV, 2007.   [2] O. Ronneberger, P. Fischer, T. Brox, \u201cU-Net: Convolutional networks for biomedical image segmentation,\u201d MICCAI, 2015.", 7.2, False, INK, B, 2)],
    [("[3] K. He, X. Zhang, S. Ren, J. Sun, \u201cDeep residual learning for image recognition,\u201d CVPR, 2016.   [4] L.-C. Chen et al., \u201cRethinking atrous convolution for semantic image segmentation,\u201d arXiv:1706.05587, 2017.", 7.2, False, INK, B, 0)],
], spacing=1.05)
text(9.4, FY + 0.06, 2.7, 0.56, [
    [("Code and data instructions:", 8.6, True, INK, B, 1)],
    [("github.com/Rahul-1-eng/person-segmentation-pennfudan", 8.6, True, NAVY, B, 0)],
])
slide.shapes.add_picture("/app/outputs/poster/qr_github.png", Inches(12.25), Inches(FY + 0.05), height=Inches(0.62))

prs.save(OUT)
print("PPTX written:", OUT)
