# -*- coding: utf-8 -*-
"""Render the project report as PDF (reportlab) from report_content.py."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
                                PageBreak, Table, TableStyle, Image, KeepTogether, XPreformatted)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.platypus.flowables import Flowable
import report_content as C

ART = "/app/work/artifacts"
FIG_DIRS = [os.path.join(ART, "figures"), os.path.join(ART, "v1"), os.path.join(ART, "v1", "overlays"), os.path.join(ART, "overlays")]

FIGMAP = {
    "figures/eda.png": "figures/eda.png",
    "figures/architecture.png": "figures/architecture.png",
    "figures/metric_comparison.png": "v1/metric_comparison.png",
    "figures/training_curves.png": "figures/training_curves.png",
    "figures/qualitative.png": "figures/qualitative.png",
    "figures/robustness.png": "figures/robustness.png",
    "figures/overlays_augmented_worst.png": "overlays/augmented_worst.png",
}

def resolve(name):
    rel = FIGMAP.get(name, name)
    for d in FIG_DIRS:
        p = os.path.join(d, os.path.basename(rel))
        if os.path.exists(p):
            return p
    return None

NAVY = colors.HexColor("#1F3A5F")
LIGHT = colors.HexColor("#EAF1F8")
ROW = colors.HexColor("#F5F8FC")

body = ParagraphStyle("body", fontName="Times-Roman", fontSize=11, leading=14.6, alignment=TA_JUSTIFY, spaceAfter=5.5)
item = ParagraphStyle("item", parent=body, leftIndent=16, spaceAfter=4)
h1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=17, leading=21, textColor=NAVY, spaceBefore=14, spaceAfter=8)
h2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=13, leading=17, textColor=NAVY, spaceBefore=11, spaceAfter=6)
cap = ParagraphStyle("cap", fontName="Times-Roman", fontSize=9.5, leading=12.5, alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceBefore=3, spaceAfter=12)
cell = ParagraphStyle("cell", fontName="Times-Roman", fontSize=8.6, leading=11)
cellh = ParagraphStyle("cellh", fontName="Times-Bold", fontSize=8.8, leading=11, textColor=colors.white)
algostyle = ParagraphStyle("algo", fontName="Courier", fontSize=8.8, leading=12.5)
titlestyle = ParagraphStyle("title", fontName="Times-Bold", fontSize=21, leading=27, alignment=TA_CENTER, textColor=NAVY)
metastyle = ParagraphStyle("meta", fontName="Times-Roman", fontSize=12.5, leading=19, alignment=TA_CENTER)
abstractstyle = ParagraphStyle("abstract", parent=body, fontSize=11.5, leading=17)
linkstyle = ParagraphStyle("link", fontName="Times-Bold", fontSize=11.5, leading=16, alignment=TA_LEFT, textColor=colors.HexColor("#1155CC"), spaceBefore=6, spaceAfter=10)

fig_registry = []   # (num, caption)
tab_registry = []

def next_fig():
    fig_registry.append(len(fig_registry) + 1)
    return len(fig_registry)

def next_tab():
    tab_registry.append(len(tab_registry) + 1)
    return len(tab_registry)


class Heading(Paragraph):
    def __init__(self, text, style, level):
        self.level = level
        self.toc_text = text
        super().__init__(text, style)

def H(text, level, toc=True, num=None):
    disp = f"{num}&nbsp;&nbsp;{text}" if num else text
    p = Heading(disp, h1 if level == 0 else h2, level)
    p.toc_text = f"{num}  {text}" if num else text
    p.toc_level = level
    p.notify_toc = toc
    return p


class DocTemplate(BaseDocTemplate):
    def afterFlowable(self, fl):
        if isinstance(fl, Heading) and getattr(fl, "notify_toc", True):
            self.notify("TOCEntry", (fl.toc_level, fl.toc_text, self.page))

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(A4[0] / 2, 1.05 * cm, str(doc.page))
    if doc.page > 1:
        canvas.drawString(2 * cm, A4[1] - 1.25 * cm, "Binary Person Segmentation on Penn-Fudan")
        canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1.25 * cm, "R. K. Sahoo, 2401CS09")
        canvas.setStrokeColor(colors.HexColor("#CCCCCC"))
        canvas.setLineWidth(0.5)
        canvas.line(2 * cm, A4[1] - 1.45 * cm, A4[0] - 2 * cm, A4[1] - 1.45 * cm)
    canvas.restoreState()


def build_table(spec, width=17 * cm):
    num = next_tab()
    tab_registry[-1] = (num, spec["caption"])
    header = [Paragraph(h, cellh) for h in spec["header"]]
    rows = [[Paragraph(str(c), cell) for c in r] for r in spec["rows"]]
    t = Table([header] + rows, colWidths=None)
    # equal columns is fine for our tables; results table gets custom widths below
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9FB3C8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    if len(spec["header"]) == 6:
        widths = [4.6 * cm, 2.4 * cm, 2.6 * cm, 3.2 * cm, 1.7 * cm, 1.7 * cm]
        t._argW = widths
    return [KeepTogether([t, Paragraph(f"Table {num}: {spec['caption'].split(': ', 1)[-1]}", cap)])]


FIGW = {
    "figures/eda.png": 15.0 * cm,
    "figures/architecture.png": 15.0 * cm,
    "figures/metric_comparison.png": 11.5 * cm,
    "figures/training_curves.png": 15.0 * cm,
    "figures/qualitative.png": 10.8 * cm,
    "figures/robustness.png": 15.5 * cm,
    "figures/overlays_augmented_worst.png": 11.5 * cm,
}

def add_figure(story, name, caption, width=16.5 * cm):
    path = resolve(name)
    if not path:
        return
    from PIL import Image as PILImage
    iw, ih = PILImage.open(path).size
    w = FIGW.get(name, width)
    h = w * ih / iw
    max_h = 20 * cm
    if h > max_h:
        h = max_h
        w = h * iw / ih
    num = next_fig()
    fig_registry[-1] = (num, caption)
    story.append(KeepTogether([
        Image(path, width=w, height=h),
        Paragraph(f"Figure {num}: {caption}", cap),
    ]))


def story_elements():
    S = []
    # ---- title page ----
    S.append(Spacer(1, 3.2 * cm))
    S.append(Paragraph("Binary Person Segmentation on Penn-Fudan with U-Net, Augmentation and Synthetic Reflection/Shadow Distractors", titlestyle))
    S.append(Spacer(1, 1.1 * cm))
    S.append(Paragraph("A Machine Learning Course Project Report", metastyle))
    S.append(Spacer(1, 1.6 * cm))
    S.append(Paragraph("<b>" + C.AUTHOR + "</b>", metastyle))
    S.append(Paragraph(C.ROLL, metastyle))
    S.append(Spacer(1, 0.5 * cm))
    S.append(Paragraph(C.DEPT, metastyle))
    S.append(Paragraph(C.INST, metastyle))
    S.append(Spacer(1, 2.2 * cm))
    tbl = Table([[Paragraph("Code repository (public):", ParagraphStyle("m2", parent=metastyle, fontName="Times-Bold")), Paragraph(C.GITHUB.replace("https://", ""), ParagraphStyle("m3", parent=metastyle, textColor=colors.HexColor("#1155CC")))]], colWidths=[6 * cm, 10 * cm])
    tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    S.append(tbl)
    S.append(PageBreak())

    # ---- abstract ----
    S.append(H("Abstract", 0, toc=False, num=None))
    S.append(Paragraph(C.ABSTRACT, abstractstyle))
    S.append(PageBreak())

    # ---- ToC / LoF / LoT placeholders (filled by multiBuild) ----
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("toc0", fontName="Times-Bold", fontSize=10.2, leading=12.4, leftIndent=6, firstLineIndent=-6, spaceBefore=1.5),
        ParagraphStyle("toc1", fontName="Times-Roman", fontSize=9.3, leading=11.4, leftIndent=20, firstLineIndent=-8, spaceBefore=0),
    ]
    S.append(H("Table of Contents", 0, toc=False, num=None))
    S.append(toc)
    S.append(Spacer(1, 0.55 * cm))
    S.append(H("List of Figures and Tables", 0, toc=False, num=None))
    S.append(Paragraph("<i>All figures and tables are numbered in order of appearance and referenced from the text.</i>",
                       ParagraphStyle("note", parent=body, fontSize=9.2, textColor=colors.HexColor("#666666"), spaceAfter=4)))
    S.append(Spacer(1, 0.45 * cm))

    ch = 0
    fig_total = 0
    for chap in C.CHAPTERS:
        ch += 1
        S.append(H(chap["title"], 0, num=f"Chapter {ch}"))
        for si, sec in enumerate(chap["sections"], 1):
            S.append(H(sec["title"], 1, num=f"{ch}.{si}"))
            for t in sec["text"]:
                S.append(Paragraph(t, body))
            if "table" in sec:
                S.extend(build_table(C.TABLES[sec["table"]]))
            if "algorithm" in sec:
                box = Table([[XPreformatted(sec["algorithm"], algostyle)]], colWidths=[17 * cm])
                box.setStyle(TableStyle([
                    ("BOX", (0, 0), (-1, -1), 1, NAVY),
                    ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ]))
                S.append(KeepTogether([box, Spacer(1, 10)]))
            if "figure" in sec:
                add_figure(S, sec["figure"][0], sec["figure"][1])
            if "figure2" in sec:
                add_figure(S, sec["figure2"][0], sec["figure2"][1])
        if ch == 2:
            pass
        S.append(Spacer(1, 0.3 * cm))

    # ---- references (two columns to save space) ----
    S.append(H("References", 0, toc=True, num=None))
    left_refs = [Paragraph(f"[{i}]&nbsp;&nbsp;{C.REFERENCES[i-1]}", ParagraphStyle("ref", parent=body, leftIndent=18, firstLineIndent=-18, spaceAfter=5, fontSize=9.2, leading=12)) for i in (1, 2, 3, 4)]
    right_refs = [Paragraph(f"[{i}]&nbsp;&nbsp;{C.REFERENCES[i-1]}", ParagraphStyle("ref", parent=body, leftIndent=18, firstLineIndent=-18, spaceAfter=5, fontSize=9.2, leading=12)) for i in (5, 6, 7, 8)]
    rt = Table([[left_refs, right_refs]], colWidths=[8.7 * cm, 8.7 * cm])
    rt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    S.append(rt)
    S.append(Spacer(1, 10))

    # ---- appendix (flows right after references) ----
    S.append(H(C.APPENDIX["title"], 0, toc=True, num=None))
    S.append(Paragraph(C.APPENDIX["intro"], body))
    S.append(Paragraph(f'<a href="{C.GITHUB}" color="#1155CC"><u>{C.GITHUB}</u></a>', linkstyle))
    S.append(Paragraph("A.1&nbsp;&nbsp;<b>What the repository contains</b>", h2))
    S.append(Paragraph(C.APPENDIX["layout"], body))
    S.append(Paragraph("A.2&nbsp;&nbsp;<b>Commands to reproduce the results</b>", h2))
    box = Table([[XPreformatted(C.APPENDIX["commands"], algostyle)]], colWidths=[17 * cm])
    box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1, NAVY),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    S.append(box)
    S.append(Spacer(1, 8))
    S.append(Paragraph("A.3&nbsp;&nbsp;<b>Where each number in this report comes from</b>", h2))
    S.append(Paragraph(C.APPENDIX["mapping"], body))
    return S


def _no_toc():
    pass


class PatchedTOC(TableOfContents):
    pass

def late_fill(story, toc):
    # fill list of figures/tables placeholders after numbering is known
    out = []
    for el in story:
        out.append(el)
    return out


def build():
    out = "/app/outputs/report/Person_Segmentation_Report.pdf"
    doc = DocTemplate(out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2.2 * cm, bottomMargin=1.9 * cm,
                      title=C.TITLE, author=C.AUTHOR)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=on_page)])

    S = story_elements()

    # Build once to learn figure/table numbering, then splice LoF/LoT and rebuild
    elems = []
    figs = fig_registry
    tabs = [t for t in tab_registry]
    lof_idx = None
    for i, el in enumerate(S):
        if isinstance(el, Heading) and el.toc_text == "List of Figures and Tables":
            lof_idx = i
    all_lof = []
    for num, capt in figs:
        short = capt.split(". ")[0] if ". " in capt[:40] else capt
        if len(short) > 95:
            short = short[:92] + "..."
        all_lof.append(f"Figure {num}: {short}")
    for num, capt in tabs:
        all_lof.append(f"Table {num}: {capt.split(': ', 1)[-1]}")
    half = (len(all_lof) + 1) // 2
    lof_left = [Paragraph(t, ParagraphStyle("lof", parent=body, fontSize=8.8, leading=11.2, leftIndent=12, firstLineIndent=-12, spaceAfter=2)) for t in all_lof[:half]]
    lof_right = [Paragraph(t, ParagraphStyle("lof", parent=body, fontSize=8.8, leading=11.2, leftIndent=12, firstLineIndent=-12, spaceAfter=2)) for t in all_lof[half:]]
    lot = Table([[lof_left, lof_right]], colWidths=[8.7 * cm, 8.7 * cm])
    lot.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    lof_paras = [lot]
    S[lof_idx + 2:lof_idx + 2] = lof_paras

    doc.multiBuild(S)
    print("PDF written:", out)


if __name__ == "__main__":
    build()
