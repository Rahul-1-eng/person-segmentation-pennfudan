# -*- coding: utf-8 -*-
"""Render the project report as DOCX (python-docx) from report_content.py."""
import os
import docx
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import report_content as C

ART = "/app/work/artifacts"
FIGMAP = {
    "figures/eda.png": ("figures", "eda.png"),
    "figures/architecture.png": ("figures", "architecture.png"),
    "figures/metric_comparison.png": ("v1", "metric_comparison.png"),
    "figures/training_curves.png": ("figures", "training_curves.png"),
    "figures/qualitative.png": ("figures", "qualitative.png"),
    "figures/robustness.png": ("figures", "robustness.png"),
    "figures/overlays_augmented_worst.png": ("overlays", "augmented_worst.png"),
}

def resolve(name):
    d, f = FIGMAP.get(name, os.path.split(name))
    p = os.path.join(ART, d, f)
    return p if os.path.exists(p) else None

NAVY = RGBColor(0x1F, 0x3A, 0x5F)

fig_n = 0
tab_n = 0

def set_cell_bg(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)

def p_text(doc, text, style=None, size=11, bold=False, align=None, space_after=7, color=None, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    r.font.name = "Times New Roman"
    if color:
        r.font.color.rgb = color
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.25
    return p

def add_hyperlink(paragraph, url, text):
    part = paragraph.part
    r_id = part.relate_to(url, docx.opc.constants.RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    c = OxmlElement("w:color")
    c.set(qn("w:val"), "1155CC")
    rPr.append(c)
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rPr.append(u)
    new_run.append(rPr)
    t = OxmlElement("w:t")
    t.text = text
    new_run.append(t)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)

def add_table(doc, spec):
    global tab_n
    tab_n += 1
    t = doc.add_table(rows=1 + len(spec["rows"]), cols=len(spec["header"]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(spec["header"]):
        cell = t.rows[0].cells[j]
        cell.text = h
        for par in cell.paragraphs:
            for r in par.runs:
                r.bold = True
                r.font.size = Pt(8.5)
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                r.font.name = "Times New Roman"
        set_cell_bg(cell, "1F3A5F")
    for i, row in enumerate(spec["rows"], 1):
        for j, v in enumerate(row):
            cell = t.rows[i].cells[j]
            cell.text = str(v)
            for par in cell.paragraphs:
                for r in par.runs:
                    r.font.size = Pt(8.5)
                    r.font.name = "Times New Roman"
        if i % 2 == 0:
            for c2 in t.rows[i].cells:
                set_cell_bg(c2, "F2F6FB")
    cap = p_text(doc, f"Table {tab_n}: {spec['caption'].split(': ', 1)[-1]}", size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12, italic=False)
    for r in cap.runs:
        r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

def add_figure(doc, name, caption, width_cm=15.5):
    global fig_n
    path = resolve(name)
    if not path:
        return
    fig_n += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Cm(width_cm))
    cap = p_text(doc, f"Figure {fig_n}: {caption}", size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)
    for r in cap.runs:
        r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

def add_algo(doc, text):
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    cell = t.rows[0].cells[0]
    cell.text = ""
    for i, line in enumerate(text.split("\n")):
        par = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        r = par.add_run(line)
        r.font.name = "Courier New"
        r.font.size = Pt(8.8)
        par.paragraph_format.space_after = Pt(2)
    set_cell_bg(cell, "EAF1F8")

def build():
    global fig_n, tab_n
    fig_n = tab_n = 0
    figs_registry = []
    tabs_registry = []

    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(2.2)
        sec.bottom_margin = Cm(2.0)
        sec.left_margin = Cm(2.3)
        sec.right_margin = Cm(2.3)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)

    # ---- title page ----
    for _ in range(4):
        p_text(doc, "", space_after=12)
    p_text(doc, C.TITLE, size=20, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, color=NAVY, space_after=18)
    p_text(doc, "A Machine Learning Course Project Report", size=12.5, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=22)
    p_text(doc, C.AUTHOR, size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    p_text(doc, C.ROLL, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)
    p_text(doc, C.DEPT, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    p_text(doc, C.INST, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=26)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Code repository (public): ")
    r.bold = True
    r.font.name = "Times New Roman"
    add_hyperlink(p, C.GITHUB, C.GITHUB)

    doc.add_page_break()

    # ---- abstract ----
    h = doc.add_heading("Abstract", level=1)
    for r in h.runs:
        r.font.color.rgb = NAVY
        r.font.name = "Times New Roman"
    p_text(doc, C.ABSTRACT, size=11.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    doc.add_page_break()

    # ---- ToC field ----
    h = doc.add_heading("Table of Contents", level=1)
    for r in h.runs:
        r.font.color.rgb = NAVY
    par = doc.add_paragraph()
    fldChar = OxmlElement("w:fldChar"); fldChar.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText"); instrText.set(qn("xml:space"), "preserve"); instrText.text = "TOC \\o \"1-2\" \\h \\z \\u"
    fldChar2 = OxmlElement("w:fldChar"); fldChar2.set(qn("w:fldCharType"), "separate")
    t = OxmlElement("w:t"); t.text = "Right-click and choose Update Field to build the table of contents."
    fldChar3 = OxmlElement("w:fldChar"); fldChar3.set(qn("w:fldCharType"), "end")
    run = par.add_run()
    run._r.append(fldChar); run._r.append(instrText); run._r.append(fldChar2); run._r.append(t); run._r.append(fldChar3)
    doc.add_page_break()

    ch = 0
    for chap in C.CHAPTERS:
        ch += 1
        h = doc.add_heading(f"Chapter {ch}  {chap['title']}", level=1)
        for r in h.runs:
            r.font.color.rgb = NAVY
            r.font.name = "Times New Roman"
        for si, sec in enumerate(chap["sections"], 1):
            h = doc.add_heading(f"{ch}.{si}  {sec['title']}", level=2)
            for r in h.runs:
                r.font.color.rgb = NAVY
            for t in sec["text"]:
                p_text(doc, t, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
            if "table" in sec:
                tabs_registry.append(C.TABLES[sec["table"]])
                add_table(doc, C.TABLES[sec["table"]])
            if "algorithm" in sec:
                add_algo(doc, sec["algorithm"])
                p_text(doc, "", space_after=8)
            if "figure" in sec:
                figs_registry.append(sec["figure"])
                add_figure(doc, sec["figure"][0], sec["figure"][1])
            if "figure2" in sec:
                figs_registry.append(sec["figure2"])
                add_figure(doc, sec["figure2"][0], sec["figure2"][1])

    # ---- references ----
    h = doc.add_heading("References", level=1)
    for r in h.runs:
        r.font.color.rgb = NAVY
    for i, r_ in enumerate(C.REFERENCES, 1):
        p_text(doc, f"[{i}]  {r_}", size=10.5, space_after=4)

    doc.add_page_break()
    h = doc.add_heading(C.APPENDIX["title"], level=1)
    for r in h.runs:
        r.font.color.rgb = NAVY
    p_text(doc, C.APPENDIX["intro"], align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    p = doc.add_paragraph()
    add_hyperlink(p, C.GITHUB, C.GITHUB)
    h = doc.add_heading("A.1  What the repository contains", level=2)
    for r in h.runs:
        r.font.color.rgb = NAVY
    p_text(doc, C.APPENDIX["layout"], align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    h = doc.add_heading("A.2  Commands to reproduce the results", level=2)
    for r in h.runs:
        r.font.color.rgb = NAVY
    for line in C.APPENDIX["commands"].split("\n"):
        if not line.strip():
            continue
        par = doc.add_paragraph()
        r = par.add_run(line.strip())
        r.font.name = "Courier New"
        r.font.size = Pt(9)
        par.paragraph_format.space_after = Pt(1)
    h = doc.add_heading("A.3  Where each number in this report comes from", level=2)
    for r in h.runs:
        r.font.color.rgb = NAVY
    p_text(doc, C.APPENDIX["mapping"], align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    out = "/app/outputs/report/Person_Segmentation_Report.docx"
    doc.save(out)
    print("DOCX written:", out)

if __name__ == "__main__":
    build()
