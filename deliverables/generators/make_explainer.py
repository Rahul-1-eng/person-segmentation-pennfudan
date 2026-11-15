# -*- coding: utf-8 -*-
"""Render explainer doc as PDF + DOCX."""
import re
import report_content as C
import explainer_content as E
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle

NAVY = colors.HexColor("#1F3A5F")
body = ParagraphStyle("body", fontName="Times-Roman", fontSize=11.5, leading=16.5, alignment=TA_JUSTIFY, spaceAfter=8)
qa = ParagraphStyle("qa", parent=body, spaceAfter=6)
h1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=15, leading=19, textColor=NAVY, spaceBefore=13, spaceAfter=7)
titlestyle = ParagraphStyle("t", fontName="Times-Bold", fontSize=24, leading=30, alignment=TA_CENTER, textColor=NAVY)

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(A4[0] / 2, 1.05 * cm, str(canvas.getPageNumber()))
    canvas.drawString(2 * cm, A4[1] - 1.25 * cm, "Person Segmentation, Explained From Zero")
    canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1.25 * cm, "Companion document")
    canvas.setStrokeColor(colors.HexColor("#CCCCCC"))
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, A4[1] - 1.45 * cm, A4[0] - 2 * cm, A4[1] - 1.45 * cm)
    canvas.restoreState()

def build_pdf():
    S = []
    S.append(Spacer(1, 3.4 * cm))
    S.append(Paragraph(E.E_TITLE, titlestyle))
    S.append(Spacer(1, 0.9 * cm))
    S.append(Paragraph(E.E_SUB, ParagraphStyle("sub", parent=body, fontSize=13, leading=19, alignment=TA_CENTER)))
    S.append(Spacer(1, 1.4 * cm))
    S.append(Paragraph(E.E_AUTHOR, ParagraphStyle("m", parent=body, alignment=TA_CENTER, fontSize=11.5)))
    S.append(Paragraph("Code: " + E.E_GITHUB.replace("https://", ""), ParagraphStyle("m2", parent=body, alignment=TA_CENTER, fontSize=11.5, textColor=colors.HexColor("#1155CC"))))
    S.append(PageBreak())
    S.append(Paragraph("How to read this document", h1))
    S.append(Paragraph(E.INTRO, body))
    for sec in E.SECTIONS:
        S.append(Paragraph(sec["title"], h1))
        for t in sec["text"]:
            style = qa if t.startswith("Q") else body
            # bold the Q part
            m = re.match(r"^(Q\d+\..+?\?)\s(.*)$", t, re.S)
            if m:
                S.append(Paragraph(f"<b>{m.group(1)}</b> {m.group(2)}", qa))
            else:
                S.append(Paragraph(t, body))
    out = "/app/outputs/explainer/Person_Segmentation_Explained.pdf"
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=2.2 * cm, bottomMargin=1.9 * cm, title=E.E_TITLE, author="Rahul Kumar Sahoo")
    doc.build(S, onFirstPage=on_page, onLaterPages=on_page)
    print("explainer PDF:", out)

def build_docx():
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    NAVYD = RGBColor(0x1F, 0x3A, 0x5F)
    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(2.2); sec.bottom_margin = Cm(2.0); sec.left_margin = Cm(2.3); sec.right_margin = Cm(2.3)
    st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11.5)

    def para(text, size=11.5, bold=False, align=None, after=7, color=None):
        p = doc.add_paragraph(); r = p.add_run(text)
        r.font.size = Pt(size); r.bold = bold; r.font.name = "Times New Roman"
        if color: r.font.color.rgb = color
        if align: p.alignment = align
        p.paragraph_format.space_after = Pt(after)
        p.paragraph_format.line_spacing = 1.25
        return p

    para("", 12, after=40)
    para(E.E_TITLE, size=22, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, color=NAVYD, after=14)
    para(E.E_SUB, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, after=18)
    para(E.E_AUTHOR, size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    para("Code: " + E.E_GITHUB, size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER, after=30)
    h = doc.add_heading("How to read this document", level=1)
    for r in h.runs: r.font.color.rgb = NAVYD
    para(E.INTRO, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    for sec in E.SECTIONS:
        h = doc.add_heading(sec["title"], level=1)
        for r in h.runs: r.font.color.rgb = NAVYD
        for t in sec["text"]:
            m = re.match(r"^(Q\d+\..+?\?)\s(.*)$", t, re.S)
            if m:
                p = doc.add_paragraph()
                r1 = p.add_run(m.group(1) + " "); r1.bold = True; r1.font.size = Pt(11.5)
                r2 = p.add_run(m.group(2)); r2.font.size = Pt(11.5)
                p.paragraph_format.space_after = Pt(6)
            else:
                para(t, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    out = "/app/outputs/explainer/Person_Segmentation_Explained.docx"
    doc.save(out)
    print("explainer DOCX:", out)

if __name__ == "__main__":
    build_pdf()
    build_docx()
