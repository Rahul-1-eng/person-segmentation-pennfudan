import re
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as PdfImage,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from PIL import Image

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "report.md"
OUTPUT = ROOT / "Person_Segmentation_Report.pdf"
PAGE_WIDTH = A4[0] - 1.3 * inch


def inline(text):
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return text


def build_table(lines, styles):
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    width = max(len(row) for row in rows)
    data = [
        [Paragraph(inline(cell), styles["TableCell"]) for cell in row + [""] * (width - len(row))]
        for row in rows
    ]
    table = Table(data, colWidths=[PAGE_WIDTH / width] * width, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE9F7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def build_image(path, caption, styles):
    target = ROOT / path
    if not target.exists():
        return [Paragraph(f"<i>[Figure pending: {escape(path)} has not been generated yet]</i>", styles["BodySmall"])]
    with Image.open(target) as image:
        width, height = image.size
    display_width = min(PAGE_WIDTH, 6.2 * inch)
    display_height = display_width * height / width
    if display_height > 6.2 * inch:
        display_height = 6.2 * inch
        display_width = display_height * width / height
    return [
        PdfImage(str(target), width=display_width, height=display_height),
        Paragraph(f"<i>{inline(caption)}</i>", styles["Caption"]),
    ]


def expand(lines):
    for raw in lines:
        match = re.fullmatch(r"\{\{TABLE:(.+?)\}\}", raw.strip())
        if match:
            path = ROOT / match.group(1)
            if path.exists():
                yield from path.read_text(encoding="utf-8").splitlines()
            else:
                yield f"*[Table pending: {match.group(1)} has not been generated yet]*"
        else:
            yield raw


def build():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=17, leading=22, spaceAfter=14))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=13, leading=17, textColor="#14365C", spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=9.5, leading=13, spaceAfter=5))
    styles.add(ParagraphStyle(name="TableCell", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=8.5, leading=11, alignment=TA_CENTER, spaceAfter=8, textColor="#444444"))
    story = []
    lines = list(expand(SOURCE.read_text(encoding="utf-8").splitlines()))
    index = 0
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if raw.startswith("```"):
            block = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                block.append(lines[index])
                index += 1
            story.append(Preformatted("\n".join(block), ParagraphStyle(name="Code", fontName="Courier", fontSize=8, leading=10, backColor="#F3F3F3", borderPadding=4, spaceAfter=8)))
        elif raw.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.append(build_table(table_lines, styles))
            story.append(Spacer(1, 6))
            continue
        elif not stripped:
            story.append(Spacer(1, 3))
        elif raw.startswith("# "):
            story.append(Paragraph(inline(stripped[2:]), styles["ReportTitle"]))
        elif raw.startswith("### "):
            story.append(Paragraph(inline(stripped[4:]), styles["Heading3"]))
        elif raw.startswith("## "):
            story.append(Paragraph(inline(stripped[3:]), styles["Section"]))
        elif re.match(r"!\[(.*)\]\((.+)\)", stripped):
            caption, path = re.match(r"!\[(.*)\]\((.+)\)", stripped).groups()
            story.extend(build_image(path, caption, styles))
        elif stripped.startswith("- "):
            story.append(Paragraph("• " + inline(stripped[2:]), styles["BodySmall"]))
        elif re.match(r"\d+\. ", stripped):
            story.append(Paragraph(inline(stripped), styles["BodySmall"]))
        elif stripped.startswith("*[") and stripped.endswith("]*"):
            story.append(Paragraph(f"<i>{escape(stripped[1:-1])}</i>", styles["BodySmall"]))
        else:
            story.append(Paragraph(inline(stripped), styles["BodySmall"]))
        index += 1
    output = OUTPUT
    try:
        with output.open("wb"):
            pass
    except PermissionError:
        output = ROOT / f"Person_Segmentation_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        print(f"Original PDF is locked; writing fallback file: {output}")
    SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=0.65 * inch, leftMargin=0.65 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch, title="Binary Person Segmentation on Penn-Fudan",
    ).build(story)
    print(output)


if __name__ == "__main__":
    build()
