"""Build the deep-learning guide PDF from its Markdown source."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_DIR / "docs" / "deep-learning-guide.md"
PDF_PATH = PROJECT_DIR / "docs" / "deep-learning-guide.pdf"


def paragraph_markup(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    return re.sub(r"`(.+?)`", r'<font name="Courier">\1</font>', escaped)


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "GuideTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            textColor=colors.HexColor("#17365d"),
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            "GuideHeading",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#17365d"),
            spaceBefore=14,
            spaceAfter=7,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            "GuideSubheading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#24527a"),
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            "GuideBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "GuideBullet",
            parent=styles["GuideBody"],
            leftIndent=16,
            firstLineIndent=-10,
            bulletIndent=0,
        )
    )
    styles.add(
        ParagraphStyle(
            "GuideCode",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            backColor=colors.HexColor("#f1f4f8"),
            borderPadding=7,
            leftIndent=8,
            rightIndent=8,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    return styles


def markdown_to_flowables(markdown: str, styles) -> list:
    flowables = []
    paragraph_lines: list[str] = []
    code_lines: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        if paragraph_lines:
            flowables.append(
                Paragraph(paragraph_markup(" ".join(paragraph_lines)), styles["GuideBody"])
            )
            paragraph_lines.clear()

    for line in markdown.splitlines():
        if line.startswith("```"):
            flush_paragraph()
            if in_code:
                flowables.append(Preformatted("\n".join(code_lines), styles["GuideCode"]))
                code_lines.clear()
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line.strip():
            flush_paragraph()
        elif line.startswith("# "):
            flush_paragraph()
            flowables.append(Spacer(1, 0.25 * inch))
            flowables.append(
                Paragraph(paragraph_markup(line[2:]), styles["GuideTitle"])
            )
        elif line.startswith("## "):
            flush_paragraph()
            flowables.append(
                Paragraph(paragraph_markup(line[3:]), styles["GuideHeading"])
            )
        elif line.startswith("### "):
            flush_paragraph()
            flowables.append(
                Paragraph(paragraph_markup(line[4:]), styles["GuideSubheading"])
            )
        elif line.startswith("- "):
            flush_paragraph()
            flowables.append(
                Paragraph(
                    "- " + paragraph_markup(line[2:]),
                    styles["GuideBullet"],
                )
            )
        else:
            paragraph_lines.append(line.strip())

    flush_paragraph()
    if in_code:
        raise ValueError("Markdown contains an unclosed fenced code block.")
    return flowables


def add_page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(0.7 * inch, 0.45 * inch, "Malware MLP learning guide")
    canvas.drawRightString(7.8 * inch, 0.45 * inch, f"Page {document.page}")
    canvas.restoreState()


def main() -> None:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"Guide source not found: {SOURCE_PATH}")
    styles = make_styles()
    document = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Deep Learning for Malware Classification: A Project Guide",
        author="Malware Classification Framework",
    )
    content = SOURCE_PATH.read_text(encoding="utf-8")
    document.build(
        markdown_to_flowables(content, styles),
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )
    print(f"Created {PDF_PATH}")


if __name__ == "__main__":
    main()
