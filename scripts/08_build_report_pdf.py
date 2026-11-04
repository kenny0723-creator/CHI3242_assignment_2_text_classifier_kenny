"""將 report.md 轉成排版整齊的 report.pdf（A4、中文字體、無 Markdown 符號）。"""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "report.md"
REPORT_PDF = ROOT / "report.pdf"

FONT_REGULAR = "MSJH"
FONT_BOLD = "MSJHB"
pdfmetrics.registerFont(TTFont(FONT_REGULAR, r"C:\Windows\Fonts\msjh.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont(FONT_BOLD, r"C:\Windows\Fonts\msjhbd.ttc", subfontIndex=0))

BODY = ParagraphStyle(
    "Body",
    fontName=FONT_REGULAR,
    fontSize=10.5,
    leading=17,
    alignment=TA_JUSTIFY,
    spaceAfter=7,
    firstLineIndent=0,
)
H1 = ParagraphStyle(
    "H1", parent=BODY, fontName=FONT_BOLD, fontSize=17, leading=24,
    spaceBefore=6, spaceAfter=12,
)
H2 = ParagraphStyle(
    "H2", parent=BODY, fontName=FONT_BOLD, fontSize=13.5, leading=20,
    spaceBefore=14, spaceAfter=8,
)
CODE = ParagraphStyle(
    "Code", parent=BODY, fontName=FONT_REGULAR, fontSize=9, leading=13,
    backColor="#f3f4f0", borderPadding=5, spaceBefore=4, spaceAfter=8,
)
def md_inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"<font face='MSJH' backColor='#f3f4f0'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def parse_blocks(markdown_text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    lines = markdown_text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            blocks.append(("code", "\n".join(code_lines)))
            continue
        if line.startswith("# "):
            blocks.append(("h1", line[2:].strip()))
        elif line.startswith("## "):
            blocks.append(("h2", line[3:].strip()))
        elif line.strip():
            blocks.append(("p", line.strip()))
        index += 1
    return blocks


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_story(blocks: list[tuple[str, str]]) -> list:
    story: list = []
    for kind, content in blocks:
        if kind == "code":
            story.append(Paragraph(escape(content).replace("\n", "<br/>"), CODE))
        elif kind == "h1":
            story.append(Paragraph(escape(content), H1))
        elif kind == "h2":
            story.append(Paragraph(escape(content), H2))
        else:
            story.append(Paragraph(md_inline(escape(content)), BODY))
    return story


def main() -> None:
    markdown_text = REPORT_MD.read_text(encoding="utf-8")
    blocks = parse_blocks(markdown_text)
    doc = BaseDocTemplate(
        str(REPORT_PDF),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.2 * cm,
        title="具體語言與抽象語言：以邏輯迴歸分類中文小說句子",
        author="",
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main"
    )
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame])])
    doc.build(build_story(blocks))
    print(f"已產生 {REPORT_PDF.name}（{REPORT_PDF.stat().st_size:,} bytes）")


if __name__ == "__main__":
    main()
