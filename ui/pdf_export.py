"""Экспорт Markdown-отчёта агента в PDF (fpdf2 + шрифты DejaVu из поставки matplotlib — кириллица без системных шрифтов).

Поддерживается то, что пишет агент: заголовки #/##/###, списки, таблицы Markdown, абзацы, **жирный**/`код` (как текст).
"""
from __future__ import annotations

import re
from pathlib import Path

try:
    import matplotlib
    from fpdf import FPDF
    from fpdf.enums import WrapMode
    FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    AVAILABLE = (FONT_DIR / "DejaVuSans.ttf").exists()
except Exception:  # noqa: BLE001 — без fpdf2 кнопка PDF просто не показывается
    AVAILABLE = False

_INLINE = re.compile(r"\*\*|__|`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _clean(s: str) -> str:
    s = _LINK.sub(r"\1 (\2)", s)
    return _INLINE.sub("", s).replace("\t", "    ").strip()


def _table_row(line: str) -> list[str] | None:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
        return None                                   # строка-разделитель |---|---|
    return [_clean(c) for c in cells]


def markdown_to_pdf(md: str, title: str = "Отчёт WindAgent") -> bytes:
    if not AVAILABLE:
        raise RuntimeError("PDF-экспорт недоступен: нет fpdf2")
    pdf = FPDF(format="A4")
    pdf.set_margins(16, 14, 16)
    pdf.set_auto_page_break(True, margin=14)
    pdf.add_font("DejaVu", "", str(FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.add_font("DejaVuMono", "", str(FONT_DIR / "DejaVuSansMono.ttf"))
    pdf.set_title(title)
    pdf.set_author("WindAgent")
    pdf.add_page()
    w = pdf.epw

    def para(text: str, size: float = 10, style: str = "", font: str = "DejaVu", gap: float = 1.2, indent: float = 0):
        if not text:
            return
        pdf.set_font(font, style, size)
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(w - indent, size * 0.5, text, wrapmode=WrapMode.CHAR, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(gap)

    in_code = False
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            para(line, 8, font="DejaVuMono", gap=0)
            continue
        s = line.strip()
        if not s:
            pdf.ln(1.5)
            continue
        if s.startswith("#"):
            level = len(s) - len(s.lstrip("#"))
            size = {1: 16, 2: 13, 3: 11.5}.get(level, 11)
            pdf.ln(2 if level > 1 else 0)
            para(_clean(s.lstrip("#")), size, "B", gap=2)
            continue
        if s.startswith("|"):
            row = _table_row(s)
            if row:
                para(" · ".join(c for c in row if c), 9, gap=0.4)
            continue
        m = re.match(r"^([-*+]|\d+[.)])\s+(.*)", s)
        if m:
            indent = min(len(line) - len(line.lstrip()), 8) * 1.2
            mark = "•" if m.group(1) in "-*+" else m.group(1)
            para(f"{mark} {_clean(m.group(2))}", 10, gap=0.6, indent=3 + indent)
            continue
        if s.startswith(">"):
            para(_clean(s.lstrip("> ")), 9.5, gap=1, indent=4)
            continue
        if re.fullmatch(r"[-*_]{3,}", s):
            y = pdf.get_y() + 1
            pdf.line(pdf.l_margin, y, pdf.l_margin + w, y)
            pdf.ln(3)
            continue
        para(_clean(s), 10)
    return bytes(pdf.output())
