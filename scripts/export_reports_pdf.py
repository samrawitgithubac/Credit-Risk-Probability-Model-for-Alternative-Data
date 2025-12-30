"""
Export markdown reports in /reports to PDF.

This is a simple exporter that renders markdown as *plain text* in PDF pages.
It is reliable on Windows and avoids heavyweight dependencies like pandoc/latex.

Usage:
  python scripts/export_reports_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


def _wrap_lines(text: str, max_width_pts: float, font_name: str, font_size: int) -> list[str]:
    lines_out: list[str] = []
    for raw_line in text.splitlines():
        # Preserve blank lines
        if raw_line.strip() == "":
            lines_out.append("")
            continue

        words = raw_line.split(" ")
        current = ""
        for w in words:
            candidate = (current + " " + w).strip() if current else w
            if stringWidth(candidate, font_name, font_size) <= max_width_pts:
                current = candidate
            else:
                if current:
                    lines_out.append(current)
                current = w
        if current:
            lines_out.append(current)
    return lines_out


def md_to_pdf(md_path: Path, pdf_path: Path) -> None:
    page_w, page_h = LETTER
    margin = 0.75 * inch
    font_name = "Helvetica"
    font_size = 10
    line_h = 12

    c = canvas.Canvas(str(pdf_path), pagesize=LETTER)
    c.setFont(font_name, font_size)

    text = md_path.read_text(encoding="utf-8")
    max_width = page_w - 2 * margin
    lines = _wrap_lines(text, max_width, font_name, font_size)

    x = margin
    y = page_h - margin

    for line in lines:
        if y < margin + line_h:
            c.showPage()
            c.setFont(font_name, font_size)
            y = page_h - margin

        c.drawString(x, y, line)
        y -= line_h

    c.save()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    reports_dir = root / "reports"
    interim_md = reports_dir / "interim_report.md"
    final_md = reports_dir / "final_report.md"

    if not interim_md.exists() or not final_md.exists():
        raise FileNotFoundError("Missing reports/*.md files. Expected interim_report.md and final_report.md.")

    md_to_pdf(interim_md, reports_dir / "interim_report.pdf")
    md_to_pdf(final_md, reports_dir / "final_report.pdf")

    print(f"Wrote: {reports_dir / 'interim_report.pdf'}")
    print(f"Wrote: {reports_dir / 'final_report.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


