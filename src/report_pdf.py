"""Reusable PDF report builder for ML experiment reports (ReportLab).

The builder is deliberately independent from ``ExperimentRunner``: it only
knows how to lay out headings, paragraphs, tables (from DataFrames) and
figures (from PNG files) in a consistent, professional style. The runner
decides *what* goes into the report.

Features
--------
* Cover page, automatic table of contents, page numbers, PDF bookmarks.
* Unicode support (DejaVu fonts bundled with matplotlib; Helvetica fallback).
* Tables built from DataFrames with wrapped cells (long experiment names and
  paths never overflow), repeated header rows, column splitting for very wide
  tables and optional temporary switch to landscape pages.
* Figures scaled to fit the page, kept together with their captions.

Install:  pip install reportlab
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

NOT_AVAILABLE = "Not Available"

MARGIN = 16 * mm
HEADER_SPACE = 8 * mm
FOOTER_SPACE = 9 * mm

PRIMARY = colors.HexColor("#1F3A5F")
ACCENT = colors.HexColor("#3B6EA5")
LIGHT_ROW = colors.HexColor("#F2F5F9")
GRID = colors.HexColor("#C9D1DC")
MUTED = colors.HexColor("#5F6B7A")


# ----------------------------------------------------------------
# Fonts and styles
# ----------------------------------------------------------------

_FONTS: Dict[str, str] = {}


def register_fonts() -> Dict[str, str]:
    """Register Unicode fonts once and return the font-name mapping.

    Uses the DejaVu family shipped with matplotlib (no extra download). If
    the files cannot be found, falls back to the built-in Helvetica/Courier
    fonts, which only cover Latin-1 characters.
    """

    if _FONTS:
        return _FONTS

    fonts = {
        "regular": "Helvetica",
        "bold": "Helvetica-Bold",
        "italic": "Helvetica-Oblique",
        "bold_italic": "Helvetica-BoldOblique",
        "mono": "Courier",
    }

    try:
        font_dir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")

        files = {
            "DejaVuSans": "DejaVuSans.ttf",
            "DejaVuSans-Bold": "DejaVuSans-Bold.ttf",
            "DejaVuSans-Oblique": "DejaVuSans-Oblique.ttf",
            "DejaVuSans-BoldOblique": "DejaVuSans-BoldOblique.ttf",
            "DejaVuSansMono": "DejaVuSansMono.ttf",
        }

        for font_name, file_name in files.items():
            pdfmetrics.registerFont(
                TTFont(font_name, os.path.join(font_dir, file_name))
            )

        pdfmetrics.registerFontFamily(
            "DejaVuSans",
            normal="DejaVuSans",
            bold="DejaVuSans-Bold",
            italic="DejaVuSans-Oblique",
            boldItalic="DejaVuSans-BoldOblique",
        )

        fonts = {
            "regular": "DejaVuSans",
            "bold": "DejaVuSans-Bold",
            "italic": "DejaVuSans-Oblique",
            "bold_italic": "DejaVuSans-BoldOblique",
            "mono": "DejaVuSansMono",
        }

    except Exception:  # pragma: no cover - depends on the environment
        pass

    _FONTS.update(fonts)

    return _FONTS


def _make_styles(fonts: Dict[str, str]) -> Dict[str, ParagraphStyle]:
    """Create the paragraph styles used throughout the report."""

    regular, bold, italic = fonts["regular"], fonts["bold"], fonts["italic"]

    return {
        "cover_title": ParagraphStyle(
            "cover_title", fontName=bold, fontSize=30, leading=36,
            textColor=PRIMARY, alignment=TA_LEFT, spaceAfter=8,
        ),
        "cover_subtitle": ParagraphStyle(
            "cover_subtitle", fontName=regular, fontSize=14, leading=19,
            textColor=MUTED, alignment=TA_LEFT, spaceAfter=6,
        ),
        "h1": ParagraphStyle(
            "h1", fontName=bold, fontSize=18, leading=22,
            textColor=PRIMARY, spaceBefore=0, spaceAfter=10,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "h2", fontName=bold, fontSize=13, leading=16,
            textColor=ACCENT, spaceBefore=10, spaceAfter=6,
            wordWrap="CJK",
        ),
        "h3": ParagraphStyle(
            "h3", fontName=bold, fontSize=10.5, leading=13,
            textColor=colors.black, spaceBefore=8, spaceAfter=4,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "body", fontName=regular, fontSize=9.5, leading=13.5,
            spaceAfter=5, wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "bullet", fontName=regular, fontSize=9.5, leading=13.5,
            leftIndent=14, bulletIndent=4, spaceAfter=2, wordWrap="CJK",
        ),
        "note": ParagraphStyle(
            "note", fontName=italic, fontSize=8.5, leading=11.5,
            textColor=MUTED, spaceAfter=5, wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "caption", fontName=italic, fontSize=8.5, leading=11,
            textColor=MUTED, alignment=TA_CENTER, spaceBefore=3,
            spaceAfter=10, wordWrap="CJK",
        ),
        "table_caption": ParagraphStyle(
            "table_caption", fontName=italic, fontSize=8.5, leading=11,
            textColor=MUTED, alignment=TA_LEFT, spaceBefore=6,
            spaceAfter=3, wordWrap="CJK",
        ),
        "toc_title": ParagraphStyle(
            "toc_title", fontName=bold, fontSize=18, leading=22,
            textColor=PRIMARY, spaceAfter=12,
        ),
    }


# ----------------------------------------------------------------
# Document template
# ----------------------------------------------------------------

class _ReportDocTemplate(BaseDocTemplate):
    """Document with cover / portrait / landscape page templates."""

    def __init__(
        self,
        filename: str,
        title: str,
        author: str,
        header_text: str,
        fonts: Dict[str, str],
    ) -> None:

        super().__init__(
            filename,
            pagesize=A4,
            title=title,
            author=author,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=MARGIN,
            bottomMargin=MARGIN,
        )

        self.header_text = header_text
        self.fonts = fonts
        self._bookmark_counter = 0

        portrait_w, portrait_h = A4
        land_w, land_h = landscape(A4)

        def make_frame(width: float, height: float, frame_id: str) -> Frame:
            return Frame(
                MARGIN,
                MARGIN + FOOTER_SPACE,
                width - 2 * MARGIN,
                height - 2 * MARGIN - FOOTER_SPACE - HEADER_SPACE,
                id=frame_id,
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
            )

        self.addPageTemplates([
            PageTemplate(
                id="cover",
                frames=[make_frame(portrait_w, portrait_h, "cover")],
                pagesize=A4,
            ),
            PageTemplate(
                id="portrait",
                frames=[make_frame(portrait_w, portrait_h, "portrait")],
                onPage=self._decorate,
                pagesize=A4,
            ),
            PageTemplate(
                id="landscape",
                frames=[make_frame(land_w, land_h, "landscape")],
                onPage=self._decorate,
                pagesize=landscape(A4),
            ),
        ])

    def _decorate(self, canvas, doc) -> None:
        """Draw running header and page number."""

        width, height = canvas._pagesize

        canvas.saveState()
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)

        canvas.setFont(self.fonts["regular"], 7.5)
        canvas.setFillColor(MUTED)

        canvas.drawString(
            MARGIN, height - MARGIN + 2 * mm, self.header_text[:120]
        )
        canvas.line(
            MARGIN, height - MARGIN, width - MARGIN, height - MARGIN
        )

        canvas.line(MARGIN, MARGIN + 6 * mm, width - MARGIN, MARGIN + 6 * mm)
        canvas.drawCentredString(
            width / 2, MARGIN, f"Page {doc.page}"
        )

        canvas.restoreState()

    def beforeDocument(self) -> None:
        """Reset per-pass state so TOC entries are identical across passes."""

        self._bookmark_counter = 0

    def afterFlowable(self, flowable) -> None:
        """Feed headings into the table of contents and PDF outline."""

        level = getattr(flowable, "_toc_level", None)

        if level is None:
            return

        text = flowable.getPlainText()
        key = f"heading_{self._bookmark_counter}"
        self._bookmark_counter += 1

        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page, key))


# ----------------------------------------------------------------
# Builder
# ----------------------------------------------------------------

class PdfReportBuilder:
    """Collects report content and renders it to a PDF file.

    A builder instance is single-use: create it, add content, call
    :meth:`build` once.
    """

    def __init__(
        self,
        title: str,
        author: str = "Experiment Runner",
    ) -> None:

        self.title = title
        self.author = author
        self.fonts = register_fonts()
        self.styles = _make_styles(self.fonts)

        self.story: List[Any] = []
        self._mode = "portrait"
        self._figure_count = 0
        self._table_count = 0

        self.toc = TableOfContents()
        self.toc.levelStyles = [
            ParagraphStyle(
                "toc0", fontName=self.fonts["bold"], fontSize=10.5,
                leading=15, leftIndent=0, spaceBefore=4,
            ),
            ParagraphStyle(
                "toc1", fontName=self.fonts["regular"], fontSize=9,
                leading=12, leftIndent=14,
            ),
        ]

    # ------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------

    @property
    def available_width(self) -> float:
        """Usable content width of the current page mode (points)."""

        page_w = landscape(A4)[0] if self._mode == "landscape" else A4[0]

        return page_w - 2 * MARGIN

    @property
    def available_height(self) -> float:
        """Usable content height of the current page mode (points)."""

        page_h = landscape(A4)[1] if self._mode == "landscape" else A4[1]

        return page_h - 2 * MARGIN - FOOTER_SPACE - HEADER_SPACE

    def _set_mode(self, mode: str) -> None:
        """Start a new page using the given page template."""

        self.story.append(NextPageTemplate(mode))
        self.story.append(PageBreak())
        self._mode = mode

    # ------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------

    def add_cover(
        self,
        subtitle: str,
        info_rows: Sequence[Tuple[str, str]],
    ) -> None:
        """Add the cover page followed by the table of contents."""

        st = self.styles

        self.story.append(Spacer(1, 55 * mm))
        self.story.append(Paragraph(escape(self.title), st["cover_title"]))
        self.story.append(Paragraph(escape(subtitle), st["cover_subtitle"]))
        self.story.append(Spacer(1, 18 * mm))

        self.add_key_value_table(info_rows, width=A4[0] - 2 * MARGIN)

        # Table of contents on the next (portrait) page.
        self._set_mode("portrait")
        self.story.append(Paragraph("Table of Contents", st["toc_title"]))
        self.story.append(self.toc)

    def begin_section(self, title: str, landscape_page: bool = False) -> None:
        """Start a top-level section on a new page."""

        self._set_mode("landscape" if landscape_page else "portrait")
        self._heading(title, "h1", toc_level=0)

    def add_subsection(self, title: str, toc: bool = False) -> None:
        """Add a second-level heading (optionally listed in the TOC)."""

        self.story.append(CondPageBreak(45 * mm))
        self._heading(title, "h2", toc_level=1 if toc else None)

    def add_subsubsection(self, title: str) -> None:
        """Add a third-level heading (never listed in the TOC)."""

        self.story.append(CondPageBreak(30 * mm))
        self._heading(title, "h3", toc_level=None)

    def _heading(self, text: str, style: str, toc_level: Optional[int]) -> None:

        paragraph = Paragraph(escape(text), self.styles[style])

        if toc_level is not None:
            paragraph._toc_level = toc_level

        self.story.append(paragraph)

    # ------------------------------------------------------------
    # Text
    # ------------------------------------------------------------

    def add_paragraph(
        self,
        text: str,
        style: str = "body",
        markup: bool = False,
    ) -> None:
        """Add a paragraph. ``markup=True`` allows ReportLab inline tags."""

        self.story.append(
            Paragraph(text if markup else escape(text), self.styles[style])
        )

    def add_note(self, text: str) -> None:
        """Add a small italic grey note."""

        self.add_paragraph(text, style="note")

    def add_bullets(self, items: Sequence[str]) -> None:
        """Add a bulleted list (items are plain text)."""

        for item in items:
            self.story.append(
                Paragraph(escape(str(item)), self.styles["bullet"],
                          bulletText="\u2022")
            )

        self.story.append(Spacer(1, 4))

    def add_spacer(self, height_mm: float = 4) -> None:
        self.story.append(Spacer(1, height_mm * mm))

    def add_page_break(self) -> None:
        self.story.append(PageBreak())

    def add_preformatted(self, text: str, font_size: float = 7.2) -> None:
        """Add monospaced text (e.g. a classification report)."""

        text = text.rstrip("\n")
        lines = text.split("\n")
        longest = max((len(line) for line in lines), default=1)

        # Shrink the font so the longest line fits the page width.
        fit_size = self.available_width / (max(longest, 1) * 0.61)
        size = max(4.5, min(font_size, fit_size))

        style = ParagraphStyle(
            "pre", fontName=self.fonts["mono"], fontSize=size,
            leading=size * 1.3, backColor=LIGHT_ROW, borderPadding=4,
            spaceBefore=4, spaceAfter=8,
        )

        block = Preformatted(text, style)

        self.story.append(KeepTogether([block]) if len(lines) < 45 else block)

    # ------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------

    @staticmethod
    def _format_cell(column: str, value: Any, decimals: Optional[int]) -> str:
        """Format one table value as display text."""

        if value is None:
            return NOT_AVAILABLE

        if isinstance(value, float) and math.isnan(value):
            return NOT_AVAILABLE

        if isinstance(value, bool):
            return "Yes" if value else "No"

        if isinstance(value, int):
            return f"{value:,}"

        if isinstance(value, float):

            if abs(value) >= 1000:
                return f"{value:,.0f}"

            digits = decimals

            if digits is None:
                digits = 2 if any(
                    tag in column for tag in ("(%)", "(s)", "(ms)", "Gap")
                ) else 4

            return f"{value:.{digits}f}"

        return str(value)

    def _build_table(
        self,
        df: pd.DataFrame,
        width: float,
        font_size: float,
        decimals: Optional[int],
        id_cols: int,
    ) -> Table:

        fonts = self.fonts

        head_style = ParagraphStyle(
            "th", fontName=fonts["bold"], fontSize=font_size,
            leading=font_size * 1.25, textColor=colors.white,
            alignment=TA_CENTER,
        )

        cell_left = ParagraphStyle(
            "td_l", fontName=fonts["regular"], fontSize=font_size,
            leading=font_size * 1.25, alignment=TA_LEFT, wordWrap="CJK",
        )

        cell_center = ParagraphStyle(
            "td_c", parent=cell_left, alignment=TA_CENTER, wordWrap=None,
        )

        columns = [str(c) for c in df.columns]

        data: List[List[Any]] = [
            [Paragraph(escape(c), head_style) for c in columns]
        ]

        texts: List[List[str]] = []

        for row in df.itertuples(index=False):

            formatted = [
                self._format_cell(col, val, decimals)
                for col, val in zip(columns, row)
            ]

            texts.append(formatted)

            data.append([
                Paragraph(
                    escape(text),
                    cell_left if i < id_cols else cell_center,
                )
                for i, text in enumerate(formatted)
            ])

        # Column widths proportional to content length.
        weights = []

        for i, col in enumerate(columns):

            longest_word = max((len(w) for w in col.split()), default=4)
            cell_len = max((len(t[i]) for t in texts), default=4)

            weights.append(
                min(max(cell_len, longest_word + 2, 6), 42 if i < id_cols else 24)
            )

        total = float(sum(weights))
        col_widths = [width * w / total for w in weights]

        table = Table(data, colWidths=col_widths, repeatRows=1)

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_ROW]),
            ("GRID", (0, 0), (-1, -1), 0.4, GRID),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))

        return table

    def add_table(
        self,
        df: Optional[pd.DataFrame],
        caption: Optional[str] = None,
        landscape_page: bool = False,
        decimals: Optional[int] = None,
        id_cols: int = 1,
        font_size: Optional[float] = None,
        keep_together_rows: int = 16,
    ) -> None:
        """Add a DataFrame as a formatted table.

        Very wide tables are split column-wise into several tables, each
        repeating the first ``id_cols`` identifier columns.

        Parameters
        ----------
        landscape_page : temporarily switch to landscape pages for this
            table and return to the previous page mode afterwards.
        """

        if df is None or len(df) == 0 or len(df.columns) == 0:
            self.add_note(
                f"{caption or 'Table'}: {NOT_AVAILABLE}."
            )
            return

        previous_mode = self._mode
        switched = landscape_page and previous_mode != "landscape"

        if switched:
            self._set_mode("landscape")

        width = self.available_width
        max_cols = 13 if self._mode == "landscape" else 8
        id_cols = min(id_cols, len(df.columns) - 1) if len(df.columns) > 1 else 1

        data_cols = list(df.columns[id_cols:])
        step = max(1, max_cols - id_cols)
        chunks = [data_cols[i:i + step] for i in range(0, len(data_cols), step)] or [[]]

        size = font_size or (7.5 if len(df.columns) > 6 else 8.5)

        for part, chunk in enumerate(chunks, start=1):

            self._table_count += 1

            sub = df[list(df.columns[:id_cols]) + chunk]
            table = self._build_table(sub, width, size, decimals, id_cols)

            label = f"Table {self._table_count}."

            if caption:
                label += f" {caption}"

            if len(chunks) > 1:
                label += f" (part {part} of {len(chunks)})"

            cap = Paragraph(escape(label), self.styles["table_caption"])

            if len(sub) <= keep_together_rows:
                self.story.append(KeepTogether([cap, table, Spacer(1, 8)]))
            else:
                self.story.append(CondPageBreak(40 * mm))
                self.story.append(cap)
                self.story.append(table)
                self.story.append(Spacer(1, 8))

        if switched:
            self._set_mode(previous_mode)

    def add_key_value_table(
        self,
        rows: Sequence[Tuple[str, str]],
        width: Optional[float] = None,
    ) -> None:
        """Add a two-column key/value table without header."""

        if not rows:
            return

        width = width or self.available_width
        fonts = self.fonts

        key_style = ParagraphStyle(
            "kv_k", fontName=fonts["bold"], fontSize=9.5, leading=13,
            textColor=PRIMARY, wordWrap="CJK",
        )
        val_style = ParagraphStyle(
            "kv_v", fontName=fonts["regular"], fontSize=9.5, leading=13,
            wordWrap="CJK",
        )

        data = [
            [Paragraph(escape(str(k)), key_style),
             Paragraph(escape(str(v)), val_style)]
            for k, v in rows
        ]

        table = Table(data, colWidths=[width * 0.32, width * 0.68])

        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ]))

        self.story.append(table)
        self.story.append(Spacer(1, 8))

    # ------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------

    def add_figure(
        self,
        path: Optional[str],
        caption: str,
        max_height_fraction: float = 0.82,
    ) -> None:
        """Add a PNG figure scaled to the page, kept with its caption.

        If the image is missing the caption is emitted with a
        "Not Available" notice instead.
        """

        self._figure_count += 1
        label = f"Figure {self._figure_count}. {caption}"

        if not path or not os.path.isfile(path):

            self.add_note(f"{label} ({NOT_AVAILABLE}: figure was not generated.)")
            return

        try:
            img_w, img_h = ImageReader(path).getSize()
        except Exception:
            self.add_note(f"{label} ({NOT_AVAILABLE}: image could not be read.)")
            return

        scale = min(
            self.available_width / img_w,
            self.available_height * max_height_fraction / img_h,
        )

        image = Image(path, width=img_w * scale, height=img_h * scale)
        image.hAlign = "CENTER"

        self.story.append(
            KeepTogether([image, Paragraph(escape(label), self.styles["caption"])])
        )

    # ------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------

    def build(self, path: str, header_text: Optional[str] = None) -> str:
        """Render the PDF to ``path`` (written atomically) and return it."""

        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)

        tmp_path = path + ".tmp"

        doc = _ReportDocTemplate(
            tmp_path,
            title=self.title,
            author=self.author,
            header_text=header_text or self.title,
            fonts=self.fonts,
        )

        try:
            doc.multiBuild(self.story)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return path
