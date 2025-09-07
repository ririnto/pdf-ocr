from __future__ import annotations

from typing import List, Dict, Any, Optional

import fitz
import unicodedata as _ud

from progress import ProgressSink


class IncrementalOverlayWriter:
    """
    Incrementally overlay invisible text into a PDF and save changes.

    Uses ``render_mode=3`` for invisible text and saves incrementally via ``saveIncr``.
    """

    def __init__(
            self,
            input_pdf: str,
            output_pdf: str,
            font_path: Optional[str],
            dpi: int = 300,
            debug_visible: bool = False
    ) -> None:
        """
        Prepare documents for incremental updates.

        :param input_pdf: Source PDF path.
        :param output_pdf: Target PDF path.
        :param font_path: Font file path used for all inserted text.
        :param dpi: Rendering DPI used for coordinate conversion.
        :param debug_visible: Whether to maintain a parallel visible overlay PDF.
        """
        import shutil
        self.scale = dpi / 72.0
        self.font_path = font_path
        self.output_pdf = output_pdf
        self.debug_visible = debug_visible
        shutil.copyfile(input_pdf, output_pdf)
        self.doc = fitz.open(output_pdf)
        self.dbg_doc = None
        self.dbg_path = None
        if debug_visible:
            self.dbg_path = output_pdf.replace('.pdf', '_debug.pdf')
            shutil.copyfile(input_pdf, self.dbg_path)
            self.dbg_doc = fitz.open(self.dbg_path)

    def _rect(self, poly) -> tuple[float, float, float, float]:
        """
        Compute bounding rectangle for a polygon.

        :param poly: Sequence of ``(x, y)``.
        :return: ``(x0, y0, x1, y1)``.
        """
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        return min(xs), min(ys), max(xs), max(ys)

    def _apply_one(self, doc: fitz.Document, page_no: int, items: List[Dict[str, Any]], visible: bool) -> None:
        """
        Apply overlay to a single page of the document.

        :param doc: Opened PyMuPDF document.
        :param page_no: 0-based page index.
        :param items: OCR items list.
        :param visible: Whether to draw visible text.
        """
        page = doc.load_page(page_no)
        page.wrap_contents()
        if self.font_path:
            page.insert_text((0, 0), " ", fontname="ocrfont", fontfile=self.font_path, fontsize=1, render_mode=3)
        for it in items:
            raw = it.get("text")
            poly = it.get("poly")
            if not raw or poly is None:
                continue
            text = _ud.normalize("NFC", raw)
            x0, y0, x1, y1 = self._rect(poly)
            x0_pt, y0_pt = x0 / self.scale, y0 / self.scale
            x1_pt, y1_pt = x1 / self.scale, y1 / self.scale
            width_pt = max(0.1, x1_pt - x0_pt)
            if self.font_path:
                font = fitz.Font(fontname="ocrfont", fontfile=self.font_path)
                font_name = "ocrfont"
            else:
                font = fitz.Font("helv")
                font_name = "helv"
            w_at_1 = max(1e-6, font.text_length(text, fontsize=1))
            font_size = width_pt / w_at_1
            baseline_y = y1_pt
            desc = font.descender
            asc = font.ascender
            if asc - desc != 0:
                baseline_y += (desc / (asc - desc)) * font_size
            insert_pt = fitz.Point(x0_pt, baseline_y)
            page.insert_text(
                insert_pt, text,
                fontname=font_name,
                fontfile=(self.font_path if self.font_path else None),
                fontsize=font_size,
                color=(1, 0, 0) if visible else None,
                render_mode=0 if visible else 3,
                overlay=True,
            )

    def apply_and_save(self, page_no: int, items: List[Dict[str, Any]], sink: Optional[ProgressSink] = None) -> None:
        """
        Apply overlays for one page and save incrementally.

        :param page_no: 0-based page index.
        :param items: OCR items to overlay.
        :param sink: Optional progress sink for overlay increment.
        """
        self._apply_one(self.doc, page_no, items, visible=False)
        if self.dbg_doc is not None:
            self._apply_one(self.dbg_doc, page_no, items, visible=True)
        self.doc.saveIncr()
        if self.dbg_doc is not None:
            self.dbg_doc.saveIncr()
        if sink is not None:
            sink.on_overlay_advance(1)

    def close(self) -> None:
        """Close all opened documents."""
        self.doc.close()
        if self.dbg_doc is not None:
            self.dbg_doc.close()
