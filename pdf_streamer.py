from __future__ import annotations

from typing import Iterable, Iterator, List, Tuple, Optional

import fitz
import numpy as np

from progress import ProgressSink


class PdfStreamer:
    """
    Stream pages of a PDF as RGB NumPy arrays.

    This class yields pages one by one to avoid loading the entire document into memory.
    """

    def __init__(self, pdf_path: str, dpi: int = 300) -> None:
        """
        Initialize the streamer.

        :param pdf_path: Path to input PDF.
        :param dpi: Rendering DPI.
        """
        self.pdf_path = pdf_path
        self.dpi = dpi

    @staticmethod
    def select_pages(
            total_pages: int,
            page_range: Optional[str] = None,
            pages: Optional[str] = None,
            step: int = 1
    ) -> List[int]:
        """
        Compute selected 0-based page indices.

        :param total_pages: Total number of pages.
        :param page_range: Inclusive 1-based range like ``"10-50"``.
        :param pages: Comma-separated 1-based list like ``"1,5,9"``.
        :param step: Sampling interval.
        :return: Sorted indices.
        """
        chosen: set[int] = set()
        if page_range:
            a, b = page_range.split("-")
            start = max(1, int(a))
            end = int(b)
            chosen.update(range(start - 1, end))
        if pages:
            for tok in pages.split(','):
                tok = tok.strip()
                if tok:
                    chosen.add(max(0, int(tok) - 1))
        if not page_range and not pages:
            chosen.update(range(total_pages))
        idx = sorted(i for i in chosen if 0 <= i < total_pages)
        if step > 1:
            idx = idx[::step]
        return idx

    def iter_pages(
            self,
            page_indices: Iterable[int],
            sink: Optional[ProgressSink] = None
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Yield ``(page_no, image)`` for each selected page.

        :param page_indices: 0-based page indices to render.
        :param sink: Optional progress sink to advance render count.
        :yield: Tuple of page number and RGB ``uint8`` array of shape ``(H, W, 3)``.
        """
        doc = fitz.open(self.pdf_path)
        try:
            for pno in page_indices:
                page = doc.load_page(pno)
                pix = page.get_pixmap(dpi=self.dpi, colorspace=fitz.csRGB, alpha=False)
                buf = pix.samples
                h, w, n = pix.height, pix.width, pix.n
                arr = np.frombuffer(buf, dtype=np.uint8)
                if n == 4 and pix.alpha:
                    arr = arr.reshape(h, w, 4)[:, :, :3]
                elif n >= 3:
                    arr = arr.reshape(h, w, n)[:, :, :3]
                else:
                    arr = arr.reshape(h, w, 1)
                    arr = np.repeat(arr, 3, axis=2)
                if sink is not None:
                    sink.on_render_advance(1)
                yield pno, np.ascontiguousarray(arr)
        finally:
            doc.close()
