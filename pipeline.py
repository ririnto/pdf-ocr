from __future__ import annotations

from typing import Optional

import pymupdf

from ocr_engine import OcrEngine
from overlay_writer import IncrementalOverlayWriter
from pdf_streamer import PdfStreamer
from progress import ProgressSink, TqdmProgressSink
from result_writers import CsvStreamWriter, NdjsonStreamWriter


class OCRPipeline:
    """
    End-to-end streaming pipeline: render → OCR → overlay.

    Handles progress reporting and streaming saves.
    """

    def __init__(
            self,
            input_pdf: str,
            output_pdf: str,
            *,
            dpi: int = 300,
            device: Optional[str] = "auto",
            lang: Optional[str] = "korean",
            rec_model: Optional[str] = "auto",
            font_path: Optional[str] = None,
            page_range: Optional[str] = None,
            pages: Optional[str] = None,
            page_step: int = 1,
            batch_size: int = 1,
            save_csv: Optional[str] = None,
            save_ndjson: Optional[str] = None,
            debug_visible: bool = False,
            sink: Optional[ProgressSink] = None,
    ) -> None:
        """
        Configure the streaming pipeline.

        :param input_pdf: Source PDF path.
        :param output_pdf: Output PDF path.
        :param dpi: Rendering DPI.
        :param device: Device string or ``"auto"``.
        :param lang: Language for runtime model selection.
        :param rec_model: ``"auto"`` or explicit recognition model name.
        :param font_path: Font file used for all text.
        :param page_range: Range filter like ``"10-50"``.
        :param pages: Comma-separated 1-based selection list.
        :param page_step: Sampling interval.
        :param batch_size: OCR batch size.
        :param save_csv: CSV output path.
        :param save_ndjson: NDJSON output path.
        :param debug_visible: Whether to also write a visible overlay PDF.
        :param sink: Progress sink implementation.
        """
        self.input_pdf = input_pdf
        self.output_pdf = output_pdf
        self.dpi = dpi
        self.font_path = font_path
        self.batch_size = max(1, batch_size)
        self.debug_visible = debug_visible
        self.sink = sink or TqdmProgressSink()
        with pymupdf.open(input_pdf) as d:
            total = d.page_count
        self.page_indices = PdfStreamer.select_pages(total, page_range=page_range, pages=pages, step=page_step)
        self.streamer = PdfStreamer(input_pdf, dpi=dpi)
        self.ocr = OcrEngine(device=device, lang=lang, rec_model=rec_model)
        self.writer = IncrementalOverlayWriter(
            input_pdf,
            output_pdf,
            font_path=font_path,
            dpi=dpi,
            debug_visible=debug_visible
        )
        self.csvw = CsvStreamWriter(save_csv)
        self.ndjw = NdjsonStreamWriter(save_ndjson)

    def run(self) -> None:
        """
        Execute the pipeline.

        :return: ``None``
        """
        total = len(self.page_indices)
        self.sink.set_totals(render_total=total, ocr_total=total, overlay_total=total)
        try:
            page_img_iter = self.streamer.iter_pages(self.page_indices, sink=self.sink)
            for pno, items in self.ocr.stream(page_img_iter, batch_size=self.batch_size, sink=self.sink):
                self.csvw.write_page(pno + 1, items)
                self.ndjw.write_page(pno + 1, items)
                self.writer.apply_and_save(pno, items, sink=self.sink)
        finally:
            self.csvw.close()
            self.ndjw.close()
            self.writer.close()
            self.sink.close()
