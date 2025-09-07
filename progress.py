from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Optional

from tqdm import tqdm


class ProgressSink(ABC):
    """
    Interface for streaming-friendly progress reporting.

    Methods are called incrementally during streaming pipelines.
    """

    @abstractmethod
    def set_totals(
            self,
            *,
            render_total: Optional[int],
            ocr_total: Optional[int],
            overlay_total: Optional[int]
    ) -> None:
        """
        Set total counts for each phase.

        :param render_total: Total pages to render.
        :param ocr_total: Total pages to OCR.
        :param overlay_total: Total pages to overlay.
        """
        raise NotImplementedError

    @abstractmethod
    def on_render_advance(self, n: int = 1) -> None:
        """
        Advance render progress by ``n``.

        :param n: Increment size.
        """
        raise NotImplementedError

    @abstractmethod
    def on_ocr_advance(self, n: int = 1) -> None:
        """
        Advance OCR progress by ``n``.

        :param n: Increment size.
        """
        raise NotImplementedError

    @abstractmethod
    def on_overlay_advance(self, n: int = 1) -> None:
        """
        Advance overlay progress by ``n``.

        :param n: Increment size.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Finalize the progress sink."""
        raise NotImplementedError


class TqdmProgressSink(ProgressSink, AbstractContextManager):
    """
    ``tqdm``-based progress sink suitable for streaming pipelines.

    Creates up to three bars for render, OCR, and overlay phases.
    """

    def __init__(self) -> None:
        self._bar_render: Optional[tqdm] = None
        self._bar_ocr: Optional[tqdm] = None
        self._bar_overlay: Optional[tqdm] = None

    def set_totals(
            self,
            *,
            render_total: Optional[int],
            ocr_total: Optional[int],
            overlay_total: Optional[int]
    ) -> None:
        if render_total is not None:
            self._bar_render = tqdm(total=render_total, desc="Render", unit="page")
        if ocr_total is not None:
            self._bar_ocr = tqdm(total=ocr_total, desc="OCR", unit="page")
        if overlay_total is not None:
            self._bar_overlay = tqdm(total=overlay_total, desc="Overlay", unit="page")

    def on_render_advance(self, n: int = 1) -> None:
        if self._bar_render is not None:
            self._bar_render.update(n)

    def on_ocr_advance(self, n: int = 1) -> None:
        if self._bar_ocr is not None:
            self._bar_ocr.update(n)

    def on_overlay_advance(self, n: int = 1) -> None:
        if self._bar_overlay is not None:
            self._bar_overlay.update(n)

    def close(self) -> None:
        for b in (self._bar_render, self._bar_ocr, self._bar_overlay):
            if b is not None:
                b.close()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
