from __future__ import annotations

import csv
import json
import os
from typing import List, Dict, Any, Optional


class CsvStreamWriter:
    """
    Streaming CSV writer for OCR results.

    Writes one row per OCR item as pages are processed.
    """

    def __init__(self, path: Optional[str]) -> None:
        """
        Initialize the writer.

        :param path: Output CSV path. If ``None``, the writer is disabled.
        """
        self.path = path
        self.fp = None
        self.w = None
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.fp = open(path, "w", newline="", encoding="utf-8")
            self.w = csv.writer(self.fp)
            self.w.writerow(["page", "x0", "y0", "x1", "y1", "score", "text"])

    def write_page(self, page_no_one_based: int, items: List[Dict[str, Any]]) -> None:
        """
        Append a page worth of results.

        :param page_no_one_based: Human-readable page number starting at 1.
        :param items: OCR items for the page.
        """
        if not self.w:
            return
        for it in items:
            poly = it.get("poly") or []
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            if xs and ys:
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            else:
                x0 = y0 = x1 = y1 = ""
            self.w.writerow([page_no_one_based, x0, y0, x1, y1, float(it.get("score", 0.0)), it.get("text")])

    def close(self) -> None:
        """Close the underlying file handle."""
        if self.fp:
            self.fp.close()


class NdjsonStreamWriter:
    """
    Streaming NDJSON writer for OCR results.

    Emits one JSON object per line for each OCR item.
    """

    def __init__(self, path: Optional[str]) -> None:
        """
        Initialize the writer.

        :param path: Output NDJSON path. If ``None``, the writer is disabled.
        """
        self.path = path
        self.fp = None
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.fp = open(path, "w", encoding="utf-8")

    def write_page(self, page_no_one_based: int, items: List[Dict[str, Any]]) -> None:
        """
        Append a page worth of results.

        :param page_no_one_based: Human-readable page number starting at 1.
        :param items: OCR items for the page.
        """
        if not self.fp:
            return
        for it in items:
            rec = {
                "page": page_no_one_based,
                "text": it.get("text"),
                "score": float(it.get("score", 0.0)),
                "poly": [[float(x), float(y)] for x, y in (it.get("poly") or [])],
            }
            self.fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def close(self) -> None:
        """Close the underlying file handle."""
        if self.fp:
            self.fp.close()
