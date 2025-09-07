from __future__ import annotations

import shutil
import subprocess
from typing import Iterator, Iterable, List, Tuple, Dict, Any, Optional

import numpy as np
from paddleocr import PaddleOCR

from progress import ProgressSink


class OcrEngine:
    """
    Wrapper around PaddleOCR with runtime model selection and batch streaming.

    Provides incremental OCR over an image stream with progress reporting.
    """

    def __init__(
            self,
            *,
            device: Optional[str] = None,
            lang: Optional[str] = "korean",
            rec_model: Optional[str] = "auto",
            use_doc_orientation_classify: bool = False,
            use_textline_orientation: bool = False,
    ) -> None:
        """
        Initialize the OCR engine.

        :param device: Device string like ``"gpu:0"`` or ``"cpu"``. ``None`` selects automatically.
        :param lang: Language code used by PaddleOCR for runtime model selection.
        :param rec_model: ``"auto"`` or explicit recognition model name.
        :param use_doc_orientation_classify: Enable document orientation classifier.
        :param use_textline_orientation: Enable text line orientation classifier.
        """
        if device in (None, "auto"):
            device = self._auto_select_device()
        kwargs = dict(
            device=device,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_textline_orientation=use_textline_orientation,
            use_doc_unwarping=False,
        )
        if lang:
            kwargs["lang"] = lang
        if rec_model and rec_model != "auto":
            kwargs["text_recognition_model_name"] = rec_model
        self._ocr = PaddleOCR(**kwargs)

    @staticmethod
    def _auto_select_device() -> str:
        """
        Choose the best available device.

        :return: ``"gpu:{i}"`` if CUDA is available, otherwise ``"cpu"``.
        """
        try:
            import paddle
            if getattr(paddle.device, "is_compiled_with_cuda", lambda: False)():
                try:
                    n = paddle.device.cuda.device_count()
                except Exception:
                    n = 0
                if n and n > 0:
                    idx = 0
                    try:
                        if shutil.which("nvidia-smi"):
                            out = subprocess.check_output(
                                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                text=True,
                            )
                            free = [int(x.strip()) for x in out.strip().splitlines() if x.strip().isdigit()]
                            if free:
                                idx = max(range(min(len(free), n)), key=lambda i: free[i])
                    except Exception:
                        pass
                    return f"gpu:{idx}"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def _parse_result_obj(res: Any) -> Dict[str, Any]:
        """
        Normalize PaddleOCR result across versions.

        :param res: PaddleOCR result object.
        :return: Dictionary containing fields like ``rec_texts``, ``rec_scores``, ``rec_polys``.
        """
        data = None
        if hasattr(res, "json"):
            data = res.json
            if isinstance(data, dict) and "res" in data and isinstance(data["res"], dict):
                data = data["res"]
        elif hasattr(res, "res"):
            data = res.res
        return data if isinstance(data, dict) else {}

    def stream(
            self,
            page_img_iter: Iterable[Tuple[int, np.ndarray]],
            batch_size: int,
            sink: Optional[ProgressSink] = None
    ) -> Iterator[Tuple[int, List[Dict[str, Any]]]]:
        """
        Run OCR in batches over a page image stream.

        :param page_img_iter: Iterable of ``(page_no, image)``.
        :param batch_size: Number of pages per OCR batch.
        :param sink: Optional progress sink to advance OCR count.
        :yield: ``(page_no, items)`` per page, where items are dicts with ``poly``, ``text``, ``score``.
        """
        if batch_size < 1:
            batch_size = 1
        batch_pages: List[int] = []
        batch_imgs: List[np.ndarray] = []

        def _flush():
            nonlocal batch_pages, batch_imgs
            if not batch_imgs:
                return
            res_list = self._ocr.predict(batch_imgs)
            for i, res in enumerate(res_list):
                data = self._parse_result_obj(res)
                texts = data.get("rec_texts") or []
                scores = data.get("rec_scores") or []
                polys = data.get("rec_polys") or []
                items: List[Dict[str, Any]] = []
                for k, t in enumerate(texts):
                    poly = polys[k] if k < len(polys) else None
                    sc = float(scores[k]) if k < len(scores) else 0.0
                    if poly is None or t is None:
                        continue
                    items.append({"poly": poly, "text": t, "score": sc})
                if sink is not None:
                    sink.on_ocr_advance(1)
                yield batch_pages[i], items
            batch_pages, batch_imgs = [], []

        for pno, img in page_img_iter:
            batch_pages.append(pno)
            batch_imgs.append(img)
            if len(batch_imgs) >= batch_size:
                yield from _flush()
        yield from _flush()
