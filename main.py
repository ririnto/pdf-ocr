from __future__ import annotations

import argparse

from pipeline import OCRPipeline


def main() -> None:
    """
    CLI entrypoint for the class-based streaming OCR pipeline.

    :return: ``None``
    """
    p = argparse.ArgumentParser(description="Scanned PDF → searchable PDF (streaming, class-based)")
    p.add_argument("input_pdf")
    p.add_argument("output_pdf")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--lang", type=str, default="korean")
    p.add_argument("--rec-model", type=str, default="auto")
    p.add_argument("--font", dest="font_path", type=str, default=None)
    p.add_argument("--page-range", type=str, default=None)
    p.add_argument("--pages", type=str, default=None)
    p.add_argument("--page-step", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--save-csv", type=str, default=None)
    p.add_argument("--save-ndjson", type=str, default=None)
    p.add_argument("--debug-visible", action="store_true")

    args = p.parse_args()

    pipe = OCRPipeline(
        input_pdf=args.input_pdf,
        output_pdf=args.output_pdf,
        dpi=args.dpi,
        device=args.device,
        lang=args.lang,
        rec_model=args.rec_model,
        font_path=args.font_path,
        page_range=args.page_range,
        pages=args.pages,
        page_step=args.page_step,
        batch_size=args.batch_size,
        save_csv=args.save_csv,
        save_ndjson=args.save_ndjson,
        debug_visible=args.debug_visible,
    )
    pipe.run()


if __name__ == "__main__":
    main()
