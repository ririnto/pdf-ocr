# python-pdf-ocr

> [**한국어**](#한국어-ko) / [**English**](#english-en)

---

## 한국어 (KO)

### 📚 목차

* [소개](#소개)
* [주요 특징](#주요-특징)
* [구성 (Architecture)](#구성-architecture)
* [사전 준비 (Prerequisites)](#사전-준비-prerequisites)
    * [CUDA 사용 시 사전 설치](#cuda-사용-시-사전-설치)
    * [권장 폰트](#권장-폰트)
* [설치 (Installation)](#설치-installation)
* [빠른 시작 (Quick Start)](#빠른-시작-quick-start)
* [옵션 요약 (테이블)](#옵션-요약-테이블)
* [출력물 (Outputs)](#출력물-outputs)
* [페이지 선택](#페이지-선택)
* [동작 원리](#동작-원리)
* [트러블슈팅](#트러블슈팅)
* [개발 정보](#개발-정보)
* [라이선스](#라이선스)
* [체크리스트](#체크리스트)

---

## 소개

고품질 스캔 PDF를 **검색 가능한 PDF**로 변환하는 **스트리밍 OCR 파이프라인**입니다.
PaddleOCR + PyMuPDF 기반으로 페이지를 한 장씩 처리하며, 인식 텍스트를 **비가시(invisible) 텍스트 레이어**로 원본 PDF에 덧입혀 저장합니다.

## 주요 특징

* **스트리밍 처리**: 전체 파일을 메모리에 올리지 않고 페이지 단위 처리/증분 저장.
* **진행률 표시**: Render / OCR / Overlay 3단계 진행바.
* **배치 OCR**: 배치 크기 조절로 처리량 최적화.
* **가시/비가시 오버레이**: 디버그용 **가시 텍스트 PDF** 병행 생성 옵션.
* **결과 내보내기**: CSV와 NDJSON 스트리밍 기록 지원.
* **언어/모델 런타임 선택**: PaddleOCR 언어/모델을 실행 시 지정 가능.

## 구성 (Architecture)

```
main.py → OCRPipeline
           ├─ PdfStreamer (PDF → RGB ndarray)
           ├─ OcrEngine   (PaddleOCR batch predict)
           ├─ IncrementalOverlayWriter (invisible / visible overlay, saveIncr)
           ├─ CsvStreamWriter / NdjsonStreamWriter
           └─ TqdmProgressSink (progress bars)
```

* `pdf_streamer.py` — 페이지 선택/렌더 스트리밍
* `ocr_engine.py` — 디바이스 자동선택, 배치 `predict`
* `overlay_writer.py` — invisible/visible 텍스트 오버레이 및 증분 저장
* `result_writers.py` — CSV/NDJSON 스트리밍 기록
* `progress.py` — `tqdm` 기반 3바 진행 표시
* `pipeline.py` — 엔드-투-엔드 파이프라인 조립
* `main.py` — CLI 진입점

## 사전 준비 (Prerequisites)

* **Python 3.9–3.12** 권장
* OS: Linux / Windows / macOS (Apple Silicon은 CPU 전용 권장)
* (선택) **GPU 가속**: NVIDIA **CUDA**
* **CJK 폰트** 권장 (한국어 정확한 매핑을 위해 폰트 지정 권장)

### CUDA 사용 시 사전 설치

CUDA로 PaddleOCR/PaddlePaddle GPU를 사용하려면 아래 요소가 **사전에** 설치되어 있어야 합니다.

1. **NVIDIA 그래픽 드라이버** (CUDA 버전과 호환)
2. **CUDA Toolkit** (예: 11.x 또는 12.x)

> `requirements.txt`의 `--extra-index-url`은 예시 CUDA 인덱스입니다. **시스템 CUDA 버전에 맞게 수정**하세요. 설치 후 환경을 점검해보세요.

```bash
# 드라이버/디바이스 확인
nvidia-smi

# 파이썬에서 Paddle 디바이스 확인
python - <<'PY'
import paddle
print('Device:', paddle.device.get_device())
print('Is Compiled With CUDA:', paddle.is_compiled_with_cuda())
PY
```

> GPU가 없거나 호환되지 않으면 본 프로젝트는 **자동으로 CPU 모드**로 동작합니다. 문제 발생 시 `--device cpu`로 강제할 수 있습니다.

### 권장 폰트

한국어/중국어/일본어 텍스트 매핑 품질을 위해 CJK 폰트 사용을 권장합니다.

* **Noto CJK 폰트(권장)**: [https://github.com/notofonts/noto-cjk](https://github.com/notofonts/noto-cjk)
    * 예) `NotoSansCJKkr-Regular.otf`, `NotoSerifCJKkr-Regular.otf`, 또는 TTC/OTC 동등 파일
* 사용 예시:

```bash
--font /path/to/NotoSansCJKkr-Regular.otf
```

## 설치 (Installation)

### 가상환경 생성 (권장)

```bash
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 의존성 설치

**GPU 환경**

```bash
pip install --upgrade pip
# (필요 시 requirements.txt의 CUDA 인덱스 조정 후)
pip install -r requirements.txt
```

**CPU 전용**

```bash
pip install --upgrade pip
pip install paddleocr pymupdf tqdm paddlepaddle
```

## 빠른 시작 (Quick Start)

```bash
python main.py INPUT.pdf OUTPUT.pdf \
  --dpi 300 \
  --device auto \
  --lang korean \
  --rec-model auto \
  --font /path/to/NotoSansCJKkr-Regular.otf \
  --batch-size 1 \
  --save-csv out/results.csv \
  --save-ndjson out/results.ndjson \
  --debug-visible
```

## 옵션 요약 (테이블)

| 옵션                | 설명                       | 기본값      | 예시                                       |
|-------------------|--------------------------|----------|------------------------------------------|
| `--dpi`           | 페이지 렌더 해상도               | `300`    | `--dpi 400`                              |
| `--device`        | 사용 디바이스                  | `auto`   | `--device gpu:0` / `--device cpu`        |
| `--lang`          | 언어 코드                    | `korean` | `--lang en` / `--lang japan`             |
| `--rec-model`     | 인식 모델 이름                 | `auto`   | `--rec-model ch_PP-OCRv4`                |
| `--font`          | 오버레이 폰트 경로 (TTF/OTF/TTC) | `없음`     | `--font /path/NotoSansCJKkr-Regular.otf` |
| `--page-range`    | 1-based 범위(포함)           | `없음`     | `--page-range 10-50`                     |
| `--pages`         | 1-based 개별 페이지           | `없음`     | `--pages 1,5,9`                          |
| `--page-step`     | 샘플링 간격                   | `1`      | `--page-step 2`                          |
| `--batch-size`    | OCR 배치 크기                | `1`      | `--batch-size 4`                         |
| `--save-csv`      | CSV 결과 경로                | `없음`     | `--save-csv out/res.csv`                 |
| `--save-ndjson`   | NDJSON 결과 경로             | `없음`     | `--save-ndjson out/res.ndjson`           |
| `--debug-visible` | 가시 텍스트 디버그 PDF 생성        | `끄기`     | `--debug-visible`                        |

## 출력물 (Outputs)

* **OUTPUT.pdf**: 비가시 텍스트 레이어가 얹힌 검색 가능한 PDF
* **OUTPUT\_debug.pdf** (옵션): 가시 텍스트 디버그 PDF
* **CSV / NDJSON** (옵션): 각 항목의 페이지/좌표/신뢰도/텍스트 기록
    * CSV: `page,x0,y0,x1,y1,score,text`
    * NDJSON: `{ page, text, score, poly }`

## 페이지 선택

* 전체 처리(기본)
* 범위: `--page-range 10-50`
* 개별: `--pages 1,5,9`
* 샘플링: `--page-step 2` (2장마다 1장 처리)

## 동작 원리

1. **Render**: PyMuPDF로 RGB 배열 생성(`dpi` 반영)
2. **OCR**: PaddleOCR `predict` 배치 API로 텍스트/정확도/폴리곤 획득
3. **Overlay**: 바운딩 박스 기반 글꼴 크기 산출 → **보이지 않게** 텍스트 삽입(`render_mode=3`) → `saveIncr()`로 증분 저장
4. (옵션) CSV/NDJSON 스트리밍 기록

## 트러블슈팅

* **CUDA 인덱스 오류**: `requirements.txt`의 인덱스를 시스템 CUDA에 맞게 조정
* **GPU 미사용**: 드라이버/CUDA 상태 확인. 실패 시 `--device cpu` 강제
* **CJK 폰트 이슈**: 네모(□) 표시/검색 부정확 시 `--font`로 CJK 폰트 지정
* **메모리 사용량**: 매우 큰 페이지는 `--dpi` 다운 또는 `--batch-size` 조정
* **성능**: GPU, 배치 크기 확대, 적절한 `dpi` 선택

## 개발 정보

* **PaddleOCR**: `predict` 경로 사용(버전 간 반환형 차이 대응 정규화 포함)
* **증분 저장**: PyMuPDF `saveIncr()`로 대용량에서도 안정적
* **모듈**:
    * `pdf_streamer.py`
    * `ocr_engine.py`
    * `overlay_writer.py`
    * `result_writers.py`
    * `progress.py`
    * `pipeline.py`
    * `main.py`

## 라이선스

사용 라이브러리(PaddleOCR, PyMuPDF 등) 라이선스를 확인하여 조직 정책에 맞는 라이선스를 추가하십시오.

## 체크리스트

* [ ] Python 3.9–3.12 가상환경
* [ ] (선택) CUDA 설치 및 `requirements.txt` 인덱스 확인
* [ ] `pip install -r requirements.txt` 또는 CPU 전용 설치
* [ ] CJK 폰트 준비 및 `--font` 지정 (권장: Noto CJK)
* [ ] 샘플 명령으로 변환 테스트

[⬆️ 위로](#python-pdf-ocr) · [English로 이동](#english-en)

---

## English (EN)

### Table of Contents (English)

* [Overview](#overview)
* [Highlights](#highlights)
* [Architecture](#architecture)
* [Prerequisites](#prerequisites)
    * [Pre-install for CUDA](#pre-install-for-cuda)
    * [Recommended Fonts](#recommended-fonts)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Options Summary (Table)](#options-summary-table)
* [Outputs](#outputs)
* [Page Selection](#page-selection)
* [How It Works](#how-it-works)
* [Troubleshooting](#troubleshooting)
* [Development Notes](#development-notes)
* [License](#license)
* [Checklist](#checklist)

---

## Overview

A **streaming OCR pipeline** that turns scanned PDFs into **searchable PDFs**.
It uses PaddleOCR + PyMuPDF, processes pages incrementally, and overlays **invisible text** back into the source PDF.

## Highlights

* **Streaming** page-by-page processing with incremental saves
* **Progress bars** for Render / OCR / Overlay
* **Batched OCR** with adjustable batch size
* **Visible/Invisible overlays** (optional **visible debug PDF**)
* **CSV & NDJSON** streamed exports
* **Runtime language/model selection** for PaddleOCR

## Architecture

```
main.py → OCRPipeline
           ├─ PdfStreamer (PDF → RGB ndarray)
           ├─ OcrEngine   (PaddleOCR batch predict)
           ├─ IncrementalOverlayWriter (invisible / visible overlay, saveIncr)
           ├─ CsvStreamWriter / NdjsonStreamWriter
           └─ TqdmProgressSink (progress bars)
```

* `pdf_streamer.py` — page selection & streaming render
* `ocr_engine.py` — device auto-select, batched `predict`
* `overlay_writer.py` — invisible/visible overlays & incremental saves
* `result_writers.py` — CSV/NDJSON streaming writers
* `progress.py` — 3-track progress via `tqdm`
* `pipeline.py` — end-to-end composition
* `main.py` — CLI entrypoint

## Prerequisites

* **Python 3.9–3.12** recommended
* OS: Linux / Windows / macOS (Apple Silicon: CPU-only recommended)
* (Optional) **GPU acceleration**: NVIDIA **CUDA**
* **CJK fonts** recommended for Korean/JP/CN text fidelity

### Pre-install for CUDA

Before using CUDA with PaddleOCR/PaddlePaddle GPU, make sure you have:

1. **NVIDIA GPU driver** (compatible with your CUDA version)
2. **CUDA Toolkit** (e.g., 11.x or 12.x)

> The `--extra-index-url` in `requirements.txt` points to an example CUDA wheel index.
> **Adjust it to match your system’s CUDA**. After installation, verify your setup:

```bash
# Driver/device check
nvidia-smi

# Verify Paddle sees CUDA
python - <<'PY'
import paddle
print('Device:', paddle.device.get_device())
print('Is Compiled With CUDA:', paddle.is_compiled_with_cuda())
PY
```

> If CUDA isn’t available or compatible, the project **falls back to CPU automatically**.
> You can force CPU via `--device cpu`.

### Recommended Fonts

For better CJK text mapping quality, use a CJK font and pass it with `--font`.

* **Noto CJK (recommended)**: [https://github.com/notofonts/noto-cjk](https://github.com/notofonts/noto-cjk)
    * e.g., `NotoSansCJKkr-Regular.otf`, `NotoSerifCJKkr-Regular.otf`, or equivalent TTC/OTC files
* Example:

```bash
--font /path/to/NotoSansCJKkr-Regular.otf
```

## Installation

### Create a virtual environment

```bash
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### Install dependencies

**GPU setup**

```bash
pip install --upgrade pip
# (Adjust CUDA index in requirements.txt if needed)
pip install -r requirements.txt
```

**CPU-only**

```bash
pip install --upgrade pip
pip install paddleocr pymupdf tqdm paddlepaddle
```

## Quick Start

```bash
python main.py INPUT.pdf OUTPUT.pdf \
  --dpi 300 \
  --device auto \
  --lang korean \
  --rec-model auto \
  --font /path/to/NotoSansCJKkr-Regular.otf \
  --batch-size 1 \
  --save-csv out/results.csv \
  --save-ndjson out/results.ndjson \
  --debug-visible
```

## Options Summary (Table)

| Option            | Description                     | Default  | Example                                  |
|-------------------|---------------------------------|----------|------------------------------------------|
| `--dpi`           | Page render resolution          | `300`    | `--dpi 400`                              |
| `--device`        | Compute device                  | `auto`   | `--device gpu:0` / `--device cpu`        |
| `--lang`          | Language code                   | `korean` | `--lang en` / `--lang japan`             |
| `--rec-model`     | Recognition model name          | `auto`   | `--rec-model ch_PP-OCRv4`                |
| `--font`          | Overlay font path (TTF/OTF/TTC) | `None`   | `--font /path/NotoSansCJKkr-Regular.otf` |
| `--page-range`    | Inclusive 1-based range         | `None`   | `--page-range 10-50`                     |
| `--pages`         | Specific 1-based pages          | `None`   | `--pages 1,5,9`                          |
| `--page-step`     | Sampling stride                 | `1`      | `--page-step 2`                          |
| `--batch-size`    | OCR batch size                  | `1`      | `--batch-size 4`                         |
| `--save-csv`      | CSV output path                 | `None`   | `--save-csv out/res.csv`                 |
| `--save-ndjson`   | NDJSON output path              | `None`   | `--save-ndjson out/res.ndjson`           |
| `--debug-visible` | Write visible-text debug PDF    | `Off`    | `--debug-visible`                        |

## Outputs

* **OUTPUT.pdf**: searchable PDF with an **invisible text** layer
* **OUTPUT\_debug.pdf** (optional): visible-text debug PDF
* **CSV / NDJSON** (optional): page/coords/score/text per detection
    * CSV: `page,x0,y0,x1,y1,score,text`
    * NDJSON: `{ page, text, score, poly }`

## Page Selection

* All pages by default
* Range: `--page-range 10-50`
* Specific pages: `--pages 1,5,9`
* Sampling: `--page-step 2` (every other page)

## How It Works

1. **Render**: PyMuPDF rasterizes each page to RGB (`dpi` applied)
2. **OCR**: PaddleOCR `predict` (batched) returns text / confidence / polygon
3. **Overlay**: compute font size from bbox and insert **invisible** text (`render_mode=3`),
   saving incrementally via `saveIncr()`
4. (Optional) stream results to CSV/NDJSON

## Troubleshooting

* **CUDA index mismatch**: fix `requirements.txt` to your CUDA version
* **GPU not used**: verify driver/CUDA; fallback to `--device cpu`
* **CJK font issues**: specify a CJK font via `--font`
* **Memory**: lower `--dpi` or tweak `--batch-size`
* **Performance**: prefer GPU, larger batches, reasonable `dpi`

## Development Notes

* Uses PaddleOCR `predict` path with normalization for version differences
* Incremental saving via PyMuPDF `saveIncr()` for stability on large PDFs
* Modules:
    * `pdf_streamer.py`
    * `ocr_engine.py`
    * `overlay_writer.py`
    * `result_writers.py`
    * `progress.py`
    * `pipeline.py`
    * `main.py`

## License

Add a project license that aligns with the dependencies (PaddleOCR, PyMuPDF, etc.).

## Checklist

* [ ] Virtualenv on Python 3.9–3.12
* [ ] (Optional) CUDA installed; `requirements.txt` index adjusted
* [ ] `pip install -r requirements.txt` or CPU-only install
* [ ] Provide a CJK font (recommended: Noto CJK) and pass it to `--font`
* [ ] Test with the sample command

[⬆️ Back to top](#python-pdf-ocr) · [한국어로 이동](#한국어-ko)
