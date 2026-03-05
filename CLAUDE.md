# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

PaddleOCR is a production-ready OCR and document AI engine built on PaddlePaddle. It does text detection, recognition, document structure analysis, and information extraction.

## Build & Verify

```bash
pip install -e ".[all]"       # Dev install (paddlepaddle installed separately)
pytest tests/                  # Tests (resource-intensive skipped by default)
pre-commit run --all-files     # Lint/format
```

## Architecture

Two layers — understand which you're working in:

- **`paddleocr/`** — Public API (3.x). `_pipelines/` has high-level pipelines (OCR, PPStructureV3), `_models/` has individual model wrappers (TextDetection, TextRecognition). Users import from here.
- **`ppocr/`** — Internal training framework. Model architectures, data loading, losses, metrics, postprocessing. Used by `tools/train.py`, not by end users.

Other directories: `tools/` (train/infer/eval scripts), `configs/` (YAML configs by task), `deploy/` (C++, Docker, ONNX, mobile), `tests/` (models/ + pipelines/).

## Critical: 3.x API Only

PaddleOCR 3.x is **not backwards compatible** with 2.x. Never generate 2.x-style code:
- Use `.predict()` not `.ocr()` (deprecated)
- Results are objects with `.print()`, `.save_to_img()`, `.save_to_json()` — not nested lists
- `PPStructure` is removed — use `PPStructureV3`
- For single-task inference, use model classes (`TextDetection`, `TextRecognition`) not `det`/`rec` params

## Detailed Docs

Read these as needed — don't load them all upfront:
- `agent_docs/inference_api.md` — Pipelines, models, constructor params, CLI, usage patterns
- `agent_docs/training.md` — Training commands, config YAML structure, internal framework
