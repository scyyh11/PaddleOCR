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

## Project Structure

```
PaddleOCR/
├── paddleocr/              # Public API (3.x) — what users import
│   ├── __init__.py         # Top-level exports (__all__ is the source of truth)
│   ├── _pipelines/         # High-level pipelines (OCR, PPStructureV3, etc.)
│   ├── _models/            # Individual model wrappers (TextDetection, etc.)
│   └── _cli.py             # CLI entry point
├── ppocr/                  # Internal training framework (not user-facing)
│   ├── modeling/           # Model architectures (Backbone, Neck, Head)
│   ├── data/               # Data loading and augmentation
│   ├── losses/             # Loss functions
│   ├── metrics/            # Evaluation metrics
│   └── postprocess/        # Post-processing
├── tools/                  # Train/infer/eval scripts (tools/train.py)
├── configs/                # YAML configs organized by task (det/, rec/, table/, etc.)
├── deploy/                 # Deployment (C++, Docker, ONNX, mobile)
├── tests/                  # Tests (models/ + pipelines/)
└── agent_docs/             # Detailed AI-readable documentation
```

Two layers — understand which you're working in:

- **`paddleocr/`** — Public API (3.x). `_pipelines/` has high-level pipelines, `_models/` has individual model wrappers. Users import from here.
- **`ppocr/`** — Internal training framework. Used by `tools/train.py`, not by end users.

## Discovering Available Pipelines & Models

**Do NOT rely on hardcoded lists.** Always discover dynamically from source:

- **Pipelines**: Read `__all__` in `paddleocr/_pipelines/__init__.py`
- **Models**: Read `__all__` in `paddleocr/_models/__init__.py`
- **All public exports**: Read `__all__` in `paddleocr/__init__.py`

Each pipeline inherits from `PaddleXPipelineWrapper` (in `_pipelines/base.py`).
Each model inherits from `PaddleXPredictorWrapper` (in `_models/base.py`).

To understand a specific pipeline or model, read its source file in the corresponding directory.

## Critical: 3.x API Only

PaddleOCR 3.x is **not backwards compatible** with 2.x. Never generate 2.x-style code:
- Use `.predict()` not `.ocr()` (deprecated)
- Results are objects with `.print()`, `.save_to_img()`, `.save_to_json()` — not nested lists
- `PPStructure` is removed — use `PPStructureV3`
- For single-task inference, use model classes (`TextDetection`, `TextRecognition`) not `det`/`rec` params

## Code Style & Conventions

- Follow existing patterns in the file you're modifying
- Use type hints for function signatures
- Use `pre-commit run --all-files` to lint before committing — this runs ruff, trailing whitespace fixes, and other checks
- Error messages should be clear and actionable
- No `eval()`, `exec()`, or `pickle` on user-controlled input

## Testing

- Tests live in `tests/` with subdirectories `models/` and `pipelines/`
- Run with `pytest tests/` — resource-intensive tests are skipped by default
- When adding a new pipeline or model, add corresponding tests
- Test the public API (`.predict()`, result object methods), not internal implementation details

## PR & Commit Guidelines

- PR titles: concise, lowercase, descriptive of what changed
- PR descriptions: explain the "why", not just the "what"
- Keep PRs focused — one logical change per PR
- Ensure `pre-commit run --all-files` passes before pushing

## Detailed Docs

Read these as needed — don't load them all upfront:
- `agent_docs/inference_api.md` — Pipelines, models, constructor params, CLI, usage patterns
- `agent_docs/training.md` — Training commands, config YAML structure, internal framework
