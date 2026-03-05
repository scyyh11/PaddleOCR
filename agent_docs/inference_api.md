# Inference API Reference (3.x)

## Quick Start

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR()
result = ocr.predict("./image.png")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

## Available Pipelines

All imported from `paddleocr`:

| Pipeline | Purpose |
|----------|---------|
| `PaddleOCR` | Full OCR (detection + recognition) |
| `PaddleOCRVL` | Vision-language OCR (v1, v1.5) |
| `PPStructureV3` | Document structure: tables, formulas, layout |
| `PPChatOCRv4Doc` | LLM-powered document analysis |
| `DocUnderstanding` | VLM-based document QA |
| `FormulaRecognitionPipeline` | Math formula recognition |
| `SealRecognition` | Seal text detection + recognition |
| `TableRecognitionPipelineV2` | Table structure recognition |
| `DocPreprocessor` | Orientation, unwarping |
| `PPDocTranslation` | Document translation |

## Available Individual Models

| Model | Purpose |
|-------|---------|
| `TextDetection` | Detect text regions |
| `TextRecognition` | Recognize text content |
| `LayoutDetection` | Detect document layout regions |
| `TableClassification` | Classify table types |
| `TableCellsDetection` | Detect table cells |
| `TableStructureRecognition` | Recognize table structure |
| `SealTextDetection` | Detect seal text |
| `FormulaRecognition` | Recognize formulas |
| `ChartParsing` | Parse charts |
| `DocVLM` | Document vision-language model |
| `DocImgOrientationClassification` | Classify document orientation |
| `TextImageUnwarping` | Unwarp distorted text images |
| `TextLineOrientationClassification` | Classify text line orientation |

## PaddleOCR Constructor Parameters

```python
PaddleOCR(
    lang="ch",                          # "ch", "en", "japan", "france", etc.
    ocr_version="PP-OCRv5",            # "PP-OCRv3", "PP-OCRv4", "PP-OCRv5"
    device="gpu",                       # "gpu", "cpu", "npu", "xpu"

    # Model overrides
    text_detection_model_name=None,     # or text_detection_model_dir
    text_recognition_model_name=None,   # or text_recognition_model_dir

    # Detection tuning
    text_det_limit_side_len=None,
    text_det_thresh=None,
    text_det_box_thresh=None,
    text_det_unclip_ratio=None,

    # Recognition tuning
    text_rec_score_thresh=None,
    return_word_box=None,

    # Preprocessing toggles
    use_doc_orientation_classify=None,
    use_doc_unwarping=None,
    use_textline_orientation=None,
)
```

## PPStructureV3 Constructor Parameters

Inherits all OCR params, plus:
- `use_table_recognition`, `use_formula_recognition`, `use_chart_recognition` — toggle sub-pipelines
- `layout_*` params — layout detection tuning (threshold, nms, unclip_ratio, merge_bboxes_mode)
- `format_block_content`, `markdown_ignore_labels` — output formatting

## Individual Model Interface

All models share this interface:
```python
model = TextDetection(model_name="PP-OCRv5_server_det", device="gpu")
results = model.predict("image.png")       # batch predict, returns list
results = model.predict_iter("image.png")  # streaming predict, returns generator
model.close()                              # cleanup resources
```

## Usage Patterns

```python
# Language/version selection
ocr = PaddleOCR(lang="en", ocr_version="PP-OCRv5")

# Document structure analysis
from paddleocr import PPStructureV3
structure = PPStructureV3()
result = structure.predict("document.png", use_table_recognition=True)

# Standalone models
from paddleocr import TextDetection
det = TextDetection(limit_side_len=960)
det_results = det.predict("image.png")

# Vision-language document understanding
from paddleocr import DocUnderstanding
doc = DocUnderstanding()
result = doc.predict({"image": "doc.png", "query": "What is the total?"})
```

## CLI

```bash
paddleocr ocr -i ./image.png
paddleocr text_detection -i ./image.png
paddleocr pp_structurev3 -i ./doc.png
paddleocr doc_vlm -i '{"image": "doc.png", "query": "describe"}'
```
