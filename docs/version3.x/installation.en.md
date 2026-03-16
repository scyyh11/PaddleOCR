---
comments: true
---

# Installation

## 1. Install PaddleOCR Inference Dependencies

### 1.1 Prepare dependencies by inference engine

PaddleOCR 3.5 introduces a unified inference-engine configuration concept. Different engines require different runtime dependencies, so prepare them according to your actual usage:

- If you use `paddle`, `paddle_static`, `paddle_dynamic`, or compatibility arguments such as `enable_hpi`, `use_tensorrt`, `precision`, `enable_mkldnn`, `mkldnn_cache_capacity`, `cpu_threads`, and `enable_cinn`, install the Paddle framework first;
- If you use `transformers`, install the `transformers` package, for example with `python -m pip install transformers`;
- If `engine` is not explicitly specified, explicitly choose an available non-Paddle engine when PaddlePaddle is not installed in the environment;
- In the CLI, only `--engine` is exposed directly for now, while `engine_config` must be configured through a PaddleX YAML file.

The recommended installation order is: first install the dependencies required by the inference engine, then install `paddleocr`.

For the engine concept, prerequisites by engine, supported values of `engine` / `engine_config`, and usage examples, see [Inference Engine and Configuration](./inference_engine.en.md).

For the concrete Paddle framework installation steps, see the dedicated document: [Paddle Framework Installation](./paddlepaddle_installation.en.md).

### 1.2 Install the PaddleOCR Inference Package

After the inference-engine dependencies are ready, install the PaddleOCR inference package.

Install the latest PaddleOCR inference package from PyPI:

```bash
# If you only want to use the basic text recognition feature (returning text position coordinates and content)
python -m pip install paddleocr
# If you want to use all functionalities, such as document parsing, document understanding, document translation, and key information extraction
# python -m pip install "paddleocr[all]"
```

Or install from source (default is the development branch):

```bash
# If you only want to use the basic text recognition feature (returning text position coordinates and content)
python -m pip install "paddleocr@git+https://github.com/PaddlePaddle/PaddleOCR.git"
# If you want to use all functionalities, such as document parsing, document understanding, document translation, and key information extraction
# python -m pip install "paddleocr[all]@git+https://github.com/PaddlePaddle/PaddleOCR.git"
```

In addition to the `all` dependency group shown above, PaddleOCR also supports installing selected optional capabilities through other dependency groups. The available groups are:

| Dependency Group Name | Corresponding Functionality |
| - | - |
| `doc-parser` | Document parsing: extract layout elements such as tables, formulas, stamps, and images from documents; includes models such as PP-StructureV3 |
| `ie` | Information extraction: extract key information such as names, dates, addresses, and amounts from documents; includes models such as PP-ChatOCRv4 |
| `trans` | Document translation: translate documents from one language to another; includes models such as PP-DocTranslation |
| `all` | Full functionality |

The general OCR pipeline (such as PP-OCRv3/v4/v5) and the document image preprocessing pipeline can be used without any additional dependency groups. Besides these two pipelines, each remaining pipeline belongs to exactly one dependency group. You can refer to the usage documentation of each pipeline to see which group it belongs to. For individual modules, installing any dependency group that contains the corresponding module is sufficient for basic usage.

## 2. Install Training Dependencies

If you plan to perform development tasks such as model training and exporting, you need to install the training dependencies. Installing both the inference package and the training dependencies in the same environment is allowed, and environment isolation is not required.

Training and export workflows depend on the Paddle framework, so install PaddlePaddle first by following [Paddle Framework Installation](./paddlepaddle_installation.en.md).

To perform model training, exporting, etc., clone the repository to your local machine:

```bash
# Recommended method
git clone https://github.com/PaddlePaddle/PaddleOCR

# (Optional) Switch to a specific branch
git checkout release/3.2

# If you encounter network issues preventing successful cloning, you can also use the repository on Gitee:
git clone https://gitee.com/paddlepaddle/PaddleOCR

# Note: The code hosted on Gitee may not be synchronized in real-time with updates from this GitHub project, with a delay of 3~5 days. Please prioritize using the recommended method.
```

Run the following command to install the remaining training dependencies:

```bash
python -m pip install -r requirements.txt
```
