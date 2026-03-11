# Config System

## Overview

Training configs are YAML files in `configs/`, organized by task: `det/`, `rec/`, `cls/`, `table/`, `e2e/`, `kie/`, `sr/`. Config loading logic lives in `tools/program.py`.

## Running with Configs

```bash
# Basic training
python tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml

# Override any value with -o (dot notation for nested keys)
python tools/train.py -c config.yml -o Global.use_gpu=false -o Optimizer.lr.learning_rate=0.001

# Multi-GPU
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c config.yml
```

The `-o` flag parses values as YAML: `true` → bool, `[1,2,3]` → list, `0.001` → float.

## Top-Level Sections

Every config has these top-level keys:

```yaml
Global:        # Device, epochs, checkpoints, character set
Architecture:  # Model definition (algorithm, Backbone, Neck, Head)
Loss:          # Loss function
Optimizer:     # Optimizer, learning rate schedule, regularizer
PostProcess:   # Post-processing (decode predictions to output format)
Train:         # Training dataset and data transforms
Eval:          # Evaluation dataset (optional)
Metric:        # Evaluation metric
```

## Global

Common keys:

```yaml
Global:
  use_gpu: true
  epoch_num: 500
  save_model_dir: ./output/model_name/
  save_epoch_step: 3
  eval_batch_step: [0, 2000]       # [start_iter, interval]
  print_batch_step: 10
  log_smooth_window: 20
  cal_metric_during_train: true

  # Pretrained weights / resume
  pretrained_model: null            # Path or URL
  checkpoints: null                 # Resume from checkpoint

  # Character set (recognition tasks)
  character_dict_path: ppocr/utils/ppocr_keys_v1.txt
  max_text_length: 25
  use_space_char: true

  # Export
  save_inference_dir: ./inference/
  model_name: model_name            # Name for static export

  # AMP (optional)
  use_amp: false
  amp_level: O2                     # O1 or O2
  amp_dtype: float16                # float16 or bfloat16
```

## Architecture

Defines the model as composable components:

```yaml
Architecture:
  model_type: det                    # det, rec, cls, table, e2e, kie, sr
  algorithm: DB                      # Algorithm name (DB, CRNN, SLANet, etc.)
  Transform: null                    # Optional spatial transformer (TPS, STN)
  Backbone:
    name: ResNet_vd
    layers: 50
  Neck:
    name: DBFPN
    out_channels: 256
  Head:
    name: DBHead
    k: 50
```

Components are resolved via `support_dict` registries in `ppocr/modeling/`. To see available backbones, necks, heads — read the `support_dict` in the corresponding `__init__.py` under `ppocr/modeling/backbones/`, `ppocr/modeling/necks/`, `ppocr/modeling/heads/`.

**Distillation** uses a special structure:

```yaml
Architecture:
  name: DistillationModel
  algorithm: Distillation
  model_type: det
  Models:
    Student:
      algorithm: DB
      Backbone: { name: MobileNetV3, scale: 0.5 }
      Neck: { name: DBFPN, out_channels: 96 }
      Head: { name: DBHead, k: 50 }
    Teacher:
      algorithm: DB
      Backbone: { name: ResNet_vd, layers: 18 }
      Neck: { name: DBFPN, out_channels: 256 }
      Head: { name: DBHead, k: 50 }
```

## Optimizer

```yaml
Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  clip_norm_global: 5.0             # Gradient clipping
  lr:
    name: Cosine                    # LR scheduler
    learning_rate: 0.001
    warmup_epoch: 2
  regularizer:
    name: L2
    factor: 0.0005
```

Available LR schedulers (defined in `ppocr/optimizer/learning_rate.py`):
- `Const`, `Linear`, `Cosine`, `Step`, `Piecewise`, `MultiStepDecay`
- `LinearWarmupCosine`, `CyclicalCosine`, `OneCycle`, `TwoStepCosine`

## Train / Eval (Dataset + Transforms)

```yaml
Train:
  dataset:
    name: SimpleDataSet              # SimpleDataSet, LMDBDataSet, PubTabDataSet, etc.
    data_dir: ./train_data/
    label_file_list:
      - ./train_data/labels.txt
    ratio_list: [1.0]                # Sampling ratio per file
    transforms:                      # Ordered pipeline
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - ResizeImg:
          image_shape: [3, 32, 100]
      - NormalizeImage:
          scale: 1./255.
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
      - KeepKeys:
          keep_keys: ['image', 'label']
  loader:
    shuffle: true
    batch_size_per_card: 256
    drop_last: true
    num_workers: 8
    use_shared_memory: true
```

Transforms execute in order. Each is `{ TransformName: { params } }`. Available transforms are in `ppocr/data/imaug/`.

## YAML Anchors

Configs use YAML anchors to avoid repeating values:

```yaml
Global:
  max_text_length: &max_text_length 500

Architecture:
  Head:
    max_text_length: *max_text_length

Train:
  dataset:
    transforms:
      - SomeTransform:
          max_text_length: *max_text_length
```

## Config Loading Flow

```
config.yml → load_config() → dict
                                ↓
                         merge_config() ← -o CLI overrides
                                ↓
                         preprocess() → device setup, validation
                                ↓
                    ┌─── build_dataloader(config, "Train")
                    ├─── build_model(config["Architecture"])
                    ├─── build_loss(config["Loss"])
                    ├─── build_optimizer(config["Optimizer"], ...)
                    ├─── build_post_process(config["PostProcess"])
                    └─── build_metric(config["Metric"])
```

Config loading: `tools/program.py` (`load_config`, `merge_config`, `ArgsParser`).
Builder functions: `ppocr/data/__init__.py`, `ppocr/modeling/architectures/__init__.py`, `ppocr/losses/__init__.py`, `ppocr/optimizer/__init__.py`, `ppocr/postprocess/__init__.py`, `ppocr/metrics/__init__.py`.

## Key File Locations

| What | Where |
|------|-------|
| Config files | `configs/{det,rec,cls,table,e2e,kie,sr}/` |
| Config loading | `tools/program.py` |
| LR schedulers | `ppocr/optimizer/learning_rate.py` |
| Data transforms | `ppocr/data/imaug/` |
| Component registries | `ppocr/modeling/{backbones,necks,heads}/__init__.py` |
