# PaddleOCR-VL High-Performance Service Deployment (Beta)

[简体中文](README.md)

This directory provides a high-performance service deployment solution for PaddleOCR-VL with concurrent request processing support.

## Architecture

```
Client → FastAPI Gateway → Triton Server → vLLM Server
```

| Component       | Description                                          |
|-----------------|------------------------------------------------------|
| FastAPI Gateway | Unified access point, simplified client calls, concurrency control |
| Triton Server   | Model management, dynamic batching, inference scheduling |
| vLLM Server     | Continuous batching, VLM inference                   |

**Triton Models:**

| Model | Device | Description |
|-------|--------|-------------|
| `layout-parsing` | GPU | Layout parsing inference |
| `restructure-pages` | CPU | Multi-page result post-processing (cross-page table merging, title level reassignment) |

## Requirements

- x64 CPU
- NVIDIA GPU, Compute Capability >= 8.0 and < 12.0
- NVIDIA driver supporting CUDA 12.6
- Docker >= 19.03
- Docker Compose >= 2.0

## Quick Start

1. Clone the PaddleOCR repository and navigate to this directory:

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR/deploy/paddleocr_vl_docker/hps
```

2. Download and prepare necessary files:

```bash
bash prepare.sh
```

3. Start the services:

```bash
docker compose up
```

The above command will start 3 containers in sequence:

| Service | Description | Port |
|---------|-------------|------|
| `paddleocr-vl-api` | FastAPI gateway (external entry point) | 8080 |
| `paddleocr-vl-tritonserver` | Triton inference server | 8000 (internal) |
| `paddleocr-vlm-server` | vLLM-based VLM inference service | 8080 (internal) |

> The first startup will automatically download and build images, which takes longer. Subsequent startups will use local images and start faster.

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and modify as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `HPS_MAX_CONCURRENT_INFERENCE_REQUESTS` | 16 | Max concurrent inference requests (layout parsing) |
| `HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS` | 64 | Max concurrent non-inference requests (page restructuring) |
| `HPS_INFERENCE_TIMEOUT` | 600 | Request timeout in seconds |
| `HPS_HEALTH_CHECK_TIMEOUT` | 5 | Health check timeout in seconds |
| `HPS_VLM_URL` | http://paddleocr-vlm-server:8080 | VLM server URL (for health checks) |
| `HPS_LOG_LEVEL` | INFO | Log level (DEBUG, INFO, WARNING, ERROR) |
| `HPS_FILTER_HEALTH_ACCESS_LOG` | true | Whether to filter health check access logs |
| `UVICORN_WORKERS` | 4 | Number of gateway worker processes |
| `GPU_DEVICE_ID` | 0 | GPU device ID to use |

### High-Throughput Configuration Example

```bash
# .env
HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=32
HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS=128
UVICORN_WORKERS=8
```

### Low-Latency Configuration Example

```bash
# .env
HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=8
HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS=32
HPS_INFERENCE_TIMEOUT=300
UVICORN_WORKERS=2
```

## API Usage

### Health Checks

```bash
# Liveness check
curl http://localhost:8080/health

# Readiness check (verifies Triton and VLM services)
curl http://localhost:8080/health/ready
```

### Layout Parsing

```bash
curl -X POST http://localhost:8080/layout-parsing \
  -H "Content-Type: application/json" \
  -d '{
    "file": "base64-encoded image or PDF",
    "fileType": 1
  }'
```

### Multi-Page Result Restructuring

Post-processes layout parsing results for multi-page documents, supporting cross-page table merging and title level reassignment.

```bash
curl -X POST http://localhost:8080/restructure-pages \
  -H "Content-Type: application/json" \
  -d '{
    "pages": [
      {
        "prunedResult": {
          "parsing_res_list": [...],
          "layout_det_res": [...]
        },
        "markdownImages": {"img_0": "base64..."}
      }
    ],
    "mergeTables": true,
    "relevelTitles": true,
    "concatenatePages": false
  }'
```

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pages` | array | required | List of layout parsing results for each page |
| `mergeTables` | bool | true | Whether to merge cross-page tables |
| `relevelTitles` | bool | true | Whether to reassign title levels |
| `concatenatePages` | bool | false | Whether to generate merged Markdown |

> Note: This endpoint is a non-inference operation and does not consume inference device resources.

### API Documentation

After starting the service, you can access the interactive API documentation:

- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

## Performance Tuning

### Concurrency Settings

The gateway uses separate semaphores for inference and non-inference operations:

- **`HPS_MAX_CONCURRENT_INFERENCE_REQUESTS`** (default 16): Controls concurrency for inference operations such as `layout-parsing` (layout parsing)
  - Too low (4): Underutilized inference device, requests queue unnecessarily
  - Too high (64): May overload Triton, causing OOM or timeouts
  - Default value of 16 allows enough requests to queue for the next batch while the current batch is being processed
  - If inference device resources are limited, consider lowering this value
- **`HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS`** (default 64): Controls concurrency for non-inference operations such as `restructure-pages` (page restructuring)
  - Non-inference operations do not consume inference device resources and can be set to a higher concurrency level
  - Adjust based on CPU cores and available memory

### Worker Processes

Each Uvicorn worker is an independent process with its own event loop:

- **1 worker**: Simple, but limited to a single process
- **4 workers**: Suitable for most scenarios
- **8+ workers**: Suitable for high-concurrency scenarios with many small requests

### Triton Dynamic Batching

Triton automatically batches requests to improve inference device utilization. Batch size is configured in the model repository (default: 8).

## Troubleshooting

### Service Fails to Start

```bash
# View logs
docker compose logs paddleocr-vl-api
docker compose logs paddleocr-vl-tritonserver
docker compose logs paddleocr-vlm-server

# Check health status
curl http://localhost:8080/health/ready
```

### Timeout Errors

- Increase `HPS_INFERENCE_TIMEOUT` (for complex documents)
- Check GPU memory usage: `nvidia-smi`
- If the inference device is overloaded, reduce `HPS_MAX_CONCURRENT_INFERENCE_REQUESTS`

### Out of Memory

- Reduce `HPS_MAX_CONCURRENT_INFERENCE_REQUESTS`
- Ensure only one service runs per GPU
- Check `shm_size` in compose.yaml (default: 4GB)

## Development & Debugging

### Running Locally (Without Docker)

```bash
cd gateway
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080 --workers 4
```

### Concurrency Testing

```bash
# Concurrent request test
for i in {1..20}; do
  curl -X POST http://localhost:8080/layout-parsing \
    -H "Content-Type: application/json" \
    -d '{"file": "...", "fileType": 1}' &
done
wait
```
