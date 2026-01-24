# PaddleOCR-VL High Performance Server (HPS)

A high-performance deployment solution for PaddleOCR-VL with support for concurrent request processing.

## Architecture

```
┌──────────┐     ┌─────────────────┐     ┌────────────────┐     ┌─────────────┐
│  Client  │────►│ FastAPI Gateway │────►│ Triton Server  │────►│ vLLM Server │
└──────────┘     └─────────────────┘     └────────────────┘     └─────────────┘
                   - Async I/O            - Dynamic batching    - Continuous batching
                   - Concurrency control  - GPU scheduling      - VLM inference
                   - Rate limiting        - Model management
```

## System Requirements

- x64 CPU
- NVIDIA GPU, Compute Capability >= 8.0 and < 12.0
- NVIDIA driver supporting CUDA 12.6
- Docker >= 19.03
- Docker Compose >= 2.0

## Quick Start

1. Clone the repository and navigate to the HPS directory:

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR/deploy/paddleocr_vl_docker/hps
```

2. Download and prepare the required SDK files:

```bash
bash prepare.sh
```

3. Start the services:

```bash
docker compose up
```

This will start 3 containers:

| Service | Description | Port |
|---------|-------------|------|
| `paddleocr-vl-api` | FastAPI gateway (entry point) | 8080 |
| `paddleocr-vl-tritonserver` | Triton Inference Server | 8000 (internal) |
| `paddleocr-vlm-server` | vLLM-based VLM inference | 8080 (internal) |

> **Note**: First startup will download and build images, which takes longer. Subsequent starts use cached images.

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and modify as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `HPS_MAX_CONCURRENT_REQUESTS` | 16 | Max concurrent requests to Triton |
| `HPS_INFERENCE_TIMEOUT` | 600 | Request timeout in seconds |
| `HPS_LOG_LEVEL` | INFO | Log level (DEBUG, INFO, WARNING, ERROR) |
| `UVICORN_WORKERS` | 4 | Number of gateway worker processes |
| `GPU_DEVICE_ID` | 0 | GPU device to use |

### Example: High Throughput Configuration

```bash
# .env
HPS_MAX_CONCURRENT_REQUESTS=32
UVICORN_WORKERS=8
```

### Example: Low Latency Configuration

```bash
# .env
HPS_MAX_CONCURRENT_REQUESTS=8
HPS_INFERENCE_TIMEOUT=300
UVICORN_WORKERS=2
```

## API Usage

### Health Check

```bash
# Liveness check
curl http://localhost:8080/health

# Readiness check (verifies Triton connectivity)
curl http://localhost:8080/health/ready
```

### Layout Parsing

```bash
curl -X POST http://localhost:8080/layout-parsing \
  -H "Content-Type: application/json" \
  -d '{
    "file": "base64_encoded_image_or_pdf",
    "fileType": 1
  }'
```

### API Documentation

When the service is running, access the interactive API documentation at:

- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

## Performance Tuning

### Concurrency Settings

The gateway uses a semaphore to limit concurrent requests to Triton:

- **Too low** (`HPS_MAX_CONCURRENT_REQUESTS=4`): Underutilizes GPU, requests queue unnecessarily
- **Too high** (`HPS_MAX_CONCURRENT_REQUESTS=64`): May overwhelm Triton, causing OOM or timeouts
- **Recommended**: Start with 16, adjust based on GPU memory and workload

### Worker Processes

Each Uvicorn worker is a separate process with its own event loop:

- **1 worker**: Simple, but limited by single process
- **4 workers**: Good balance for most cases
- **8+ workers**: For high-concurrency scenarios with many small requests

### Triton Dynamic Batching

Triton automatically batches requests for efficient GPU utilization. The batch size is configured in the model repository (default: 8).

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker compose logs paddleocr-vl-api
docker compose logs paddleocr-vl-tritonserver
docker compose logs paddleocr-vlm-server

# Check health
curl http://localhost:8080/health/ready
```

### Timeout Errors

- Increase `HPS_INFERENCE_TIMEOUT` for complex documents
- Check GPU memory usage: `nvidia-smi`
- Reduce `HPS_MAX_CONCURRENT_REQUESTS` if GPU is overloaded

### Out of Memory

- Reduce `HPS_MAX_CONCURRENT_REQUESTS`
- Ensure only one service uses each GPU
- Check `shm_size` in compose.yaml (default: 4GB)

## Development

### Running Locally (without Docker)

```bash
cd gateway
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080 --workers 4
```

### Running Tests

```bash
# Concurrent request test
for i in {1..20}; do
  curl -X POST http://localhost:8080/layout-parsing \
    -H "Content-Type: application/json" \
    -d '{"file": "...", "fileType": 1}' &
done
wait
```

## License

Apache License 2.0
