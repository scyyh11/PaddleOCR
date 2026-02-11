# PaddleOCR-VL 高性能服务化部署（Beta）

本目录提供一套支持并发请求处理的 PaddleOCR-VL 高性能服务化部署方案。

## 架构

```
客户端 → FastAPI 网关 → Triton 服务器 → vLLM 服务器
```

| 组件           | 说明                                   |
|----------------|----------------------------------------|
| FastAPI 网关   | 统一访问入口、简化客户端调用、并发控制 |
| Triton 服务器  | 模型管理、动态批处理、GPU 调度         |
| vLLM 服务器    | 连续批处理、VLM 推理                   |

**Triton 模型：**

| 模型 | 设备 | 说明 |
|------|------|------|
| `layout-parsing` | GPU | 版面解析推理 |
| `restructure-pages` | CPU | 多页结果后处理（跨页表格合并、标题层级重分配） |

## 环境要求

- x64 CPU
- NVIDIA GPU，Compute Capability >= 8.0 且 < 12.0
- NVIDIA 驱动支持 CUDA 12.6
- Docker >= 19.03
- Docker Compose >= 2.0

## 快速开始

1. 拉取 PaddleOCR 源码并切换到当前目录：

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR/deploy/paddleocr_vl_docker/hps
```

2. 下载并准备必要文件：

```bash
bash prepare.sh
```

3. 启动服务：

```bash
docker compose up
```

上述命令将依次启动 3 个容器：

| 服务 | 说明 | 端口 |
|------|------|------|
| `paddleocr-vl-api` | FastAPI 网关（对外入口） | 8080 |
| `paddleocr-vl-tritonserver` | Triton 推理服务器 | 8000（内部） |
| `paddleocr-vlm-server` | 基于 vLLM 的 VLM 推理服务 | 8080（内部） |

> 首次启动会自动下载并构建镜像，耗时较长；从第二次启动起将直接使用本地镜像，启动速度更快。

## 配置说明

### 环境变量

复制 `.env.example` 到 `.env` 并根据需要修改：

```bash
cp .env.example .env
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HPS_MAX_CONCURRENT_REQUESTS` | 16 | 最大并发请求数 |
| `HPS_INFERENCE_TIMEOUT` | 600 | 请求超时时间（秒） |
| `HPS_HEALTH_CHECK_TIMEOUT` | 5 | 健康检查超时时间（秒） |
| `HPS_VLM_URL` | http://paddleocr-vlm-server:8080 | VLM 服务器地址（用于健康检查） |
| `HPS_LOG_LEVEL` | INFO | 日志级别（DEBUG, INFO, WARNING, ERROR） |
| `HPS_FILTER_HEALTH_ACCESS_LOG` | true | 是否过滤健康检查的访问日志 |
| `UVICORN_WORKERS` | 4 | 网关 Worker 进程数 |
| `GPU_DEVICE_ID` | 0 | 使用的 GPU 设备 ID |

### 高吞吐配置示例

```bash
# .env
HPS_MAX_CONCURRENT_REQUESTS=32
UVICORN_WORKERS=8
```

### 低延迟配置示例

```bash
# .env
HPS_MAX_CONCURRENT_REQUESTS=8
HPS_INFERENCE_TIMEOUT=300
UVICORN_WORKERS=2
```

## API 使用

### 健康检查

```bash
# 存活检查
curl http://localhost:8080/health

# 就绪检查（验证 Triton 和 VLM 服务）
curl http://localhost:8080/health/ready
```

### 版面解析

```bash
curl -X POST http://localhost:8080/layout-parsing \
  -H "Content-Type: application/json" \
  -d '{
    "file": "base64编码的图片或PDF",
    "fileType": 1
  }'
```

### 多页结果重组

对多页文档的版面解析结果进行后处理，支持跨页表格合并和标题层级重新分配。

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

**请求参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pages` | array | 必填 | 各页的版面解析结果列表 |
| `mergeTables` | bool | true | 是否合并跨页表格 |
| `relevelTitles` | bool | true | 是否重新分配标题层级 |
| `concatenatePages` | bool | false | 是否生成合并后的 Markdown |

> 注意：此接口为 CPU 操作，不占用 GPU 资源。

### API 文档

服务启动后，可访问交互式 API 文档：

- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

## 性能调优

### 并发设置

网关使用信号量限制到 Triton 的并发请求数：

- **过低**（`HPS_MAX_CONCURRENT_REQUESTS=4`）：GPU 利用率不足，请求不必要地排队
- **过高**（`HPS_MAX_CONCURRENT_REQUESTS=64`）：可能导致 Triton 过载，出现 OOM 或超时
- **默认值 16 的选择依据**：
  - Triton 动态批处理的默认最大批大小为 8
  - 16 允许在当前批次处理时有足够请求排队形成下一批次
  - 如使用显存较小的 GPU，建议适当降低此值

### Worker 进程数

每个 Uvicorn Worker 是独立的进程，有自己的事件循环：

- **1 个 Worker**：简单，但受限于单进程
- **4 个 Worker**：适合大多数场景
- **8+ 个 Worker**：适用于高并发、大量小请求的场景

### Triton 动态批处理

Triton 自动将请求批处理以提高 GPU 利用率。批处理大小在模型仓库中配置（默认：8）。

## 故障排查

### 服务无法启动

```bash
# 查看日志
docker compose logs paddleocr-vl-api
docker compose logs paddleocr-vl-tritonserver
docker compose logs paddleocr-vlm-server

# 检查健康状态
curl http://localhost:8080/health/ready
```

### 超时错误

- 增加 `HPS_INFERENCE_TIMEOUT`（针对复杂文档）
- 检查 GPU 显存使用：`nvidia-smi`
- 如果 GPU 过载，减少 `HPS_MAX_CONCURRENT_REQUESTS`

### 显存不足

- 减少 `HPS_MAX_CONCURRENT_REQUESTS`
- 确保每个 GPU 只运行一个服务
- 检查 compose.yaml 中的 `shm_size`（默认：4GB）

## 开发调试

### 本地运行（不使用 Docker）

```bash
cd gateway
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080 --workers 4
```

### 并发测试

```bash
# 并发请求测试
for i in {1..20}; do
  curl -X POST http://localhost:8080/layout-parsing \
    -H "Content-Type: application/json" \
    -d '{"file": "...", "fileType": 1}' &
done
wait
```
