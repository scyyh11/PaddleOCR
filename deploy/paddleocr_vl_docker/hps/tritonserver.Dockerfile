# Build args for hardware flexibility
# For CPU-only or non-NVIDIA hardware, override these at build time:
#   docker build --build-arg BASE_IMAGE=<cpu-image> --build-arg DEVICE_TYPE=cpu ...
ARG BASE_IMAGE=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.3-gpu
ARG DEVICE_TYPE=gpu

FROM ${BASE_IMAGE}
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install PaddleX for Python backend models (if not already in base image)
RUN pip install --no-cache-dir paddlex>=3.4.0 || true

WORKDIR /app
COPY paddlex_hps_PaddleOCR-VL_sdk/server .

# TODO: Remove this once the SDK includes restructure-pages model
# Copy additional Python backend model for CPU-only restructure-pages operation
# Note: restructure-pages uses KIND_CPU in config.pbtxt, so it works on all hardware
COPY model_repository/restructure-pages model_repository/restructure-pages

ARG DEVICE_TYPE
ENV PADDLEX_HPS_DEVICE_TYPE=${DEVICE_TYPE}
CMD ["/bin/bash", "server.sh"]
