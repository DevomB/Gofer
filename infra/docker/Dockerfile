# syntax=docker/dockerfile:1
# Gofer training image: engine (Go + in-process ONNX Runtime) + learner (PyTorch)
# + orchestrator. One image runs the whole loop on any GPU/CPU host.
#
#   GPU (default):  docker build -f infra/docker/Dockerfile -t gofer-train .
#   CPU only:       docker build -f infra/docker/Dockerfile -t gofer-train:cpu \
#                     --build-arg BASE=python:3.11-slim --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
#   Run:            docker run --gpus all -v $PWD/runs:/gofer/runs gofer-train \
#                     run --config configs/pipeline-gpu.toml --no-build
ARG BASE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ARG GO_VERSION=1.22
ARG ORT_VERSION=1.26.0

# ---- engine: CGO build with the onnx tag (links ORT at runtime via dlopen) ----
FROM golang:${GO_VERSION}-bookworm AS engine
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd ./cmd
# default.pgo is git-ignored (a local `make pgo-profile` artifact): use it when present.
COPY default.pg[o] ./
RUN PGO=off; [ -f default.pgo ] && PGO=default.pgo; \
    CGO_ENABLED=1 go build -tags=onnx -pgo="$PGO" -o /out/gofer ./cmd/gofer

# ---- runtime ----
FROM ${BASE}
ARG ORT_VERSION
ARG TORCH_INDEX=""
ENV PYTHONUNBUFFERED=1 \
    ONNXRUNTIME_SHARED_LIBRARY_PATH=/opt/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so.${ORT_VERSION}
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    | tar -xz -C /opt \
 && test -f "$ONNXRUNTIME_SHARED_LIBRARY_PATH"
WORKDIR /gofer
COPY training/requirements.txt training/requirements.txt
# The CUDA base already ships torch; the CPU base pulls the small CPU wheel.
RUN if [ -n "$TORCH_INDEX" ]; then pip install --no-cache-dir torch --index-url "$TORCH_INDEX"; fi \
 && pip install --no-cache-dir -r training/requirements.txt pytest
COPY . .
COPY --from=engine /out/gofer bin/gofer
RUN python -m training.pipeline config --config configs/pipeline-gpu.toml > /dev/null
VOLUME ["/gofer/runs"]
ENTRYPOINT ["python", "-m", "training.pipeline"]
CMD ["run", "--config", "configs/pipeline-gpu.toml", "--no-build"]
