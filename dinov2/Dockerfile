FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libglib2.0-0 \
        libgl1 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# xFormers is required by train/ssl_meta_arch.py. This version matches torch 2.4.1.
RUN python -m pip install xformers==0.0.28.post1 --no-deps

RUN python -m pip install \
    fvcore \
    monai==1.4.0 \
    nibabel \
    omegaconf \
    scikit-learn \
    torchio \
    torchmetrics

COPY . /workspace/dinov2

WORKDIR /workspace

RUN python - <<'PY'
import torch
import torchvision
import xformers
import monai
import dinov2

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("xformers", xformers.__version__)
print("monai", monai.__version__)
print("CUDA available at build time:", torch.cuda.is_available())
PY

CMD ["python", "dinov2/train/train3d.py", "--help"]
