# CUDA + cuDNN runtime (Ubuntu 22.04)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_WORKERS=1

WORKDIR /workspace

# ---- System deps (OpenSlide for WSI) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git ca-certificates curl \
    libopenslide0 openslide-tools libglib2.0-0 \
    libjpeg-turbo8 libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps: Torch (CUDA 12.1) first, then libs ----
RUN python3 -m pip install --upgrade pip
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# (Optional but useful)
RUN pip install runpod requests pillow numpy tqdm huggingface_hub openslide-python

# ---- Bring in the repo code (assumes your fork has the RunPod files) ----
# If you're building from GitHub in Hub, RunPod will clone for you.
# For local testing, uncomment these two lines:
# RUN git clone https://github.com/YOUR_USERNAME/prov-gigapath.git
# WORKDIR /workspace/prov-gigapath

# Install repo Python package/requirements if provided
# Try common patterns; ignore failures if files don't exist
RUN bash -lc 'if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi'
RUN bash -lc 'if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then pip install -e . || true; fi'

# ---- Add the serverless handler ----
COPY handler.py /workspace/handler.py

# (Optional) Cache/weights directory
RUN mkdir -p /workspace/model

# ---- Serverless entrypoint ----
CMD ["python3", "/workspace/handler.py"]
