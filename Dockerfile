# Day0 Krea2 Trainer image — Made by Yewcake
# Heavy, stable dependencies live here. Trainer + UI code is pulled from
# GitHub at pod boot (see docker/start.sh), so code updates never need a rebuild.

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.hf_cache \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    HF_XET_HIGH_PERFORMANCE=1

RUN apt-get update && apt-get install --no-install-recommends -y \
    git curl wget unzip zip rsync tmux htop nvtop ffmpeg \
    p7zip-full p7zip-rar \
    python3.12 python3.12-venv python3-pip python3-dev \
    openssh-server ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    || (apt-get update && apt-get install --no-install-recommends -y \
    git curl wget unzip zip rsync tmux htop nvtop ffmpeg \
    p7zip-full \
    python3.12 python3.12-venv python3-pip python3-dev \
    openssh-server ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*)

# One shared venv for trainer + UI. Baked into the image so jobs start instantly.
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install -U pip wheel packaging && \
    pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# Diffusers/Transformers from main for fresh Krea2 support. Pin by rebuilding
# the image when you want to advance these.
RUN pip install -U \
    git+https://github.com/huggingface/diffusers.git \
    git+https://github.com/huggingface/transformers.git && \
    pip install -U accelerate peft safetensors pillow tqdm wandb bitsandbytes \
    hf-transfer sentencepiece protobuf requests \
    fastapi "uvicorn[standard]" python-multipart

COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

WORKDIR /workspace
EXPOSE 8888
CMD ["/start.sh"]
