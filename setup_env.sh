#!/usr/bin/env bash
#set -euo pipefail
# ------------------------------------------------ CONFIG --------------------
export HF_HOME="/workspace/.cache"      # where HF downloads go
PYTHON_BIN="python3.11"                     # host interpreter
CUDA_TAG="cu128"                            # CUDA 12.8
TORCH_VERSION="2.7.1"                       # exact micro that wheels were built for

# ------------------------------------------------ BASE TOOLS ----------------
$PYTHON_BIN -m pip install -qU uv           # tiny bootstrap
uv venv .venv
source .venv/bin/activate                   # Windows: .\.venv\Scripts\activate

# ------------------------------------------------ CORE DEPENDENCIES ---------
uv pip install  \
  "torch==${TORCH_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

uv pip install  \
  datasets==2.19.1 \
  trl==0.8.1 \
  accelerate==0.28.0 \
  tensorboard==2.16.2 \
  huggingface_hub==0.23.2 \
  pandas==2.2.2 \
  numpy==1.26.4 \
  nnsight==0.4.6 \
  scikit-learn \
  matplotlib \
  jupyter \
  peft \
  transformers


uv pip install  bitsandbytes>=0.45.5 || echo "⚠️  bitsandbytes wheel failed; continuing"


uv pip install -U transformers
uv pip install -U peft

uv pip install sae-lens

#uv pip install "${FLASH_WHEEL}"
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.18/flash_attn-2.7.4+cu128torch2.7-cp311-cp311-linux_x86_64.whl
# ---------------------------------------------------------------------------
#  5. lock the environment ---------------------------------------------------
# ---------------------------------------------------------------------------
{ echo '--extra-index-url https://download.pytorch.org/whl/cu128';
  python -m uv pip freeze --all | sort -f; } > requirements.txt
echo "✅  locked environment written to requirements.lock"
# ---------------------------------------------------------------------------
#  6. (optional) Hugging Face login -----------------------------------------
# ---------------------------------------------------------------------------
echo
echo "👉  Run HF login (press Enter to skip)…"

uv pip install -U transformers trl fire tensorboard
uv pip install -U torch

huggingface-cli login --token #

