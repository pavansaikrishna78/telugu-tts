#!/usr/bin/env bash
# Reproduces the Telugu TTS environment. Safe to re-run: every step checks first.
# Never upgrades the system, never touches the NVIDIA driver, never uses the system Python.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT/.venv"
PY_VERSION="3.10"
# Driver 570 reports CUDA 12.8; cu124 wheels run on it. torch<2.6 keeps the older
# torch.load default that many checkpoints expect.
TORCH_VERSION="2.5.1"
TORCH_CUDA="cu124"
# Pinned for reproducibility. Change deliberately.
INDICF5_COMMIT="13f7c4d627cc10111aea8fe9c0039462cacacdc7"

cd "$PROJECT"
mkdir -p refs input output logs hf-cache tests

# --- 1. System packages: ffmpeg and sox only ---------------------------------
APT_MISSING=()
command -v ffmpeg >/dev/null || APT_MISSING+=(ffmpeg)
command -v sox    >/dev/null || APT_MISSING+=(sox)
APT_PENDING=0
if [ "${#APT_MISSING[@]}" -gt 0 ]; then
  if sudo -n true 2>/dev/null || [ -t 0 ]; then
    sudo apt-get update
    sudo apt-get install -y --no-upgrade "${APT_MISSING[@]}"
  else
    APT_PENDING=1
    echo "WARN: ${APT_MISSING[*]} missing and sudo needs a password. Continuing with the rest."
  fi
fi

# --- 2. uv for this user -----------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
uv --version

# --- 3. Python 3.10 venv -----------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python "$PY_VERSION" "$VENV"
fi
"$VENV/bin/python" --version | grep -q "^Python $PY_VERSION" \
  || { echo "ERROR: $VENV is not Python $PY_VERSION"; exit 1; }

# --- 4. Keep HF_HOME inside the project whenever the venv is active ----------
MARKER="# telugu-tts env"
if ! grep -qF "$MARKER" "$VENV/bin/activate"; then
  cat >> "$VENV/bin/activate" <<EOF

$MARKER
export HF_HOME="$PROJECT/hf-cache"
EOF
fi
export HF_HOME="$PROJECT/hf-cache"
export VIRTUAL_ENV="$VENV"

# --- 5. PyTorch for the matching CUDA ----------------------------------------
if ! "$VENV/bin/python" -c "import torch,sys; sys.exit(0 if torch.__version__=='$TORCH_VERSION+$TORCH_CUDA' else 1)" 2>/dev/null; then
  uv pip install "torch==$TORCH_VERSION+$TORCH_CUDA" "torchaudio==$TORCH_VERSION+$TORCH_CUDA" \
    --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
fi

# --- 6. IndicF5 and tools ----------------------------------------------------
# setup.py leaves transformers unpinned while requirements.txt says <4.50, so the
# pins are enforced here. uv treats a re-run as a no-op when they are satisfied.
uv pip install \
  "f5_tts @ git+https://github.com/AI4Bharat/IndicF5.git@$INDICF5_COMMIT" \
  "numpy<=1.26.4" "transformers<4.50" "pyarrow<21" \
  "torch==$TORCH_VERSION+$TORCH_CUDA" "torchaudio==$TORCH_VERSION+$TORCH_CUDA" \
  soundfile librosa pyloudnorm gradio pytest faster-whisper "huggingface_hub>=0.34" \
  --extra-index-url "https://download.pytorch.org/whl/$TORCH_CUDA" \
  --index-strategy unsafe-best-match

# --- 7. Verify ---------------------------------------------------------------
"$VENV/bin/python" - <<'PY'
import numpy, transformers, torch
assert torch.cuda.is_available(), "torch cannot see CUDA"
name = torch.cuda.get_device_name(0)
assert "Tesla T4" in name, f"unexpected GPU: {name}"
assert tuple(int(x) for x in numpy.__version__.split(".")[:2]) <= (1, 26), numpy.__version__
assert tuple(int(x) for x in transformers.__version__.split(".")[:2]) < (4, 50), transformers.__version__
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | {name}")
print(f"numpy {numpy.__version__} | transformers {transformers.__version__}")
PY
[ -x "$VENV/bin/hf" ] || { echo "ERROR: hf CLI missing from venv"; exit 1; }

if [ "$APT_PENDING" = 1 ]; then
  echo "INCOMPLETE: run this yourself, then re-run setup.sh:"
  echo "  sudo apt-get update && sudo apt-get install -y --no-upgrade ffmpeg sox"
  exit 2
fi
echo "setup.sh finished OK"
