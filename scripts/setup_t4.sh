#!/usr/bin/env bash
# One-time setup of Ollama + models on a Linux machine with an NVIDIA T4 (16 GB).
# Works on a cloud VM (Ubuntu + NVIDIA driver), Google Colab and Kaggle.
#
#   bash scripts/setup_t4.sh            # install/configure Ollama, pull models, install Python packages
#
# Models come from utils.py (OLLAMA_MODEL, OLLAMA_EMBED_MODEL, OLLAMA_PARALLEL, OLLAMA_CONTEXT).
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf "\n\033[1m%s\033[0m\n" "$*"; }
setting() { python3 -c "import importlib.util as u; s=u.spec_from_file_location('s','utils.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print(getattr(m,'$1','$2'))"; }

MODEL=$(setting OLLAMA_MODEL qwen3:14b)
EMBED=$(setting OLLAMA_EMBED_MODEL nomic-embed-text)
PARALLEL=$(setting OLLAMA_PARALLEL 2)
CONTEXT=$(setting OLLAMA_CONTEXT 16384)
SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO="sudo"

# Server settings tuned for a 16 GB T4:
#  - flash attention + 8-bit KV cache: halves context memory with negligible quality loss
#  - 2 parallel requests: batched decoding roughly doubles throughput on a memory-bound GPU
#  - keep both models (chat + embeddings) loaded so nothing reloads between steps
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_NUM_PARALLEL="$PARALLEL"
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_CONTEXT_LENGTH="$CONTEXT"

say "1/5 GPU"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: nvidia-smi not found. Ollama will run on the CPU, which is far too slow for this app."
fi

say "2/5 Ollama"
if ! command -v ollama >/dev/null; then
  command -v zstd >/dev/null || { $SUDO apt-get update -qq && $SUDO apt-get install -y -qq zstd pciutils lshw >/dev/null || true; }
  curl -fsSL https://ollama.com/install.sh | sh
fi
ollama --version || true

say "3/5 Start the Ollama server with T4 settings"
if command -v systemctl >/dev/null && systemctl list-unit-files 2>/dev/null | grep -q '^ollama.service'; then
  $SUDO mkdir -p /etc/systemd/system/ollama.service.d
  $SUDO tee /etc/systemd/system/ollama.service.d/fineprint-t4.conf >/dev/null <<EOF
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_NUM_PARALLEL=$PARALLEL"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_KEEP_ALIVE=30m"
Environment="OLLAMA_CONTEXT_LENGTH=$CONTEXT"
EOF
  $SUDO systemctl daemon-reload && $SUDO systemctl restart ollama
else
  # Colab / Kaggle / containers: no systemd, run the server in the background.
  pkill -f "ollama serve" 2>/dev/null || true
  setsid nohup ollama serve > ollama.log 2>&1 < /dev/null &
fi
for _ in $(seq 1 60); do curl -sf http://localhost:11434/api/tags >/dev/null && break; sleep 1; done
curl -sf http://localhost:11434/api/tags >/dev/null || { echo "Ollama did not start; see ollama.log"; exit 1; }
echo "Ollama is running."

say "4/5 Pull models: $MODEL and $EMBED"
ollama pull "$MODEL"
ollama pull "$EMBED"
# Warm up: load the chat model onto the GPU now instead of during the first review.
curl -s http://localhost:11434/api/generate -d "{\"model\":\"$MODEL\",\"prompt\":\"ok\",\"stream\":false,\"think\":false,\"options\":{\"num_ctx\":$CONTEXT,\"num_predict\":1}}" >/dev/null || true
ollama ps

say "5/5 Python packages"
python3 -m pip install -q -r requirements.txt

say "Done. Check speed with:  python3 scripts/benchmark_ollama.py"
echo "Start the app with:        bash scripts/start_app.sh   (add --public for a shareable link)"
