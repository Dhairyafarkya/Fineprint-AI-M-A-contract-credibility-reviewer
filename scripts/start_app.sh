#!/usr/bin/env bash
# Start the Fine Print app.
#
#   bash scripts/start_app.sh             # http://localhost:8501 (or the machine's IP on port 8501)
#   bash scripts/start_app.sh --public    # also prints a temporary public https link (Cloudflare tunnel),
#                                         # needed on Colab/Kaggle where port 8501 is not reachable
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8501}"

curl -sf http://localhost:11434/api/tags >/dev/null || echo "Note: Ollama is not running. Run scripts/setup_t4.sh first (or set LLM_PROVIDER = \"gemini\" in utils.py)."

setsid python3 -m streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0 --server.headless true \
  --browser.gatherUsageStats false > streamlit.log 2>&1 &
APP_PID=$!
for _ in $(seq 1 30); do curl -sf "http://localhost:$PORT/_stcore/health" >/dev/null && break; sleep 1; done
echo "App running on http://localhost:$PORT (log: streamlit.log, pid $APP_PID)"

if [ "${1:-}" = "--public" ]; then
  if [ ! -x ./cloudflared ]; then
    curl -fsSL -o cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x cloudflared
  fi
  setsid ./cloudflared tunnel --url "http://localhost:$PORT" --no-autoupdate > tunnel.log 2>&1 < /dev/null &
  for _ in $(seq 1 30); do
    URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' tunnel.log | head -1 || true)
    [ -n "$URL" ] && break; sleep 1
  done
  echo "Public link: ${URL:-see tunnel.log}"
fi
# Keep running in a terminal; in a notebook, run this script in the background instead.
[ -t 1 ] && wait $APP_PID || true
