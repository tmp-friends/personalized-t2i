#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
PATCHED_UPSTREAM="$ROOT/.work/upstream"
DATA="$ROOT/data/raw/user_data"
OUTPUTS="$ROOT/outputs/official"
PORT="${TV_LLM_PORT:-8000}"
BASE="http://127.0.0.1:${PORT}/v1"
LOG="${TV_LLM_LOG:-/tmp/tv_local_llm.log}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PATCHED_UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  echo "usage: $0 {demo|main|serve|stop} [args...]" >&2
}

require_upstream() {
  if [[ ! -f "$PATCHED_UPSTREAM/main.py" ]]; then
    echo "run 'python scripts/prepare_upstream.py' first" >&2
    exit 1
  fi
}

start_server() {
  if curl -sf --max-time 2 "${BASE}/models" >/dev/null 2>&1; then
    echo "[run] local LLM already listening on ${BASE}"
    return
  fi
  echo "[run] starting local LLM server (log: ${LOG}) ..."
  nohup "$PY" "$SCRIPT_DIR/serve_local_llm.py" --port "$PORT" >"$LOG" 2>&1 &
  for _ in $(seq 1 120); do
    curl -sf --max-time 2 "${BASE}/models" >/dev/null 2>&1 && { echo "[run] ready"; return; }
    sleep 2
  done
  echo "[run] server did not come up; see ${LOG}" >&2
  exit 1
}

cmd="${1:-demo}"
if [[ "$cmd" == "--help" || "$cmd" == "-h" || "$cmd" == "help" ]]; then
  usage
  exit 2
fi
shift || true

case "$cmd" in
  serve)
    exec "$PY" "$SCRIPT_DIR/serve_local_llm.py" --port "$PORT" "$@"
    ;;
  demo)
    require_upstream
    start_server
    mkdir -p "$OUTPUTS"
    ln -sfn "$DATA" "$OUTPUTS/user_data"
    prompt="${1:-a cat}"
    shift || true
    cd "$OUTPUTS"
    TV_OPENAI_BASE="$BASE" TV_OPENAI_KEY=local \
      exec "$PY" "$PATCHED_UPSTREAM/demo.py" --input_prompt="$prompt" "$@"
    ;;
  main)
    require_upstream
    start_server
    mkdir -p "$OUTPUTS"
    cd "$OUTPUTS"
    TV_OPENAI_BASE="$BASE" TV_OPENAI_KEY=local \
      exec "$PY" "$PATCHED_UPSTREAM/main.py" --retrieval=ebr --num_retrieval=3 \
        --rewrite_method=ICL --ICL_shot=1 --data_folder="$DATA" "$@"
    ;;
  stop)
    pkill -f "serve_local_llm.py --port ${PORT}" && echo "[run] stopped" || echo "[run] not running"
    ;;
  *)
    usage
    exit 2
    ;;
esac
