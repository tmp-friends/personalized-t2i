#!/usr/bin/env bash
# Convenience wrapper: starts the local LLM server if it isn't up, then runs the
# official code against it. Everything it does is spelled out in SETUP.md; this
# just saves opening a second terminal.
#
#   ./run.sh demo 'a cat'                 # demo.py, rewrite + 2 images
#   ./run.sh demo 'a cat' --no-t2i        # rewrite only
#   ./run.sh main --limit_users=5         # main.py over the PIP dataset
#   ./run.sh serve                        # just the LLM server, in the foreground
#
# To use the real ChatGPT instead, skip this script and follow SETUP.md §1:
#   export TV_OPENAI_KEY=sk-... ; unset TV_OPENAI_BASE ; python demo.py ...
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-../.venv/bin/python}
PORT=${TV_LLM_PORT:-8000}
BASE="http://127.0.0.1:${PORT}/v1"
LOG=${TV_LLM_LOG:-/tmp/tv_local_llm.log}

start_server() {
  if curl -sf --max-time 2 "${BASE}/models" >/dev/null 2>&1; then
    echo "[run] local LLM already listening on ${BASE}"
    return
  fi
  echo "[run] starting local LLM server (log: ${LOG}) ..."
  nohup "$PY" serve_local_llm.py --port "$PORT" >"$LOG" 2>&1 &
  # First start loads a 4B model; give it room.
  for _ in $(seq 1 120); do
    curl -sf --max-time 2 "${BASE}/models" >/dev/null 2>&1 && { echo "[run] ready"; return; }
    sleep 2
  done
  echo "[run] server did not come up; see ${LOG}" >&2
  exit 1
}

cmd=${1:-demo}; shift || true

case "$cmd" in
  serve)
    exec "$PY" serve_local_llm.py --port "$PORT"
    ;;
  demo)
    start_server
    prompt=${1:-a cat}; shift || true
    TV_OPENAI_BASE="$BASE" TV_OPENAI_KEY=local \
      exec "$PY" demo.py --input_prompt="$prompt" "$@"
    ;;
  main)
    start_server
    TV_OPENAI_BASE="$BASE" TV_OPENAI_KEY=local \
      exec "$PY" main.py --retrieval=ebr --num_retrieval=3 \
                         --rewrite_method=ICL --ICL_shot=1 "$@"
    ;;
  stop)
    pkill -f "serve_local_llm.py --port ${PORT}" && echo "[run] stopped" || echo "[run] not running"
    ;;
  *)
    echo "usage: $0 {demo|main|serve|stop} [args...]" >&2
    exit 2
    ;;
esac
