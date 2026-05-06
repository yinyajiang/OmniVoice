#!/usr/bin/env bash
#
# OmniVoice gRPC deployment. Run from anywhere:  bash serve/deploy.sh
# Configuration: serve/.env (see serve/.env.example)

set -euo pipefail

SERVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SERVE_DIR}/.." && pwd)"
if [[ -f "${SERVE_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SERVE_DIR}/.env"
  set +a
fi

# Priority: runtime env > serve/.env > defaults.
: "${BASE_PORT:=50551}"
: "${PID_DIR:=/tmp/deploy_pids}"
: "${WORKERS:=1}"
# num <= workers
: "${LOAD_ASR_WORKERS_COUNT:=1}"
: "${GATEWAY_PORT:=50550}"

cd "$ROOT"

if ((LOAD_ASR_WORKERS_COUNT < 0 || LOAD_ASR_WORKERS_COUNT > WORKERS)); then
    echo "LOAD_ASR_WORKERS_COUNT must be between 0 and WORKERS." >&2
    exit 1
fi

echo "BASE_PORT: $BASE_PORT"
echo "PID_DIR: $PID_DIR"
echo "WORKERS: $WORKERS"
echo "LOAD_ASR_WORKERS_COUNT: $LOAD_ASR_WORKERS_COUNT"
echo "GATEWAY_PORT: $GATEWAY_PORT"

stop_workers() {
    if [[ -d "$PID_DIR" ]]; then
        for pidfile in "$PID_DIR"/*.pid; do
            [[ -f "$pidfile" ]] || continue
            pid=$(<"$pidfile")
            kill "$pid" 2>/dev/null || true
            rm -f "$pidfile"
        done
    fi
}

stop_workers 2>/dev/null || true
mkdir -p "$PID_DIR"

# ── Launch workers ───────────────────────────────────────────────────────
WORKER_ADDRS=()
for ((worker_idx=0; worker_idx<WORKERS; worker_idx++)); do
    port=$((BASE_PORT + worker_idx))
    base_addr="127.0.0.1:${port}"
    load_asr=false
    if ((worker_idx < LOAD_ASR_WORKERS_COUNT)); then
        load_asr=true
    fi
    advertised_addr="${base_addr}"
    if [[ "${load_asr}" == "true" ]]; then
        advertised_addr="${advertised_addr}-asr"
    fi
    WORKER_ADDRS+=("${advertised_addr}")

    echo "Starting worker $worker_idx on port $port (load_asr=${load_asr}) ..."

    PYTHONPATH="${ROOT}/serve:${PYTHONPATH:-}" \
    uv run python -m serve.server --port "$port" --load-asr "$load_asr" &
    worker_pid=$!

    echo "$worker_pid" > "$PID_DIR/worker_${worker_idx}.pid"

    echo "Waiting for worker $worker_idx (${base_addr}) ..."
    ready=0
    for _ in $(seq 1 300); do
        if PYTHONPATH="${ROOT}/serve:${PYTHONPATH:-}" \
           uv run python -m serve.client --server "${base_addr}" health >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 1
    done
    if [[ $ready -ne 1 ]]; then
        echo "Worker $worker_idx failed to start within 300s." >&2
        exit 1
    fi
    echo "Worker $worker_idx ready."
done


# ── Launch gateway ───────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Deployment"
echo "  Gateway port:     $GATEWAY_PORT"
echo "  Workers:          $WORKERS"
echo "  Worker addresses: ${WORKER_ADDRS[@]}"
echo "============================================"
echo ""

PYTHONPATH="${ROOT}/serve:${PYTHONPATH:-}" \
uv run python -m serve.gateway \
    --port "$GATEWAY_PORT" \
    --workers "${WORKER_ADDRS[@]}"
