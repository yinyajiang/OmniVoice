#!/bin/bash

# 并发 TTS 压测脚本 (gRPC)
# 从项目根目录:  ./serve/concurrent_tts_test.sh 5
# 或:            bash /path/to/serve/concurrent_tts_test.sh 5
# 或:            cd serve && ./concurrent_tts_test.sh 5
# 服务地址:      serve/.env 中 CFG_GRPC_ADDR

SERVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SERVE_DIR}/.." && pwd)"
if [[ -f "${SERVE_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SERVE_DIR}/.env"
  set +a
fi

cd "$ROOT"

CONCURRENCY=${1:-5}
REF_AUDIO=${2:-}
MAX_DURATION_MS=${3:-1000000}
SERVER="${CFG_GRPC_ADDR:-127.0.0.1:50050}"
TEXT='King Charles III visit to the US was meant to be a celebration of America 250th anniversary, enduring Anglo American ties, and the special relationship. But it has also been billed as a rescue mission. The current state of US UK relations is strained, reflecting British reluctance to fully back the joint US Israeli war against Iran. The King goal has been to ease those tensions with a royal charm offensive, most notably with his joint address to Congress.'

if [ -n "$REF_AUDIO" ] && [ ! -f "$REF_AUDIO" ]; then
  echo "参考音频不存在: $REF_AUDIO" >&2
  exit 1
fi

echo "=========================================="
echo " 并发 TTS 压测 (gRPC)"
echo " 并发数:   $CONCURRENCY"
echo " 服务地址: $SERVER"
echo " max_duration_ms: $MAX_DURATION_MS"
if [ -n "$REF_AUDIO" ]; then
  echo " 模式:     声音克隆"
  echo " 参考音频: $REF_AUDIO"
else
  echo " 模式:     普通 TTS"
fi
echo "=========================================="

CLIENT_ARGS=(
  --server "$SERVER"
  bench
  --text "$TEXT"
  --language English
  --max-duration-ms "$MAX_DURATION_MS"
  -n "$CONCURRENCY"
)

if [ -n "$REF_AUDIO" ]; then
  CLIENT_ARGS+=(--ref-audio "$REF_AUDIO")
fi

uv run python -m serve.client "${CLIENT_ARGS[@]}"
