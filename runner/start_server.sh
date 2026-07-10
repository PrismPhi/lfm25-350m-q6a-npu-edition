#!/usr/bin/env bash
set -euo pipefail

RUNNER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${LFM25_STATE_DIR:-${XDG_DATA_HOME:-${HOME}/.local/share}/lfm25-350m-q6a-npu-edition}"
PYTHON="${LFM25_PYTHON:-python3}"
HOST="${LFM25_HOST:-127.0.0.1}"
PORT="${LFM25_PORT:-18080}"

exec "$PYTHON" "$RUNNER_DIR/scripts/server.py" \
  --host "$HOST" \
  --port "$PORT" \
  --log-root "$STATE_DIR/logs" \
  --chunk-context "$STATE_DIR/contexts/chunk/chunk_epcontext.onnx" \
  --decode-context "$STATE_DIR/contexts/decode/decode_epcontext.onnx" \
  --tokenizer "$STATE_DIR/models/tokenizer/tokenizer.json" \
  --rope-cache "$STATE_DIR/models/host/rope_cache.npz" \
  --embedding-int8-dir "$STATE_DIR/models/host/embedding_int8_rowwise" \
  --v0-runner-dir "$RUNNER_DIR/scripts" \
  --chunk 16 \
  --total-len 2048 \
  --default-profile chat
