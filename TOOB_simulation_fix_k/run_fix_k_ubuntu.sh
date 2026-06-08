#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

sanitize_threads() {
  local name="$1"
  local value="${!name:-}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    export "$name=1"
  fi
}

sanitize_threads OMP_NUM_THREADS
sanitize_threads MKL_NUM_THREADS
sanitize_threads OPENBLAS_NUM_THREADS
sanitize_threads NUMEXPR_NUM_THREADS

PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" TOOB_simulation_fix_k/fix_k_pipeline.py "$MODE"
