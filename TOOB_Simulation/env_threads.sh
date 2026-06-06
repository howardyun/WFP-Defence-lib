#!/usr/bin/env bash

# Normalize BLAS/OpenMP thread environment variables before Python imports
# numpy/torch. Some container images export invalid values such as "auto",
# empty strings, or zero, which makes libgomp abort during import.

THREAD_COUNT="${THREAD_COUNT:-1}"
if ! [[ "$THREAD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "warning: invalid THREAD_COUNT='$THREAD_COUNT'; using 1" >&2
  THREAD_COUNT=1
fi
export THREAD_COUNT

sanitize_thread_env() {
  local name="$1"
  local value="${!name:-}"
  if [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    export "$name=$value"
    return
  fi
  if [ -n "$value" ]; then
    echo "warning: invalid $name='$value'; using $THREAD_COUNT" >&2
  fi
  export "$name=$THREAD_COUNT"
}

sanitize_thread_env OMP_NUM_THREADS
sanitize_thread_env MKL_NUM_THREADS
sanitize_thread_env OPENBLAS_NUM_THREADS
sanitize_thread_env NUMEXPR_NUM_THREADS
