#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET="${DATASET:-TOOB_Simulation/data/raw/test.npz}"
DATA_KEY="${DATA_KEY:-X}"
LABELS_KEY="${LABELS_KEY:-y}"
MAPPING="${MAPPING:-}"
DF_BUILDER="${DF_BUILDER:-TOOB_Simulation/checkpoints/df/DF.py:DF}"
DF_CHECKPOINT="${DF_CHECKPOINT:-TOOB_Simulation/checkpoints/df/max_f1.pth}"

OUT_DIR="${OUT_DIR:-TOOB_Simulation/outputs}"
MAX_BURSTS="${MAX_BURSTS:-2000}"
TRACE_LEN="${TRACE_LEN:-5000}"
NUM_CLASSES="${NUM_CLASSES:-96}"
PROJECTION_CHUNK_SIZE="${PROJECTION_CHUNK_SIZE:-64}"
SOFT_PROJECTION_TAU="${SOFT_PROJECTION_TAU:-1.5}"
SET_SIZE="${SET_SIZE:-30}"
CLUSTER_ROUNDS="${CLUSTER_ROUNDS:-1}"
PROFILE_METHOD="${PROFILE_METHOD:-super}"
EXCLUDE_LABELS="${EXCLUDE_LABELS:-95}"

SMOKE_LIMIT="${SMOKE_LIMIT:-200}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-4}"
SMOKE_NOISE_DIM="${SMOKE_NOISE_DIM:-64}"

FULL_EPOCHS="${FULL_EPOCHS:-30}"
FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-32}"
FULL_NOISE_DIM="${FULL_NOISE_DIM:-256}"

require_file() {
  if [ ! -f "$1" ]; then
    echo "Missing required file: $1" >&2
    echo "Copy the dataset/checkpoint into the project or override the path with an environment variable." >&2
    exit 1
  fi
}

require_python_deps() {
  "$PYTHON_BIN" - <<'PY'
import importlib.util
missing = [name for name in ("numpy", "torch", "tqdm") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))
PY
}

if [ "$MODE" = "smoke" ]; then
  RUN_DIR="${OUT_DIR}_smoke"
  LIMIT_ARGS=(--limit "$SMOKE_LIMIT")
  EPOCHS="$SMOKE_EPOCHS"
  BATCH_SIZE="$SMOKE_BATCH_SIZE"
  NOISE_DIM="$SMOKE_NOISE_DIM"
  PSEUDO_ARGS=()
elif [ "$MODE" = "full" ]; then
  RUN_DIR="$OUT_DIR"
  LIMIT_ARGS=()
  EPOCHS="$FULL_EPOCHS"
  BATCH_SIZE="$FULL_BATCH_SIZE"
  NOISE_DIM="$FULL_NOISE_DIM"
  PSEUDO_ARGS=()
elif [ "$MODE" = "one" ]; then
  RUN_DIR="${OUT_DIR}_one"
  LIMIT_ARGS=()
  EPOCHS="${ONE_EPOCHS:-1}"
  BATCH_SIZE="${ONE_BATCH_SIZE:-4}"
  NOISE_DIM="${ONE_NOISE_DIM:-64}"
  PSEUDO_LABEL="${PSEUDO_LABEL:-0}"
  PSEUDO_ARGS=(--pseudo-labels "$PSEUDO_LABEL")
else
  echo "Usage: bash TOOB_Simulation/run_exp_ubuntu.sh [smoke|one|full]"
  echo ""
  echo "Environment overrides:"
  echo "  PYTHON_BIN DATASET DF_BUILDER DF_CHECKPOINT OUT_DIR"
  echo "  SET_SIZE CLUSTER_ROUNDS PROFILE_METHOD EXCLUDE_LABELS"
  echo "  FULL_EPOCHS FULL_BATCH_SIZE FULL_NOISE_DIM"
  echo "  SMOKE_LIMIT SMOKE_EPOCHS SMOKE_BATCH_SIZE SMOKE_NOISE_DIM"
  echo "  PSEUDO_LABEL for mode 'one'"
  exit 2
fi

require_file "$DATASET"
if [ -n "$MAPPING" ]; then
  require_file "$MAPPING"
  MAPPING_ARGS=(--mapping "$MAPPING")
else
  MAPPING_ARGS=()
fi
require_file "${DF_BUILDER%%:*}"
require_file "$DF_CHECKPOINT"
require_python_deps

BURST_NPZ="${RUN_DIR}/burst_dataset.npz"
PSEUDO_NPZ="${RUN_DIR}/pseudo_labels.npz"
PSEUDO_JSON="${RUN_DIR}/pseudo_labels.json"
GENERATOR_DIR="${RUN_DIR}/generators"
ADV_DIRECTION_NPZ="${RUN_DIR}/toob_adv_direction.npz"

mkdir -p "$RUN_DIR"

echo "[0/4] Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "[0/4] TOOB imports"
"$PYTHON_BIN" TOOB_Simulation/EXP/00_check_imports.py

echo "[1/4] Direction sequence -> burst dataset"
"$PYTHON_BIN" TOOB_Simulation/EXP/01_make_burst_dataset.py \
  --input "$DATASET" \
  --output "$BURST_NPZ" \
  --data-key "$DATA_KEY" \
  --labels-key "$LABELS_KEY" \
  --max-bursts "$MAX_BURSTS" \
  "${LIMIT_ARGS[@]}"

EXCLUDE_ARGS=()
if [ -n "$EXCLUDE_LABELS" ]; then
  # shellcheck disable=SC2206
  EXCLUDE_ARGS=(--exclude-labels $EXCLUDE_LABELS)
fi

echo "[2/4] Cluster burst profiles -> pseudo labels"
"$PYTHON_BIN" TOOB_Simulation/EXP/02_make_pseudo_labels.py \
  --labels-npz "$BURST_NPZ" \
  --output "$PSEUDO_NPZ" \
  --json-output "$PSEUDO_JSON" \
  --set-size "$SET_SIZE" \
  --rounds "$CLUSTER_ROUNDS" \
  --profile-method "$PROFILE_METHOD" \
  --drop-unmapped \
  "${MAPPING_ARGS[@]}" \
  "${EXCLUDE_ARGS[@]}"

echo "[3/4] Train cluster-wise burst generators"
"$PYTHON_BIN" TOOB_Simulation/EXP/03_train_generators.py \
  --burst-npz "$BURST_NPZ" \
  --pseudo-npz "$PSEUDO_NPZ" \
  --detector-builder "$DF_BUILDER" \
  --detector-checkpoint "$DF_CHECKPOINT" \
  --num-classes "$NUM_CLASSES" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --noise-dim "$NOISE_DIM" \
  --detector-input-kind direction \
  --detector-input-layout ncl \
  --detector-input-length "$TRACE_LEN" \
  --soft-projection-tau "$SOFT_PROJECTION_TAU" \
  --projection-chunk-size "$PROJECTION_CHUNK_SIZE" \
  --output-dir "$GENERATOR_DIR" \
  "${PSEUDO_ARGS[@]}"

echo "[4/4] Export defended direction dataset"
"$PYTHON_BIN" TOOB_Simulation/EXP/04_generate_dataset.py \
  --burst-npz "$BURST_NPZ" \
  --pseudo-npz "$PSEUDO_NPZ" \
  --generator-dir "$GENERATOR_DIR" \
  --output "$ADV_DIRECTION_NPZ" \
  --output-kind direction \
  --max-trace-len "$TRACE_LEN" \
  --batch-size 256 \
  --round

echo "Done."
echo "Output dataset: $ADV_DIRECTION_NPZ"
echo "Pseudo-label JSON: $PSEUDO_JSON"
