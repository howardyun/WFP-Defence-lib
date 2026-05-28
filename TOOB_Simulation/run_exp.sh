#!/usr/bin/env sh
set -eu

MODE="${1:-smoke}"

if [ -d "/d/Git/usr/bin" ]; then
  PATH="/d/Git/usr/bin:$PATH"
  export PATH
fi

if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_EXE="$PYTHON_BIN"
elif [ -x "/d/Anaconda/envs/mytorch/python.exe" ]; then
  PYTHON_EXE="/d/Anaconda/envs/mytorch/python.exe"
elif [ -x "/mnt/d/Anaconda/envs/mytorch/python.exe" ]; then
  PYTHON_EXE="/mnt/d/Anaconda/envs/mytorch/python.exe"
elif [ -x "D:/Anaconda/envs/mytorch/python.exe" ]; then
  PYTHON_EXE="D:/Anaconda/envs/mytorch/python.exe"
else
  PYTHON_EXE="python"
fi

DATASET="${DATASET:-TOOB_Simulation/data/raw/test.npz}"
DATA_KEY="${DATA_KEY:-X}"
LABELS_KEY="${LABELS_KEY:-y}"
MAPPING="${MAPPING:-TOOB_Simulation/assets/website_to_set_1000_30_1.json}"
DF_BUILDER="${DF_BUILDER:-TOOB_Simulation/checkpoints/df/DF.py:DF}"
DF_CHECKPOINT="${DF_CHECKPOINT:-TOOB_Simulation/checkpoints/df/max_f1.pth}"

OUT_DIR="${OUT_DIR:-TOOB_Simulation/outputs}"
MAX_BURSTS="${MAX_BURSTS:-2000}"
TRACE_LEN="${TRACE_LEN:-5000}"
NUM_CLASSES="${NUM_CLASSES:-96}"
PROJECTION_CHUNK_SIZE="${PROJECTION_CHUNK_SIZE:-64}"
SOFT_PROJECTION_TAU="${SOFT_PROJECTION_TAU:-1.5}"

SMOKE_LIMIT="${SMOKE_LIMIT:-200}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-4}"
SMOKE_NOISE_DIM="${SMOKE_NOISE_DIM:-64}"

FULL_EPOCHS="${FULL_EPOCHS:-30}"
FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-32}"
FULL_NOISE_DIM="${FULL_NOISE_DIM:-256}"

if [ "$MODE" = "smoke" ]; then
  RUN_DIR="${OUT_DIR}_smoke"
  LIMIT_ARGS="--limit ${SMOKE_LIMIT}"
  EPOCHS="$SMOKE_EPOCHS"
  BATCH_SIZE="$SMOKE_BATCH_SIZE"
  NOISE_DIM="$SMOKE_NOISE_DIM"
  PSEUDO_ARGS=""
elif [ "$MODE" = "full" ]; then
  RUN_DIR="$OUT_DIR"
  LIMIT_ARGS=""
  EPOCHS="$FULL_EPOCHS"
  BATCH_SIZE="$FULL_BATCH_SIZE"
  NOISE_DIM="$FULL_NOISE_DIM"
  PSEUDO_ARGS=""
elif [ "$MODE" = "one" ]; then
  RUN_DIR="${OUT_DIR}_one"
  LIMIT_ARGS=""
  EPOCHS="${ONE_EPOCHS:-1}"
  BATCH_SIZE="${ONE_BATCH_SIZE:-4}"
  NOISE_DIM="${ONE_NOISE_DIM:-64}"
  PSEUDO_LABEL="${PSEUDO_LABEL:-0}"
  PSEUDO_ARGS="--pseudo-labels ${PSEUDO_LABEL}"
else
  echo "Usage: sh TOOB_Simulation/run_exp.sh [smoke|one|full]"
  echo ""
  echo "Environment overrides:"
  echo "  PYTHON_BIN DATASET MAPPING DF_BUILDER DF_CHECKPOINT OUT_DIR"
  echo "  FULL_EPOCHS FULL_BATCH_SIZE FULL_NOISE_DIM"
  echo "  SMOKE_LIMIT SMOKE_EPOCHS SMOKE_BATCH_SIZE SMOKE_NOISE_DIM"
  echo "  PSEUDO_LABEL for mode 'one'"
  exit 2
fi

BURST_NPZ="${RUN_DIR}/burst_dataset.npz"
PSEUDO_NPZ="${RUN_DIR}/pseudo_labels.npz"
PSEUDO_JSON="${RUN_DIR}/pseudo_labels.json"
GENERATOR_DIR="${RUN_DIR}/generators"
ADV_DIRECTION_NPZ="${RUN_DIR}/toob_adv_direction.npz"

mkdir -p "$RUN_DIR"

echo "[1/4] Direction sequence -> burst dataset"
"$PYTHON_EXE" TOOB_Simulation/EXP/01_make_burst_dataset.py \
  --input "$DATASET" \
  --output "$BURST_NPZ" \
  --data-key "$DATA_KEY" \
  --labels-key "$LABELS_KEY" \
  --max-bursts "$MAX_BURSTS" \
  $LIMIT_ARGS

echo "[2/4] Palette mapping -> pseudo labels"
"$PYTHON_EXE" TOOB_Simulation/EXP/02_make_pseudo_labels.py \
  --labels-npz "$BURST_NPZ" \
  --mapping "$MAPPING" \
  --output "$PSEUDO_NPZ" \
  --json-output "$PSEUDO_JSON" \
  --drop-unmapped

echo "[3/4] Train cluster-wise burst generators"
"$PYTHON_EXE" TOOB_Simulation/EXP/03_train_generators.py \
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
  $PSEUDO_ARGS

echo "[4/4] Export defended direction dataset"
"$PYTHON_EXE" TOOB_Simulation/EXP/04_generate_dataset.py \
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
