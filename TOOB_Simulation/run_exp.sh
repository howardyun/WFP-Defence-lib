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
TRAIN_DATASET="${TRAIN_DATASET:-$DATASET}"
TRAIN_DATA_KEY="${TRAIN_DATA_KEY:-$DATA_KEY}"
TRAIN_LABELS_KEY="${TRAIN_LABELS_KEY:-$LABELS_KEY}"
VALID_DATASET="${VALID_DATASET:-}"
VALID_DATA_KEY="${VALID_DATA_KEY:-$DATA_KEY}"
VALID_LABELS_KEY="${VALID_LABELS_KEY:-$LABELS_KEY}"
VALID_LIMIT="${VALID_LIMIT:-}"
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
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_METRICS="${EVAL_METRICS:-accuracy precision recall f1}"
EVAL_AVERAGE="${EVAL_AVERAGE:-macro}"

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
  if [ -n "$VALID_DATASET" ] && [ -z "$VALID_LIMIT" ]; then
    VALID_LIMIT="$SMOKE_LIMIT"
  fi
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
  echo "  PYTHON_BIN DATASET TRAIN_DATASET VALID_DATASET DF_BUILDER DF_CHECKPOINT OUT_DIR"
  echo "  DATA_KEY LABELS_KEY TRAIN_DATA_KEY TRAIN_LABELS_KEY VALID_DATA_KEY VALID_LABELS_KEY VALID_LIMIT"
  echo "  SET_SIZE CLUSTER_ROUNDS PROFILE_METHOD EXCLUDE_LABELS"
  echo "  RUN_EVAL EVAL_METRICS EVAL_AVERAGE"
  echo "  FULL_EPOCHS FULL_BATCH_SIZE FULL_NOISE_DIM"
  echo "  SMOKE_LIMIT SMOKE_EPOCHS SMOKE_BATCH_SIZE SMOKE_NOISE_DIM"
  echo "  PSEUDO_LABEL for mode 'one'"
  exit 2
fi

VALID_LIMIT_ARGS=""
if [ -n "$VALID_LIMIT" ]; then
  VALID_LIMIT_ARGS="--limit ${VALID_LIMIT}"
fi

BURST_NPZ="${RUN_DIR}/burst_dataset.npz"
PSEUDO_NPZ="${RUN_DIR}/pseudo_labels.npz"
PSEUDO_JSON="${RUN_DIR}/pseudo_labels.json"
VALID_BURST_NPZ="${RUN_DIR}/valid_burst_dataset.npz"
VALID_PSEUDO_NPZ="${RUN_DIR}/valid_pseudo_labels.npz"
VALID_PSEUDO_JSON="${RUN_DIR}/valid_pseudo_labels.json"
GENERATOR_DIR="${RUN_DIR}/generators"
if [ -n "$VALID_DATASET" ]; then
  ADV_DIRECTION_NPZ="${RUN_DIR}/toob_valid_adv_direction.npz"
else
  ADV_DIRECTION_NPZ="${RUN_DIR}/toob_adv_direction.npz"
fi
EVAL_JSON="${RUN_DIR}/defense_eval_metrics.json"

mkdir -p "$RUN_DIR"

echo "[0/5] TOOB imports"
"$PYTHON_EXE" TOOB_Simulation/EXP/00_check_imports.py

echo "[1/5] Train direction sequence -> burst dataset"
"$PYTHON_EXE" TOOB_Simulation/EXP/01_make_burst_dataset.py \
  --input "$TRAIN_DATASET" \
  --output "$BURST_NPZ" \
  --data-key "$TRAIN_DATA_KEY" \
  --labels-key "$TRAIN_LABELS_KEY" \
  --max-bursts "$MAX_BURSTS" \
  $LIMIT_ARGS

MAPPING_ARGS=""
if [ -n "$MAPPING" ]; then
  MAPPING_ARGS="--mapping ${MAPPING}"
fi
EXCLUDE_ARGS=""
if [ -n "$EXCLUDE_LABELS" ]; then
  EXCLUDE_ARGS="--exclude-labels ${EXCLUDE_LABELS}"
fi

echo "[2/5] Cluster burst profiles -> pseudo labels"
"$PYTHON_EXE" TOOB_Simulation/EXP/02_make_pseudo_labels.py \
  --labels-npz "$BURST_NPZ" \
  --output "$PSEUDO_NPZ" \
  --json-output "$PSEUDO_JSON" \
  --set-size "$SET_SIZE" \
  --rounds "$CLUSTER_ROUNDS" \
  --profile-method "$PROFILE_METHOD" \
  --drop-unmapped \
  $MAPPING_ARGS \
  $EXCLUDE_ARGS

echo "[3/5] Train cluster-wise burst generators"
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

DEFENSE_BURST_NPZ="$BURST_NPZ"
DEFENSE_PSEUDO_NPZ="$PSEUDO_NPZ"
if [ -n "$VALID_DATASET" ]; then
  echo "[4/5] Valid direction sequence -> burst dataset"
  "$PYTHON_EXE" TOOB_Simulation/EXP/01_make_burst_dataset.py \
    --input "$VALID_DATASET" \
    --output "$VALID_BURST_NPZ" \
    --data-key "$VALID_DATA_KEY" \
    --labels-key "$VALID_LABELS_KEY" \
    --max-bursts "$MAX_BURSTS" \
    $VALID_LIMIT_ARGS

  echo "[4/5] Map valid labels with train pseudo-label mapping"
  "$PYTHON_EXE" TOOB_Simulation/EXP/02_make_pseudo_labels.py \
    --labels-npz "$VALID_BURST_NPZ" \
    --mapping "$PSEUDO_JSON" \
    --output "$VALID_PSEUDO_NPZ" \
    --json-output "$VALID_PSEUDO_JSON" \
    --drop-unmapped

  DEFENSE_BURST_NPZ="$VALID_BURST_NPZ"
  DEFENSE_PSEUDO_NPZ="$VALID_PSEUDO_NPZ"
fi

echo "[4/5] Export defended direction dataset"
"$PYTHON_EXE" TOOB_Simulation/EXP/04_generate_dataset.py \
  --burst-npz "$DEFENSE_BURST_NPZ" \
  --pseudo-npz "$DEFENSE_PSEUDO_NPZ" \
  --generator-dir "$GENERATOR_DIR" \
  --output "$ADV_DIRECTION_NPZ" \
  --output-kind direction \
  --max-trace-len "$TRACE_LEN" \
  --batch-size 256 \
  --round

if [ "$RUN_EVAL" = "1" ]; then
  echo "[5/5] Evaluate defended dataset"
  "$PYTHON_EXE" TOOB_Simulation/EXP/05_evaluate_defense.py \
    --input-npz "$ADV_DIRECTION_NPZ" \
    --input-kind direction \
    --data-key data \
    --labels-key labels \
    --detector-builder "$DF_BUILDER" \
    --detector-checkpoint "$DF_CHECKPOINT" \
    --num-classes "$NUM_CLASSES" \
    --detector-input-kind direction \
    --detector-input-layout ncl \
    --max-trace-len "$TRACE_LEN" \
    --metrics $EVAL_METRICS \
    --average "$EVAL_AVERAGE" \
    --output-json "$EVAL_JSON"
fi

echo "Done."
echo "Output dataset: $ADV_DIRECTION_NPZ"
if [ -n "$VALID_DATASET" ]; then
  echo "Valid pseudo-label JSON: $VALID_PSEUDO_JSON"
fi
if [ "$RUN_EVAL" = "1" ]; then
  echo "Evaluation metrics: $EVAL_JSON"
fi
