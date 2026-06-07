#!/usr/bin/env bash
set -euo pipefail

# Pipeline mode: smoke for quick checks, one for one pseudo-label, full for full-scale training.
MODE="${1:-smoke}"

# Directory containing this runner script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repository root; all relative paths below are resolved from here.
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/env_threads.sh"

# Python executable used for every step.
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Backward-compatible default dataset path used when TRAIN_DATASET is not set.
DATASET="${DATASET:-TOOB_Simulation/data/raw/train.npz}"
# Key for direction traces in DATASET.
DATA_KEY="${DATA_KEY:-X}"
# Key for labels in DATASET.
LABELS_KEY="${LABELS_KEY:-y}"
# Training dataset used for clustering and generator training.
TRAIN_DATASET="${TRAIN_DATASET:-$DATASET}"
# Key for direction traces in TRAIN_DATASET.
TRAIN_DATA_KEY="${TRAIN_DATA_KEY:-$DATA_KEY}"
# Key for labels in TRAIN_DATASET.
TRAIN_LABELS_KEY="${TRAIN_LABELS_KEY:-$LABELS_KEY}"
# Optional validation dataset used for final defended-dataset evaluation.
VALID_DATASET="${VALID_DATASET:-TOOB_Simulation/data/raw/valid.npz}"
# Key for direction traces in VALID_DATASET.
VALID_DATA_KEY="${VALID_DATA_KEY:-$DATA_KEY}"
# Key for labels in VALID_DATASET.
VALID_LABELS_KEY="${VALID_LABELS_KEY:-$LABELS_KEY}"
# Optional validation sample limit; empty means use the full validation set.
VALID_LIMIT="${VALID_LIMIT:-}"
# Optional precomputed website-to-pseudo-label mapping JSON/NPY.
MAPPING="${MAPPING:-}"
# Reuse existing burst/pseudo-label intermediate files when they already exist.
REUSE_INTERMEDIATES="${REUSE_INTERMEDIATES:-1}"
# CW DF detector model builder in file.py:function or module:function format.
DF_BUILDER="${DF_BUILDER:-TOOB_Simulation/toob/wflib_df.py:DF}"
# CW DF checkpoint attacked during generator training and used for evaluation.
DF_CHECKPOINT="${DF_CHECKPOINT:-TOOB_Simulation/checkpoints/df_cw/max_f1.pth}"

# Base output directory; smoke and one modes append suffixes.
OUT_DIR="${OUT_DIR:-TOOB_Simulation/outputs}"
# Maximum number of bursts kept per trace after direction-to-burst conversion.
MAX_BURSTS="${MAX_BURSTS:-2000}"
# Maximum direction sequence length exported and evaluated by DF.
TRACE_LEN="${TRACE_LEN:-5000}"
# Number of detector output classes. CW DF has labels 0..94.
NUM_CLASSES="${NUM_CLASSES:-95}"
# Soft projection chunk size; lower values use less GPU memory.
PROJECTION_CHUNK_SIZE="${PROJECTION_CHUNK_SIZE:-64}"
# Burst-to-direction projection used while training: ste uses hard forward values and soft gradients.
PROJECTION_MODE="${PROJECTION_MODE:-ste}"
# Soft burst-to-direction sharpness; lower is closer to hard expansion but can be less stable.
SOFT_PROJECTION_TAU="${SOFT_PROJECTION_TAU:-1.5}"
# Target number of original website labels per pseudo-label cluster.
SET_SIZE="${SET_SIZE:-30}"
# Number of clustering rounds; TOOB currently expects 1 because each site maps to one pseudo-label.
CLUSTER_ROUNDS="${CLUSTER_ROUNDS:-1}"
# Burst profile used for website clustering: super, mean_abs, or mean_signed.
PROFILE_METHOD="${PROFILE_METHOD:-super}"
# Labels excluded before clustering; 95 is usually the open-world label.
EXCLUDE_LABELS="${EXCLUDE_LABELS:-95}"
# Labels excluded before detector evaluation; default mirrors clustering exclusions.
EVAL_EXCLUDE_LABELS="${EVAL_EXCLUDE_LABELS:-$EXCLUDE_LABELS}"
# Whether to run detector evaluation after exporting the defended dataset.
RUN_EVAL="${RUN_EVAL:-1}"
# Metrics reported by the evaluator.
EVAL_METRICS="${EVAL_METRICS:-accuracy precision recall f1}"
# Averaging mode for precision, recall, and F1.
EVAL_AVERAGE="${EVAL_AVERAGE:-macro}"

# Generator optimizer learning rate.
LR="${LR:-1e-4}"
# Per-sample overhead budget where hinge penalty starts; raise this if you can spend more bandwidth.
OVERHEAD_THRESHOLD="${OVERHEAD_THRESHOLD:-0.22}"
# Weight of the overhead penalty; lower values let the attack spend bandwidth more freely.
LAMBDA_OVERHEAD="${LAMBDA_OVERHEAD:-1.0}"
# Bandwidth loss mode: hinge keeps overhead under a cap; target_l1/target_l2 fit a target budget.
OVERHEAD_LOSS="${OVERHEAD_LOSS:-hinge}"
# Allowed +/- band around OVERHEAD_THRESHOLD when OVERHEAD_LOSS=band.
OVERHEAD_TOLERANCE="${OVERHEAD_TOLERANCE:-0.02}"
# Weight of total-variation smoothing on generated burst padding.
LAMBDA_TV="${LAMBDA_TV:-0.001}"
# Generator attack objective: true_prob, true_logit, or negative_ce.
ATTACK_LOSS="${ATTACK_LOSS:-true_prob}"

# Smoke-mode sample limit.
SMOKE_LIMIT="${SMOKE_LIMIT:-200}"
# Smoke-mode training epochs.
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
# Smoke-mode batch size.
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-4}"
# Smoke-mode generator input noise dimension.
SMOKE_NOISE_DIM="${SMOKE_NOISE_DIM:-64}"

# Full-mode training epochs.
FULL_EPOCHS="${FULL_EPOCHS:-30}"
# Full-mode batch size.
FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-32}"
# Full-mode generator input noise dimension.
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
  # Smoke outputs are isolated from full outputs.
  RUN_DIR="${OUT_DIR}_smoke"
  # Smoke mode only keeps a small prefix of the training dataset.
  LIMIT_ARGS=(--limit "$SMOKE_LIMIT")
  if [ -n "$VALID_DATASET" ] && [ -z "$VALID_LIMIT" ]; then
    # Match validation limit to smoke limit unless explicitly overridden.
    VALID_LIMIT="$SMOKE_LIMIT"
  fi
  # Training epochs used in smoke mode.
  EPOCHS="$SMOKE_EPOCHS"
  # Batch size used in smoke mode.
  BATCH_SIZE="$SMOKE_BATCH_SIZE"
  # Generator noise dimension used in smoke mode.
  NOISE_DIM="$SMOKE_NOISE_DIM"
  # Smoke mode trains all pseudo labels present in the limited data.
  PSEUDO_ARGS=()
elif [ "$MODE" = "full" ]; then
  # Full outputs are written directly under OUT_DIR.
  RUN_DIR="$OUT_DIR"
  # Full mode uses all training samples.
  LIMIT_ARGS=()
  # Training epochs used in full mode.
  EPOCHS="$FULL_EPOCHS"
  # Batch size used in full mode.
  BATCH_SIZE="$FULL_BATCH_SIZE"
  # Generator noise dimension used in full mode.
  NOISE_DIM="$FULL_NOISE_DIM"
  # Full mode trains all pseudo labels.
  PSEUDO_ARGS=()
elif [ "$MODE" = "one" ]; then
  # One-label outputs are isolated from smoke and full outputs.
  RUN_DIR="${OUT_DIR}_one"
  # One mode uses all samples for the selected pseudo label.
  LIMIT_ARGS=()
  # One-mode training epochs.
  EPOCHS="${ONE_EPOCHS:-1}"
  # One-mode batch size.
  BATCH_SIZE="${ONE_BATCH_SIZE:-4}"
  # One-mode generator noise dimension.
  NOISE_DIM="${ONE_NOISE_DIM:-64}"
  # Pseudo label trained in one mode.
  PSEUDO_LABEL="${PSEUDO_LABEL:-0}"
  # Restrict generator training to the selected pseudo label.
  PSEUDO_ARGS=(--pseudo-labels "$PSEUDO_LABEL")
else
  echo "Usage: bash TOOB_Simulation/run_exp_ubuntu.sh [smoke|one|full]"
  echo ""
  echo "Environment overrides:"
  echo "  PYTHON_BIN DATASET TRAIN_DATASET VALID_DATASET DF_BUILDER DF_CHECKPOINT OUT_DIR"
  echo "  DATA_KEY LABELS_KEY TRAIN_DATA_KEY TRAIN_LABELS_KEY VALID_DATA_KEY VALID_LABELS_KEY VALID_LIMIT"
  echo "  BURST_NPZ PSEUDO_NPZ PSEUDO_JSON VALID_BURST_NPZ VALID_PSEUDO_NPZ VALID_PSEUDO_JSON REUSE_INTERMEDIATES"
  echo "  SET_SIZE CLUSTER_ROUNDS PROFILE_METHOD EXCLUDE_LABELS"
  echo "  LR OVERHEAD_THRESHOLD LAMBDA_OVERHEAD OVERHEAD_LOSS OVERHEAD_TOLERANCE LAMBDA_TV ATTACK_LOSS"
  echo "  PROJECTION_MODE SOFT_PROJECTION_TAU PROJECTION_CHUNK_SIZE"
  echo "  RUN_EVAL EVAL_METRICS EVAL_AVERAGE"
  echo "  FULL_EPOCHS FULL_BATCH_SIZE FULL_NOISE_DIM"
  echo "  SMOKE_LIMIT SMOKE_EPOCHS SMOKE_BATCH_SIZE SMOKE_NOISE_DIM"
  echo "  PSEUDO_LABEL for mode 'one'"
  exit 2
fi

VALID_LIMIT_ARGS=()
if [ -n "$VALID_LIMIT" ]; then
  # Optional validation limit passed only when VALID_LIMIT is non-empty.
  VALID_LIMIT_ARGS=(--limit "$VALID_LIMIT")
fi

require_file "$TRAIN_DATASET"
if [ -n "$VALID_DATASET" ]; then
  require_file "$VALID_DATASET"
fi
if [ -n "$MAPPING" ]; then
  require_file "$MAPPING"
  # Reuse an existing website-to-pseudo-label mapping instead of reclustering.
  MAPPING_ARGS=(--mapping "$MAPPING")
else
  # Empty mapping args means Step 2 will cluster from the training burst dataset.
  MAPPING_ARGS=()
fi
require_file "${DF_BUILDER%%:*}"
require_file "$DF_CHECKPOINT"
require_python_deps

# Train burst dataset path produced by Step 1.
BURST_NPZ="${BURST_NPZ:-${RUN_DIR}/burst_dataset.npz}"
# Train pseudo-label NPZ produced by Step 2.
PSEUDO_NPZ="${PSEUDO_NPZ:-${RUN_DIR}/pseudo_labels.npz}"
# Human-readable train pseudo-label mapping produced by Step 2.
PSEUDO_JSON="${PSEUDO_JSON:-${RUN_DIR}/pseudo_labels.json}"
# Validation burst dataset path produced when VALID_DATASET is set.
VALID_BURST_NPZ="${VALID_BURST_NPZ:-${RUN_DIR}/valid_burst_dataset.npz}"
# Validation pseudo-label NPZ produced when VALID_DATASET is set.
VALID_PSEUDO_NPZ="${VALID_PSEUDO_NPZ:-${RUN_DIR}/valid_pseudo_labels.npz}"
# Human-readable validation pseudo-label mapping produced when VALID_DATASET is set.
VALID_PSEUDO_JSON="${VALID_PSEUDO_JSON:-${RUN_DIR}/valid_pseudo_labels.json}"
# Directory containing generator_pseudo_*.pt checkpoints.
GENERATOR_DIR="${RUN_DIR}/generators"
if [ -n "$VALID_DATASET" ]; then
  # Defended validation direction dataset.
  ADV_DIRECTION_NPZ="${RUN_DIR}/toob_valid_adv_direction.npz"
else
  # Defended training/input direction dataset when no validation dataset is set.
  ADV_DIRECTION_NPZ="${RUN_DIR}/toob_adv_direction.npz"
fi
# Evaluation JSON report path.
EVAL_JSON="${RUN_DIR}/defense_eval_metrics.json"

mkdir -p "$RUN_DIR"

echo "[0/5] Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "[0/5] TOOB imports"
"$PYTHON_BIN" TOOB_Simulation/EXP/00_check_imports.py

echo "[1/5] Train direction sequence -> burst dataset"
if [ "$REUSE_INTERMEDIATES" = "1" ] && [ -f "$BURST_NPZ" ]; then
  echo "reuse: $BURST_NPZ"
else
  "$PYTHON_BIN" TOOB_Simulation/EXP/01_make_burst_dataset.py \
    --input "$TRAIN_DATASET" \
    --output "$BURST_NPZ" \
    --data-key "$TRAIN_DATA_KEY" \
    --labels-key "$TRAIN_LABELS_KEY" \
    --max-bursts "$MAX_BURSTS" \
    "${LIMIT_ARGS[@]}"
fi

EXCLUDE_ARGS=()
if [ -n "$EXCLUDE_LABELS" ]; then
  # shellcheck disable=SC2206
  # Split EXCLUDE_LABELS into separate --exclude-labels values.
  EXCLUDE_ARGS=(--exclude-labels $EXCLUDE_LABELS)
fi

echo "[2/5] Cluster burst profiles -> pseudo labels"
if [ "$REUSE_INTERMEDIATES" = "1" ] && [ -f "$PSEUDO_NPZ" ] && [ -f "$PSEUDO_JSON" ]; then
  echo "reuse: $PSEUDO_NPZ"
  echo "reuse: $PSEUDO_JSON"
else
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
fi

echo "[3/5] Train cluster-wise burst generators"
"$PYTHON_BIN" TOOB_Simulation/EXP/03_train_generators.py \
  --burst-npz "$BURST_NPZ" \
  --pseudo-npz "$PSEUDO_NPZ" \
  --detector-builder "$DF_BUILDER" \
  --detector-checkpoint "$DF_CHECKPOINT" \
  --num-classes "$NUM_CLASSES" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --noise-dim "$NOISE_DIM" \
  --overhead-threshold "$OVERHEAD_THRESHOLD" \
  --lambda-overhead "$LAMBDA_OVERHEAD" \
  --overhead-loss "$OVERHEAD_LOSS" \
  --overhead-tolerance "$OVERHEAD_TOLERANCE" \
  --lambda-tv "$LAMBDA_TV" \
  --attack-loss "$ATTACK_LOSS" \
  --detector-input-kind direction \
  --detector-input-layout ncl \
  --detector-input-length "$TRACE_LEN" \
  --projection-mode "$PROJECTION_MODE" \
  --soft-projection-tau "$SOFT_PROJECTION_TAU" \
  --projection-chunk-size "$PROJECTION_CHUNK_SIZE" \
  --output-dir "$GENERATOR_DIR" \
  "${PSEUDO_ARGS[@]}"

DEFENSE_BURST_NPZ="$BURST_NPZ"
DEFENSE_PSEUDO_NPZ="$PSEUDO_NPZ"
if [ -n "$VALID_DATASET" ]; then
  echo "[4/5] Valid direction sequence -> burst dataset"
  if [ "$REUSE_INTERMEDIATES" = "1" ] && [ -f "$VALID_BURST_NPZ" ]; then
    echo "reuse: $VALID_BURST_NPZ"
  else
    "$PYTHON_BIN" TOOB_Simulation/EXP/01_make_burst_dataset.py \
      --input "$VALID_DATASET" \
      --output "$VALID_BURST_NPZ" \
      --data-key "$VALID_DATA_KEY" \
      --labels-key "$VALID_LABELS_KEY" \
      --max-bursts "$MAX_BURSTS" \
      "${VALID_LIMIT_ARGS[@]}"
  fi

  echo "[4/5] Map valid labels with train pseudo-label mapping"
  if [ "$REUSE_INTERMEDIATES" = "1" ] && [ -f "$VALID_PSEUDO_NPZ" ] && [ -f "$VALID_PSEUDO_JSON" ]; then
    echo "reuse: $VALID_PSEUDO_NPZ"
    echo "reuse: $VALID_PSEUDO_JSON"
  else
    "$PYTHON_BIN" TOOB_Simulation/EXP/02_make_pseudo_labels.py \
      --labels-npz "$VALID_BURST_NPZ" \
      --mapping "$PSEUDO_JSON" \
      --output "$VALID_PSEUDO_NPZ" \
      --json-output "$VALID_PSEUDO_JSON" \
      --drop-unmapped
  fi

  DEFENSE_BURST_NPZ="$VALID_BURST_NPZ"
  DEFENSE_PSEUDO_NPZ="$VALID_PSEUDO_NPZ"
fi

echo "[4/5] Export defended direction dataset"
"$PYTHON_BIN" TOOB_Simulation/EXP/04_generate_dataset.py \
  --burst-npz "$DEFENSE_BURST_NPZ" \
  --pseudo-npz "$DEFENSE_PSEUDO_NPZ" \
  --generator-dir "$GENERATOR_DIR" \
  --output "$ADV_DIRECTION_NPZ" \
  --output-kind direction \
  --max-trace-len "$TRACE_LEN" \
  --batch-size 256 \
  --round

if [ "$RUN_EVAL" = "1" ]; then
  # shellcheck disable=SC2206
  # Split EVAL_METRICS into separate metric arguments.
  EVAL_METRIC_ARGS=($EVAL_METRICS)
  EVAL_EXCLUDE_ARGS=()
  if [ -n "$EVAL_EXCLUDE_LABELS" ]; then
    # shellcheck disable=SC2206
    # Split EVAL_EXCLUDE_LABELS into separate --exclude-labels values.
    EVAL_EXCLUDE_ARGS=(--exclude-labels $EVAL_EXCLUDE_LABELS)
  fi
  echo "[5/5] Evaluate defended dataset"
  "$PYTHON_BIN" TOOB_Simulation/EXP/05_evaluate_defense.py \
    --input-npz "$ADV_DIRECTION_NPZ" \
    --input-kind direction \
    --data-key data \
    --labels-key labels \
    "${EVAL_EXCLUDE_ARGS[@]}" \
    --detector-builder "$DF_BUILDER" \
    --detector-checkpoint "$DF_CHECKPOINT" \
    --num-classes "$NUM_CLASSES" \
    --detector-input-kind direction \
    --detector-input-layout ncl \
    --max-trace-len "$TRACE_LEN" \
    --metrics "${EVAL_METRIC_ARGS[@]}" \
    --average "$EVAL_AVERAGE" \
    --output-json "$EVAL_JSON"
fi

echo "Done."
echo "Output dataset: $ADV_DIRECTION_NPZ"
echo "Pseudo-label JSON: $PSEUDO_JSON"
if [ -n "$VALID_DATASET" ]; then
  echo "Valid pseudo-label JSON: $VALID_PSEUDO_JSON"
fi
if [ "$RUN_EVAL" = "1" ]; then
  echo "Evaluation metrics: $EVAL_JSON"
fi
