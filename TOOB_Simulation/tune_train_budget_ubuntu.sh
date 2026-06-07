#!/usr/bin/env bash
set -euo pipefail

# Train-time bandwidth tuning for TOOB.
# This script retrains generators for each loss-parameter combination, evaluates
# every run, and summarizes the bandwidth-accuracy tradeoff for paper tables.

# Directory containing this runner script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repository root; all relative paths below are resolved from here.
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/env_threads.sh"

# Python executable used for config writing and result collection.
PYTHON_BIN="${PYTHON_BIN:-python3}"
# Runner used for each candidate training run.
RUNNER="${RUNNER:-TOOB_Simulation/run_exp_ubuntu.sh}"
# run_exp_ubuntu.sh mode used per candidate: smoke, one, or full.
if [ "$#" -gt 0 ]; then
  TUNE_MODE="$1"
else
  TUNE_MODE="${TUNE_MODE:-full}"
fi
# Directory containing one subdirectory per candidate run.
TUNE_DIR="${TUNE_DIR:-TOOB_Simulation/outputs_train_tuning}"

# Target training overhead budgets. 0.20 means around 20% extra packets.
OVERHEAD_TARGETS="${OVERHEAD_TARGETS:-0.10 0.20 0.30}"
# Candidate penalty weights. Higher values enforce the budget more strongly.
LAMBDA_OVERHEADS="${LAMBDA_OVERHEADS:-5.0 10.0 20.0 50.0}"
# Training-time overhead loss modes: hinge, target_l1, target_l2, or band.
OVERHEAD_LOSSES="${OVERHEAD_LOSSES:-target_l2 band}"
# Allowed +/- band when OVERHEAD_LOSS=band.
OVERHEAD_TOLERANCE="${OVERHEAD_TOLERANCE:-0.02}"
# Candidate attack objectives: true_prob, true_logit, or negative_ce.
ATTACK_LOSSES="${ATTACK_LOSSES:-true_prob}"
# Weight of total-variation smoothing on generated burst padding.
LAMBDA_TV="${LAMBDA_TV:-0.001}"
# Burst-to-direction projection used while training: soft or ste.
PROJECTION_MODE="${PROJECTION_MODE:-ste}"
# Soft burst-to-direction sharpness used during training.
SOFT_PROJECTION_TAU="${SOFT_PROJECTION_TAU:-1.5}"
# Generator optimizer learning rate.
LR="${LR:-1e-4}"
# Target number of original website labels per pseudo-label cluster.
SET_SIZE="${SET_SIZE:-30}"
# Burst profile used for website clustering.
PROFILE_METHOD="${PROFILE_METHOD:-super}"
# Labels excluded before clustering.
EXCLUDE_LABELS="${EXCLUDE_LABELS:-95}"
# CW DF detector model builder used by each candidate run.
DF_BUILDER="${DF_BUILDER:-TOOB_Simulation/toob/wflib_df.py:DF}"
# CW DF checkpoint used by each candidate run.
DF_CHECKPOINT="${DF_CHECKPOINT:-TOOB_Simulation/checkpoints/df_cw/max_f1.pth}"
# CW DF output classes: labels 0..94.
NUM_CLASSES="${NUM_CLASSES:-95}"
# Labels excluded before detector evaluation; default mirrors clustering exclusions.
EVAL_EXCLUDE_LABELS="${EVAL_EXCLUDE_LABELS:-$EXCLUDE_LABELS}"
# Optional precomputed website-to-pseudo-label mapping JSON/NPY.
MAPPING="${MAPPING:-}"
# Optional fixed pseudo-label cluster count used for reporting when MAPPING is set.
CLUSTER_COUNT="${CLUSTER_COUNT:-}"
# Shared directory for burst datasets and pseudo labels reused across candidates.
if [ -n "$MAPPING" ]; then
  mapping_name="$(basename "$MAPPING")"
  mapping_name="${mapping_name%.*}"
  CLUSTER_CACHE_DIR="${CLUSTER_CACHE_DIR:-${TUNE_DIR}/cluster_cache_mapping_${mapping_name}}"
else
  CLUSTER_CACHE_DIR="${CLUSTER_CACHE_DIR:-${TUNE_DIR}/cluster_cache_set${SET_SIZE}_${PROFILE_METHOD}_exclude${EXCLUDE_LABELS// /_}}"
fi

# Full-mode defaults forwarded to run_exp_ubuntu.sh unless already overridden.
FULL_EPOCHS="${FULL_EPOCHS:-30}"
FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-32}"
FULL_NOISE_DIM="${FULL_NOISE_DIM:-256}"
# Smoke-mode defaults forwarded when TUNE_MODE=smoke.
SMOKE_LIMIT="${SMOKE_LIMIT:-200}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-4}"
SMOKE_NOISE_DIM="${SMOKE_NOISE_DIM:-64}"
# One-mode defaults forwarded when TUNE_MODE=one.
ONE_EPOCHS="${ONE_EPOCHS:-1}"
ONE_BATCH_SIZE="${ONE_BATCH_SIZE:-4}"
ONE_NOISE_DIM="${ONE_NOISE_DIM:-64}"
PSEUDO_LABEL="${PSEUDO_LABEL:-0}"

# Skip a candidate when its defense_eval_metrics.json already exists.
SKIP_EXISTING="${SKIP_EXISTING:-1}"
# Allowed overhead overshoot when selecting best_by_target rows.
BUDGET_SLACK="${BUDGET_SLACK:-0.02}"
# Metric minimized when choosing the best row under each target budget.
SELECT_METRIC="${SELECT_METRIC:-accuracy}"

tag_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/_}"
  echo "$value"
}

mode_epochs() {
  case "$TUNE_MODE" in
    full) echo "$FULL_EPOCHS" ;;
    smoke) echo "$SMOKE_EPOCHS" ;;
    one) echo "$ONE_EPOCHS" ;;
    *) echo "" ;;
  esac
}

mode_batch_size() {
  case "$TUNE_MODE" in
    full) echo "$FULL_BATCH_SIZE" ;;
    smoke) echo "$SMOKE_BATCH_SIZE" ;;
    one) echo "$ONE_BATCH_SIZE" ;;
    *) echo "" ;;
  esac
}

mode_noise_dim() {
  case "$TUNE_MODE" in
    full) echo "$FULL_NOISE_DIM" ;;
    smoke) echo "$SMOKE_NOISE_DIM" ;;
    one) echo "$ONE_NOISE_DIM" ;;
    *) echo "" ;;
  esac
}

mkdir -p "$TUNE_DIR"
mkdir -p "$CLUSTER_CACHE_DIR"

CACHE_BURST_NPZ="${BURST_NPZ:-${CLUSTER_CACHE_DIR}/burst_dataset.npz}"
CACHE_PSEUDO_NPZ="${PSEUDO_NPZ:-${CLUSTER_CACHE_DIR}/pseudo_labels.npz}"
CACHE_PSEUDO_JSON="${PSEUDO_JSON:-${CLUSTER_CACHE_DIR}/pseudo_labels.json}"
CACHE_VALID_BURST_NPZ="${VALID_BURST_NPZ:-${CLUSTER_CACHE_DIR}/valid_burst_dataset.npz}"
CACHE_VALID_PSEUDO_NPZ="${VALID_PSEUDO_NPZ:-${CLUSTER_CACHE_DIR}/valid_pseudo_labels.npz}"
CACHE_VALID_PSEUDO_JSON="${VALID_PSEUDO_JSON:-${CLUSTER_CACHE_DIR}/valid_pseudo_labels.json}"

echo "Train-time tuning"
echo "  mode: $TUNE_MODE"
echo "  tune dir: $TUNE_DIR"
echo "  cluster cache dir: $CLUSTER_CACHE_DIR"
echo "  burst npz: $CACHE_BURST_NPZ"
if [ -n "$MAPPING" ]; then
  echo "  mapping: $MAPPING"
fi
if [ -n "$CLUSTER_COUNT" ]; then
  echo "  cluster count: $CLUSTER_COUNT"
fi
echo "  targets: $OVERHEAD_TARGETS"
echo "  lambdas: $LAMBDA_OVERHEADS"
echo "  overhead losses: $OVERHEAD_LOSSES"
echo "  attack losses: $ATTACK_LOSSES"
echo "  lambda tv: $LAMBDA_TV"
echo "  projection mode: $PROJECTION_MODE"
echo "  lr: $LR"
echo "  detector builder: $DF_BUILDER"
echo "  detector checkpoint: $DF_CHECKPOINT"
echo "  num classes: $NUM_CLASSES"
echo "  eval exclude labels: $EVAL_EXCLUDE_LABELS"

run_epochs="$(mode_epochs)"
run_batch_size="$(mode_batch_size)"
run_noise_dim="$(mode_noise_dim)"
CONFIG_CLUSTER_ARGS=()
if [ -n "$CLUSTER_COUNT" ]; then
  CONFIG_CLUSTER_ARGS=(--cluster-count "$CLUSTER_COUNT")
fi
CONFIG_MAPPING_ARGS=()
if [ -n "$MAPPING" ]; then
  CONFIG_MAPPING_ARGS=(--mapping "$MAPPING")
fi

for target in $OVERHEAD_TARGETS; do
  target_tag="$(tag_value "$target")"
  for lambda_overhead in $LAMBDA_OVERHEADS; do
    lambda_tag="$(tag_value "$lambda_overhead")"
    for overhead_loss in $OVERHEAD_LOSSES; do
      for attack_loss in $ATTACK_LOSSES; do
        run_name="run_t${target_tag}_l${lambda_tag}_${overhead_loss}_${attack_loss}"
        run_dir="${TUNE_DIR}/${run_name}"
        metrics_json="${run_dir}/defense_eval_metrics.json"
        if [ "$SKIP_EXISTING" = "1" ] && [ -f "$metrics_json" ]; then
          echo "[skip] $run_name"
          continue
        fi

        mkdir -p "$run_dir"
        if [ "${VALID_DATASET:-TOOB_Simulation/data/raw/valid.npz}" = "" ]; then
          output_npz="${run_dir}/toob_adv_direction.npz"
        else
          output_npz="${run_dir}/toob_valid_adv_direction.npz"
        fi

        "$PYTHON_BIN" TOOB_Simulation/EXP/06_collect_tuning.py \
          --write-config "${run_dir}/run_config.json" \
          --run-name "$run_name" \
          --overhead-threshold "$target" \
          --lambda-overhead "$lambda_overhead" \
          --overhead-loss "$overhead_loss" \
          --overhead-tolerance "$OVERHEAD_TOLERANCE" \
          --attack-loss "$attack_loss" \
          --lambda-tv "$LAMBDA_TV" \
          --projection-mode "$PROJECTION_MODE" \
          --soft-projection-tau "$SOFT_PROJECTION_TAU" \
          --lr "$LR" \
          --detector-builder "$DF_BUILDER" \
          --detector-checkpoint "$DF_CHECKPOINT" \
          --num-classes "$NUM_CLASSES" \
          --eval-exclude-labels "$EVAL_EXCLUDE_LABELS" \
          --set-size "$SET_SIZE" \
          "${CONFIG_CLUSTER_ARGS[@]}" \
          "${CONFIG_MAPPING_ARGS[@]}" \
          --epochs "$run_epochs" \
          --batch-size "$run_batch_size" \
          --noise-dim "$run_noise_dim" \
          --output-npz "$output_npz" \
          --metrics-json "$metrics_json"

        echo "[run] $run_name"
        OUT_DIR="$run_dir" \
        BURST_NPZ="$CACHE_BURST_NPZ" \
        PSEUDO_NPZ="$CACHE_PSEUDO_NPZ" \
        PSEUDO_JSON="$CACHE_PSEUDO_JSON" \
        VALID_BURST_NPZ="$CACHE_VALID_BURST_NPZ" \
        VALID_PSEUDO_NPZ="$CACHE_VALID_PSEUDO_NPZ" \
        VALID_PSEUDO_JSON="$CACHE_VALID_PSEUDO_JSON" \
        REUSE_INTERMEDIATES=1 \
        OVERHEAD_THRESHOLD="$target" \
        LAMBDA_OVERHEAD="$lambda_overhead" \
        OVERHEAD_LOSS="$overhead_loss" \
        OVERHEAD_TOLERANCE="$OVERHEAD_TOLERANCE" \
        ATTACK_LOSS="$attack_loss" \
        LAMBDA_TV="$LAMBDA_TV" \
        PROJECTION_MODE="$PROJECTION_MODE" \
        SOFT_PROJECTION_TAU="$SOFT_PROJECTION_TAU" \
        LR="$LR" \
        SET_SIZE="$SET_SIZE" \
        PROFILE_METHOD="$PROFILE_METHOD" \
        EXCLUDE_LABELS="$EXCLUDE_LABELS" \
        EVAL_EXCLUDE_LABELS="$EVAL_EXCLUDE_LABELS" \
        DF_BUILDER="$DF_BUILDER" \
        DF_CHECKPOINT="$DF_CHECKPOINT" \
        NUM_CLASSES="$NUM_CLASSES" \
        MAPPING="$MAPPING" \
        FULL_EPOCHS="$FULL_EPOCHS" \
        FULL_BATCH_SIZE="$FULL_BATCH_SIZE" \
        FULL_NOISE_DIM="$FULL_NOISE_DIM" \
        SMOKE_LIMIT="$SMOKE_LIMIT" \
        SMOKE_EPOCHS="$SMOKE_EPOCHS" \
        SMOKE_BATCH_SIZE="$SMOKE_BATCH_SIZE" \
        SMOKE_NOISE_DIM="$SMOKE_NOISE_DIM" \
        ONE_EPOCHS="$ONE_EPOCHS" \
        ONE_BATCH_SIZE="$ONE_BATCH_SIZE" \
        ONE_NOISE_DIM="$ONE_NOISE_DIM" \
        PSEUDO_LABEL="$PSEUDO_LABEL" \
        bash "$RUNNER" "$TUNE_MODE"
      done
    done
  done
done

"$PYTHON_BIN" TOOB_Simulation/EXP/06_collect_tuning.py \
  --tune-dir "$TUNE_DIR" \
  --output-csv "${TUNE_DIR}/summary.csv" \
  --output-json "${TUNE_DIR}/summary.json" \
  --best-csv "${TUNE_DIR}/best_by_target.csv" \
  --best-json "${TUNE_DIR}/best_by_target.json" \
  --budget-slack "$BUDGET_SLACK" \
  --select-metric "$SELECT_METRIC"

echo "Done."
echo "All runs: ${TUNE_DIR}/summary.csv"
echo "Best by target: ${TUNE_DIR}/best_by_target.csv"
