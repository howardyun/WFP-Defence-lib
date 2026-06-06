#!/usr/bin/env bash
set -euo pipefail

# Fixed-K train-time bandwidth tuning for TOOB.
# This runner fixes the number of pseudo-label clusters first, creates one
# website-to-pseudo-label mapping per K, then delegates training to
# tune_train_budget_ubuntu.sh.

# Directory containing this runner script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repository root; all relative paths below are resolved from here.
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/env_threads.sh"

# Python executable used for burst conversion, K probing, and result collection.
PYTHON_BIN="${PYTHON_BIN:-python3}"
# Runner that performs bandwidth/loss tuning for one fixed mapping.
TUNE_RUNNER="${TUNE_RUNNER:-TOOB_Simulation/tune_train_budget_ubuntu.sh}"
# Runner used internally by TUNE_RUNNER for one candidate.
RUNNER="${RUNNER:-TOOB_Simulation/run_exp_ubuntu.sh}"
# run_exp_ubuntu.sh mode used per candidate: smoke, one, or full.
if [ "$#" -gt 0 ]; then
  TUNE_MODE="$1"
else
  TUNE_MODE="${TUNE_MODE:-full}"
fi
# Parent directory containing one k*/ tuning directory per fixed K.
TUNE_ROOT="${TUNE_ROOT:-TOOB_Simulation/outputs_train_tuning_fixed_k}"
# Fixed pseudo-label cluster counts to evaluate.
CLUSTER_COUNTS="${CLUSTER_COUNTS:-4 5 6}"

# Target training overhead budgets. ALERT uses 0.22; multiple values give a bandwidth curve.
OVERHEAD_TARGETS="${OVERHEAD_TARGETS:-0.10 0.20 0.30 0.40}"
# ALERT-style default: keep attack and overhead terms equally weighted.
LAMBDA_OVERHEADS="${LAMBDA_OVERHEADS:-1.0}"
# ALERT-style overhead loss: max(0, overhead - target).
OVERHEAD_LOSSES="${OVERHEAD_LOSSES:-hinge}"
# Allowed +/- band when OVERHEAD_LOSS=band.
OVERHEAD_TOLERANCE="${OVERHEAD_TOLERANCE:-0.02}"
# ALERT's Keras model outputs softmax probabilities, so true_prob is the closest PyTorch match.
ATTACK_LOSSES="${ATTACK_LOSSES:-true_prob}"
# ALERT does not use an explicit TV smoothing term.
LAMBDA_TV="${LAMBDA_TV:-0.0}"
# Soft burst-to-direction sharpness used during training.
SOFT_PROJECTION_TAU="${SOFT_PROJECTION_TAU:-1.5}"
# Generator optimizer learning rate; ALERT uses Adam(1e-4).
LR="${LR:-1e-4}"
# Full-mode training epochs.
FULL_EPOCHS="${FULL_EPOCHS:-30}"
# Full-mode batch size; ALERT uses 64.
FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-64}"
# Full-mode generator input noise dimension; ALERT uses data_length=2000.
FULL_NOISE_DIM="${FULL_NOISE_DIM:-2000}"
# Allowed overhead overshoot when selecting best_by_target rows.
BUDGET_SLACK="${BUDGET_SLACK:-0.02}"
# Metric minimized when choosing the best row under each target budget.
SELECT_METRIC="${SELECT_METRIC:-accuracy}"

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
# Optional validation dataset forwarded to the normal runner.
VALID_DATASET="${VALID_DATASET:-TOOB_Simulation/data/raw/valid.npz}"
# Key for direction traces in VALID_DATASET.
VALID_DATA_KEY="${VALID_DATA_KEY:-$DATA_KEY}"
# Key for labels in VALID_DATASET.
VALID_LABELS_KEY="${VALID_LABELS_KEY:-$LABELS_KEY}"
# Optional validation sample limit; empty means use the full validation set.
VALID_LIMIT="${VALID_LIMIT:-}"

# Maximum number of bursts kept per trace after direction-to-burst conversion.
MAX_BURSTS="${MAX_BURSTS:-2000}"
# Reuse existing burst/mapping/tuning intermediate files when possible.
REUSE_INTERMEDIATES="${REUSE_INTERMEDIATES:-1}"
# Shared train burst dataset used to create all fixed-K mappings.
SHARED_BURST_NPZ="${BURST_NPZ:-${TUNE_ROOT}/shared_cache/burst_dataset.npz}"

# Burst profile used for website clustering: super, mean_abs, or mean_signed.
PROFILE_METHOD="${PROFILE_METHOD:-super}"
# Profile normalization used by the fixed-K K-means probe: zscore, l2, or none.
NORMALIZE="${NORMALIZE:-zscore}"
# Labels excluded before clustering; 95 is usually the open-world label.
EXCLUDE_LABELS="${EXCLUDE_LABELS:-95}"
# Random seed for fixed-K K-means.
SEED="${SEED:-1}"
# K-means restarts per K; higher is slower but more stable.
KMEANS_RESTARTS="${KMEANS_RESTARTS:-20}"
# Maximum K-means iterations per restart.
KMEANS_MAX_ITER="${KMEANS_MAX_ITER:-100}"
# Only used for reporting/selection inside the probe; with one K, it will not change K.
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-1}"

# Smoke-mode sample limit. Used only when TUNE_MODE=smoke and BURST_LIMIT is empty.
SMOKE_LIMIT="${SMOKE_LIMIT:-200}"
# Optional sample limit for the shared burst cache. Empty means full training data.
BURST_LIMIT="${BURST_LIMIT:-}"
if [ -z "$BURST_LIMIT" ] && [ "$TUNE_MODE" = "smoke" ]; then
  BURST_LIMIT="$SMOKE_LIMIT"
fi

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
missing = [name for name in ("numpy",) if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))
PY
}

tag_value() {
  local value="$1"
  value="${value// /_}"
  value="${value//./p}"
  value="${value//-/_}"
  echo "$value"
}

require_file "$TRAIN_DATASET"
require_python_deps

mkdir -p "$(dirname "$SHARED_BURST_NPZ")"
mkdir -p "$TUNE_ROOT"

BURST_LIMIT_ARGS=()
if [ -n "$BURST_LIMIT" ]; then
  BURST_LIMIT_ARGS=(--limit "$BURST_LIMIT")
fi

EXCLUDE_ARGS=()
if [ -n "$EXCLUDE_LABELS" ]; then
  # shellcheck disable=SC2206
  # Split EXCLUDE_LABELS into separate --exclude-labels values.
  EXCLUDE_ARGS=(--exclude-labels $EXCLUDE_LABELS)
fi

echo "Fixed-K train-time tuning"
echo "  mode: $TUNE_MODE"
echo "  tune root: $TUNE_ROOT"
echo "  cluster counts: $CLUSTER_COUNTS"
echo "  shared burst npz: $SHARED_BURST_NPZ"
echo "  profile method: $PROFILE_METHOD"
echo "  normalize: $NORMALIZE"
echo "  exclude labels: $EXCLUDE_LABELS"
echo "  targets: $OVERHEAD_TARGETS"
echo "  lambdas: $LAMBDA_OVERHEADS"
echo "  overhead losses: $OVERHEAD_LOSSES"
echo "  attack losses: $ATTACK_LOSSES"
echo "  lambda tv: $LAMBDA_TV"
echo "  lr: $LR"
echo "  full epochs/batch/noise: $FULL_EPOCHS/$FULL_BATCH_SIZE/$FULL_NOISE_DIM"

echo "[0/3] Prepare shared burst dataset"
if [ "$REUSE_INTERMEDIATES" = "1" ] && [ -f "$SHARED_BURST_NPZ" ]; then
  echo "reuse: $SHARED_BURST_NPZ"
else
  "$PYTHON_BIN" TOOB_Simulation/EXP/01_make_burst_dataset.py \
    --input "$TRAIN_DATASET" \
    --output "$SHARED_BURST_NPZ" \
    --data-key "$TRAIN_DATA_KEY" \
    --labels-key "$TRAIN_LABELS_KEY" \
    --max-bursts "$MAX_BURSTS" \
    "${BURST_LIMIT_ARGS[@]}"
fi

for cluster_count in $CLUSTER_COUNTS; do
  if ! [[ "$cluster_count" =~ ^[0-9]+$ ]] || [ "$cluster_count" -lt 2 ]; then
    echo "Invalid cluster count: $cluster_count" >&2
    exit 2
  fi

  k_tag="k$(tag_value "$cluster_count")"
  exclude_tag="$(tag_value "$EXCLUDE_LABELS")"
  if [ -z "$exclude_tag" ]; then
    exclude_tag="none"
  fi
  k_tune_dir="${TUNE_ROOT}/${k_tag}"
  k_cache_dir="${k_tune_dir}/cluster_cache_fixed_${k_tag}_${PROFILE_METHOD}_exclude${exclude_tag}"
  k_probe_dir="${k_tune_dir}/cluster_probe"
  k_mapping="${k_probe_dir}/mapping_recommended.json"

  mkdir -p "$k_tune_dir"
  mkdir -p "$k_cache_dir"

  echo "[1/3] Build fixed-K mapping: K=$cluster_count"
  if [ "$REUSE_INTERMEDIATES" = "1" ] && [ -f "$k_mapping" ]; then
    echo "reuse: $k_mapping"
  else
    "$PYTHON_BIN" TOOB_Simulation/cluster_probe/probe_cluster_count.py \
      --burst-npz "$SHARED_BURST_NPZ" \
      --output-dir "$k_probe_dir" \
      --k-values "$cluster_count" \
      --profile-method "$PROFILE_METHOD" \
      --normalize "$NORMALIZE" \
      --seed "$SEED" \
      --restarts "$KMEANS_RESTARTS" \
      --max-iter "$KMEANS_MAX_ITER" \
      --min-cluster-size "$MIN_CLUSTER_SIZE" \
      "${EXCLUDE_ARGS[@]}"
  fi
  require_file "$k_mapping"

  echo "[2/3] Tune bandwidth losses with fixed K=$cluster_count"
  PYTHON_BIN="$PYTHON_BIN" \
  RUNNER="$RUNNER" \
  TUNE_MODE="$TUNE_MODE" \
  TUNE_DIR="$k_tune_dir" \
  CLUSTER_CACHE_DIR="$k_cache_dir" \
  BURST_NPZ="$SHARED_BURST_NPZ" \
  MAPPING="$k_mapping" \
  CLUSTER_COUNT="$cluster_count" \
  PROFILE_METHOD="$PROFILE_METHOD" \
  EXCLUDE_LABELS="$EXCLUDE_LABELS" \
  OVERHEAD_TARGETS="$OVERHEAD_TARGETS" \
  LAMBDA_OVERHEADS="$LAMBDA_OVERHEADS" \
  OVERHEAD_LOSSES="$OVERHEAD_LOSSES" \
  OVERHEAD_TOLERANCE="$OVERHEAD_TOLERANCE" \
  ATTACK_LOSSES="$ATTACK_LOSSES" \
  LAMBDA_TV="$LAMBDA_TV" \
  SOFT_PROJECTION_TAU="$SOFT_PROJECTION_TAU" \
  LR="$LR" \
  FULL_EPOCHS="$FULL_EPOCHS" \
  FULL_BATCH_SIZE="$FULL_BATCH_SIZE" \
  FULL_NOISE_DIM="$FULL_NOISE_DIM" \
  BUDGET_SLACK="$BUDGET_SLACK" \
  SELECT_METRIC="$SELECT_METRIC" \
  DATASET="$DATASET" \
  DATA_KEY="$DATA_KEY" \
  LABELS_KEY="$LABELS_KEY" \
  TRAIN_DATASET="$TRAIN_DATASET" \
  TRAIN_DATA_KEY="$TRAIN_DATA_KEY" \
  TRAIN_LABELS_KEY="$TRAIN_LABELS_KEY" \
  VALID_DATASET="$VALID_DATASET" \
  VALID_DATA_KEY="$VALID_DATA_KEY" \
  VALID_LABELS_KEY="$VALID_LABELS_KEY" \
  VALID_LIMIT="$VALID_LIMIT" \
  MAX_BURSTS="$MAX_BURSTS" \
  REUSE_INTERMEDIATES="$REUSE_INTERMEDIATES" \
  bash "$TUNE_RUNNER"
done

echo "[3/3] Aggregate fixed-K summaries"
"$PYTHON_BIN" - "$TUNE_ROOT" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path


def collect_csv(root: Path, source_name: str, output_name: str) -> None:
    rows = []
    fieldnames = ["cluster_count"]
    for csv_path in sorted(root.glob(f"k*/{source_name}")):
        cluster_count = csv_path.parent.name.removeprefix("k")
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                merged = {"cluster_count": cluster_count}
                merged.update(row)
                rows.append(merged)
                for key in merged:
                    if key not in fieldnames:
                        fieldnames.append(key)

    output_path = root / output_name
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


tune_root = Path(sys.argv[1])
collect_csv(tune_root, "summary.csv", "fixed_k_summary.csv")
collect_csv(tune_root, "best_by_target.csv", "fixed_k_best_by_target.csv")
print(f"saved summary: {tune_root / 'fixed_k_summary.csv'}")
print(f"saved best: {tune_root / 'fixed_k_best_by_target.csv'}")
PY

echo "Done."
echo "All fixed-K runs: ${TUNE_ROOT}/fixed_k_summary.csv"
echo "Best by target across fixed K: ${TUNE_ROOT}/fixed_k_best_by_target.csv"
