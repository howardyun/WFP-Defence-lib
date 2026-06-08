# TOOB Fixed-K Pipeline

This folder is an isolated runner for the fixed-K TOOB experiment. It does not
call the older tuning shell scripts.

Default command:

```bash
bash TOOB_simulation_fix_k/run_fix_k_ubuntu.sh full
```

The default run uses:

```bash
CLUSTER_COUNTS="4"
OVERHEAD_TARGETS="0.20 0.30 0.40"
LAMBDA_OVERHEADS="1.0"
OVERHEAD_LOSSES="hinge"
ATTACK_LOSSES="true_prob"
PROJECTION_MODE="ste"
NUM_CLASSES="95"
DF_CHECKPOINT="TOOB_simulation_fix_k/checkpoints/df_cw/max_f1.pth"
```

Useful overrides:

```bash
CLUSTER_COUNTS="4 5 6" bash TOOB_simulation_fix_k/run_fix_k_ubuntu.sh full
OVERHEAD_TARGETS="0.30 0.40" bash TOOB_simulation_fix_k/run_fix_k_ubuntu.sh full
FULL_NOISE_DIM=256 bash TOOB_simulation_fix_k/run_fix_k_ubuntu.sh full
```

Outputs are written to:

```text
TOOB_simulation_fix_k/outputs/
```

Important files:

```text
k4/mapping/mapping_recommended.json
k4/cache/train_pseudo_labels.json
k4/run_*/run_config.json
k4/run_*/defense_eval_metrics.json
summary.csv
best_by_target.csv
```

The code is standalone. It does not import or call the older TOOB code. By
default it expects data and checkpoint files under this folder:

```text
TOOB_simulation_fix_k/data/raw/train.npz
TOOB_simulation_fix_k/data/raw/valid.npz
TOOB_simulation_fix_k/checkpoints/df_cw/max_f1.pth
```

You can also point to existing files explicitly:

```bash
TRAIN_DATASET=/path/to/train.npz \
VALID_DATASET=/path/to/valid.npz \
DF_CHECKPOINT=/path/to/max_f1.pth \
bash TOOB_simulation_fix_k/run_fix_k_ubuntu.sh full
```
