# TOOB Cluster Probe

This folder is intentionally separate from the main TOOB training pipeline.
It reads datasets produced by the main flow, evaluates candidate pseudo-label
cluster counts, and writes standalone reports.

Typical usage:

```bash
python TOOB_Simulation/cluster_probe/probe_cluster_count.py \
  --burst-npz TOOB_Simulation/outputs_train_tuning/cluster_cache_set30_super_exclude95/burst_dataset.npz \
  --k-values 3 4 5 6 7 8 9 10 \
  --exclude-labels 95
```

Outputs:

```text
TOOB_Simulation/cluster_probe/results/cluster_count_probe.csv
TOOB_Simulation/cluster_probe/results/cluster_count_probe.json
TOOB_Simulation/cluster_probe/results/recommendation.json
TOOB_Simulation/cluster_probe/results/mapping_recommended.json
```

The recommendation reports a candidate `K` and an approximate `SET_SIZE`:

```text
SET_SIZE = ceil(num_websites / K)
```

Use that `SET_SIZE` in the main runner only after reviewing the CSV/JSON.

To run the main tuning flow with the recommended natural, possibly unbalanced
mapping:

```bash
MAPPING=TOOB_Simulation/cluster_probe/results/mapping_recommended.json \
TUNE_DIR=TOOB_Simulation/outputs_train_tuning_natural_k \
bash TOOB_Simulation/tune_train_budget_ubuntu.sh
```

This bypasses the main flow's `SET_SIZE` clustering and reuses the mapping from
the probe.
