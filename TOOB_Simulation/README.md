# TOOB Simulation

This folder contains a first simulation-only version of the cluster-wise burst
perturbation idea.

The current design intentionally does not implement a real pluggable transport.
It creates defended/adversarial datasets that can be evaluated by website
fingerprinting detectors.

## Idea

1. Convert packet direction traces into burst sequences.
2. Use Palette clustering output as pseudo labels.
3. Train one generator `G_c` for each pseudo label `c`.
4. `G_c(z)` outputs a non-negative burst perturbation vector.
5. Apply the perturbation with:

```text
b_adv = b + G_c(z) * sign(b)
```

The real website label is preserved in the exported dataset. Pseudo labels are
only used to choose which generator is used.

## Inputs

The scripts expect `.npz` datasets with:

```text
data   - numpy array, usually direction sequences or burst sequences
labels - integer labels or one-hot labels
```

For DF training/evaluation, this code uses a detector adapter. Your current DF
class can be loaded directly from:

```text
TOOB_Simulation/checkpoints/df/DF.py:DF
```

and the checkpoint path you provided is:

```text
TOOB_Simulation/checkpoints/df/max_f1.pth
```

The cluster pseudo-label mapping is bundled with TOOB:

```text
TOOB_Simulation/assets/website_to_set_1000_30_1.json
```

## One Command Runner

Ubuntu runner:

```bash
bash TOOB_Simulation/run_exp_ubuntu.sh smoke
```

```bash
bash TOOB_Simulation/run_exp_ubuntu.sh full
```

If your Ubuntu Python is not `python3`, override it:

```bash
PYTHON_BIN=/path/to/python bash TOOB_Simulation/run_exp_ubuntu.sh smoke
```

PowerShell runner, recommended on this machine:

```powershell
powershell -ExecutionPolicy Bypass -File TOOB_Simulation/run_exp.ps1 smoke
```

```powershell
powershell -ExecutionPolicy Bypass -File TOOB_Simulation/run_exp.ps1 full
```

On this Windows machine, Git Bash is available at `D:/Git/bin/sh.exe`.

Smoke test, using 200 samples:

```powershell
D:\Git\bin\sh.exe TOOB_Simulation/run_exp.sh smoke
```

Train one pseudo label on the full dataset:

```powershell
$env:PSEUDO_LABEL="0"
D:\Git\bin\sh.exe TOOB_Simulation/run_exp.sh one
```

Run the full experiment:

```powershell
D:\Git\bin\sh.exe TOOB_Simulation/run_exp.sh full
```

Useful overrides:

```powershell
$env:FULL_EPOCHS="50"
$env:FULL_BATCH_SIZE="16"
$env:OUT_DIR="TOOB_Simulation/outputs_v1"
```

## Step 1: Direction To Burst

```powershell
python TOOB_Simulation/EXP/01_make_burst_dataset.py `
  --input TOOB_Simulation/data/raw/test.npz `
  --output TOOB_Simulation/outputs/burst_dataset.npz `
  --data-key X `
  --labels-key y `
  --max-bursts 2000
```

## Step 2: Pseudo Labels

Use Palette's `website_to_set_*.npy` mapping:

```powershell
python TOOB_Simulation/EXP/02_make_pseudo_labels.py `
  --labels-npz TOOB_Simulation/outputs/burst_dataset.npz `
  --mapping TOOB_Simulation/assets/website_to_set_1000_30_1.json `
  --output TOOB_Simulation/outputs/pseudo_labels.npz `
  --json-output TOOB_Simulation/outputs/pseudo_labels.json `
  --drop-unmapped
```

This step also writes a readable JSON summary with pseudo-label counts,
website-to-set mapping, and per-sample label assignments.

## Step 3: Train Generators With DF

```powershell
python TOOB_Simulation/EXP/03_train_generators.py `
  --burst-npz TOOB_Simulation/outputs/burst_dataset.npz `
  --pseudo-npz TOOB_Simulation/outputs/pseudo_labels.npz `
  --detector-builder TOOB_Simulation/checkpoints/df/DF.py:DF `
  --detector-checkpoint TOOB_Simulation/checkpoints/df/max_f1.pth `
  --num-classes 96 `
  --detector-input-kind direction `
  --detector-input-layout ncl `
  --detector-input-length 5000 `
  --output-dir TOOB_Simulation/outputs/generators
```

This DF architecture consumes direction sequences with shape `[N, 1, 5000]`.
During training, TOOB uses a differentiable soft burst-to-direction projection
so gradients can flow from DF back into `G_c`. The provided checkpoint has 96
output classes, so use `--num-classes 96`. The pseudo-label step can drop the
open-world label if it is not present in the Palette mapping.

For a quick smoke test, train only one pseudo label and one epoch:

```powershell
python TOOB_Simulation/EXP/03_train_generators.py `
  --burst-npz TOOB_Simulation/outputs/burst_dataset.npz `
  --pseudo-npz TOOB_Simulation/outputs/pseudo_labels.npz `
  --detector-builder TOOB_Simulation/checkpoints/df/DF.py:DF `
  --detector-checkpoint TOOB_Simulation/checkpoints/df/max_f1.pth `
  --num-classes 96 `
  --pseudo-labels 0 `
  --epochs 1 `
  --batch-size 8 `
  --detector-input-kind direction `
  --detector-input-layout ncl `
  --detector-input-length 5000 `
  --output-dir TOOB_Simulation/outputs/generators_smoke
```

If GPU memory is tight, lower these together:

```text
--batch-size 4 --projection-chunk-size 64
```

`--soft-projection-tau` controls how sharp the differentiable burst expansion is.
Smaller values are closer to hard direction sequences but can make gradients
more brittle. Start with `1.5`, then try `1.0` or `0.5` if the attack loss does
not move.

## Step 4: Generate Dataset

```powershell
python TOOB_Simulation/EXP/04_generate_dataset.py `
  --burst-npz TOOB_Simulation/outputs/burst_dataset.npz `
  --pseudo-npz TOOB_Simulation/outputs/pseudo_labels.npz `
  --generator-dir TOOB_Simulation/outputs/generators `
  --output TOOB_Simulation/outputs/toob_adv_burst.npz `
  --output-kind burst `
  --round
```

To export direction sequences for other detectors:

```powershell
python TOOB_Simulation/EXP/04_generate_dataset.py `
  --burst-npz TOOB_Simulation/outputs/burst_dataset.npz `
  --pseudo-npz TOOB_Simulation/outputs/pseudo_labels.npz `
  --generator-dir TOOB_Simulation/outputs/generators `
  --output TOOB_Simulation/outputs/toob_adv_direction.npz `
  --output-kind direction `
  --max-trace-len 5000 `
  --round
```
