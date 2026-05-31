# TOOB Simulation

This folder contains a first simulation-only version of the cluster-wise burst
perturbation idea.

The current design intentionally does not implement a real pluggable transport.
It creates defended/adversarial datasets that can be evaluated by website
fingerprinting detectors.

## Idea

1. Convert packet direction traces into burst sequences.
2. Cluster website burst profiles into pseudo-label anonymity sets.
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

Output:

```text
TOOB_Simulation/outputs/burst_dataset.npz
```

This file contains:

```text
data   - burst sequences converted from the original direction traces
labels - original website labels, unchanged
```

`data` is the burst-domain training representation used by the later TOOB
steps. `labels` still means the real website class, such as `0..94`; this step
does not create pseudo labels yet.

## Step 2: Pseudo Labels

Cluster website burst profiles and create pseudo labels:

```powershell
python TOOB_Simulation/EXP/02_make_pseudo_labels.py `
  --labels-npz TOOB_Simulation/outputs/burst_dataset.npz `
  --output TOOB_Simulation/outputs/pseudo_labels.npz `
  --json-output TOOB_Simulation/outputs/pseudo_labels.json `
  --set-size 30 `
  --rounds 1 `
  --profile-method super `
  --exclude-labels 95 `
  --drop-unmapped
```

Output:

```text
TOOB_Simulation/outputs/pseudo_labels.npz
TOOB_Simulation/outputs/pseudo_labels.json
```

`pseudo_labels.npz` is the file used by later scripts. It contains:

```text
labels        - original website labels after filtering, still the real labels
pseudo_labels - cluster IDs assigned to each kept sample
keep_indices  - row indices kept from burst_dataset.npz
```

`pseudo_labels.json` is a readable summary for inspection. The important fields
are:

```text
website_to_pseudo_label - original website label -> pseudo label
pseudo_label_to_websites - pseudo label -> grouped website labels
pseudo_label_counts - number of samples under each pseudo label
cluster - clustering settings and metadata
samples - per-sample original label and pseudo label assignment
```

Conceptually, this step is where the original monitored website labels are
merged into fewer pseudo-label groups. For example, website labels `0`, `5`, and
`8` may all map to pseudo label `0`. Later, one generator `G_0` is trained for
all samples whose pseudo label is `0`.

`--exclude-labels 95` removes the open-world class from pseudo-label clustering.
The real labels are preserved so the final defended dataset can still be
evaluated by normal detectors.

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

Output:

```text
TOOB_Simulation/outputs/generators/generator_pseudo_0.pt
TOOB_Simulation/outputs/generators/generator_pseudo_1.pt
...
TOOB_Simulation/outputs/generators/manifest.json
```

Each `generator_pseudo_*.pt` file is one trained generator `G_c` for a pseudo
label `c`. The script reads `burst_dataset.npz` for burst inputs and
`pseudo_labels.npz` to decide which samples belong to each pseudo label.

`manifest.json` records the detector checkpoint, training arguments, and the
generator files produced during the run.

This DF architecture consumes direction sequences with shape `[N, 1, 5000]`.
During training, TOOB uses a differentiable soft burst-to-direction projection
so gradients can flow from DF back into `G_c`. The provided checkpoint has 96
output classes, so use `--num-classes 96`. The pseudo-label step excludes the
open-world label `95` by default in the runners.

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

Output:

```text
TOOB_Simulation/outputs/toob_adv_direction.npz
```

For `--output-kind direction`, the output contains:

```text
data          - defended direction sequences, ready for detector evaluation
labels        - original website labels, unchanged
pseudo_labels - pseudo label used to choose the generator
overhead      - perturbation overhead for each sample
```

For `--output-kind burst`, `data` and `burst_data` are defended burst data
instead of direction sequences. For `--output-kind both`, the file stores
`burst_data` and `direction_data`; `data` points to the burst representation.

This is the final simulation dataset. Other detectors should evaluate against
`labels`, not `pseudo_labels`. The pseudo labels only explain which cluster-wise
generator produced the perturbation.

## Runner Output Layout

The one-command runners use the same four steps and write to a run directory:

```text
smoke -> TOOB_Simulation/outputs_smoke/
one   -> TOOB_Simulation/outputs_one/
full  -> TOOB_Simulation/outputs/
```

Inside a run directory, the main artifacts are:

```text
burst_dataset.npz          - Step 1 burst representation
pseudo_labels.npz          - Step 2 sample-level pseudo labels for training
pseudo_labels.json         - Step 2 readable cluster mapping and summary
generators/                - Step 3 trained G_c checkpoints
toob_adv_direction.npz     - Step 4 final defended direction dataset
```
