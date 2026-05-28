param(
    [ValidateSet("smoke", "one", "full")]
    [string]$Mode = "smoke"
)

$ErrorActionPreference = "Stop"

$PythonExe = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "D:\Anaconda\envs\mytorch\python.exe" }
$Dataset = if ($env:DATASET) { $env:DATASET } else { "TOOB_Simulation\data\raw\test.npz" }
$DataKey = if ($env:DATA_KEY) { $env:DATA_KEY } else { "X" }
$LabelsKey = if ($env:LABELS_KEY) { $env:LABELS_KEY } else { "y" }
$Mapping = if ($env:MAPPING) { $env:MAPPING } else { "TOOB_Simulation\assets\website_to_set_1000_30_1.json" }
$DfBuilder = if ($env:DF_BUILDER) { $env:DF_BUILDER } else { "TOOB_Simulation\checkpoints\df\DF.py:DF" }
$DfCheckpoint = if ($env:DF_CHECKPOINT) { $env:DF_CHECKPOINT } else { "TOOB_Simulation\checkpoints\df\max_f1.pth" }

$OutDir = if ($env:OUT_DIR) { $env:OUT_DIR } else { "TOOB_Simulation\outputs" }
$MaxBursts = if ($env:MAX_BURSTS) { $env:MAX_BURSTS } else { "2000" }
$TraceLen = if ($env:TRACE_LEN) { $env:TRACE_LEN } else { "5000" }
$NumClasses = if ($env:NUM_CLASSES) { $env:NUM_CLASSES } else { "96" }
$ProjectionChunkSize = if ($env:PROJECTION_CHUNK_SIZE) { $env:PROJECTION_CHUNK_SIZE } else { "64" }
$SoftProjectionTau = if ($env:SOFT_PROJECTION_TAU) { $env:SOFT_PROJECTION_TAU } else { "1.5" }

if ($Mode -eq "smoke") {
    $RunDir = "${OutDir}_smoke"
    $LimitArgs = @("--limit", $(if ($env:SMOKE_LIMIT) { $env:SMOKE_LIMIT } else { "200" }))
    $Epochs = if ($env:SMOKE_EPOCHS) { $env:SMOKE_EPOCHS } else { "1" }
    $BatchSize = if ($env:SMOKE_BATCH_SIZE) { $env:SMOKE_BATCH_SIZE } else { "4" }
    $NoiseDim = if ($env:SMOKE_NOISE_DIM) { $env:SMOKE_NOISE_DIM } else { "64" }
    $PseudoArgs = @()
} elseif ($Mode -eq "full") {
    $RunDir = $OutDir
    $LimitArgs = @()
    $Epochs = if ($env:FULL_EPOCHS) { $env:FULL_EPOCHS } else { "30" }
    $BatchSize = if ($env:FULL_BATCH_SIZE) { $env:FULL_BATCH_SIZE } else { "32" }
    $NoiseDim = if ($env:FULL_NOISE_DIM) { $env:FULL_NOISE_DIM } else { "256" }
    $PseudoArgs = @()
} else {
    $RunDir = "${OutDir}_one"
    $LimitArgs = @()
    $Epochs = if ($env:ONE_EPOCHS) { $env:ONE_EPOCHS } else { "1" }
    $BatchSize = if ($env:ONE_BATCH_SIZE) { $env:ONE_BATCH_SIZE } else { "4" }
    $NoiseDim = if ($env:ONE_NOISE_DIM) { $env:ONE_NOISE_DIM } else { "64" }
    $PseudoLabel = if ($env:PSEUDO_LABEL) { $env:PSEUDO_LABEL } else { "0" }
    $PseudoArgs = @("--pseudo-labels", $PseudoLabel)
}

$BurstNpz = Join-Path $RunDir "burst_dataset.npz"
$PseudoNpz = Join-Path $RunDir "pseudo_labels.npz"
$PseudoJson = Join-Path $RunDir "pseudo_labels.json"
$GeneratorDir = Join-Path $RunDir "generators"
$AdvDirectionNpz = Join-Path $RunDir "toob_adv_direction.npz"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

Write-Host "[1/4] Direction sequence -> burst dataset"
& $PythonExe TOOB_Simulation\EXP\01_make_burst_dataset.py `
    --input $Dataset `
    --output $BurstNpz `
    --data-key $DataKey `
    --labels-key $LabelsKey `
    --max-bursts $MaxBursts `
    @LimitArgs

Write-Host "[2/4] Palette mapping -> pseudo labels"
& $PythonExe TOOB_Simulation\EXP\02_make_pseudo_labels.py `
    --labels-npz $BurstNpz `
    --mapping $Mapping `
    --output $PseudoNpz `
    --json-output $PseudoJson `
    --drop-unmapped

Write-Host "[3/4] Train cluster-wise burst generators"
& $PythonExe TOOB_Simulation\EXP\03_train_generators.py `
    --burst-npz $BurstNpz `
    --pseudo-npz $PseudoNpz `
    --detector-builder $DfBuilder `
    --detector-checkpoint $DfCheckpoint `
    --num-classes $NumClasses `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --noise-dim $NoiseDim `
    --detector-input-kind direction `
    --detector-input-layout ncl `
    --detector-input-length $TraceLen `
    --soft-projection-tau $SoftProjectionTau `
    --projection-chunk-size $ProjectionChunkSize `
    --output-dir $GeneratorDir `
    @PseudoArgs

Write-Host "[4/4] Export defended direction dataset"
& $PythonExe TOOB_Simulation\EXP\04_generate_dataset.py `
    --burst-npz $BurstNpz `
    --pseudo-npz $PseudoNpz `
    --generator-dir $GeneratorDir `
    --output $AdvDirectionNpz `
    --output-kind direction `
    --max-trace-len $TraceLen `
    --batch-size 256 `
    --round

Write-Host "Done."
Write-Host "Output dataset: $AdvDirectionNpz"
