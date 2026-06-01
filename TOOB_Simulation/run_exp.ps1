param(
    [ValidateSet("smoke", "one", "full")]
    [string]$Mode = "smoke"
)

$ErrorActionPreference = "Stop"

$PythonExe = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "D:\Anaconda\envs\mytorch\python.exe" }
$Dataset = if ($env:DATASET) { $env:DATASET } else { "TOOB_Simulation\data\raw\test.npz" }
$DataKey = if ($env:DATA_KEY) { $env:DATA_KEY } else { "X" }
$LabelsKey = if ($env:LABELS_KEY) { $env:LABELS_KEY } else { "y" }
$Mapping = if ($env:MAPPING) { $env:MAPPING } else { "" }
$DfBuilder = if ($env:DF_BUILDER) { $env:DF_BUILDER } else { "TOOB_Simulation\checkpoints\df\DF.py:DF" }
$DfCheckpoint = if ($env:DF_CHECKPOINT) { $env:DF_CHECKPOINT } else { "TOOB_Simulation\checkpoints\df\max_f1.pth" }

$OutDir = if ($env:OUT_DIR) { $env:OUT_DIR } else { "TOOB_Simulation\outputs" }
$MaxBursts = if ($env:MAX_BURSTS) { $env:MAX_BURSTS } else { "2000" }
$TraceLen = if ($env:TRACE_LEN) { $env:TRACE_LEN } else { "5000" }
$NumClasses = if ($env:NUM_CLASSES) { $env:NUM_CLASSES } else { "96" }
$ProjectionChunkSize = if ($env:PROJECTION_CHUNK_SIZE) { $env:PROJECTION_CHUNK_SIZE } else { "64" }
$SoftProjectionTau = if ($env:SOFT_PROJECTION_TAU) { $env:SOFT_PROJECTION_TAU } else { "1.5" }
$SetSize = if ($env:SET_SIZE) { $env:SET_SIZE } else { "30" }
$ClusterRounds = if ($env:CLUSTER_ROUNDS) { $env:CLUSTER_ROUNDS } else { "1" }
$ProfileMethod = if ($env:PROFILE_METHOD) { $env:PROFILE_METHOD } else { "super" }
$ExcludeLabels = if ($env:EXCLUDE_LABELS) { $env:EXCLUDE_LABELS } else { "95" }
$RunEval = if ($env:RUN_EVAL) { $env:RUN_EVAL } else { "1" }
$EvalMetrics = if ($env:EVAL_METRICS) { $env:EVAL_METRICS } else { "accuracy precision recall f1" }
$EvalAverage = if ($env:EVAL_AVERAGE) { $env:EVAL_AVERAGE } else { "macro" }

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
$EvalJson = Join-Path $RunDir "defense_eval_metrics.json"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

Write-Host "[0/5] TOOB imports"
& $PythonExe TOOB_Simulation\EXP\00_check_imports.py

Write-Host "[1/5] Direction sequence -> burst dataset"
& $PythonExe TOOB_Simulation\EXP\01_make_burst_dataset.py `
    --input $Dataset `
    --output $BurstNpz `
    --data-key $DataKey `
    --labels-key $LabelsKey `
    --max-bursts $MaxBursts `
    @LimitArgs

Write-Host "[2/5] Cluster burst profiles -> pseudo labels"
$MappingArgs = @()
if ($Mapping) {
    $MappingArgs = @("--mapping", $Mapping)
}
$ExcludeArgs = @()
if ($ExcludeLabels) {
    $ExcludeArgs = @("--exclude-labels") + $ExcludeLabels.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
}
& $PythonExe TOOB_Simulation\EXP\02_make_pseudo_labels.py `
    --labels-npz $BurstNpz `
    --output $PseudoNpz `
    --json-output $PseudoJson `
    --set-size $SetSize `
    --rounds $ClusterRounds `
    --profile-method $ProfileMethod `
    --drop-unmapped `
    @MappingArgs `
    @ExcludeArgs

Write-Host "[3/5] Train cluster-wise burst generators"
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

Write-Host "[4/5] Export defended direction dataset"
& $PythonExe TOOB_Simulation\EXP\04_generate_dataset.py `
    --burst-npz $BurstNpz `
    --pseudo-npz $PseudoNpz `
    --generator-dir $GeneratorDir `
    --output $AdvDirectionNpz `
    --output-kind direction `
    --max-trace-len $TraceLen `
    --batch-size 256 `
    --round

if ($RunEval -eq "1") {
    Write-Host "[5/5] Evaluate defended dataset"
    $EvalMetricArgs = $EvalMetrics.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
    & $PythonExe TOOB_Simulation\EXP\05_evaluate_defense.py `
        --input-npz $AdvDirectionNpz `
        --input-kind direction `
        --data-key data `
        --labels-key labels `
        --detector-builder $DfBuilder `
        --detector-checkpoint $DfCheckpoint `
        --num-classes $NumClasses `
        --detector-input-kind direction `
        --detector-input-layout ncl `
        --max-trace-len $TraceLen `
        --metrics @EvalMetricArgs `
        --average $EvalAverage `
        --output-json $EvalJson
}

Write-Host "Done."
Write-Host "Output dataset: $AdvDirectionNpz"
if ($RunEval -eq "1") {
    Write-Host "Evaluation metrics: $EvalJson"
}
