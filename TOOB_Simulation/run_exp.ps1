param(
    [ValidateSet("smoke", "one", "full")]
    [string]$Mode = "smoke"
)

$ErrorActionPreference = "Stop"

$PythonExe = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "D:\Anaconda\envs\mytorch\python.exe" }
$Dataset = if ($env:DATASET) { $env:DATASET } else { "TOOB_Simulation\data\raw\test.npz" }
$DataKey = if ($env:DATA_KEY) { $env:DATA_KEY } else { "X" }
$LabelsKey = if ($env:LABELS_KEY) { $env:LABELS_KEY } else { "y" }
$TrainDataset = if ($env:TRAIN_DATASET) { $env:TRAIN_DATASET } else { $Dataset }
$TrainDataKey = if ($env:TRAIN_DATA_KEY) { $env:TRAIN_DATA_KEY } else { $DataKey }
$TrainLabelsKey = if ($env:TRAIN_LABELS_KEY) { $env:TRAIN_LABELS_KEY } else { $LabelsKey }
$ValidDataset = if ($env:VALID_DATASET) { $env:VALID_DATASET } else { "" }
$ValidDataKey = if ($env:VALID_DATA_KEY) { $env:VALID_DATA_KEY } else { $DataKey }
$ValidLabelsKey = if ($env:VALID_LABELS_KEY) { $env:VALID_LABELS_KEY } else { $LabelsKey }
$ValidLimit = if ($env:VALID_LIMIT) { $env:VALID_LIMIT } else { "" }
$Mapping = if ($env:MAPPING) { $env:MAPPING } else { "" }
$DfBuilder = if ($env:DF_BUILDER) { $env:DF_BUILDER } else { "TOOB_Simulation\toob\wflib_df.py:DF" }
$DfCheckpoint = if ($env:DF_CHECKPOINT) { $env:DF_CHECKPOINT } else { "TOOB_Simulation\checkpoints\df_cw\max_f1.pth" }

$OutDir = if ($env:OUT_DIR) { $env:OUT_DIR } else { "TOOB_Simulation\outputs" }
$MaxBursts = if ($env:MAX_BURSTS) { $env:MAX_BURSTS } else { "2000" }
$TraceLen = if ($env:TRACE_LEN) { $env:TRACE_LEN } else { "5000" }
$NumClasses = if ($env:NUM_CLASSES) { $env:NUM_CLASSES } else { "95" }
$ProjectionChunkSize = if ($env:PROJECTION_CHUNK_SIZE) { $env:PROJECTION_CHUNK_SIZE } else { "64" }
$SoftProjectionTau = if ($env:SOFT_PROJECTION_TAU) { $env:SOFT_PROJECTION_TAU } else { "1.5" }
$SetSize = if ($env:SET_SIZE) { $env:SET_SIZE } else { "30" }
$ClusterRounds = if ($env:CLUSTER_ROUNDS) { $env:CLUSTER_ROUNDS } else { "1" }
$ProfileMethod = if ($env:PROFILE_METHOD) { $env:PROFILE_METHOD } else { "super" }
$ExcludeLabels = if ($env:EXCLUDE_LABELS) { $env:EXCLUDE_LABELS } else { "95" }
$EvalExcludeLabels = if ($env:EVAL_EXCLUDE_LABELS) { $env:EVAL_EXCLUDE_LABELS } else { $ExcludeLabels }
$RunEval = if ($env:RUN_EVAL) { $env:RUN_EVAL } else { "1" }
$EvalMetrics = if ($env:EVAL_METRICS) { $env:EVAL_METRICS } else { "accuracy precision recall f1" }
$EvalAverage = if ($env:EVAL_AVERAGE) { $env:EVAL_AVERAGE } else { "macro" }

if ($Mode -eq "smoke") {
    $RunDir = "${OutDir}_smoke"
    $SmokeLimitValue = if ($env:SMOKE_LIMIT) { $env:SMOKE_LIMIT } else { "200" }
    $LimitArgs = @("--limit", $SmokeLimitValue)
    if ($ValidDataset -and -not $ValidLimit) {
        $ValidLimit = $SmokeLimitValue
    }
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

$ValidLimitArgs = @()
if ($ValidLimit) {
    $ValidLimitArgs = @("--limit", $ValidLimit)
}

$BurstNpz = Join-Path $RunDir "burst_dataset.npz"
$PseudoNpz = Join-Path $RunDir "pseudo_labels.npz"
$PseudoJson = Join-Path $RunDir "pseudo_labels.json"
$ValidBurstNpz = Join-Path $RunDir "valid_burst_dataset.npz"
$ValidPseudoNpz = Join-Path $RunDir "valid_pseudo_labels.npz"
$ValidPseudoJson = Join-Path $RunDir "valid_pseudo_labels.json"
$GeneratorDir = Join-Path $RunDir "generators"
if ($ValidDataset) {
    $AdvDirectionNpz = Join-Path $RunDir "toob_valid_adv_direction.npz"
} else {
    $AdvDirectionNpz = Join-Path $RunDir "toob_adv_direction.npz"
}
$EvalJson = Join-Path $RunDir "defense_eval_metrics.json"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

Write-Host "[0/5] TOOB imports"
& $PythonExe TOOB_Simulation\EXP\00_check_imports.py

Write-Host "[1/5] Train direction sequence -> burst dataset"
& $PythonExe TOOB_Simulation\EXP\01_make_burst_dataset.py `
    --input $TrainDataset `
    --output $BurstNpz `
    --data-key $TrainDataKey `
    --labels-key $TrainLabelsKey `
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

$DefenseBurstNpz = $BurstNpz
$DefensePseudoNpz = $PseudoNpz
if ($ValidDataset) {
    Write-Host "[4/5] Valid direction sequence -> burst dataset"
    & $PythonExe TOOB_Simulation\EXP\01_make_burst_dataset.py `
        --input $ValidDataset `
        --output $ValidBurstNpz `
        --data-key $ValidDataKey `
        --labels-key $ValidLabelsKey `
        --max-bursts $MaxBursts `
        @ValidLimitArgs

    Write-Host "[4/5] Map valid labels with train pseudo-label mapping"
    & $PythonExe TOOB_Simulation\EXP\02_make_pseudo_labels.py `
        --labels-npz $ValidBurstNpz `
        --mapping $PseudoJson `
        --output $ValidPseudoNpz `
        --json-output $ValidPseudoJson `
        --drop-unmapped

    $DefenseBurstNpz = $ValidBurstNpz
    $DefensePseudoNpz = $ValidPseudoNpz
}

Write-Host "[4/5] Export defended direction dataset"
& $PythonExe TOOB_Simulation\EXP\04_generate_dataset.py `
    --burst-npz $DefenseBurstNpz `
    --pseudo-npz $DefensePseudoNpz `
    --generator-dir $GeneratorDir `
    --output $AdvDirectionNpz `
    --output-kind direction `
    --max-trace-len $TraceLen `
    --batch-size 256 `
    --round

if ($RunEval -eq "1") {
    Write-Host "[5/5] Evaluate defended dataset"
    $EvalMetricArgs = $EvalMetrics.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
    $EvalExcludeArgs = @()
    if ($EvalExcludeLabels) {
        $EvalExcludeArgs = @("--exclude-labels") + $EvalExcludeLabels.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
    }
    & $PythonExe TOOB_Simulation\EXP\05_evaluate_defense.py `
        --input-npz $AdvDirectionNpz `
        --input-kind direction `
        --data-key data `
        --labels-key labels `
        @EvalExcludeArgs `
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
if ($ValidDataset) {
    Write-Host "Valid pseudo-label JSON: $ValidPseudoJson"
}
if ($RunEval -eq "1") {
    Write-Host "Evaluation metrics: $EvalJson"
}
