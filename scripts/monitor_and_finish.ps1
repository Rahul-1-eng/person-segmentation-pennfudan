$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$logRoot = Join-Path $projectRoot "artifacts\logs"
$pipelineLog = Join-Path $logRoot "pipeline.log"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

if (-not (Test-Path $pythonExe)) {
    throw "Project Python was not found at $pythonExe"
}

$env:PYTHONPATH = Join-Path $projectRoot "src"

function Write-PipelineLog([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Message"
    Add-Content -LiteralPath $pipelineLog -Value $line -Encoding UTF8
}

# A previous manager may have exited while its Python child kept training. Wait
# for any such run before scheduling work so a restart cannot duplicate it.
while ($true) {
    $activeTrains = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'person_segmentation\.train.*--experiment' })
    if ($activeTrains.Count -eq 0) { break }
    $ids = @($activeTrains | ForEach-Object { [int]$_.ProcessId })
    Write-PipelineLog "Waiting for active training process(es): $($ids -join ', ')."
    Wait-Process -Id $ids -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

function Test-RunComplete([string]$Experiment, [int]$Seed) {
    $runName = if ($Seed -eq 20261103) { $Experiment } else { "${Experiment}_seed$Seed" }
    $runDir = Join-Path (Join-Path $projectRoot "artifacts") $runName
    return ((Test-Path (Join-Path $runDir "summary.json")) -and
            (Test-Path (Join-Path $runDir "history.csv")))
}

function Invoke-TrainingRun([string]$Experiment, [int]$Seed) {
    if (Test-RunComplete $Experiment $Seed) {
        Write-PipelineLog "Skipping completed $Experiment seed $Seed."
        return
    }

    $runName = if ($Seed -eq 20261103) { $Experiment } else { "${Experiment}_seed$Seed" }
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $runLog = Join-Path $logRoot "train_${runName}_${stamp}.log"

    foreach ($threadLimit in @(4, 1)) {
        $env:OMP_NUM_THREADS = [string]$threadLimit
        $env:MKL_NUM_THREADS = [string]$threadLimit
        Write-PipelineLog "Starting $Experiment seed $Seed with CPU thread limit $threadLimit."
        & $pythonExe -m person_segmentation.train --config config.yaml --experiment $Experiment --seed $Seed 2>&1 |
            Tee-Object -FilePath $runLog -Append
        $exitCode = $LASTEXITCODE

        if (($exitCode -eq 0) -and (Test-RunComplete $Experiment $Seed)) {
            Write-PipelineLog "Completed $Experiment seed $Seed."
            return
        }

        $logText = if (Test-Path $runLog) { Get-Content -LiteralPath $runLog -Raw } else { "" }
        $memoryFailure = $logText -match '(?i)DefaultCPUAllocator|not enough memory|out of memory|bad_alloc'
        if (($threadLimit -eq 4) -and $memoryFailure) {
            Write-PipelineLog "Memory error in $Experiment seed $Seed; retrying once with one CPU thread."
            continue
        }

        Write-PipelineLog "FAILED $Experiment seed $Seed with exit code $exitCode. See $runLog"
        throw "Training failed for $Experiment seed $Seed. See $runLog"
    }
}

$plan = @(
    @{ Experiment = "augmented"; Seed = 1 },
    @{ Experiment = "augmented"; Seed = 2 }
)
foreach ($experiment in @(
    "advanced",
    "advanced_ablation",
    "strong_aug",
    "distract",
    "resnet34",
    "resnet34_distract",
    "deeplab_ft"
)) {
    foreach ($seed in @(20261103, 1, 2)) {
        $plan += @{ Experiment = $experiment; Seed = $seed }
    }
}

foreach ($run in $plan) {
    Invoke-TrainingRun $run.Experiment $run.Seed
}

$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"

function Invoke-LoggedPython([string]$Label, [string[]]$Arguments) {
    Write-PipelineLog "Starting post-processing step: $Label"
    & $pythonExe @Arguments 2>&1 | Tee-Object -FilePath $pipelineLog -Append
    if ($LASTEXITCODE -ne 0) {
        Write-PipelineLog "FAILED post-processing step: $Label (exit code $LASTEXITCODE)."
        throw "Post-processing failed at $Label. See $pipelineLog"
    }
}

Invoke-LoggedPython "environment capture" @("-m", "pip", "freeze") |
    Out-File -LiteralPath (Join-Path $projectRoot "artifacts\environment.txt") -Encoding utf8
Invoke-LoggedPython "trivial baselines" @("-m", "person_segmentation.baselines", "--config", "config.yaml")
Invoke-LoggedPython "model evaluation" @("-m", "person_segmentation.evaluate", "--config", "config.yaml", "--zero-shot")
Invoke-LoggedPython "robustness evaluation" @("-m", "person_segmentation.robustness", "--config", "config.yaml", "--zero-shot")
Invoke-LoggedPython "tables and paired comparisons" @(
    "scripts/make_tables.py", "--config", "config.yaml",
    "--pair", "baseline", "augmented",
    "--pair", "augmented", "advanced",
    "--pair", "augmented", "advanced_ablation",
    "--pair", "augmented", "strong_aug",
    "--pair", "strong_aug", "distract",
    "--pair", "strong_aug", "resnet34",
    "--pair", "resnet34", "resnet34_distract",
    "--pair", "resnet34", "deeplab_ft"
)
Invoke-LoggedPython "worst-image overlays" @("scripts/overlay.py", "--config", "config.yaml", "--experiment", "augmented")
Invoke-LoggedPython "figures" @("scripts/make_figures.py", "--config", "config.yaml")

Write-PipelineLog "Training, evaluation, tables, overlays, and figures completed. Report narrative and final PDF/PPTX still need final review."
Set-Content -LiteralPath (Join-Path $logRoot "pipeline_complete.txt") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss") -Encoding UTF8
