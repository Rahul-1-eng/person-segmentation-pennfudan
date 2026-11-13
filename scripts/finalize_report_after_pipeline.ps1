param(
    [Parameter(Mandatory = $true)]
    [int]$ManagerPid
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$logRoot = Join-Path $projectRoot "artifacts\logs"
$statusPath = Join-Path $logRoot "report_finalizer.log"
$completeMarker = Join-Path $logRoot "pipeline_complete.txt"
$pipelineLog = Join-Path $logRoot "pipeline.log"
$skillDir = "C:\Users\Rahul Kumar Sahoo\.codex\plugins\cache\openai-primary-runtime\pdf\26.1015.12040\skills\pdf"

function Write-Status([string]$Message) {
    Add-Content -LiteralPath $statusPath -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Message" -Encoding UTF8
}

while (-not (Test-Path -LiteralPath $completeMarker)) {
    $manager = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'monitor_and_finish\.ps1' } |
        Select-Object -First 1
    $activeTrains = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'person_segmentation\.train.*--experiment' })
    if (-not $manager -and $activeTrains.Count -eq 0) {
        Write-Status "Pipeline manager exited and no training process remains before the completion marker; report PDF was not rebuilt."
        exit 1
    }
    $recentLog = if (Test-Path -LiteralPath $pipelineLog) { (Get-Content -LiteralPath $pipelineLog -Tail 8 | Out-String) } else { "" }
    if ($recentLog -match "(?i)\bFAILED\b|Training failed|Post-processing failed") {
        Write-Status "Pipeline logged a failure; waiting for the manager to stop before deciding whether report inputs are complete."
    }
    Start-Sleep -Seconds 45
}

try {
    Write-Status "Finalizing report narrative from generated results."
    & $pythonExe scripts/finalize_report.py 2>&1 | Tee-Object -FilePath $statusPath -Append
    if ($LASTEXITCODE -ne 0) { throw "Report narrative finalization failed with exit code $LASTEXITCODE." }

    Push-Location $skillDir
    try {
        & node container_tools/mark_artifact_operation_started.mjs --operation-kind edit --expected-output-count 1 --output-format pdf
        if ($LASTEXITCODE -ne 0) { throw "PDF edit marker failed with exit code $LASTEXITCODE." }
    }
    finally {
        Pop-Location
    }

    Write-Status "Building the final PDF with generated figures and tables."
    $buildStarted = Get-Date
    $buildOutput = & $pythonExe scripts/create_report_pdf.py 2>&1
    $buildOutput | Tee-Object -FilePath $statusPath -Append
    if ($LASTEXITCODE -ne 0) { throw "PDF build failed with exit code $LASTEXITCODE." }
    $printedPdf = $buildOutput | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ -match '(?i)\.pdf\s*$' } | Select-Object -Last 1
    if (-not $printedPdf) { throw "PDF builder did not report the output path." }
    $pdf = $printedPdf.Trim()
    if (-not (Test-Path -LiteralPath $pdf)) { throw "Reported PDF was not created: $pdf" }
    if ((Get-Item -LiteralPath $pdf).LastWriteTime -lt $buildStarted) { throw "Reported PDF is stale: $pdf" }
    Write-Status "Final report PDF built at $pdf."
}
catch {
    Write-Status "ERROR: $_"
    exit 1
}
