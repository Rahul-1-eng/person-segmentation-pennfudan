$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot
$staging = Join-Path $projectRoot "artifacts\submission_staging"
$outputZip = Join-Path $projectRoot "MLproject_person_segmentation.zip"
$tempZip = Join-Path $projectRoot "MLproject_person_segmentation.tmp.zip"
$pipelineDone = Join-Path $projectRoot "artifacts\logs\pipeline_complete.txt"
$finalizerLog = Join-Path $projectRoot "artifacts\logs\report_finalizer.log"

if (-not (Test-Path -LiteralPath $pipelineDone)) { throw "The training and evaluation pipeline is not complete." }
if (-not (Test-Path -LiteralPath $finalizerLog)) { throw "The report finalizer log is missing." }

$reportLine = Get-Content -LiteralPath $finalizerLog | Where-Object { $_ -match "Final report PDF built at " } | Select-Object -Last 1
if (-not $reportLine) { throw "The finalizer has not confirmed a fresh report PDF." }
$reportPdf = ($reportLine -replace '^.*Final report PDF built at ', '').Trim()
if (-not (Test-Path -LiteralPath $reportPdf)) { throw "The finalized PDF does not exist: $reportPdf" }

$presentation = Join-Path $projectRoot "Person_Segmentation_Viva_Presentation.pptx"
if (-not (Test-Path -LiteralPath $presentation)) { throw "The final presentation is missing." }
if ((Get-Item -LiteralPath $presentation).LastWriteTime -lt (Get-Item -LiteralPath $reportPdf).LastWriteTime) {
    throw "The presentation predates the final report. Rebuild and review the deck before packaging."
}

$rootFiles = @(
    "README.md", "report.md", "slides.md", "presenter_notes.md", "STUDY_GUIDE.md",
    "submission_checklist.md", "config.yaml", "requirements.txt", "Person_Segmentation_Full_Walkthrough.ipynb",
    "Person_Segmentation_Viva_Presentation.pptx"
)
$codeFolders = @("scripts", "src")
$artifactFiles = @(
    "artifacts\.gitkeep", "artifacts\dataset_audit.csv", "artifacts\environment.txt", "artifacts\splits.json",
    "artifacts\baselines_summary.csv", "artifacts\results_summary.csv", "artifacts\results_summary.json",
    "artifacts\results_table.csv", "artifacts\results_table.md", "artifacts\paired_comparison.csv",
    "artifacts\paired_comparison.md", "artifacts\robustness_metrics.csv", "artifacts\metric_comparison.png",
    "artifacts\overlays\augmented_worst.png"
)

if (Test-Path -LiteralPath $staging) {
    $resolvedStaging = (Resolve-Path -LiteralPath $staging).Path
    if ($resolvedStaging -ne $staging) { throw "Refusing to clear unexpected staging path: $resolvedStaging" }
    Remove-Item -LiteralPath $resolvedStaging -Recurse -Force
}
New-Item -ItemType Directory -Path $staging -Force | Out-Null

foreach ($relative in $rootFiles) {
    $source = Join-Path $projectRoot $relative
    if (-not (Test-Path -LiteralPath $source)) { throw "Required submission file missing: $relative" }
    Copy-Item -LiteralPath $source -Destination (Join-Path $staging $relative) -Force
}
Copy-Item -LiteralPath $reportPdf -Destination (Join-Path $staging "Person_Segmentation_Report.pdf") -Force

foreach ($folder in $codeFolders) {
    Copy-Item -LiteralPath (Join-Path $projectRoot $folder) -Destination (Join-Path $staging $folder) -Recurse -Force
}
Get-ChildItem -LiteralPath $staging -Recurse -Directory -Filter "__pycache__" | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Recurse -Force
}
Get-ChildItem -LiteralPath $staging -Recurse -File -Filter "*.pyc" | Remove-Item -Force
foreach ($relative in $artifactFiles) {
    $source = Join-Path $projectRoot $relative
    if (Test-Path -LiteralPath $source) {
        $destination = Join-Path $staging $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }
}

foreach ($directory in @("artifacts\figures", "artifacts\per_image", "artifacts\v1")) {
    $source = Join-Path $projectRoot $directory
    if (Test-Path -LiteralPath $source) {
        $destination = Join-Path $staging $directory
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        Get-ChildItem -LiteralPath $source -Recurse -File | Where-Object {
            $_.Extension -in @(".png", ".csv", ".json") -and $_.FullName -notmatch "\\predictions\\|\\__pycache__\\"
        } | ForEach-Object {
            $relative = $_.FullName.Substring($projectRoot.TrimEnd("\").Length + 1)
            $target = Join-Path $staging $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $target -Force
        }
    }
}

Get-ChildItem -LiteralPath (Join-Path $projectRoot "artifacts") -Directory | ForEach-Object {
    $summary = Join-Path $_.FullName "summary.json"
    $history = Join-Path $_.FullName "history.csv"
    $progress = Join-Path $_.FullName "progress.json"
    if ((Test-Path -LiteralPath $summary) -and (Test-Path -LiteralPath $history)) {
        $targetDir = Join-Path $staging (Join-Path "artifacts" $_.Name)
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        Copy-Item -LiteralPath $summary -Destination $targetDir -Force
        Copy-Item -LiteralPath $history -Destination $targetDir -Force
        if (Test-Path -LiteralPath $progress) { Copy-Item -LiteralPath $progress -Destination $targetDir -Force }
    }
}

@(
    "This package contains completed run summaries and histories, evaluation CSV/JSON files, and figures.",
    "Model checkpoints, prediction image dumps, the dataset itself, and run logs are intentionally excluded."
) | Set-Content -LiteralPath (Join-Path $staging "PACKAGE_CONTENTS.txt") -Encoding UTF8

if (Test-Path -LiteralPath $tempZip) { Remove-Item -LiteralPath $tempZip -Force }
Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $tempZip -CompressionLevel Optimal

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($tempZip)
try {
    $entries = @($archive.Entries | ForEach-Object FullName)
    if ($entries | Where-Object { $_ -match '(?i)\.pt$|__pycache__|\.pyc$' }) {
        throw "Archive unexpectedly contains checkpoints or Python cache files."
    }
    foreach ($required in @("Person_Segmentation_Report.pdf", "Person_Segmentation_Viva_Presentation.pptx", "Person_Segmentation_Full_Walkthrough.ipynb", "report.md", "artifacts/results_table.csv", "artifacts/metric_comparison.png")) {
        if ($required -notin $entries) { throw "Archive is missing required output: $required" }
    }
}
finally {
    $archive.Dispose()
}

Move-Item -LiteralPath $tempZip -Destination $outputZip -Force
Write-Output "Created $outputZip without model checkpoints."
