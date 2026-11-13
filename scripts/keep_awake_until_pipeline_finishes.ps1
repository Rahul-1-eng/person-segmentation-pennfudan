param(
    [Parameter(Mandatory = $true)]
    [int]$ManagerPid
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$completeMarker = Join-Path $projectRoot "artifacts\logs\pipeline_complete.txt"
$pipelineLog = Join-Path $projectRoot "artifacts\logs\pipeline.log"
$powerLog = Join-Path $projectRoot "artifacts\logs\keep_awake.log"

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class PipelinePowerRequest {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$continuous = [Convert]::ToUInt32("80000000", 16)
$systemRequired = [uint32]0x00000001
$keepSystemAwake = [Convert]::ToUInt32("80000001", 16)

try {
    Add-Content -LiteralPath $powerLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  Power request helper started for manager PID $ManagerPid."
    while ($true) {
        if (Test-Path -LiteralPath $completeMarker) { break }
        $manager = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -match 'monitor_and_finish\.ps1' } |
            Select-Object -First 1
        $activeTrains = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -match 'person_segmentation\.train.*--experiment' })
        if (-not $manager -and $activeTrains.Count -eq 0) { break }
        $recentLog = if (Test-Path -LiteralPath $pipelineLog) { (Get-Content -LiteralPath $pipelineLog -Tail 5 | Out-String) } else { "" }
        if ($recentLog -match "(?i)\bFAILED\b|Training failed|Post-processing failed") { break }
        [void][PipelinePowerRequest]::SetThreadExecutionState($keepSystemAwake)
        Start-Sleep -Seconds 45
    }
    Add-Content -LiteralPath $powerLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  Power request helper stopping."
}
catch {
    Add-Content -LiteralPath $powerLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  Power request error: $_"
    throw
}
finally {
    [void][PipelinePowerRequest]::SetThreadExecutionState($continuous)
}
