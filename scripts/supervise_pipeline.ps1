param(
    [int]$RestartLimit = 3
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot
$managerScript = Join-Path $projectRoot "scripts\monitor_and_finish.ps1"
$keepAwakeScript = Join-Path $projectRoot "scripts\keep_awake_until_pipeline_finishes.ps1"
$finalizerScript = Join-Path $projectRoot "scripts\finalize_report_after_pipeline.ps1"
$powershellExe = Join-Path $PSHOME "powershell.exe"
$logRoot = Join-Path $projectRoot "artifacts\logs"
$completeMarker = Join-Path $logRoot "pipeline_complete.txt"
$supervisorLog = Join-Path $logRoot "pipeline_supervisor.log"
$restartCount = 0

function Write-SupervisorLog([string]$Message) {
    Add-Content -LiteralPath $supervisorLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Message" -Encoding UTF8
}

function Find-Manager {
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'monitor_and_finish\.ps1' } |
        Select-Object -First 1
}

function Find-Helper([string]$ScriptName) {
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match [regex]::Escape($ScriptName) } |
        Select-Object -First 1
}

function Ensure-Helper([string]$ScriptPath, [int]$ActiveManagerPid) {
    $scriptName = Split-Path -Leaf $ScriptPath
    if (Find-Helper $scriptName) { return }

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $label = [IO.Path]::GetFileNameWithoutExtension($scriptName)
    $stdout = Join-Path $logRoot "${label}_${stamp}.out.log"
    $stderr = Join-Path $logRoot "${label}_${stamp}.err.log"
    $process = Start-Process -FilePath $powershellExe `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"' + $ScriptPath + '"'), "-ManagerPid", [string]$ActiveManagerPid) `
        -WorkingDirectory $projectRoot -WindowStyle Hidden `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Write-SupervisorLog "Started $scriptName PID $($process.Id) for manager PID $ActiveManagerPid."
}

while (-not (Test-Path -LiteralPath $completeMarker)) {
    $manager = Find-Manager
    if (-not $manager) {
        if ($restartCount -ge $RestartLimit) {
            Write-SupervisorLog "Manager stopped without completion after $restartCount restarts; stopping supervisor."
            exit 1
        }

        $restartCount++
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $stdout = Join-Path $logRoot "pipeline_manager_restart_${stamp}.out.log"
        $stderr = Join-Path $logRoot "pipeline_manager_restart_${stamp}.err.log"
        $process = Start-Process -FilePath (Join-Path $PSHOME "powershell.exe") `
            -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"' + $managerScript + '"')) `
            -WorkingDirectory $projectRoot -WindowStyle Hidden `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        Write-SupervisorLog "Started replacement manager PID $($process.Id), restart $restartCount/$RestartLimit."
    }

    $manager = Find-Manager
    if ($manager) {
        Ensure-Helper $keepAwakeScript ([int]$manager.ProcessId)
        Ensure-Helper $finalizerScript ([int]$manager.ProcessId)
    }
    Start-Sleep -Seconds 30
}

Write-SupervisorLog "Pipeline completion marker found; supervisor stopping."
