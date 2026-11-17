# Launches the paper-protocol evaluation (evaluation\evaluate_paper_protocol.py) as a detached process.
# Usage (from anywhere):
#   .\remote_scripts\run_remote_eval_paper_protocol.ps1 -Weights runs\baseline\run_1\best.pth
param(
    [string]$Weights = "runs\baseline\run_1\best.pth",
    [ValidateSet("baseline", "proposed")]
    [string]$Model = "baseline",
    [ValidateSet("train", "validate", "test")]
    [string]$Split = "test"
)

# Use the parent of this script's directory (src\) as the working directory
$workDir = Split-Path $PSScriptRoot -Parent

$tag = "${Model}_${Split}_" + [System.IO.Path]::GetFileNameWithoutExtension($Weights)
$logFile = "logs\eval_paper_${tag}_log.txt"
$errFile = "logs\eval_paper_${tag}_error.txt"

# Fail here instead of inside a detached process nobody is watching
if (-not (Test-Path (Join-Path $workDir $Weights))) {
    Write-Host "Weights not found: $workDir\$Weights" -ForegroundColor Red
    exit 1
}

$extraArgs = "--model $Model --weights $Weights --split $Split"

Write-Host "Starting detached paper-protocol evaluation..." -ForegroundColor Cyan
Write-Host "Working Directory: $workDir"
Write-Host "Arguments: $extraArgs"

New-Item -ItemType Directory -Force -Path (Join-Path $workDir "logs") | Out-Null

# conhost --headless: keeps the job alive when Windows Terminal restarts (e.g. after a WindowsAppRuntime update)
$cmd = "conhost.exe --headless cmd.exe /c cd /d `"$workDir`" && uv run python .\evaluation\evaluate_paper_protocol.py $extraArgs > $logFile 2> $errFile"
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}

if ($result.ReturnValue -eq 0) {
    $jobPid = $result.ProcessId
    Write-Host "Successfully started process!" -ForegroundColor Green
    Write-Host "Process ID (PID): $jobPid" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You can now safely close your SSH connection or terminal. The job will keep running."
    Write-Host "------------------------------------------------------"
    Write-Host "Check if running: Get-Process -Id $jobPid"
    Write-Host "View live logs:   Get-Content `"$workDir\$logFile`" -Tail 20 -Wait"
    Write-Host "View errors:      Get-Content `"$workDir\$errFile`" -Tail 20 -Wait"
    Write-Host "Stop evaluation:  Stop-Process -Id $jobPid -Force"
    Write-Host "Results JSON:     next to the weights, eval_${Split}_<weights name>.json"
    Write-Host "------------------------------------------------------"
} else {
    Write-Host "Failed to start process. Error code: $($result.ReturnValue)" -ForegroundColor Red
}
