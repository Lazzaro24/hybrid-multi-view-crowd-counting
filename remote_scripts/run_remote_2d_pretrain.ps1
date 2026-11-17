# Use the directory where this script is located as the working directory
$workDir = Split-Path $PSScriptRoot -Parent

$logFile = "logs\pretrain_2d_log.txt"
$errFile = "logs\pretrain_2d_error.txt"

# Pretraining parameters (checkpoints land in runs\train_2d\<timestamp>\resnet18_2d_pretrained_epX.pth; the epoch with
# the lowest val MAE is also copied to best.pth there)
$epochs = 15
$lr = 1e-4
# To resume, set e.g. $resume = "runs\train_2d\<timestamp>\resnet18_2d_pretrained_ep7.pth"
$resume = ""

$extraArgs = "--epochs $epochs --lr $lr"
if ($resume -ne "") { $extraArgs += " --resume $resume" }

Write-Host "Starting detached 2D pretraining job..." -ForegroundColor Cyan
Write-Host "Working Directory: $workDir"
Write-Host "Arguments: $extraArgs"

# Make sure the log directory exists, otherwise the redirection fails silently
New-Item -ItemType Directory -Force -Path (Join-Path $workDir "logs") | Out-Null

# Construct the command
$cmd = "cmd.exe /c cd /d `"$workDir`" && uv run python .\training_scripts\train_2d_pretrain.py $extraArgs > $logFile 2> $errFile"

# Spawn the detached process
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}

if ($result.ReturnValue -eq 0) {
    $jobPid = $result.ProcessId
    Write-Host "Successfully started process!" -ForegroundColor Green
    Write-Host "Process ID (PID): $jobPid" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You can now safely close your SSH connection or terminal. The job will keep running."
    Write-Host "------------------------------------------------------"
    Write-Host "Quick Command Reference (Copy/Paste these if needed):"
    Write-Host "Check if running: Get-Process -Id $jobPid"
    Write-Host "View live logs:   Get-Content `"$workDir\$logFile`" -Tail 20 -Wait"
    Write-Host "View errors:      Get-Content `"$workDir\$errFile`" -Tail 20 -Wait"
    Write-Host "Stop training:    Stop-Process -Id $jobPid -Force"
    Write-Host "Checkpoints:      Get-ChildItem `"$workDir\runs\train_2d`" | Sort-Object LastWriteTime | Select-Object -Last 1"
    Write-Host "------------------------------------------------------"
} else {
    Write-Host "Failed to start process. Error code: $($result.ReturnValue)" -ForegroundColor Red
}
