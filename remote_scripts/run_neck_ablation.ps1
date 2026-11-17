param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$Variants = "homography_256,ca,sa_ca,ca_sa,bev3",
    [string]$ExtraArgs = "",
    [int]$MaxAttempts = 10
)

# Detached neck ablation (training_scripts/neck_ablation.py) that restarts itself after a crash.
# A driver-level "CUDA error: out of memory" leaves the CUDA context unusable, so the process has to exit; every
# restart resumes the interrupted variant from its last.pth (only the current epoch is lost) and skips finished ones.
# Keep the PC awake (system sleep kills the run).

$workDir = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path "$workDir\logs")) { New-Item -ItemType Directory -Force -Path "$workDir\logs" | Out-Null }
$log = "logs\neck_ablation_${RunName}_log.txt"
$err = "logs\neck_ablation_${RunName}_error.txt"

$py = "uv run python -u .\training_scripts\neck_ablation.py --run_name $RunName --variants $Variants $ExtraArgs >> $log 2>> $err"
# cmd loop: rerun until exit code 0, at most $MaxAttempts times, 60 s pause between attempts
$loop = "for /L %i in (1,1,$MaxAttempts) do @(echo [attempt %i] %date% %time% >> $log & $py && exit /b 0 & echo [attempt %i failed] >> $log & timeout /t 60 /nobreak > nul)"
# conhost --headless: keep the job out of Windows Terminal (a Terminal restart sends Ctrl+C to its children)
$cmd = "conhost.exe --headless cmd.exe /c `"cd /d `"$workDir`" && $loop`""

$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd}
if ($result.ReturnValue -eq 0) {
    Write-Host "Started neck ablation '$RunName' ($Variants), up to $MaxAttempts attempts. PID: $($result.ProcessId)" -ForegroundColor Green
    Write-Host "Log:      Get-Content `"$workDir\$log`" -Tail 20 -Wait"
    Write-Host "Progress: Get-Content `"$workDir\$err`" -Tail 1"
    Write-Host "Summary:  uv run python training_scripts/neck_ablation.py --run_name $RunName --summarize"
    Write-Host "Stop:     Stop-Process -Id $($result.ProcessId) -Force   (then stop the python.exe processes it started)"
} else {
    Write-Host "Failed to start process. Error code: $($result.ReturnValue)" -ForegroundColor Red
}
