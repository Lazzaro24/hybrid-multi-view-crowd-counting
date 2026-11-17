param(
    [int]$Epochs1 = 30,
    [int]$Epochs2 = 30,
    [int]$Epochs3 = 30,
    [double]$TverskyAlpha = 0.7,
    [ValidateSet(1, 2, 3)][int]$StartStage = 1   # 2/3: resume the chain from the previous stage's best checkpoint
)

# Stage 1 -> 2 -> 3 on the new stride-8 2D-pretrained backbone (runs/train_2d/new_best/
# resnet18_2d_pretrained_ep4.pth, the train_proposed_multiscene.py default), fully frozen as in the
# frozen-backbone baseline. Same settings as the stride-4 reference lineage:
#   Stage 2: dense_out_weight=0.7, count_loss_weight=0.01, frames_per_scene=10 (doc/multi_training_record.md #11)
#   Stage 3: tversky_alpha=0.7 (doc/router_diagnosis.md), router WITHOUT the contrast branch
#            (--no_router_contrast; the contrast branch was a negative result). Every stage gets the flag so
#            the gating_net shape stays the same along the lineage.
# One ckpt_tag for the whole lineage; each stage writes its own log/error pair; the chain stops at the
# first failure.

$workDir = Split-Path $PSScriptRoot -Parent
$tag = "_s8_dout0.7_cw0.01_f10_a$TverskyAlpha"

Write-Host "Starting detached stride-8 Stage 1 -> 2 -> 3 pipeline..." -ForegroundColor Cyan
Write-Host "Working Directory: $workDir"
Write-Host "Lineage tag: $tag | epochs $Epochs1 / $Epochs2 / $Epochs3"

if (-not (Test-Path "$workDir\logs")) {
    New-Item -ItemType Directory -Force -Path "$workDir\logs" | Out-Null
}

$sharedFlags = "--frames_per_scene 10 --val_frames_per_scene 10 --precise_bn_frames 1 --skip_camera_validation " +
               "--no_router_contrast --ckpt_tag $tag"

$cmd1 = "uv run python -u .\training_scripts\train_proposed_multiscene.py --stage 1 " +
        "--epochs $Epochs1 --patience1 15 $sharedFlags " +
        "> logs\stage1${tag}_log.txt 2> logs\stage1${tag}_error.txt"

$cmd2 = "uv run python -u .\training_scripts\train_proposed_multiscene.py --stage 2 " +
        "--resume multiscene_best_stage_1$tag.pth --epochs $Epochs2 --patience2 15 " +
        "--dense_out_weight 0.7 --count_loss_weight 0.01 $sharedFlags " +
        "> logs\stage2${tag}_log.txt 2> logs\stage2${tag}_error.txt"

$cmd3 = "uv run python -u .\training_scripts\train_proposed_multiscene.py --stage 3 " +
        "--resume multiscene_best_stage_2$tag.pth --epochs $Epochs3 --patience3 15 " +
        "--tversky_alpha $TverskyAlpha $sharedFlags " +
        "> logs\stage3${tag}_log.txt 2> logs\stage3${tag}_error.txt"

$runChain = @("($cmd1)", "($cmd2)", "($cmd3)")[($StartStage - 1)..2] -join " && "
Write-Host "Starting at Stage $StartStage"
# conhost --headless: the console must not live in Windows Terminal -- a WindowsAppRuntime update restarted
# Terminal on 2026-10-02 08:15 and the job received Ctrl+C. (System sleep still kills the run; keep the PC awake.)
$cmd = "conhost.exe --headless cmd.exe /c `"cd /d `"$workDir`" && $runChain`""

$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}

if ($result.ReturnValue -eq 0) {
    $jobPid = $result.ProcessId
    Write-Host "Successfully started stride-8 Stage 1->2->3 chain!" -ForegroundColor Green
    Write-Host "Process ID (PID): $jobPid" -ForegroundColor Yellow
    Write-Host "------------------------------------------------------"
    Write-Host "Check if running:  Get-Process -Id $jobPid"
    Write-Host "Stage 1 log:       Get-Content `"$workDir\logs\stage1${tag}_log.txt`" -Tail 30 -Wait"
    Write-Host "Stage 2 log:       Get-Content `"$workDir\logs\stage2${tag}_log.txt`" -Tail 30 -Wait"
    Write-Host "Stage 3 log:       Get-Content `"$workDir\logs\stage3${tag}_log.txt`" -Tail 30 -Wait"
    Write-Host "Stop:              Stop-Process -Id $jobPid -Force"
    Write-Host "------------------------------------------------------"
} else {
    Write-Host "Failed to start process. Error code: $($result.ReturnValue)" -ForegroundColor Red
}
