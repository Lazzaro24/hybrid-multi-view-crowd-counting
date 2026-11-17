param(
    [string]$Neck = "homography_256",   # Stage 2 checkpoint: multiscene_best_stage_2_abl_<neck>_unmasked_joint_genloss_ep30.pth
    [int]$Kernel = 3,                   # NMS kernel of the peak-rendered sparse output
    [double]$Threshold = 0.064,
    [int]$RouteCell = 25,
    [int]$Epochs3 = 30
)

# Performance router on the sparse expert read as a detector (doc/head_ablation.md §6b): the frozen Stage 2 experts of
# the _perf_c10_sel lineage, but the sparse output is one map kernel per NMS peak (--sparse_render peaks) and the routing
# cell is 25 px. Stage 3 selection as the _sel runs (learned soft-routing count MAE, mean of the last 3 epochs), then
# the paper-protocol test evaluation of the selected checkpoint (the render setting is stored in it).
# Sequential, detached; stops at the first failure.

$workDir = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path "$workDir\logs")) { New-Item -ItemType Directory -Force -Path "$workDir\logs" | Out-Null }
$stage2 = "multiscene_best_stage_2_abl_${Neck}_unmasked_joint_genloss_ep30.pth"
if ($Neck -eq "ca_sa") { $stage2 = "multiscene_checkpoint_stage_2_abl_ca_sa_unmasked_joint_genloss_epoch_30.pth" }
if (-not (Test-Path "$workDir\$stage2")) { Write-Host "Missing $stage2" -ForegroundColor Red; exit 1 }

$tag = "_abl_${Neck}_unmasked_joint_genloss_ep30_perf_c${RouteCell}_peaks${Kernel}_sel"
$shared = "--frames_per_scene 10 --val_frames_per_scene 10 --precise_bn_frames 1 --skip_camera_validation " +
          "--no_router_contrast --route_cell $RouteCell"
# (built with += : inside @( ... ) the comma binds tighter than +, which glued both commands into one)
$steps = @()
$steps += "uv run python -u .\training_scripts\train_proposed_multiscene.py --stage 3 --resume $stage2 " +
          "--epochs $Epochs3 --patience3 15 --stage3_target performance --stage3_select soft_mae --select_window 3 " +
          "--sparse_render peaks --sparse_render_kernel $Kernel --sparse_render_threshold $Threshold " +
          "--ckpt_tag $tag $shared > logs\stage3${tag}_log.txt 2> logs\stage3${tag}_error.txt"
$steps += "uv run python -u .\evaluation\evaluate_paper_protocol.py --model proposed --weights multiscene_best_stage_3$tag.pth " +
          "> logs\eval_test${tag}_log.txt 2> logs\eval_test${tag}_error.txt"

$chain = ($steps | ForEach-Object { "($_)" }) -join " && "
$cmd = "conhost.exe --headless cmd.exe /c `"cd /d `"$workDir`" && $chain`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
if ($result.ReturnValue -eq 0) {
    Write-Host "Started Stage 3 + test eval for '$Neck' (tag $tag) -- PID $($result.ProcessId)" -ForegroundColor Green
    Write-Host "  Log:  Get-Content `"$workDir\logs\stage3${tag}_log.txt`" -Tail 30 -Wait"
    Write-Host "  Stop: taskkill /PID $($result.ProcessId) /T /F"
} else {
    Write-Host "Failed to start. Error code: $($result.ReturnValue)" -ForegroundColor Red
}
