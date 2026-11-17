param(
    [string]$Necks = "ca_sa,homography_256",   # neck-ablation variants; each needs multiscene_best_stage_1_abl_v2_<neck>.pth
    [int]$Epochs2 = 30,
    [int]$Epochs3 = 30,
    [double]$Lr = 5e-5,
    [int]$RouteCell = 10
)

# Proposed head on the neck-ablation fusions (doc/neck_ablation.md): each fusion's final ablation weights stand in for
# Stage 1 (training_scripts/make_stage1_from_ablation.py), then the current proposed recipe, identical to the legacy-neck
# lineage it is compared with:
#   Stage 2  joint unmasked genloss: dense MSE + sparse genloss (OT), neck + both experts trainable, lr 5e-5
#            (= run_remote_joint_genloss_unmasked.ps1, only --resume and --ckpt_tag differ)
#   Stage 3  performance router (--stage3_target performance) on that Stage 2's best checkpoint
# Everything runs sequentially in one detached chain (parallel runs on this PC were ~4x slower each) and stops at the
# first failure.

$workDir = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path "$workDir\logs")) { New-Item -ItemType Directory -Force -Path "$workDir\logs" | Out-Null }

$shared = "--frames_per_scene 10 --val_frames_per_scene 10 --precise_bn_frames 1 --skip_camera_validation " +
          "--no_router_contrast --route_cell $RouteCell"
$stage2Flags = "--stage2_unmasked --sparse_loss genloss --sparse_genloss_weight 1.0 --sparse_genloss_topk 1000 " +
               "--sparse_genloss_bg_weight 0.05 --sparse_genloss_count_weight 1.0 --sparse_nms_mult 1 --sparse_nms_kernel 7 " +
               "--stage2_select combined --dense_out_weight 0.7 --count_loss_weight 0.01 --lr $Lr"

$steps = @()
foreach ($neck in ($Necks -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
    $stage1 = "multiscene_best_stage_1_abl_v2_$neck.pth"
    if (-not (Test-Path "$workDir\$stage1")) {
        Write-Host "Missing $stage1 -- run: uv run python training_scripts/make_stage1_from_ablation.py --variant $neck" -ForegroundColor Red
        exit 1
    }
    $tag2 = "_abl_${neck}_unmasked_joint_genloss"
    $tag3 = "${tag2}_perf_c$RouteCell"
    $steps += "uv run python -u .\training_scripts\train_proposed_multiscene.py --stage 2 --resume $stage1 " +
              "--epochs $Epochs2 --patience2 15 $stage2Flags --ckpt_tag $tag2 $shared " +
              "> logs\stage2${tag2}_log.txt 2> logs\stage2${tag2}_error.txt"
    $steps += "uv run python -u .\training_scripts\train_proposed_multiscene.py --stage 3 " +
              "--resume multiscene_best_stage_2$tag2.pth --epochs $Epochs3 --patience3 15 --stage3_target performance " +
              "--ckpt_tag $tag3 $shared " +
              "> logs\stage3${tag3}_log.txt 2> logs\stage3${tag3}_error.txt"
}

$chain = ($steps | ForEach-Object { "($_)" }) -join " && "
$cmd = "conhost.exe --headless cmd.exe /c `"cd /d `"$workDir`" && $chain`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
if ($result.ReturnValue -eq 0) {
    Write-Host "Started $($steps.Count) steps for necks '$Necks' -- PID $($result.ProcessId)" -ForegroundColor Green
    Write-Host "  Logs: logs\stage2_abl_<neck>_unmasked_joint_genloss_log.txt, logs\stage3_abl_<neck>_unmasked_joint_genloss_perf_c${RouteCell}_log.txt"
    Write-Host "  Stop: taskkill /PID $($result.ProcessId) /T /F"
} else {
    Write-Host "Failed to start. Error code: $($result.ReturnValue)" -ForegroundColor Red
}
