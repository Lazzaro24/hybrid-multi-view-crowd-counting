param(
    [int]$WaitForPid = 0
)

# Fixed checkpoint selection (2026-10-10), applied to the three Stage 2 lineages compared in doc/neck_ablation.md:
# legacy neck (sa_ca), ablation ca_sa, ablation homography_256 -- all joint unmasked genloss Stage 2.
#   Stage 2 rule: --stage2_select experts (mean of dense-alone and sparse-alone count MAE) --select_window 3
#     The rule picks an already saved checkpoint for legacy (epoch 20) and ca_sa (epoch 30), so only the homography,
#     which the old patience rule stopped at epoch 18 while still improving, is continued to epoch 30 under it.
#     Its epoch-18 weights are copied in as the starting "best" (smoothed metric 70.12 over epochs 16-18).
#   Stage 3 rule: --stage3_select soft_mae (learned soft-routing count MAE) --select_window 3, performance router,
#     otherwise identical to the earlier Stage 3 runs.
# Then the paper-protocol test evaluation of the three Stage 3 best checkpoints. Sequential, stops at first failure.

$workDir = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path "$workDir\logs")) { New-Item -ItemType Directory -Force -Path "$workDir\logs" | Out-Null }
$py = "uv run python -u"
$shared = "--frames_per_scene 10 --val_frames_per_scene 10 --precise_bn_frames 1 --skip_camera_validation " +
          "--no_router_contrast --route_cell 10"
$stage2Flags = "--stage2_unmasked --sparse_loss genloss --sparse_genloss_weight 1.0 --sparse_genloss_topk 1000 " +
               "--sparse_genloss_bg_weight 0.05 --sparse_genloss_count_weight 1.0 --sparse_nms_mult 1 --sparse_nms_kernel 7 " +
               "--dense_out_weight 0.7 --count_loss_weight 0.01 --lr 5e-5 --stage2_select experts --select_window 3"
$stage3Flags = "--epochs 30 --patience3 15 --stage3_target performance --stage3_select soft_mae --select_window 3"

$homTag2 = "_abl_homography_256_unmasked_joint_genloss_ep30"
Copy-Item "$workDir\multiscene_final_stage_2_abl_homography_256_unmasked_joint_genloss.pth" `
          "$workDir\multiscene_best_stage_2$homTag2.pth" -Force

$stage3 = @(
    @{ from = "multiscene_best_stage_2$homTag2.pth"; tag = "${homTag2}_perf_c10_sel" },
    @{ from = "multiscene_checkpoint_stage_2_abl_ca_sa_unmasked_joint_genloss_epoch_30.pth";
       tag = "_abl_ca_sa_unmasked_joint_genloss_ep30_perf_c10_sel" },
    @{ from = "multiscene_checkpoint_stage_2_s8_dout0.7_cw0.01_f10_unmasked_joint_genloss_epoch_20.pth";
       tag = "_s8_dout0.7_cw0.01_f10_unmasked_joint_genloss_ep20_perf_c10_sel" }
)

$steps = @("$py .\training_scripts\train_proposed_multiscene.py --stage 2 " +
           "--resume multiscene_final_stage_2_abl_homography_256_unmasked_joint_genloss.pth --start_epoch 18 --epochs 30 " +
           "--patience2 15 --best_metric_so_far 70.1167 --seed_metric_history 80.15,65.3,64.9 $stage2Flags " +
           "--ckpt_tag $homTag2 $shared > logs\stage2${homTag2}_log.txt 2> logs\stage2${homTag2}_error.txt")
foreach ($s in $stage3) {
    $steps += "$py .\training_scripts\train_proposed_multiscene.py --stage 3 --resume $($s.from) $stage3Flags " +
              "--ckpt_tag $($s.tag) $shared > logs\stage3$($s.tag)_log.txt 2> logs\stage3$($s.tag)_error.txt"
}
foreach ($s in $stage3) {
    $steps += "$py .\evaluation\evaluate_paper_protocol.py --model proposed --weights multiscene_best_stage_3$($s.tag).pth " +
              "> logs\eval_test$($s.tag)_log.txt 2> logs\eval_test$($s.tag)_error.txt"
}

$chain = ($steps | ForEach-Object { "($_)" }) -join " && "
$wait = if ($WaitForPid -gt 0) { "powershell -NoProfile -Command while (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 } & " } else { "" }
$cmd = "conhost.exe --headless cmd.exe /c `"cd /d `"$workDir`" && $wait$chain`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
if ($result.ReturnValue -eq 0) {
    Write-Host "Started $($steps.Count) steps -- PID $($result.ProcessId)" -ForegroundColor Green
    Write-Host "  Stop: taskkill /PID $($result.ProcessId) /T /F"
} else {
    Write-Host "Failed to start. Error code: $($result.ReturnValue)" -ForegroundColor Red
}
