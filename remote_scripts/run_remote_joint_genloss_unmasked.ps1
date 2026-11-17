param(
    [int]$Epochs2 = 30,
    [double]$Lr = 5e-5,
    [int]$WaitForPid = 0   # >0: wait until this process (e.g. a running training chain) has exited before starting
)

# Specialised experts from the start (doc/oracle_mask_redesign.md §9): ONE Stage 2 run, both experts on the full
# patch (--stage2_unmasked), dense expert with MSE, sparse expert with genloss (OT) -- neck, dense and sparse all
# trainable together, instead of MSE-for-both followed by a sparse-only genloss fine-tune.
#
# The masked version of this was tried and killed (doc/sparse_ot_diagnose.md §15): the dense expert smeared mass
# over empty ground (bg 45-65x, leaked mass ~30x) while the OT gradient reshaped the shared neck. Unmasked, the dense
# MSE covers the whole patch, which penalises exactly that smear. Kept from that probe: lr 5e-5 and the finalised
# genloss calibration terms (sparse_ot_diagnose.md §13). Kill early if the §15 signature reappears (val-pedestal
# "dense bg" jumping to ~1, sparse F1 falling to ~0.1). Selection: per-cell routing upper bound.

$workDir = Split-Path $PSScriptRoot -Parent
$stage1 = "multiscene_best_stage_1_s8_dout0.7_cw0.01_f10_a0.7.pth"
$tag = "_s8_dout0.7_cw0.01_f10_unmasked_joint_genloss"

if (-not (Test-Path "$workDir\logs")) {
    New-Item -ItemType Directory -Force -Path "$workDir\logs" | Out-Null
}

$flags = "--frames_per_scene 10 --val_frames_per_scene 10 --precise_bn_frames 1 --skip_camera_validation " +
         "--no_router_contrast --route_cell 10 --stage2_unmasked " +
         "--sparse_loss genloss --sparse_genloss_weight 1.0 --sparse_genloss_topk 1000 " +
         "--sparse_genloss_bg_weight 0.05 --sparse_genloss_count_weight 1.0 --sparse_nms_mult 1 --sparse_nms_kernel 7 " +
         "--stage2_select combined --dense_out_weight 0.7 --count_loss_weight 0.01 --lr $Lr --ckpt_tag $tag"
$run = "uv run python -u .\training_scripts\train_proposed_multiscene.py --stage 2 " +
       "--resume $stage1 --epochs $Epochs2 --patience2 15 $flags " +
       "> logs\stage2${tag}_log.txt 2> logs\stage2${tag}_error.txt"

if ($WaitForPid -gt 0) {
    # the wait happens inside the detached job, so this script returns immediately
    $wait = "powershell -NoProfile -Command while (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 } & "
} else {
    $wait = ""
}
$cmd = "conhost.exe --headless cmd.exe /c `"cd /d `"$workDir`" && $wait($run)`""

$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
if ($result.ReturnValue -eq 0) {
    $when = if ($WaitForPid -gt 0) { "after PID $WaitForPid exits" } else { "now" }
    Write-Host "Joint unmasked genloss Stage 2 ($tag) starts $when -- PID $($result.ProcessId)" -ForegroundColor Green
    Write-Host "  Log:  Get-Content `"$workDir\logs\stage2${tag}_log.txt`" -Tail 30 -Wait"
    Write-Host "  Stop: taskkill /PID $($result.ProcessId) /T /F"
} else {
    Write-Host "Failed to start. Error code: $($result.ReturnValue)" -ForegroundColor Red
}
