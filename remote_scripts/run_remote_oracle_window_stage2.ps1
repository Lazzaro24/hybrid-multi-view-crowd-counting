param(
    [int]$Epochs2 = 30,
    [int]$Window = 25,
    [double[]]$Densities = @(0.5, 0.4),
    [int]$MinArea = 50
)

# Oracle-mask redesign ablation (doc: diagnostics/oracle_mask_extent/, analyse_oracle_mask_extent.py): Stage 2 with
# the window-density oracle mask -- dense where the mean density over a $Window px (= dense expert RF, 5 m) window
# exceeds D persons/m^2, components < $MinArea px dropped -- instead of the pointwise 1.27-peak threshold.
# Identical to the reference lineage's Stage 2 (run_remote_stage123_stride8.ps1, tag _s8_dout0.7_cw0.01_f10_a0.7)
# in every other flag, resuming from the same Stage 1 checkpoint, so the mask is the only difference.
# Densities run SEQUENTIALLY in one detached chain: two runs in parallel on the one GPU/CPU were ~4x slower each
# (20 vs 5 min/epoch, validation 10x slower -- CPU thread contention), i.e. slower than running them back to back.

$workDir = Split-Path $PSScriptRoot -Parent
$stage1 = "multiscene_best_stage_1_s8_dout0.7_cw0.01_f10_a0.7.pth"

if (-not (Test-Path "$workDir\logs")) {
    New-Item -ItemType Directory -Force -Path "$workDir\logs" | Out-Null
}

$runs = @()
$tags = @()
foreach ($d in $Densities) {
    $tag = "_s8_dout0.7_cw0.01_f10_W${Window}D${d}_ma$MinArea"
    $flags = "--frames_per_scene 10 --val_frames_per_scene 10 --precise_bn_frames 1 --skip_camera_validation " +
             "--no_router_contrast --ckpt_tag $tag " +
             "--oracle_density_window $Window --oracle_min_density $d --oracle_min_area $MinArea"
    $runs += "(uv run python -u .\training_scripts\train_proposed_multiscene.py --stage 2 " +
             "--resume $stage1 --epochs $Epochs2 --patience2 15 " +
             "--dense_out_weight 0.7 --count_loss_weight 0.01 $flags " +
             "> logs\stage2${tag}_log.txt 2> logs\stage2${tag}_error.txt)"
    $tags += $tag
}
# " & " (not " && "): the next density still runs if one fails
$chain = $runs -join " & "
# conhost --headless: keep the console out of Windows Terminal (see run_remote_stage123_stride8.ps1)
$cmd = "conhost.exe --headless cmd.exe /c `"cd /d `"$workDir`" && $chain`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
if ($result.ReturnValue -eq 0) {
    Write-Host "Started sequential Stage 2 chain -- PID $($result.ProcessId)" -ForegroundColor Green
    foreach ($tag in $tags) {
        Write-Host "  Log:  Get-Content `"$workDir\logs\stage2${tag}_log.txt`" -Tail 30 -Wait"
    }
    Write-Host "  Stop: taskkill /PID $($result.ProcessId) /T /F"
} else {
    Write-Host "Failed to start. Error code: $($result.ReturnValue)" -ForegroundColor Red
}
