param(
    [int]$Epochs = 30,
    [int]$Patience = 15,
    [ValidateSet(1, 2, 3)][int]$StartStep = 1,   # resume the chain at a later step (2 = evals only, 3 = paper protocol only)
    [switch]$Foreground                          # run in this console instead of a detached process
)

# Dense-capacity control (doc/head_ablation.md §8): does the proposed head's 64-channel bottleneck limit the count?
# Same recipe as the narrow reference (Stage 2 of the homography lineage, _abl_homography_256_unmasked_joint_genloss:
# run_remote_ablation_neck_routing.ps1), with two changes:
#   --neck_adapter none    drop the head_processor (256 -> 64 ch); the head reads the 256-ch homography_256 fusion
#   --dense_arch wide512   dense expert = the SynMVCrowd baseline decoder's stack (3x3 convs 256->512, 512->512 + ReLU,
#                          512->512 + ReLU, 512->1; ~5.9 M parameters instead of ~0.1 M), fresh init
# Selection on the dense expert alone (--stage2_select dense), no moving window. Then the dense expert alone is scored on
# the map-quality frames (K-means cameras, 50 frames per held-out scene: compare proposed__hom_dense.csv = 83.0 test and
# the baseline-head ablation run = 72.1) and with the paper protocol (compare dense alone 83.8, baseline run_1 67.0).
#
# Needs on the machine (src/ as working directory): this branch's code, multiscene_best_stage_1_abl_v2_homography_256.pth,
# the SynMVCrowd data with master_refs.pt, datasets/SynMVCrowd/optimal_cameras_filtered.json, utils/z_sign_corrections_full.json
# and the excluded-frames file. Steps are sequential and stop at the first failure.
# Bring back: multiscene_best_stage_2$tag.pth (+ final / epoch checkpoints), logs\stage2$tag_log.txt, logs\eval_*$tag*,
# diagnostics\map_quality\proposed__hom_wide512_dense.csv and eval_test_multiscene_best_stage_2${tag}_dense.json.
# Do NOT copy back diagnostics\map_quality\summary.json: here it only contains this one model.

$workDir = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path "$workDir\logs")) { New-Item -ItemType Directory -Force -Path "$workDir\logs" | Out-Null }
$stage1 = "multiscene_best_stage_1_abl_v2_homography_256.pth"
if ($StartStep -eq 1 -and -not (Test-Path "$workDir\$stage1")) {
    Write-Host "Missing $stage1" -ForegroundColor Red
    Write-Host "On a machine without the local generated files use run_remote_dense_wide512_full.ps1 (rebuilds the" -ForegroundColor Yellow
    Write-Host "reference cache, the fusions and Stage 1 itself)." -ForegroundColor Yellow
    exit 1
}

$tag = "_abl_homography_256_wide512_unmasked_joint_genloss"
$weights = "multiscene_best_stage_2$tag.pth"
$shared = "--frames_per_scene 10 --val_frames_per_scene 10 --precise_bn_frames 1 --skip_camera_validation " +
          "--no_router_contrast --route_cell 10"
$stage2Flags = "--stage2_unmasked --sparse_loss genloss --sparse_genloss_weight 1.0 --sparse_genloss_topk 1000 " +
               "--sparse_genloss_bg_weight 0.05 --sparse_genloss_count_weight 1.0 --sparse_nms_mult 1 --sparse_nms_kernel 7 " +
               "--dense_out_weight 0.7 --count_loss_weight 0.01 --lr 5e-5"

$steps = @()
$steps += "uv run python -u .\training_scripts\train_proposed_multiscene.py --stage 2 --resume $stage1 " +
          "--neck_adapter none --dense_arch wide512 --stage2_select dense --epochs $Epochs --patience2 $Patience " +
          "$stage2Flags --ckpt_tag $tag $shared > logs\stage2${tag}_log.txt 2> logs\stage2${tag}_error.txt"
$steps += "uv run python -u .\evaluation\eval_map_quality.py --models= --baselines= " +
          "--proposed hom_wide512_dense=${weights}:dense > logs\eval_map_quality${tag}_log.txt 2>&1"
$steps += "uv run python -u .\evaluation\evaluate_paper_protocol.py --model proposed --routing dense --weights $weights " +
          "--split test > logs\eval_test${tag}_dense_log.txt 2> logs\eval_test${tag}_dense_error.txt"
$steps = $steps[($StartStep - 1)..($steps.Count - 1)]

if ($Foreground) {
    Set-Location $workDir
    foreach ($s in $steps) {
        Write-Host ">> $s" -ForegroundColor Cyan
        cmd.exe /c $s
        if ($LASTEXITCODE -ne 0) { Write-Host "Step failed (exit $LASTEXITCODE)" -ForegroundColor Red; exit $LASTEXITCODE }
    }
    exit 0
}
$chain = ($steps | ForEach-Object { "($_)" }) -join " && "
$cmd = "conhost.exe --headless cmd.exe /c `"cd /d `"$workDir`" && $chain`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
if ($result.ReturnValue -eq 0) {
    Write-Host "Started $($steps.Count) steps from step $StartStep -- PID $($result.ProcessId)" -ForegroundColor Green
    Write-Host "  Training log: Get-Content `"$workDir\logs\stage2${tag}_log.txt`" -Tail 30 -Wait"
    Write-Host "  Stop:         taskkill /PID $($result.ProcessId) /T /F"
} else {
    Write-Host "Failed to start. Error code: $($result.ReturnValue)" -ForegroundColor Red
}
