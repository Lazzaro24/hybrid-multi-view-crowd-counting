param(
    [string]$Neck = "homography_256", # homography_256 | ca_sa: whose Stage 2 experts the sparse expert is trained on
    [int]$Epochs = 15,                # shallow sparse head
    [int]$DeepEpochs = 45,            # runs with --sparse_arch deep (they converge slowly)
    [int]$Patience = 10,
    [string]$Runs = "S0,S1,S2",       # runs to train
    [string]$DiagRuns = "",           # runs the final diagnostic scores; default = the runs trained here. The diagnostic
                                      # is deterministic (fixed per-scene val_seed), so earlier JSONs stay comparable:
                                      # score each model once and compare across files.
    [switch]$DiagCurrent,             # also score the current Stage 2 expert (already in sparse_S012.json)
    [switch]$DiagnosticOnly,
    [switch]$DryRun                   # print the commands and exit
)

# Sparse expert experiments (doc/head_ablation.md §6e/§9), homography_256 lineage. Each run is a sparse-only fine-tune
# from the current Stage 2 checkpoint: neck and dense expert frozen (eval mode), sparse expert re-initialised, trained on
# the whole patch (--stage2_unmasked), selected on validation peak F1 (NMS kernel 3, no moving window).
#   S0  control: the current loss (genloss, background weight 0.05, count weight 1.0)
#   S1  genloss + empty-ground mass penalty (>~1.7 m from anyone) in count units, weight 1.0 (the pixel-mean MSE form
#       is ~1e-4 vs the OT term ~5 and changed nothing: a first S1 with --sparse_genloss_bg_weight 1.0 matched S0 exactly)
#   S2  CenterNet-style penalty-reduced focal heatmap loss (sigmoid output, one peak per person)
#   S3  S1's setup, but the sparse expert is supervised (and validated) only where the local density is below
#       0.25 p/m^2 (window-density mask W25 / D0.25 / min area 50): the sparse expert's job restricted to where
#       detection is possible (doc/head_ablation.md §6e)
#   S4  S1's loss with the deep sparse head (--sparse_arch deep: 5 convs, 15 px = 3 m receptive field instead of 5 px)
#   S5  deep head with a weaker empty-ground penalty (weight 0.3): S4's precision with S1's recall on isolated people?
#       (the deep head converges slowly: run with -Epochs 45)
#   S6  S3 with the deep head: supervision only below 0.25 p/m^2, 15 px receptive field (-Epochs 45)
# Then scripts/diagnosis/diagnose_sparse_detection.py on all runs + the current expert, thresholds 0.064 / 0.2 / 0.4.
# Sequential, detached; stops at the first failure.

$workDir = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path "$workDir\logs")) { New-Item -ItemType Directory -Force -Path "$workDir\logs" | Out-Null }
# the Stage 2 checkpoints the performance routers (_perf_c10_sel) were trained on
$stage2 = @{ "homography_256" = "multiscene_best_stage_2_abl_homography_256_unmasked_joint_genloss_ep30.pth";
             "ca_sa"          = "multiscene_checkpoint_stage_2_abl_ca_sa_unmasked_joint_genloss_epoch_30.pth" }[$Neck]
if (-not $stage2) { Write-Host "Unknown neck $Neck" -ForegroundColor Red; exit 1 }
if (-not (Test-Path "$workDir\$stage2")) { Write-Host "Missing $stage2" -ForegroundColor Red; exit 1 }

$shared = "--frames_per_scene 10 --val_frames_per_scene 10 --precise_bn_frames 1 --skip_camera_validation --no_router_contrast"
$common = "--stage 2 --resume $stage2 --sparse_only --reinit_sparse --stage2_select sparse_f1 " +
          "--sparse_nms_kernel 3 --dense_out_weight 0.7 --count_loss_weight 0.01 --patience2 $Patience"
$unm = "--stage2_unmasked"            # whole patch; S3 instead trains only outside the window-density mask
$genloss = "--sparse_loss genloss --sparse_genloss_weight 1.0 --sparse_genloss_topk 1000 --sparse_genloss_count_weight 1.0 --sparse_nms_mult 1"
$defs = [ordered]@{
    "S0" = @{ tag = "_abl_${Neck}_sparse_S0_genloss_bg0.05"; flags = "$unm $genloss --sparse_genloss_bg_weight 0.05"; act = "" }
    "S1" = @{ tag = "_abl_${Neck}_sparse_S1_genloss_bgmass1.0"; flags = "$unm $genloss --sparse_genloss_bg_weight 1.0 --sparse_genloss_bg_mode mass --sparse_bg_empty_thr 1e-3"; act = "" }
    "S2" = @{ tag = "_abl_${Neck}_sparse_S2_focal";          flags = "$unm --sparse_loss focal --sparse_nms_mult 3";     act = ":sigmoid" }
    "S4" = @{ tag = "_abl_${Neck}_sparse_S4_deep_bgmass1.0"; flags = "$unm $genloss --sparse_genloss_bg_weight 1.0 --sparse_genloss_bg_mode mass --sparse_bg_empty_thr 1e-3 --sparse_arch deep"; act = "" }
    "S3" = @{ tag = "_abl_${Neck}_sparse_S3_genloss_bgmass1.0_W25D0.25"; flags = "--oracle_density_window 25 --oracle_min_density 0.25 --oracle_min_area 50 $genloss --sparse_genloss_bg_weight 1.0 --sparse_genloss_bg_mode mass --sparse_bg_empty_thr 1e-3"; act = "" }
    "S5" = @{ tag = "_abl_${Neck}_sparse_S5_deep_bgmass0.3"; flags = "$unm $genloss --sparse_genloss_bg_weight 0.3 --sparse_genloss_bg_mode mass --sparse_bg_empty_thr 1e-3 --sparse_arch deep"; act = "" }
    "S6" = @{ tag = "_abl_${Neck}_sparse_S6_deep_bgmass1.0_W25D0.25"; flags = "--oracle_density_window 25 --oracle_min_density 0.25 --oracle_min_area 50 $genloss --sparse_genloss_bg_weight 1.0 --sparse_genloss_bg_mode mass --sparse_bg_empty_thr 1e-3 --sparse_arch deep"; act = "" }
}

$steps = @()
if (-not $DiagnosticOnly) {
    foreach ($r in ($Runs -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
        $d = $defs[$r]
        $ep = if ($d.flags -match '--sparse_arch deep') { $DeepEpochs } else { $Epochs }
        $steps += "uv run python -u .\training_scripts\train_proposed_multiscene.py $common --epochs $ep $($d.flags) " +
                  "--ckpt_tag $($d.tag) $shared > logs\stage2$($d.tag)_log.txt 2> logs\stage2$($d.tag)_error.txt"
    }
}
if (-not $DiagRuns) { $DiagRuns = $Runs }
$ckpts = if ($DiagCurrent) { "--ckpt current=$stage2" } else { "" }
foreach ($r in ($DiagRuns -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
    $ckpts += " --ckpt $r=multiscene_best_stage_2$($defs[$r].tag).pth$($defs[$r].act)"
}
$diagName = "sparse_" + $(if ($Neck -ne "homography_256") { "${Neck}_" } else { "" }) + ($DiagRuns -replace ',', '')
$steps += "uv run python -u .\scripts\diagnosis\diagnose_sparse_detection.py $ckpts --threshold 0.064,0.2,0.4 " +
          "--out_name $diagName > diagnostics\sparse_detection\$diagName.log 2>&1"

if ($DryRun) { $steps | ForEach-Object { Write-Host $_ }; exit 0 }
$chain = ($steps | ForEach-Object { "($_)" }) -join " && "
$cmd = "conhost.exe --headless cmd.exe /c `"cd /d `"$workDir`" && $chain`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
if ($result.ReturnValue -eq 0) {
    Write-Host "Started $($steps.Count) steps -- PID $($result.ProcessId)" -ForegroundColor Green
    foreach ($r in $defs.Keys) { Write-Host "  $r log: logs\stage2$($defs[$r].tag)_log.txt" }
    Write-Host "  Diagnostic: diagnostics\sparse_detection\$diagName.log"
    Write-Host "  Stop: taskkill /PID $($result.ProcessId) /T /F"
} else {
    Write-Host "Failed to start. Error code: $($result.ReturnValue)" -ForegroundColor Red
}
