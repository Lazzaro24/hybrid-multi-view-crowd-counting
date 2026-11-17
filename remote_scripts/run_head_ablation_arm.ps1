param(
    [ValidateSet("h0", "h1", "h2", "h3")]
    [string]$Arm = "h1",
    [int]$FramesPerScene = 10,    # 28 train scenes x 10 frames = 280 iterations/epoch, as H0 and H3
    [int]$Epochs2 = 30,           # Stage 2 experts, masked MSE on the arm's oracle mask        (h1, h2)
    [int]$EpochsOT = 13,          # sparse-only OT step (fresh sparse expert, genloss)          (h1, h2)
    [int]$Epochs3 = 30,           # Stage 3 router                                              (h1, h2, h3)
    [int]$Epochs4 = 15,           # Stage 4 router only, fused-map loss + mask anchor           (h1, h2)
    [int]$EpochsH0 = 0,           # h0: extra decoder epochs; 0 = Epochs2 + EpochsOT + Epochs3 + Epochs4 (H1's head budget)
    [switch]$FreezeFusion,        # h0: train the decoder only (default: fusion + decoder, as H1's Stage 2 trains the fusion)
    [double]$MinDensity = 0.5,    # h2: window-density threshold in persons/m^2 (W25, min area 50 px)
    [double]$AnchorWeight = 0.1,
    [int]$SelectWindow = 3,
    [int]$StartStep = 1,          # resume the chain at a later step (the list is printed at start)
    [switch]$NoEval,              # train only: drop the final paper-protocol test step (run it later with -StartStep <last>)
    [switch]$DryRun,             # print the step list and commands, run nothing
    [switch]$Foreground           # internal: run the steps in this process (the default relaunches itself detached)
)

# Head ablation on the homography_256 fusion (doc/head_ablation.md section 1): the arms H1 and H2 that were planned but
# never run, and H3's router again with final weights, so all four arms follow one rule. Runs LOCALLY, detached, one
# step after another.
#
# Same start for every arm: the Stage 1 checkpoint made from the 30-epoch neck-ablation fusion
#   runs\head_ablation\stage1_from_ablation\multiscene_best_stage_1_abl_v2_homography_256.pth
# (H0 = that fusion's own decoder, runs\neck_ablation\ablation_v2\homography_256\last.pth, test 87.1). As in H3, the
# fusion trains along in Stage 2 (the trainer's default); Stage 3 trains the router only.
#
# Budget: one iteration = one frame, 5 views, 3 random 200x200 BEV patches, batch 1; one epoch = 280 iterations.
#   H1 / H2: Stage 2 30 + OT step 13 + Stage 3 30 + Stage 4 15 epochs = 24.6k iterations: the full recipe of h1_hom's
#            chain (run_remote_budget_ca_sa_h1.ps1) at the head ablation's short budget (the 30/30/13/30/15 epochs of
#            the legacy 99.6 lineage). Stage 4 trains the router only, on the fused map, with the Tversky anchor to the
#            arm's own oracle mask (weight 0.1), as the thesis head.
#   H3:      its Stage 2 (30 epochs, joint unmasked genloss) is reused; Stage 3 30 epochs is redone from the Stage 2
#            FINAL weights (the old _perf_c10_sel run started from the window-selected Stage 2 and was scored at the
#            window-selected Stage 3, test 91.0). No Stage 4: its router is already trained on the per-cell count error,
#            and it has no oracle mask to anchor to (Stage 4 without the anchor turns the mask grey, doc/results.md).
# Every stage hands on its FINAL weights; every stage runs all its epochs (patience = epochs). Evaluation: the paper
# protocol on the test split, learned routing, final weights of the arm's last stage only.
#
#   H0  the baseline decoder, trained on from its 30-epoch fusion run for the head's budget (neck_ablation.py
#       --init_from runs\neck_ablation\ablation_v2\homography_256\last.pth; EpochsH0 = H1's 88 epochs = 24.6k it by
#       default), so H0 and H1 have the same total budget (8.4k fusion run + 24.6k). New warmup + cosine schedule,
#       fusion trainable unless -FreezeFusion; final weights; val every 5 epochs (monitoring only), no own test run.
#   H1  pointwise oracle mask (GT density > 1.273), Tversky alpha 0.7
#   H2  window-density mask W25 / -MinDensity persons/m^2 (default 0.5) / min area 50 px, Tversky alpha 0.7 (the mask
#       flags on every stage)
#   H3  performance mask, 10 px cells, weighted BCE
#
# Usage (from src\):  .\remote_scripts\run_head_ablation_arm.ps1 -Arm h1     (-Arm h2 -MinDensity 0.4, -Arm h0, ...)
# Output: runs\head_ablation\homography_256\{h0_long | og_mask | window_mask_D<MinDensity> | perf_router\c10_final}\
#   chain_log.txt, 2_stage2\ 2_ot_sparse\ 3_router\ 4_router_anchor\ (weights, <stage>_log.txt, <stage>_error.txt,
#   val_logs\), eval\ ; h0: 1_decoder\homography_256\{last.pth, log.csv, result.json}, eval\

$workDir = Split-Path $PSScriptRoot -Parent
Set-Location $workDir
$root = "runs\head_ablation\homography_256"
$stage1 = "runs\head_ablation\stage1_from_ablation\multiscene_best_stage_1_abl_v2_homography_256.pth"
$h3Stage2 = "$root\joint_genloss\unmasked_joint_genloss_ep30\multiscene_final_stage_2_abl_homography_256_unmasked_joint_genloss_ep30.pth"
$h0Init = "runs\neck_ablation\ablation_v2\homography_256\last.pth"
$inv = [System.Globalization.CultureInfo]::InvariantCulture
if ($EpochsH0 -le 0) { $EpochsH0 = $Epochs2 + $EpochsOT + $Epochs3 + $Epochs4 }
$dStr = $MinDensity.ToString($inv)

switch ($Arm) {
    "h0" { $outDir = "$root\h0_long" + $(if ($FreezeFusion) { "_frozen" } else { "" }) }
    "h1" { $outDir = "$root\og_mask"; $lineage = "_abl_homography_256_og"; $maskFlags = "" }
    "h2" { $outDir = "$root\window_mask_D$dStr"; $lineage = "_abl_homography_256_W25D${dStr}_ma50"
           $maskFlags = "--oracle_density_window 25 --oracle_min_density $dStr --oracle_min_area 50 " }
    "h3" { $outDir = "$root\perf_router\c10_final"; $lineage = "_abl_homography_256_unmasked_joint_genloss_ep30"; $maskFlags = "" }
}
$dirs = [ordered]@{ s2 = "$outDir\2_stage2"; ot = "$outDir\2_ot_sparse"; s3 = "$outDir\3_router"; s4 = "$outDir\4_router_anchor"
                    eval = "$outDir\eval" }
if ($Arm -eq "h3") { $dirs.Remove("s2"); $dirs.Remove("ot"); $dirs.Remove("s4") }
if ($Arm -eq "h0") { $dirs = [ordered]@{ dec = "$outDir\1_decoder"; eval = "$outDir\eval" } }
$chainLog = "$outDir\chain_log.txt"

if (-not $Foreground -and -not $DryRun) {
    $needed = @("runs\train_2d\new_best\resnet18_2d_pretrained_ep4.pth") +
              $(switch ($Arm) { "h0" { @($h0Init) } "h3" { @($stage1, $h3Stage2) } default { @($stage1) } })
    foreach ($p in $needed) {
        if (-not (Test-Path "$workDir\$p")) { Write-Host "Missing $workDir\$p" -ForegroundColor Red; exit 1 }
    }
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Foreground -Arm $Arm " +
               "-FramesPerScene $FramesPerScene -Epochs2 $Epochs2 -EpochsOT $EpochsOT -Epochs3 $Epochs3 -Epochs4 $Epochs4 " +
               "-EpochsH0 $EpochsH0 -MinDensity $dStr -AnchorWeight $($AnchorWeight.ToString($inv)) " +
               "-SelectWindow $SelectWindow -StartStep $StartStep" + $(if ($FreezeFusion) { " -FreezeFusion" } else { "" }) +
               $(if ($NoEval) { " -NoEval" } else { "" })
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
         -Arguments @{CommandLine = "conhost.exe --headless powershell.exe $argList"}
    if ($r.ReturnValue -eq 0) {
        Write-Host "Started detached chain ($Arm) -- PID $($r.ProcessId)" -ForegroundColor Green
        Write-Host "  Progress: Get-Content `"$workDir\$chainLog`" -Tail 20 -Wait"
        Write-Host "  Stop:     taskkill /PID $($r.ProcessId) /T /F"
    } else { Write-Host "Failed to start. Error code: $($r.ReturnValue)" -ForegroundColor Red }
    exit 0
}

if (-not $DryRun) { foreach ($d in $dirs.Values) { New-Item -ItemType Directory -Force -Path $d | Out-Null } }
function Log($msg) { $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"; if (-not $DryRun) { Add-Content -Path $chainLog -Value $line }; Write-Host $line }

$py = "uv run python -u"
$train = "$py .\training_scripts\train_proposed_multiscene.py"
$shared = "--frames_per_scene $FramesPerScene --val_frames_per_scene 10 --precise_bn_frames 1 --skip_camera_validation " +
          "--no_router_contrast --select_window $SelectWindow"
function Redir($dir, $name) { "> $dir\${name}_log.txt 2> $dir\${name}_error.txt" }
function StageOut($dir, $name) { "--ckpt_dir $dir --val_log_dir $dir\val_logs --log_file \\.\NUL $(Redir $dir $name)" }

$steps = @()
if ($Arm -eq "h0") {
    $runName = "h0_long_homography_256"
    $steps += @{ name = "H0 decoder trained on from the 30-epoch fusion run ($EpochsH0 epochs, fusion " +
                        $(if ($FreezeFusion) { "frozen" } else { "trainable" }) + ")"
                 cmd = "$py .\training_scripts\neck_ablation.py --run_name $runName --variants homography_256 --epochs $EpochsH0 " +
                       "--frames_per_scene $FramesPerScene --init_from $h0Init --out_root $($dirs.dec) --val_every 5 --no_test " +
                       $(if ($FreezeFusion) { "--freeze_fusion " } else { "" }) +
                       "$(Redir $dirs.dec 'h0_decoder')" }
    $steps += @{ name = "paper protocol test: H0 after $EpochsH0 more epochs"
                 cmd = "$py .\evaluation\evaluate_paper_protocol.py --model ablation --weights $($dirs.dec)\homography_256\last.pth " +
                       "--split test --out $($dirs.eval)\eval_test_h0_long.json $(Redir $dirs.eval 'eval_test_h0_long')" }
} elseif ($Arm -eq "h3") {
    $tag3 = "${lineage}_perf_c10_final"
    $final3 = "$($dirs.s3)\multiscene_final_stage_3$tag3.pth"
    $steps += @{ name = "Stage 3 performance router from the Stage 2 final ($Epochs3 epochs)"
                 cmd = "$train --stage 3 --resume $h3Stage2 --epochs $Epochs3 --patience3 $Epochs3 --stage3_target performance " +
                       "--stage3_select soft_mae --route_cell 10 --ckpt_tag $tag3 $shared $(StageOut $dirs.s3 'stage3_router')" }
    $finalW = $final3
} else {
    $tag = "${lineage}_dout0.7_cw0.01"
    $tagOT = "${tag}_genloss"
    $tag3 = "${tagOT}_a0.7"
    $out2 = "$($dirs.s2)\multiscene_final_stage_2$tag.pth"
    $outOT = "$($dirs.ot)\multiscene_final_stage_2$tagOT.pth"
    $final3 = "$($dirs.s3)\multiscene_final_stage_3$tag3.pth"
    $steps += @{ name = "Stage 2 masked MSE ($Epochs2 epochs)"
                 cmd = "$train --stage 2 --resume $stage1 --epochs $Epochs2 --patience2 $Epochs2 $maskFlags" +
                       "--dense_out_weight 0.7 --count_loss_weight 0.01 --ckpt_tag $tag $shared $(StageOut $dirs.s2 'stage2_dense_mask')" }
    $steps += @{ name = "OT step, sparse only ($EpochsOT epochs)"
                 cmd = "$train --stage 2 --resume $out2 --epochs $EpochsOT --patience2 $EpochsOT --sparse_only --reinit_sparse $maskFlags" +
                       "--sparse_loss genloss --sparse_genloss_weight 1.0 --sparse_genloss_topk 1000 --sparse_genloss_bg_weight 0.05 " +
                       "--sparse_genloss_count_weight 1.0 --stage2_select sparse_f1 --sparse_nms_mult 1 --sparse_nms_kernel 7 " +
                       "--dense_out_weight 0.7 --count_loss_weight 0.01 --ckpt_tag $tagOT $shared $(StageOut $dirs.ot 'stage2_ot_sparse')" }
    $steps += @{ name = "Stage 3 router, Tversky 0.7 ($Epochs3 epochs)"
                 cmd = "$train --stage 3 --resume $outOT --epochs $Epochs3 --patience3 $Epochs3 --tversky_alpha 0.7 $maskFlags" +
                       "--ckpt_tag $tag3 $shared $(StageOut $dirs.s3 'stage3_router')" }
    $tag4 = "${tag3}_ro_anchor$($AnchorWeight.ToString($inv))_fs"
    $finalW = "$($dirs.s4)\multiscene_final_stage_4$tag4.pth"
    $steps += @{ name = "Stage 4 router only + anchor $($AnchorWeight.ToString($inv)) ($Epochs4 epochs)"
                 cmd = "$train --stage 4 --router_only --resume $final3 --epochs $Epochs4 --patience4 $Epochs4 --lr 3e-4 " +
                       "--tversky_alpha 0.7 --stage4_mask_anchor_weight $($AnchorWeight.ToString($inv)) $maskFlags" +
                       "--full_scene_val_frames 10 --full_scene_val_draws 2 --ckpt_tag $tag4 $shared " +
                       "$(StageOut $dirs.s4 'stage4_router_anchor')" }
}
if ($Arm -ne "h0") {
    $base = [System.IO.Path]::GetFileNameWithoutExtension($finalW)
    $steps += @{ name = "paper protocol test, learned routing: $base"
                 cmd = "$py .\evaluation\evaluate_paper_protocol.py --model proposed --weights $finalW --split test --routing learned " +
                       "--out $($dirs.eval)\eval_test_${base}_learned.json $(Redir $dirs.eval "eval_test_${base}_learned")" }
}
if ($NoEval) { $steps = $steps[0..($steps.Count - 2)] }  # the test step is always last

Log "=== head ablation $Arm, chain start (steps $StartStep..$($steps.Count)) | output $workDir\$outDir ==="
for ($i = 0; $i -lt $steps.Count; $i++) { Log ("  {0}. {1}" -f ($i + 1), $steps[$i].name) }
if ($DryRun) { foreach ($s in $steps) { Write-Host "`n# $($s.name)`n$($s.cmd)" }; exit 0 }
for ($i = $StartStep - 1; $i -lt $steps.Count; $i++) {
    Log ("step {0}/{1}: {2}" -f ($i + 1), $steps.Count, $steps[$i].name)
    cmd.exe /c $steps[$i].cmd
    if ($LASTEXITCODE -ne 0) { Log "  failed (exit $LASTEXITCODE)"; Log "=== chain stopped at step $($i + 1); rerun with -StartStep $($i + 1) ==="; exit 1 }
}
Log "=== chain done ==="
