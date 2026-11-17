param(
    [string]$Neck = "ca_sa",
    [int]$FramesPerScene = 50,    # 28 train scenes x 50 frames = 1400 iterations/epoch = one baseline epoch
    [int]$Epochs1 = 50,           # fusion (neck_ablation.py, baseline decoder, EMA)          -> Stage 1
    [int]$Epochs2 = 50,           # Stage 2 experts, masked MSE on the oracle zone
    [int]$EpochsOT = 25,          # sparse-only OT step (fresh sparse expert, genloss)
    [int]$Epochs3 = 50,           # Stage 3 router, Tversky on the oracle mask
    [int]$Epochs4 = 25,           # Stage 4 router-only, fused-map loss + shape anchor, whole-scene selection
    [int]$SelectWindow = 3,       # checkpoint selection on the mean of the last N validated epochs
    [double]$AnchorWeight = 0.1,
    [string]$RunTag = "bud200",   # lineage name; change it for a second seed / budget
    [string]$OutDir = "full_length_dense_mask",   # folder under src\ that receives every weight file and log (no spaces)
    [ValidateSet("final", "best")]
    [string]$Pass = "final",      # which checkpoint each stage hands to the next: last epoch, or the --select_window best
    [string]$InitCheckpoint = "", # with -StartStep 4..7: the checkpoint that step resumes from instead of the chain's own
                                  # (e.g. an earlier run's Stage 2 final, to redo only the OT step onward)
    [int]$StartStep = 1,          # resume the chain at a later step (the list is printed at start)
    [int]$MaxAttempts = 10,       # restarts of the fusion training after a crash (it resumes from last.pth)
    [switch]$DryRun,              # print the step list and commands, run nothing
    [switch]$Foreground           # internal: run the steps in this process (the default relaunches itself detached)
)

# Budget-matched run of the best proposed recipe (H1 OG mask + OT sparse + Stage 4 shape anchor, test MAE 99.6 on the
# legacy sa_ca neck, doc/results.md "Stage 4" section), now on the ca_sa fusion, for a fair comparison with the baseline.
#
# Budget. One iteration is the same unit in both trainers: one frame, 5 views, 3 random 200x200 BEV patches, batch 1.
#   Baseline run_1 (runs\baseline\run_1, test MAE 67.0): 200 epochs x 1400 it = 280k iterations (best epoch 170).
#   The 99.6 recipe: 30 + 30 + 13 + 30 + 15 = 118 epochs x 280 it = 33k iterations, ~1/8.5 of the baseline.
#   Here: --frames_per_scene 50 makes one epoch = 1400 it = one baseline epoch, and the stages keep the 99.6 recipe's
#   proportions (x ~1.7 in these epochs): 50 + 50 + 25 + 50 + 25 = 200 epochs = 280k iterations. The count includes
#   the router-only stages (cheap per step, but same samples seen); every stage runs its full budget (patience =
#   epochs, no early stop), so the cosine schedules finish and the final-epoch weights are a real end point.
#   The baseline's 5 random view draws per frame are matched by 5x more distinct frames (the proposed trainer uses
#   each scene's fixed K-means cameras); the gradient-step count is the same.
#
# Hand-over (-Pass): each stage resumes from the previous stage's FINAL weights by default. Every stage still writes
# both multiscene_best_stage_<n><tag>.pth (--select_window $SelectWindow: lowest mean of the last N validation
# metrics) and multiscene_final_stage_<n><tag>.pth. -Pass best is the first run's behaviour (the 99.6 lineage's).
# Why final: in the first run (-Pass best) the Stage 2 window kept epoch 4 of 50; on test (paper protocol, oracle
# routing) epoch 4 scored 137.7 and epoch 50 73.4, while validation ranked them 71.6 vs ~80 -- the validation metrics
# (patch MAE, sparse F1, mass error) don't rank checkpoints like the test, and the cosine schedules end at 1e-6, so the
# final weights are the stable end point. Stage 1 always passes the final EMA weights (make_stage1_from_ablation.py
# reads last.pth). The fusion's best/final test MAE is the H0 reference.
#
# Steps:
#   1  reference-point cache master_refs.pt (cache_reference.py skips scenes that already have it; stays in the data)
#   2  fusion $Neck, neck_ablation.py --epochs $Epochs1 --frames_per_scene $FramesPerScene
#   3  Stage 1 checkpoint from it
#   4  Stage 2: neck + both experts, masked MSE, --dense_out_weight 0.7 --count_loss_weight 0.01
#   5  OT step: --sparse_only --reinit_sparse, genloss (the genloss_v7_reinit recipe)
#   6  Stage 3: router only, --tversky_alpha 0.7
#   7  Stage 4: router only, --stage4_mask_anchor_weight $AnchorWeight, whole-scene selection, lr 3e-4
#   8  paper protocol test: Stage 4 final         9  Stage 4 best (for reference)
#  10  paper protocol test: the Stage 3 checkpoint that was handed on, oracle routing (the routing ceiling)
# Redo from the OT step on an earlier run's Stage 2 (into a new folder, so nothing is overwritten):
#   .\remote_scripts\run_remote_budget_ca_sa_h1.ps1 -OutDir full_length_dense_mask_final -StartStep 5 `
#       -InitCheckpoint full_length_dense_mask\2_dense_mask\multiscene_final_stage_2_bud200_ca_sa_dout0.7_cw0.01.pth
#
# Everything goes to src\$OutDir\ (bring back that one folder):
#   chain_log.txt          progress of the whole chain
#   0_cache\               log of step 1
#   1_fusion\              neck_ablation.py output (<neck>\{best,last}.pth, log.csv, result.json), its logs, and the
#                          Stage 1 checkpoint made from it
#   2_dense_mask\  2_ot_sparse\  3_router\  4_router_anchor\
#                          per stage: best / final / every-5-epoch weights, <stage>_log.txt (stdout),
#                          <stage>_error.txt (stderr, tqdm), val_logs\ (per-scene validation CSVs)
#   eval\                  paper-protocol JSON + log per evaluated checkpoint

$workDir = Split-Path $PSScriptRoot -Parent
Set-Location $workDir
$run = "${RunTag}_$Neck"
# paths relative to src\ (the working directory of every step), so the cmd lines need no quoting
$dirs = [ordered]@{ cache = "0_cache"; fusion = "1_fusion"; s2 = "2_dense_mask"; ot = "2_ot_sparse"; s3 = "3_router"
                    s4 = "4_router_anchor"; eval = "eval" }
foreach ($k in @($dirs.Keys)) { $dirs[$k] = "$OutDir\$($dirs[$k])" }
$chainLog = "$OutDir\chain_log.txt"
if ($InitCheckpoint) {
    if ($StartStep -lt 4 -or $StartStep -gt 7) { Write-Host "-InitCheckpoint needs -StartStep 4..7" -ForegroundColor Red; exit 1 }
    if (-not $DryRun -and -not (Test-Path "$workDir\$InitCheckpoint")) { Write-Host "Missing $workDir\$InitCheckpoint" -ForegroundColor Red; exit 1 }
}

if (-not $Foreground -and -not $DryRun) {
    $backbone = "$workDir\runs\train_2d\new_best\resnet18_2d_pretrained_ep4.pth"
    if (-not (Test-Path $backbone)) { Write-Host "Missing $backbone" -ForegroundColor Red; exit 1 }
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Foreground -Neck $Neck " +
               "-FramesPerScene $FramesPerScene -Epochs1 $Epochs1 -Epochs2 $Epochs2 -EpochsOT $EpochsOT " +
               "-Epochs3 $Epochs3 -Epochs4 $Epochs4 -SelectWindow $SelectWindow -AnchorWeight $($AnchorWeight.ToString($inv)) " +
               "-RunTag $RunTag -OutDir $OutDir -Pass $Pass -StartStep $StartStep -MaxAttempts $MaxAttempts" +
               $(if ($InitCheckpoint) { " -InitCheckpoint $InitCheckpoint" } else { "" })
    $cmd = "conhost.exe --headless powershell.exe $argList"
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd}
    if ($r.ReturnValue -eq 0) {
        Write-Host "Started detached chain -- PID $($r.ProcessId)" -ForegroundColor Green
        Write-Host "  Progress: Get-Content `"$workDir\$chainLog`" -Tail 20 -Wait"
        Write-Host "  Stop:     taskkill /PID $($r.ProcessId) /T /F"
    } else { Write-Host "Failed to start. Error code: $($r.ReturnValue)" -ForegroundColor Red }
    exit 0
}

if (-not $DryRun) { foreach ($d in $dirs.Values) { New-Item -ItemType Directory -Force -Path $d | Out-Null } }
function Log($msg) { $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"; if (-not $DryRun) { Add-Content -Path $chainLog -Value $line }; Write-Host $line }

$inv = [System.Globalization.CultureInfo]::InvariantCulture
$tag = "_${run}_dout0.7_cw0.01"                         # Stage 2 masked MSE (and the lineage prefix)
$tagOT = "${tag}_genloss"                               # OT step (also stage 2)
$tag3 = "${tagOT}_a0.7"
$tag4 = "${tag3}_ro_anchor$($AnchorWeight.ToString($inv))_fs"
$stage1 = "$($dirs.fusion)\multiscene_best_stage_1_abl_${run}.pth"
$out2 = "$($dirs.s2)\multiscene_${Pass}_stage_2$tag.pth"            # what each stage hands on (-Pass)
$outOT = "$($dirs.ot)\multiscene_${Pass}_stage_2$tagOT.pth"
$out3 = "$($dirs.s3)\multiscene_${Pass}_stage_3$tag3.pth"
$best4 = "$($dirs.s4)\multiscene_best_stage_4$tag4.pth"
$final4 = "$($dirs.s4)\multiscene_final_stage_4$tag4.pth"
# the checkpoint each training step (4..7) resumes from; -InitCheckpoint replaces the start step's
$from = @{ 4 = $stage1; 5 = $out2; 6 = $outOT; 7 = $out3 }
if ($InitCheckpoint) { $from[$StartStep] = $InitCheckpoint }

$py = "uv run python -u"
$shared = "--frames_per_scene $FramesPerScene --val_frames_per_scene 10 --precise_bn_frames 1 --skip_camera_validation " +
          "--no_router_contrast --select_window $SelectWindow"
$train = "$py .\training_scripts\train_proposed_multiscene.py"
function Redir($dir, $name) { "> $dir\${name}_log.txt 2> $dir\${name}_error.txt" }
# per-stage output: checkpoints and validation CSVs into the stage folder. --log_file \\.\NUL drops the trainer's own copy
# of stdout (multiscene_training_log_<time>.txt in the cwd); the redirect already captures it.
function StageOut($dir, $name) { "--ckpt_dir $dir --val_log_dir $dir\val_logs --log_file \\.\NUL $(Redir $dir $name)" }

# --- the step list: @{ name; cmd; retries } -------------------------------------------------------------------------
$steps = @()
$steps += @{ name = "reference cache master_refs.pt"; retries = 1
             cmd = "$py .\utils\cache_reference.py > $($dirs.cache)\cache_refs_log.txt 2>&1" }
$steps += @{ name = "fusion $Neck ($Epochs1 epochs x $FramesPerScene frames, run $run)"; retries = $MaxAttempts
             cmd = "$py .\training_scripts\neck_ablation.py --run_name $run --variants $Neck --epochs $Epochs1 " +
                   "--frames_per_scene $FramesPerScene --out_root $($dirs.fusion) " +
                   ">> $($dirs.fusion)\fusion_log.txt 2>> $($dirs.fusion)\fusion_error.txt" }
$steps += @{ name = "Stage 1 checkpoint from the fusion"; retries = 1
             cmd = "$py .\training_scripts\make_stage1_from_ablation.py --run $run --variant $Neck --run_dir $($dirs.fusion) " +
                   "--out $stage1 > $($dirs.fusion)\make_stage1_log.txt 2>&1" }
$steps += @{ name = "Stage 2 masked MSE ($Epochs2 epochs)"; retries = 1
             cmd = "$train --stage 2 --resume $($from[4]) --epochs $Epochs2 --patience2 $Epochs2 " +
                   "--dense_out_weight 0.7 --count_loss_weight 0.01 --ckpt_tag $tag $shared $(StageOut $dirs.s2 'stage2_dense_mask')" }
$steps += @{ name = "OT step, sparse only ($EpochsOT epochs)"; retries = 1
             cmd = "$train --stage 2 --resume $($from[5]) --epochs $EpochsOT --patience2 $EpochsOT --sparse_only --reinit_sparse " +
                   "--sparse_loss genloss --sparse_genloss_weight 1.0 --sparse_genloss_topk 1000 --sparse_genloss_bg_weight 0.05 " +
                   "--sparse_genloss_count_weight 1.0 --stage2_select sparse_f1 --sparse_nms_mult 1 --sparse_nms_kernel 7 " +
                   "--dense_out_weight 0.7 --count_loss_weight 0.01 --ckpt_tag $tagOT $shared $(StageOut $dirs.ot 'stage2_ot_sparse')" }
$steps += @{ name = "Stage 3 router, Tversky 0.7 ($Epochs3 epochs)"; retries = 1
             cmd = "$train --stage 3 --resume $($from[6]) --epochs $Epochs3 --patience3 $Epochs3 --tversky_alpha 0.7 " +
                   "--ckpt_tag $tag3 $shared $(StageOut $dirs.s3 'stage3_router')" }
$steps += @{ name = "Stage 4 router only + anchor $AnchorWeight, whole-scene selection ($Epochs4 epochs)"; retries = 1
             cmd = "$train --stage 4 --router_only --resume $($from[7]) --epochs $Epochs4 --patience4 $Epochs4 --lr 3e-4 " +
                   "--tversky_alpha 0.7 --stage4_mask_anchor_weight $($AnchorWeight.ToString($inv)) " +
                   "--full_scene_val_frames 10 --full_scene_val_draws 2 --ckpt_tag $tag4 $shared " +
                   "$(StageOut $dirs.s4 'stage4_router_anchor')" }
foreach ($e in @(@{ w = $final4; r = "learned" }, @{ w = $best4; r = "learned" }, @{ w = $out3; r = "oracle" })) {
    $base = [System.IO.Path]::GetFileNameWithoutExtension($e.w)
    $steps += @{ name = "paper protocol test: $base ($($e.r) routing)"; retries = 1
                 cmd = "$py .\evaluation\evaluate_paper_protocol.py --model proposed --weights $($e.w) --split test " +
                       "--routing $($e.r) --out $($dirs.eval)\eval_test_${base}_$($e.r).json " +
                       "$(Redir $dirs.eval "eval_test_${base}_$($e.r)")" }
}

$itPerEpoch = 28 * $FramesPerScene
$total = ($Epochs1 + $Epochs2 + $EpochsOT + $Epochs3 + $Epochs4)
Log "=== chain start (steps $StartStep..$($steps.Count)) | lineage $tag | output $workDir\$OutDir | hand-over: $Pass ==="
if ($InitCheckpoint) { Log "step $StartStep resumes from $InitCheckpoint" }
Log ("budget: {0} epochs x {1} it = {2:N0} iterations (baseline run_1: 200 x 1400 = 280,000)" -f $total, $itPerEpoch, ($total * $itPerEpoch))
for ($i = 0; $i -lt $steps.Count; $i++) { Log ("  {0,2}. {1}" -f ($i + 1), $steps[$i].name) }
if ($DryRun) { foreach ($s in $steps) { Write-Host "`n# $($s.name)`n$($s.cmd)" }; exit 0 }
for ($i = $StartStep - 1; $i -lt $steps.Count; $i++) {
    $s = $steps[$i]
    $ok = $false
    for ($a = 1; $a -le $s.retries; $a++) {
        Log ("step {0}/{1}: {2}{3}" -f ($i + 1), $steps.Count, $s.name, $(if ($s.retries -gt 1) { " (attempt $a)" } else { "" }))
        cmd.exe /c $s.cmd
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
        Log "  failed (exit $LASTEXITCODE)"
        if ($a -lt $s.retries) { Start-Sleep -Seconds 60 }
    }
    if (-not $ok) { Log "=== chain stopped at step $($i + 1); rerun with -StartStep $($i + 1) ==="; exit 1 }
}
Log "=== chain done ==="
