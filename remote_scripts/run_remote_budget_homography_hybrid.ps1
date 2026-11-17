param(
    [string]$Neck = "homography_256",
    [int]$FramesPerScene = 50,    # 28 train scenes x 50 frames = 1400 iterations/epoch = one baseline epoch
    [int]$Epochs1 = 50,           # fusion (neck_ablation.py, baseline decoder, EMA) -> Stage 1 and the H0 reference
    [int]$EpochsDense = 60,       # D2: dense expert (+ adapter) on the frozen fusion, MSE only, EMA
    [int]$EpochsSparse = 90,      # S4D2: deep sparse expert on the frozen D2, S4's loss, EMA
    [int]$EpochsControl = 0,      # H0 decoder fine-tune on the frozen fusion; 0 = EpochsDense + EpochsSparse
    [string]$Seeds = "42",        # seeds of the head (D2 + S4D2); the fusion is trained once
    [string]$RunTag = "bud200",
    [string]$OutDir = "full_length_hybrid",   # folder under src\ that receives every weight file and log (no spaces)
    [switch]$WithVal,             # also score the paper protocol on the val split
    [switch]$SkipControl,         # no H0 decoder fine-tune
    [int]$StartStep = 1,          # resume the chain at a later step (the list is printed at start)
    [int]$MaxAttempts = 10,       # restarts of the neck_ablation trainings after a crash (they resume from last.pth)
    [switch]$DryRun,              # print the step list and commands, run nothing
    [switch]$Foreground           # internal: run the steps in this process (the default relaunches itself detached)
)

# Budget-matched run of the dense-first hybrid head (doc/head_ablation.md §8: D2 + S4D2 + density rule, §8.14) on the
# homography_256 fusion, for the clean head comparison at the baseline's training budget.
#
# Budget. One iteration = one frame, 5 views, 3 random 200x200 BEV patches, batch 1 (as run_remote_budget_ca_sa_h1.ps1).
#   Baseline run_1: 200 epochs x 1400 it = 280k iterations. The short-budget recipe (§8.6) was 30 + 30 + 45 = 105
#   epochs x 280 it; here --frames_per_scene 50 makes one epoch = 1400 it, the fusion gets the same 50 epochs as the
#   ca_sa budget chain, and the head the rest in the recipe's 2 : 3 proportion: 50 + 60 + 90 = 200 epochs = 280k
#   iterations per head seed. Every stage runs its full budget (no early stop), and stages hand on their FINAL weights
#   (the trainer's best-epoch pick was unreliable, §8.1 / §8.2).
# References on the same fusion:
#   H0          the fusion run's own baseline decoder (50 epochs, 70k it, EMA final)
#   H0-control  that decoder fine-tuned on the frozen fusion for the head's iterations (EpochsDense + EpochsSparse =
#               150 epochs), so baseline decoder and hybrid head get the same total budget (§8.14 "same budget")
# Threshold. Not fixed: the rule of §8.10 is. sweep_density_rule.py on each hybrid's own val map-quality frames, threshold
#   with the lowest val GAME3, written to eval\routing_s<seed>.txt and read by the paper protocol (--routing @file).
#
# Steps:
#   1  reference-point cache master_refs.pt (skips scenes that already have it)
#   2  fusion $Neck, neck_ablation.py --epochs $Epochs1 --frames_per_scene $FramesPerScene    (= H0)
#   3  Stage 1 checkpoint from it
#   per seed: D2 (dense only, frozen fusion, MSE, EMA), then S4D2 (sparse only, deep, frozen D2, EMA)
#   H0-control (neck_ablation.py --init_from <fusion> --freeze_fusion)
#   per seed: density-rule sweep (val GAME3 selection)
#   paper protocol, test (and val with -WithVal): H0, H0-control, per seed D2 dense alone and the hybrid
#
# Everything goes to src\$OutDir\ (bring back that one folder; the .pth files can stay on the remote):
#   chain_log.txt   0_cache\   1_fusion\ (H0 run + Stage 1)   2_dense_s<seed>\   3_sparse_s<seed>\   4_h0_control\
#   eval\           density sweeps, routing_s<seed>.txt, paper-protocol JSON + logs

$workDir = Split-Path $PSScriptRoot -Parent
Set-Location $workDir
$run = "${RunTag}_$Neck"
if ($EpochsControl -le 0) { $EpochsControl = $EpochsDense + $EpochsSparse }
$seedList = @($Seeds -split '[,\s]+' | Where-Object { $_ })
$fusionDir = "$OutDir\1_fusion"
$evalDir = "$OutDir\eval"
$controlDir = "$OutDir\4_h0_control"
$chainLog = "$OutDir\chain_log.txt"

if (-not $Foreground -and -not $DryRun) {
    $backbone = "$workDir\runs\train_2d\new_best\resnet18_2d_pretrained_ep4.pth"
    if (-not (Test-Path $backbone)) { Write-Host "Missing $backbone" -ForegroundColor Red; exit 1 }
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Foreground -Neck $Neck " +
               "-FramesPerScene $FramesPerScene -Epochs1 $Epochs1 -EpochsDense $EpochsDense -EpochsSparse $EpochsSparse " +
               "-EpochsControl $EpochsControl -Seeds $($seedList -join ',') -RunTag $RunTag -OutDir $OutDir " +
               "-StartStep $StartStep -MaxAttempts $MaxAttempts" +
               $(if ($WithVal) { " -WithVal" } else { "" }) + $(if ($SkipControl) { " -SkipControl" } else { "" })
    $cmd = "conhost.exe --headless powershell.exe $argList"
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd}
    if ($r.ReturnValue -eq 0) {
        Write-Host "Started detached chain -- PID $($r.ProcessId)" -ForegroundColor Green
        Write-Host "  Progress: Get-Content `"$workDir\$chainLog`" -Tail 20 -Wait"
        Write-Host "  Stop:     taskkill /PID $($r.ProcessId) /T /F"
    } else { Write-Host "Failed to start. Error code: $($r.ReturnValue)" -ForegroundColor Red }
    exit 0
}

if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }
function Log($msg) { $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"; if (-not $DryRun) { Add-Content -Path $chainLog -Value $line }; Write-Host $line }
$py = "uv run python -u"
$train = "$py .\training_scripts\train_proposed_multiscene.py"
$pp = "$py .\evaluation\evaluate_paper_protocol.py --num_workers 8"
function Redir($dir, $name) { "> $dir\${name}_log.txt 2> $dir\${name}_error.txt" }
function StageOut($dir, $name) { "--ckpt_dir $dir --val_log_dir $dir\val_logs --log_file \\.\NUL $(Redir $dir $name)" }
$shared = "--frames_per_scene $FramesPerScene --val_frames_per_scene 10 --precise_bn_frames 1 --skip_camera_validation " +
          "--no_router_contrast --route_cell 10 --stage2_unmasked --lr 1e-4 --dense_out_weight 0.7 --ema_decay 0.999"
# S1's sparse loss + the deep head = S4 (run_remote_dense_recipe.ps1 "S4D2")
$sparseS4 = "--sparse_loss genloss --sparse_genloss_weight 1.0 --sparse_genloss_topk 1000 --sparse_genloss_count_weight 1.0 " +
            "--sparse_nms_mult 1 --sparse_nms_kernel 3 --sparse_genloss_bg_weight 1.0 --sparse_genloss_bg_mode mass " +
            "--sparse_bg_empty_thr 1e-3 --sparse_arch deep"
$stage1 = "$fusionDir\multiscene_best_stage_1_abl_${run}.pth"
$h0 = "$fusionDir\$Neck\last.pth"
$h0c = "$controlDir\$Neck\last.pth"

# --- the step list: @{ name; cmd; retries; dirs } ---------------------------------------------------------------------
$steps = @()
$steps += @{ name = "reference cache master_refs.pt"; retries = 1; dirs = @("$OutDir\0_cache")
             cmd = "$py .\utils\cache_reference.py > $OutDir\0_cache\cache_refs_log.txt 2>&1" }
$steps += @{ name = "fusion $Neck = H0 ($Epochs1 epochs x $FramesPerScene frames, run $run)"; retries = $MaxAttempts; dirs = @($fusionDir)
             cmd = "$py .\training_scripts\neck_ablation.py --run_name $run --variants $Neck --epochs $Epochs1 " +
                   "--frames_per_scene $FramesPerScene --out_root $fusionDir " +
                   ">> $fusionDir\fusion_log.txt 2>> $fusionDir\fusion_error.txt" }
$steps += @{ name = "Stage 1 checkpoint from the fusion"; retries = 1; dirs = @()
             cmd = "$py .\training_scripts\make_stage1_from_ablation.py --run $run --variant $Neck --run_dir $fusionDir " +
                   "--out $stage1 > $fusionDir\make_stage1_log.txt 2>&1" }
$hybrid = @{}; $dense = @{}
foreach ($s in $seedList) {
    $dDir = "$OutDir\2_dense_s$s"; $sDir = "$OutDir\3_sparse_s$s"
    $tagD = "_${run}_D2_s$s"; $tagS = "_${run}_S4D2_s$s"
    $dense[$s] = "$dDir\multiscene_final_stage_2$tagD.pth"
    $hybrid[$s] = "$sDir\multiscene_final_stage_2$tagS.pth"
    $steps += @{ name = "seed ${s}: D2 dense expert, frozen fusion ($EpochsDense epochs)"; retries = 1; dirs = @($dDir)
                 cmd = "$train --stage 2 --resume $stage1 --epochs $EpochsDense --patience2 $EpochsDense --seed $s " +
                       "--dense_only --stage2_select dense --sparse_loss mse --count_loss_weight 0 --ckpt_tag $tagD " +
                       "$shared $(StageOut $dDir 'stage2_dense')" }
    $steps += @{ name = "seed ${s}: S4D2 deep sparse expert on the frozen D2 ($EpochsSparse epochs)"; retries = 1; dirs = @($sDir)
                 cmd = "$train --stage 2 --resume $($dense[$s]) --epochs $EpochsSparse --patience2 $EpochsSparse --seed $s " +
                       "--sparse_only --reinit_sparse --stage2_select sparse_f1 --count_loss_weight 0.01 $sparseS4 " +
                       "--ckpt_tag $tagS $shared $(StageOut $sDir 'stage2_sparse')" }
}
if (-not $SkipControl) {
    $steps += @{ name = "H0-control: decoder fine-tune on the frozen fusion ($EpochsControl epochs)"; retries = $MaxAttempts; dirs = @($controlDir)
                 cmd = "$py .\training_scripts\neck_ablation.py --run_name ${run}_h0ctrl --variants $Neck --epochs $EpochsControl " +
                       "--frames_per_scene $FramesPerScene --init_from $h0 --freeze_fusion --out_root $controlDir " +
                       ">> $controlDir\control_log.txt 2>> $controlDir\control_error.txt" }
}
foreach ($s in $seedList) {
    $steps += @{ name = "seed ${s}: density-rule sweep, threshold by val GAME3"; retries = 1; dirs = @($evalDir)
                 cmd = "$py .\scripts\diagnosis\sweep_density_rule.py --ckpt hyb_s$s=$($hybrid[$s]) --out_dir $evalDir\density_sweep_s$s " +
                       "--select game3 --write_routing $evalDir\routing_s$s.txt > $evalDir\sweep_s${s}_log.txt 2>&1" }
}
$splits = @("test") + $(if ($WithVal) { @("validate") } else { @() })
foreach ($sp in $splits) {
    $evals = @(@{ n = "H0"; a = "--model ablation --weights $h0" })
    if (-not $SkipControl) { $evals += @{ n = "H0control"; a = "--model ablation --weights $h0c" } }
    foreach ($s in $seedList) {
        $evals += @{ n = "D2_s$s"; a = "--model proposed --routing dense --weights $($dense[$s])" }
        $evals += @{ n = "hybrid_s$s"; a = "--model proposed --routing @$evalDir\routing_s$s.txt --weights $($hybrid[$s])" }
    }
    foreach ($e in $evals) {
        $steps += @{ name = "paper protocol ${sp}: $($e.n)"; retries = 1; dirs = @($evalDir)
                     cmd = "$pp --split $sp $($e.a) --out $evalDir\eval_${sp}_$($e.n).json $(Redir $evalDir "eval_${sp}_$($e.n)")" }
    }
}

$itPerEpoch = 28 * $FramesPerScene
Log "=== chain start (steps $StartStep..$($steps.Count)) | run $run | head seeds $($seedList -join ', ') | output $workDir\$OutDir ==="
$budgetFmt = "budget per head seed: {0} + {1} + {2} = {3} epochs x {4} it = {5:N0} iterations; H0-control {0} + {6} " +
             "epochs (baseline run_1: 200 x 1400 = 280,000)"
Log ($budgetFmt -f $Epochs1, $EpochsDense, $EpochsSparse, ($Epochs1 + $EpochsDense + $EpochsSparse), $itPerEpoch,
     (($Epochs1 + $EpochsDense + $EpochsSparse) * $itPerEpoch), $EpochsControl)
for ($i = 0; $i -lt $steps.Count; $i++) { Log ("  {0,2}. {1}" -f ($i + 1), $steps[$i].name) }
if ($DryRun) { foreach ($s in $steps) { Write-Host "`n# $($s.name)`n$($s.cmd)" }; exit 0 }
foreach ($s in $steps) { foreach ($d in $s.dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null } }
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
