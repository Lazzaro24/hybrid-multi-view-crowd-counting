param(
    [string]$Neck = "homography_256",
    [int]$FramesPerScene = 50,    # 1400 iterations/epoch, as the budget chain
    [int]$EpochsOT = 40,          # OT fine-tune: 40 x 1400 = 56k steps (the Baseline (OT) fine-tune's best epoch 40 ~ 56k steps)
    [string]$Seeds = "42",        # head seeds of the budget chain to fine-tune
    [string]$RunTag = "bud200",
    [string]$OutDir = "full_length_hybrid",   # the budget chain's folder under src\ (its weights must still be there)
    [switch]$WithVal,             # also score the paper protocol on the val split
    [int]$StartStep = 1,
    [int]$MaxAttempts = 10,       # restarts of the neck_ablation fine-tune after a crash (it resumes from last.pth)
    [switch]$DryRun,
    [switch]$Foreground           # internal: run the steps in this process (the default relaunches itself detached)
)

# OT stage on top of the budget-matched homography chain (run_remote_budget_homography_hybrid.ps1, doc/head_ablation.md
# §8.18-8.19): the comparison with the Baseline (OT) (paper-protocol test MAE 46.2, F1 65.7), the SynMVCrowd baseline's
# MSE model fine-tuned with its OT loss (Adam 1e-5 constant). Both arms get that fine-tune on the frozen fusion:
#   H0-OT      the fully trained baseline decoder (H0-control, 200 epochs) + OT, decoder only
#              (neck_ablation.py --loss ot --init_from <H0-control> --freeze_fusion)
#   hybrid-OT  the hybrid's dense expert (D2) + OT, dense expert only, adapter and sparse expert frozen
#              (train_proposed_multiscene.py --dense_only --freeze_adapter --dense_loss ot, from the S4D2 final), so the
#              hybrid's sparse expert stays valid on top
# Same loss as the baseline's (losses/ot_loss/genloss_synmvcrowd.py, cost ed, scale 100; the dense map is rescaled to
# count x 100 units), same optimiser and schedule, EMA, final weights. Then the density-rule threshold by val GAME3 on
# the hybrid-OT (sweep_density_rule.py), and the paper protocol for H0-OT, D2-OT alone and the hybrid-OT.
#
# Output in src\$OutDir\ next to the budget chain's: 5_h0_ot\, 6_hybrid_ot_s<seed>\, eval\ (eval_<split>_H0ot.json,
# eval_<split>_D2ot_s<seed>.json, eval_<split>_hybridot_s<seed>.json, density_sweep_ot_s<seed>\, routing_ot_s<seed>.txt),
# chain_log_ot.txt. Bring back eval\ and chain_log_ot.txt (+ the final weights if further local analyses are wanted).

$workDir = Split-Path $PSScriptRoot -Parent
Set-Location $workDir
$run = "${RunTag}_$Neck"
$seedList = @($Seeds -split '[,\s]+' | Where-Object { $_ })
$evalDir = "$OutDir\eval"
$otDir = "$OutDir\5_h0_ot"
$chainLog = "$OutDir\chain_log_ot.txt"
$h0c = "$OutDir\4_h0_control\$Neck\last.pth"
$h0ot = "$otDir\$Neck\last.pth"

if (-not $Foreground -and -not $DryRun) {
    $need = @($h0c) + @($seedList | ForEach-Object { "$OutDir\3_sparse_s$_\multiscene_final_stage_2_${run}_S4D2_s$_.pth" })
    foreach ($f in $need) { if (-not (Test-Path "$workDir\$f")) { Write-Host "Missing $workDir\$f (run the budget chain first)" -ForegroundColor Red; exit 1 } }
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Foreground -Neck $Neck -FramesPerScene $FramesPerScene " +
               "-EpochsOT $EpochsOT -Seeds $($seedList -join ',') -RunTag $RunTag -OutDir $OutDir -StartStep $StartStep " +
               "-MaxAttempts $MaxAttempts" + $(if ($WithVal) { " -WithVal" } else { "" })
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
          "--no_router_contrast --route_cell 10 --stage2_unmasked --ema_decay 0.999"

$steps = @()
$steps += @{ name = "H0-OT: OT fine-tune of the H0-control decoder, frozen fusion ($EpochsOT epochs)"; retries = $MaxAttempts; dirs = @($otDir)
             cmd = "$py .\training_scripts\neck_ablation.py --run_name ${run}_h0ot --variants $Neck --loss ot --lr 1e-5 " +
                   "--lr_schedule constant --epochs $EpochsOT --frames_per_scene $FramesPerScene --init_from $h0c --freeze_fusion " +
                   "--out_root $otDir >> $otDir\h0ot_log.txt 2>> $otDir\h0ot_error.txt" }
$hyb = @{}
foreach ($s in $seedList) {
    $src = "$OutDir\3_sparse_s$s\multiscene_final_stage_2_${run}_S4D2_s$s.pth"
    $dir = "$OutDir\6_hybrid_ot_s$s"
    $tag = "_${run}_S4D2ot_s$s"
    $hyb[$s] = "$dir\multiscene_final_stage_2$tag.pth"
    $steps += @{ name = "seed ${s}: hybrid-OT, OT fine-tune of the dense expert only ($EpochsOT epochs)"; retries = 1; dirs = @($dir)
                 cmd = "$train --stage 2 --resume $src --epochs $EpochsOT --patience2 $EpochsOT --seed $s --lr 1e-5 " +
                       "--lr_schedule constant --dense_only --freeze_adapter --dense_loss ot --stage2_select dense " +
                       "--sparse_loss mse --count_loss_weight 0 --sparse_arch deep --ckpt_tag $tag $shared $(StageOut $dir 'stage2_dense_ot')" }
}
foreach ($s in $seedList) {
    $steps += @{ name = "seed ${s}: density-rule sweep on the hybrid-OT, threshold by val GAME3"; retries = 1; dirs = @($evalDir)
                 cmd = "$py .\scripts\diagnosis\sweep_density_rule.py --ckpt hybot_s$s=$($hyb[$s]) --out_dir $evalDir\density_sweep_ot_s$s " +
                       "--select game3 --write_routing $evalDir\routing_ot_s$s.txt > $evalDir\sweep_ot_s${s}_log.txt 2>&1" }
}
$splits = @("test") + $(if ($WithVal) { @("validate") } else { @() })
foreach ($sp in $splits) {
    $evals = @(@{ n = "H0ot"; a = "--model ablation --weights $h0ot" })
    foreach ($s in $seedList) {
        $evals += @{ n = "D2ot_s$s"; a = "--model proposed --routing dense --weights $($hyb[$s])" }
        $evals += @{ n = "hybridot_s$s"; a = "--model proposed --routing @$evalDir\routing_ot_s$s.txt --weights $($hyb[$s])" }
    }
    foreach ($e in $evals) {
        $steps += @{ name = "paper protocol ${sp}: $($e.n)"; retries = 1; dirs = @($evalDir)
                     cmd = "$pp --split $sp $($e.a) --out $evalDir\eval_${sp}_$($e.n).json $(Redir $evalDir "eval_${sp}_$($e.n)")" }
    }
}

Log "=== OT-stage chain start (steps $StartStep..$($steps.Count)) | run $run | seeds $($seedList -join ', ') | output $workDir\$OutDir ==="
Log ("OT fine-tune: {0} epochs x {1} it = {2:N0} steps per arm (Baseline (OT): best epoch 40 of its fine-tune, ~56k steps)" -f
     $EpochsOT, (28 * $FramesPerScene), ($EpochsOT * 28 * $FramesPerScene))
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
Log "=== OT-stage chain done ==="
