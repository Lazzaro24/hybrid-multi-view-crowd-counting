param(
    [string]$Neck = "ca_sa",          # ca_sa | homography_256: the ablation fusion (frozen) the dense head is trained on
    [string]$Runs = "D1,D2,D3",       # runs to train and score
    [int]$Seed = 42,
    [int]$Epochs = 30,
    [int]$DeepEpochs = 45,            # runs with the deep sparse head (S4D2): it converges slowly
    [string]$DenseBase = "D2_s42_ema",  # S4D2: the dense run whose final checkpoint is the frozen base
    [int]$Patience = 30,              # no early stop by default: the final (epoch 30) weights are scored too
    [double]$EmaDecay = 0,            # >0: validate / select / save an EMA of the weights (H0's recipe: 0.999); tags get _ema
    [string]$Snapshots = "",          # also score these periodic checkpoints, e.g. "10,15,20,25" (names get _ep<N>)
    [string]$EvalRuns = "",           # runs to score; default = -Runs (e.g. train D2,D3 but score D1,D2,D3)
    [switch]$EvalOnly,                # skip training, only score the runs' checkpoints
    [switch]$DryRun                   # print the commands and exit
)

# What limits the dense expert (doc/dense_expert_plan.md §2). Each run is a dense-only Stage 2 from the ablation's own
# Stage 1 (backbone + the ablation fusion = H0's features, no head): fusion frozen (eval mode), only the dense head
# trains (with the 64-ch adapter where there is one), whole patch (--stage2_unmasked), H0's lr 1e-4, selected on the
# dense expert alone. The sparse expert is frozen; --sparse_loss mse avoids the OT solve every step.
#   D1  64-ch adapter (BN) + dilated dense expert, MSE + count term 0.01   (A: frozen fusion vs Stage 2 joint)
#   D2  as D1, MSE only                                                    (B: the count term)
#   D3  --neck_adapter none --dense_arch wide512 = H0's decoder, MSE only  (sanity check: should reproduce H0)
#   J1  joint Stage 2 on the frozen fusion (--freeze_fusion): adapter + both experts train; dense = D2 (MSE only),
#       sparse = S1's loss (genloss + empty-ground mass penalty); also scored under the density rule (dzone0.25p)
#   S4D2 sparse-only on top of the frozen final D2 checkpoint (-DenseBase): fusion, adapter and dense expert frozen, a
#       fresh deep sparse head (S4: S1's loss + --sparse_arch deep, 45 epochs); scored under the density rule only
#       (the dense expert is D2's, already scored)
# -Neck takes a list ("ca_sa,homography_256"): the runs are trained neck after neck, then scored in one eval call.
# -EmaDecay 0.999 trains the same runs with H0's weight EMA (tags / names get the suffix _ema).
# Then eval_map_quality.py on best + final of every run (proposed__<neck>_<run>_s<seed>[_final].csv) and a rebuild of
# diagnostics\map_quality\summary.json over all models (the name lists of the summary.json found at launch + the new runs).
# Sequential, detached; stops at the first failure.

$workDir = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path "$workDir\logs")) { New-Item -ItemType Directory -Force -Path "$workDir\logs" | Out-Null }
$shortOf = @{ "ca_sa" = "casa"; "homography_256" = "hom" }
$neckList = @($Neck -split '[,\s]+' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($n in $neckList) {
    if (-not $shortOf[$n]) { Write-Host "Unknown neck $n" -ForegroundColor Red; exit 1 }
    if (-not $EvalOnly -and -not (Test-Path "$workDir\multiscene_best_stage_1_abl_v2_$n.pth")) {
        Write-Host "Missing multiscene_best_stage_1_abl_v2_$n.pth" -ForegroundColor Red; exit 1 }
}

$shared = "--frames_per_scene 10 --val_frames_per_scene 10 --precise_bn_frames 1 --skip_camera_validation " +
          "--no_router_contrast --route_cell 10"
$common = "--stage 2 --stage2_unmasked --lr 1e-4 --dense_out_weight 0.7 --patience2 $Patience --seed $Seed"
$emaSuffix = ""
if ($EmaDecay -gt 0) { $common += " --ema_decay $EmaDecay"; $emaSuffix = "_ema" }
$denseOnly = "--dense_only --stage2_select dense --sparse_loss mse"
# S1's sparse loss (run_remote_sparse_expert_s012.ps1): genloss + empty-ground mass penalty, shallow sparse expert
$sparseS1 = "--sparse_loss genloss --sparse_genloss_weight 1.0 --sparse_genloss_topk 1000 --sparse_genloss_count_weight 1.0 " +
            "--sparse_nms_mult 1 --sparse_nms_kernel 3 --sparse_genloss_bg_weight 1.0 --sparse_genloss_bg_mode mass " +
            "--sparse_bg_empty_thr 1e-3"
$defs = [ordered]@{
    "D1" = "$denseOnly --count_loss_weight 0.01"
    "D2" = "$denseOnly --count_loss_weight 0"
    "D3" = "$denseOnly --count_loss_weight 0 --neck_adapter none --dense_arch wide512"
    "J1" = "--freeze_fusion --stage2_select experts --count_loss_weight 0 $sparseS1"
    # S4's recipe exactly (run_remote_sparse_expert_s012.ps1), incl. its count weight (it reaches only the sparse expert)
    "S4D2" = "--sparse_only --reinit_sparse --stage2_select sparse_f1 --count_loss_weight 0.01 $sparseS1 --sparse_arch deep"
}
function Get-Resume($n, $r) {
    if ($r -like "S*") { return "multiscene_final_stage_2_abl_${n}_dense_${DenseBase}.pth" }
    return "multiscene_best_stage_1_abl_v2_$n.pth"
}

$runList = @($Runs -split '[,\s]+' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($r in $runList) { if (-not $defs.Contains($r)) { Write-Host "Unknown run $r" -ForegroundColor Red; exit 1 } }

$evalList = if ($EvalRuns) { @($EvalRuns -split '[,\s]+' | ForEach-Object { $_.Trim() } | Where-Object { $_ }) } else { $runList }
foreach ($r in $evalList) { if (-not $defs.Contains($r)) { Write-Host "Unknown run $r" -ForegroundColor Red; exit 1 } }
$snapList = @($Snapshots -split '[,\s]+' | ForEach-Object { $_.Trim() } | Where-Object { $_ })

$steps = @()
$proposed = @()   # already-scored CSVs are skipped by eval_map_quality.py
foreach ($n in $neckList) {
    if (-not $EvalOnly) {
        foreach ($r in $runList) {
            $tag = "_abl_${n}_dense_${r}_s${Seed}${emaSuffix}"
            $resume = Get-Resume $n $r
            # the dense base of a sparse-only run may be trained earlier in this chain (e.g. -Runs "D2,S4D2" -Seed 7
            # -DenseBase D2_s7_ema)
            $baseInChain = ($r -like "S*") -and ($runList -contains ($DenseBase -split '_')[0]) -and
                           ($DenseBase -eq "$(($DenseBase -split '_')[0])_s${Seed}${emaSuffix}")
            if (-not $baseInChain -and -not (Test-Path "$workDir\$resume")) { Write-Host "Missing $resume" -ForegroundColor Red; exit 1 }
            $ep = if ($defs[$r] -match '--sparse_arch deep') { $DeepEpochs } else { $Epochs }
            $steps += "uv run python -u .\training_scripts\train_proposed_multiscene.py --resume $resume --epochs $ep " +
                      "$common $($defs[$r]) --ckpt_tag $tag $shared > logs\stage2${tag}_log.txt 2> logs\stage2${tag}_error.txt"
        }
    }
    foreach ($r in $evalList) {
        $tag = "_abl_${n}_dense_${r}_s${Seed}${emaSuffix}"
        $name = "$($shortOf[$n])_${r}_s${Seed}${emaSuffix}"
        $ckpts = [ordered]@{ "" = "multiscene_best_stage_2${tag}.pth"; "_final" = "multiscene_final_stage_2${tag}.pth" }
        foreach ($e in $snapList) { $ckpts["_ep$e"] = "multiscene_checkpoint_stage_2${tag}_epoch_${e}.pth" }
        foreach ($k in $ckpts.Keys) {
            # joint runs: also the hybrid under the density rule (dense where its own density >= 0.25 p/m^2, sparse as
            # peaks); sparse-only runs: only the hybrid (their dense expert is the frozen base's)
            if ($r -notlike "S*") { $proposed += "${name}${k}=$($ckpts[$k]):dense" }
            if (($r -like "J*" -and $k -ne "") -or $r -like "S*") { $proposed += "${name}${k}_dz25p=$($ckpts[$k]):dzone0.25p" }
        }
    }
}
$evalTag = "_abl_$($neckList -join '_')_dense_$($evalList -join '')_s${Seed}${emaSuffix}"
$steps += "uv run python -u .\evaluation\eval_map_quality.py --models= --baselines= " +
          "--proposed $($proposed -join ',') > logs\eval_map_quality${evalTag}_log.txt 2>&1"

# summary.json is rewritten with only the models of the call above: rebuild it over everything
$summaryPath = "$workDir\diagnostics\map_quality\summary.json"
$prev = if (Test-Path $summaryPath) { (Get-Content $summaryPath -Raw | ConvertFrom-Json).args } else { $null }
$newNames = @($proposed | ForEach-Object { ($_ -split '=')[0] + "=x" })
$prevProposed = if ($prev -and $prev.proposed) { @($prev.proposed -split ',' | Where-Object { $_ -and ($newNames -notcontains $_) }) } else { @() }
$allProposed = ($prevProposed + $newNames) -join ','
$prevModels = if ($prev) { $prev.models } else { "" }
$prevBaselines = if ($prev) { $prev.baselines } else { "" }
$steps += "uv run python -u .\evaluation\eval_map_quality.py --analyze_only --models=$prevModels " +
          "--baselines=$prevBaselines --proposed=$allProposed > logs\eval_map_quality_summary_all.log 2>&1"

if ($DryRun) { $steps | ForEach-Object { Write-Host $_ }; exit 0 }
$chain = ($steps | ForEach-Object { "($_)" }) -join " && "
$cmd = "conhost.exe --headless cmd.exe /c `"cd /d `"$workDir`" && $chain`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
if ($result.ReturnValue -eq 0) {
    Write-Host "Started $($steps.Count) steps -- PID $($result.ProcessId)" -ForegroundColor Green
    foreach ($n in $neckList) { foreach ($r in $runList) { Write-Host "  $n $r log: logs\stage2_abl_${n}_dense_${r}_s${Seed}${emaSuffix}_log.txt" } }
    Write-Host "  Eval: logs\eval_map_quality${evalTag}_log.txt"
    Write-Host "  Stop: taskkill /PID $($result.ProcessId) /T /F"
} else {
    Write-Host "Failed to start. Error code: $($result.ReturnValue)" -ForegroundColor Red
}
