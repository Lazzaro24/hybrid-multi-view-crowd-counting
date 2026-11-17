param(
    [string]$Necks = "homography_256,ca_sa",
    [int]$Epochs2 = 30,
    [int]$Patience2 = 15,
    [int]$StartStep = 1,          # resume the chain at a later step (the list is printed at start)
    [switch]$SkipNarrow,          # skip the narrow (64-ch) Stage 2 controls on the remote fusions
    [int]$MaxAttempts = 10,       # restarts of the fusion training after a crash (it resumes from last.pth)
    [switch]$Foreground           # internal: run the steps in this process (the default relaunches itself detached)
)

# Dense-capacity control on a machine WITHOUT the local generated files (doc/head_ablation.md §8). Rebuilds everything
# from the data + the 2D-pretrained backbone (runs\train_2d\new_best\resnet18_2d_pretrained_ep4.pth must exist):
#   1  reference-point cache master_refs.pt (utils\cache_reference.py, default = the 4-anchor cache of ablation_v2)
#   2  the two fusions, exactly the ablation_v2 recipe (all neck_ablation.py defaults: 30 epochs, seed 1, EMA)
#      -> runs\neck_ablation\ablation_v2r\<neck>\ ; this is also the baseline-head reference H0 on these fusions
#   3  Stage 1 checkpoints from them (make_stage1_from_ablation.py) -> multiscene_best_stage_1_abl_v2r_<neck>.pth
#   4  Stage 2, wide dense head:  --neck_adapter none --dense_arch wide512 (baseline decoder stack 256->512->512->512->1)
#   5  Stage 2, narrow dense head (the current 64-ch adapter + expert), same recipe        [skip with -SkipNarrow]
#      Both: the homography lineage's Stage 2 recipe (joint unmasked, dense MSE + sparse genloss, lr 5e-5), selected on
#      the dense expert alone (--stage2_select dense), tags _abl_<neck>_{wide512,narrow}_unmasked_joint_genloss_r
#   6  map-quality frames (K-means cameras, 50 frames per held-out scene): H0 and every dense expert alone
#   7  paper protocol (random camera draws), every dense expert alone
# Everything carries an "r" suffix so it never collides with the local runs when copied back.
# Bring back: runs\neck_ablation\ablation_v2r\, multiscene_*_v2r_* and multiscene_*_r.pth, logs\*_r_*, logs\remote_full_*,
# diagnostics\map_quality\{ablation_v2r__*,proposed__*_r}.csv, eval_test_*_r_dense.json.
# Do NOT copy back diagnostics\map_quality\summary.json (here it only holds this run's models).

$workDir = Split-Path $PSScriptRoot -Parent
Set-Location $workDir
if (-not (Test-Path "$workDir\logs")) { New-Item -ItemType Directory -Force -Path "$workDir\logs" | Out-Null }
$chainLog = "logs\remote_full_wide512_chain_log.txt"

if (-not $Foreground) {
    $backbone = "$workDir\runs\train_2d\new_best\resnet18_2d_pretrained_ep4.pth"
    if (-not (Test-Path $backbone)) { Write-Host "Missing $backbone" -ForegroundColor Red; exit 1 }
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Foreground -Necks $Necks -Epochs2 $Epochs2 " +
               "-Patience2 $Patience2 -StartStep $StartStep -MaxAttempts $MaxAttempts" + $(if ($SkipNarrow) { " -SkipNarrow" } else { "" })
    $cmd = "conhost.exe --headless powershell.exe $argList"
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd}
    if ($r.ReturnValue -eq 0) {
        Write-Host "Started detached chain -- PID $($r.ProcessId)" -ForegroundColor Green
        Write-Host "  Progress: Get-Content `"$workDir\$chainLog`" -Tail 20 -Wait"
        Write-Host "  Stop:     taskkill /PID $($r.ProcessId) /T /F"
    } else { Write-Host "Failed to start. Error code: $($r.ReturnValue)" -ForegroundColor Red }
    exit 0
}

function Log($msg) { $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"; Add-Content -Path $chainLog -Value $line; Write-Host $line }

$neckList = $Necks -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$run = "ablation_v2r"
$shared = "--frames_per_scene 10 --val_frames_per_scene 10 --precise_bn_frames 1 --skip_camera_validation " +
          "--no_router_contrast --route_cell 10"
$stage2Flags = "--stage2_unmasked --sparse_loss genloss --sparse_genloss_weight 1.0 --sparse_genloss_topk 1000 " +
               "--sparse_genloss_bg_weight 0.05 --sparse_genloss_count_weight 1.0 --sparse_nms_mult 1 --sparse_nms_kernel 7 " +
               "--dense_out_weight 0.7 --count_loss_weight 0.01 --lr 5e-5 --stage2_select dense " +
               "--epochs $Epochs2 --patience2 $Patience2"
$heads = [ordered]@{ "wide512" = "--neck_adapter none --dense_arch wide512" }
if (-not $SkipNarrow) { $heads["narrow"] = "" }

# --- the step list: @{ name; cmd; retries } -------------------------------------------------------------------------
$steps = @()
$steps += @{ name = "reference cache master_refs.pt";
             cmd = "uv run python -u .\utils\cache_reference.py > logs\remote_full_cache_refs_log.txt 2>&1"; retries = 1 }
$steps += @{ name = "fusions $($neckList -join ', ') (neck_ablation.py, $run)";
             cmd = "uv run python -u .\training_scripts\neck_ablation.py --run_name $run --variants $($neckList -join ',') " +
                   ">> logs\neck_ablation_${run}_log.txt 2>> logs\neck_ablation_${run}_error.txt"; retries = $MaxAttempts }
foreach ($n in $neckList) {
    $steps += @{ name = "Stage 1 checkpoint $n";
                 cmd = "uv run python -u .\training_scripts\make_stage1_from_ablation.py --run $run --variant $n " +
                       "> logs\remote_full_make_stage1_${n}_log.txt 2>&1"; retries = 1 }
}
$weights = @()
foreach ($h in $heads.Keys) {
    foreach ($n in $neckList) {
        $tag = "_abl_${n}_${h}_unmasked_joint_genloss_r"
        $weights += @{ name = "${n}_${h}_r"; file = "multiscene_best_stage_2$tag.pth" }
        $steps += @{ name = "Stage 2 $h head on $n";
                     cmd = "uv run python -u .\training_scripts\train_proposed_multiscene.py --stage 2 " +
                           "--resume multiscene_best_stage_1_abl_v2r_$n.pth $($heads[$h]) $stage2Flags --ckpt_tag $tag $shared " +
                           "> logs\stage2${tag}_log.txt 2> logs\stage2${tag}_error.txt"; retries = 1 }
    }
}
$proposed = ($weights | ForEach-Object { "$($_.name)_dense=$($_.file):dense" }) -join ','
$models = ($neckList | ForEach-Object { "${run}:$_" }) -join ','
$steps += @{ name = "map-quality frames: H0 + dense experts";
             cmd = "uv run python -u .\evaluation\eval_map_quality.py --models $models --baselines= --proposed $proposed " +
                   "> logs\remote_full_eval_map_quality_r_log.txt 2>&1"; retries = 1 }
foreach ($w in $weights) {
    $steps += @{ name = "paper protocol, dense alone: $($w.name)";
                 cmd = "uv run python -u .\evaluation\evaluate_paper_protocol.py --model proposed --routing dense " +
                       "--weights $($w.file) --split test > logs\eval_test_$($w.name)_dense_log.txt 2>&1"; retries = 1 }
}

Log "=== chain start (steps $StartStep..$($steps.Count)) ==="
for ($i = 0; $i -lt $steps.Count; $i++) { Log ("  {0,2}. {1}" -f ($i + 1), $steps[$i].name) }
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
