param(
    [string]$Parts = "check,zeroshot,finetune",   # check = WP1 data check, zeroshot = WP3, smoke / finetune = WP4 runs
    [string]$Models = "",                         # instead of the default four: "name=model:routing:weights;..." (routing 'learned' for baseline)
    [string]$Only = "",                           # only these model names, e.g. "baseline_ot,h1_casa"
    [string]$Datasets = "wildtrack,multiviewx",
    [string]$Modes = "vnorm,raw",                 # zero-shot: vnorm = --view_norm (trained 5-view magnitude), raw = as trained
    [string]$Fractions = "10,100",                # fine-tune: percent of the train split, all 10 % runs before the 100 % ones
    [string]$Seeds = "0",                         # fine-tune: seeds, each the same for every model (it picks the 10 % subset);
                                                  # seed 0 writes <name>_<dataset>_<pct>, others <name>_<dataset>_<pct>_s<seed>
    [string]$ExtraArgs = "",                      # passed to every finetune_cross_dataset.py call, e.g. "--iters 3000 --lr 1e-5"
    [int]$MaxAttempts = 3,
    [int]$StartStep = 1,
    [switch]$DryRun,
    [switch]$Foreground                           # internal: run the steps in this process (the default relaunches itself detached)
)

# Cross-dataset evaluation and fine-tuning on Wildtrack and MultiviewX (doc/cross_dataset.md), one process per step,
# never two GPU jobs at once:
#   check     the OT baseline zero-shot, all views, raw, annotated-area map: must match doc section 2 (data copied correctly)
#   zeroshot  every model x dataset x {vnorm, raw}, all views                                    seconds per step
#   smoke     every model, 50 fine-tuning steps on Wildtrack at 100 % (7 views, the heaviest case) with -ExtraArgs:
#             train_log.csv gives s/step and peak GPU memory to fix --iters / --train_backbone; not in the default parts
#   finetune  every model x dataset at 10 %, then at 100 % of the train split (final weights, test eval at the end)
# Finished steps (their result JSON present) are skipped, a failed step is retried up to $MaxAttempts times (a
# fine-tune resumes from its last.pth), then left for later and the chain goes on. Rerun the same command to retry
# what is missing. Keep the PC awake (system sleep kills the run).
#
# Output: runs\cross_dataset\{check,zero_shot}\<name>_<dataset>[_vnorm].json,
#         runs\cross_dataset\{smoke\<name>_wildtrack, finetune\<name>_<dataset>_<pct>}\result.json + train_log.csv,
#         logs\<step>_log.txt / _error.txt, chain_log.txt.

$workDir = Split-Path $PSScriptRoot -Parent
Set-Location $workDir
$root = "runs\cross_dataset"
$chainLog = "$root\chain_log.txt"

# Weight paths as laid out on the remote machine (the camera-ablation set)
$default = "baseline_ot=baseline:learned:runs\baseline\ot\best.pth;" +
           "h1_hom=proposed:learned:full_length_dense_mask_homography\4_router_anchor\multiscene_final_stage_4_bud200_homography_256_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth;" +
           "h1_casa=proposed:learned:full_length_dense_mask_final\4_router_anchor\multiscene_final_stage_4_bud200_ca_sa_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth;" +
           "hybrid_hom=proposed:dzone0.1p:full_length_hybrid\3_sparse_s42\multiscene_final_stage_2_bud200_homography_256_S4D2_s42.pth"
$spec = if ($Models) { $Models } else { $default }
$modelList = foreach ($m in @($spec -split ';' | Where-Object { $_ })) {
    $name, $rest = $m -split '=', 2
    $model, $routing, $weights = $rest -split ':', 3
    @{ name = $name; model = $model; routing = $routing; weights = $weights }
}
if ($Only) {
    $onlyList = @($Only -split '[,\s]+' | Where-Object { $_ })
    $modelList = @($modelList | Where-Object { $onlyList -contains $_.name })
    if (-not $modelList) { Write-Host "No model matches -Only $Only" -ForegroundColor Red; exit 1 }
    $spec = ($modelList | ForEach-Object { "$($_.name)=$($_.model):$($_.routing):$($_.weights)" }) -join ';'
}
$partList = @($Parts -split '[,\s]+' | Where-Object { $_ })
$datasetList = @($Datasets -split '[,\s]+' | Where-Object { $_ })
$modeList = @($Modes -split '[,\s]+' | Where-Object { $_ })
$fractionList = @($Fractions -split '[,\s]+' | Where-Object { $_ } | Sort-Object { [int]$_ })
$seedList = @($Seeds -split '[,\s]+' | Where-Object { $_ } | ForEach-Object { [int]$_ })
$evalPy = "evaluation\evaluate_cross_dataset.py"
$tunePy = "training_scripts\finetune_cross_dataset.py"

if (-not $Foreground -and -not $DryRun) {
    $need = @(foreach ($m in $modelList) { $m.weights }) + @(foreach ($d in $datasetList) {
        @{ wildtrack = "datasets\Wildtrack\data\calibrations"; multiviewx = "datasets\MultiviewX\data\calibrations" }[$d] })
    if ($partList -contains 'finetune' -or $partList -contains 'smoke') { $need += $tunePy }
    $missing = @($need | Where-Object { -not (Test-Path "$workDir\$_") })
    if ($missing) { $missing | ForEach-Object { Write-Host "Missing $workDir\$_" -ForegroundColor Red }; exit 1 }
    if ($partList -contains 'zeroshot' -and -not (Select-String -Path $evalPy -Pattern "--routing" -SimpleMatch -Quiet)) {
        Write-Host "$evalPy has no --model proposed / --routing yet (WP3); run -Parts check for now" -ForegroundColor Red; exit 1
    }
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Foreground -Parts $($partList -join ',') " +
               "-Models `"$spec`" -Datasets $($datasetList -join ',') -Modes $($modeList -join ',') " +
               "-Fractions $($fractionList -join ',') -Seeds $($seedList -join ',') -ExtraArgs `"$ExtraArgs`" -MaxAttempts $MaxAttempts -StartStep $StartStep"
    $cmd = "conhost.exe --headless powershell.exe $argList"
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd}
    if ($r.ReturnValue -eq 0) {
        Write-Host "Started detached cross-dataset chain ($($partList -join ', ')) -- PID $($r.ProcessId)" -ForegroundColor Green
        Write-Host "  Steps:    Get-Content `"$workDir\$chainLog`" -Tail 20 -Wait"
        Write-Host "  Step log: Get-Content `"$workDir\$root\logs\<step>_log.txt`" -Tail 5"
        Write-Host "  Stop:     taskkill /PID $($r.ProcessId) /T /F"
    } else { Write-Host "Failed to start. Error code: $($r.ReturnValue)" -ForegroundColor Red }
    exit 0
}

if (-not $DryRun) { New-Item -ItemType Directory -Force -Path "$root\logs" | Out-Null }
function Log($msg) { $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"; if (-not $DryRun) { Add-Content -Path $chainLog -Value $line }; Write-Host $line }
$py = "uv run python -u"

# done = the file a finished step leaves behind
$steps = @()
if ($partList -contains 'check') {
    foreach ($d in $datasetList) {
        $out = "$root\check\baseline_ot_$d.json"
        $steps += @{ name = "check_baseline_ot_$d"; done = $out
                     cmd = "$py $evalPy --dataset $d --weights runs\baseline\ot\best.pth --min_map_px 0 --num_workers 8 --out $out" }
    }
}
if ($partList -contains 'zeroshot') {
    foreach ($mode in $modeList) {
        foreach ($d in $datasetList) {
            foreach ($m in $modelList) {
                $tag = "$($m.name)_$d$(if ($mode -eq 'vnorm') { '_vnorm' })"
                $steps += @{ name = "zeroshot_$tag"; done = "$root\zero_shot\$tag.json"
                             cmd = "$py $evalPy --dataset $d --model $($m.model) --routing $($m.routing) --weights $($m.weights) " +
                                   "--num_workers 8 $(if ($mode -eq 'vnorm') { '--view_norm' }) --out $root\zero_shot\$tag.json" }
            }
        }
    }
}
if ($partList -contains 'smoke') {
    foreach ($m in $modelList) {
        $dir = "$root\smoke\$($m.name)_wildtrack"
        # -ExtraArgs first, so the smoke's own --iters / --log_every win (argparse keeps the last value)
        $steps += @{ name = "smoke_$($m.name)"; done = "$dir\result.json"
                     cmd = "$py $tunePy --model $($m.model) --routing $($m.routing) --weights $($m.weights) --dataset wildtrack " +
                           "--fraction 1.0 --seed $($seedList[0]) --out_dir $dir $ExtraArgs --iters 50 --log_every 10" }
    }
}
if ($partList -contains 'finetune') {
    foreach ($seed in $seedList) {
        foreach ($pct in $fractionList) {
            foreach ($d in $datasetList) {
                foreach ($m in $modelList) {
                    $tag = "$($m.name)_${d}_$pct$(if ($seed -ne 0) { "_s$seed" })"
                    $dir = "$root\finetune\$tag"
                    $steps += @{ name = "finetune_$tag"; done = "$dir\result.json"
                                 cmd = "$py $tunePy --model $($m.model) --routing $($m.routing) --weights $($m.weights) --dataset $d " +
                                       "--fraction $([double]$pct / 100) --seed $seed --out_dir $dir $ExtraArgs" }
                }
            }
        }
    }
}
foreach ($dir in "$root\check", "$root\zero_shot", "$root\smoke", "$root\finetune") { if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $dir | Out-Null } }

Log "=== cross-dataset chain (steps $StartStep..$($steps.Count)) | parts $($partList -join ', ') | seeds $($seedList -join ', ') | extra args: '$ExtraArgs' ==="
for ($i = 0; $i -lt $steps.Count; $i++) { Log ("  {0,2}. {1}" -f ($i + 1), $steps[$i].name) }
if ($DryRun) { foreach ($s in $steps) { Write-Host "`n# $($s.name)`n$($s.cmd)" }; exit 0 }

$failed = @()
for ($i = $StartStep - 1; $i -lt $steps.Count; $i++) {
    $s = $steps[$i]
    if (Test-Path $s.done) { Log "step $($i + 1): $($s.name) already done"; continue }
    $ok = $false
    for ($a = 1; $a -le $MaxAttempts -and -not $ok; $a++) {
        Log ("step {0}/{1}: {2} (attempt {3}/{4})" -f ($i + 1), $steps.Count, $s.name, $a, $MaxAttempts)
        cmd.exe /c "$($s.cmd) >> $root\logs\$($s.name)_log.txt 2>> $root\logs\$($s.name)_error.txt"
        $ok = $LASTEXITCODE -eq 0
        if (-not $ok) { Log "  failed (exit $LASTEXITCODE), see logs\$($s.name)_error.txt"; Start-Sleep -Seconds 60 }
    }
    if (-not $ok) { $failed += $s.name; Log "  giving up on $($s.name) for now, continuing with the next step" }
}
if ($failed) { Log "=== done with failures: $($failed -join ', '); rerun the same command to retry them ===" }
else { Log "=== cross-dataset chain done ===" }
