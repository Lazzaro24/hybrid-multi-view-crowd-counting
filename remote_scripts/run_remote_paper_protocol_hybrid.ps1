param(
    [string]$Seeds = "42,7",          # seeds of the dense-first recipe (D2 + S4D2) to score
    [string]$Split = "test",
    [string]$Threshold = "0.15",      # density-rule threshold (p/m^2); doc/head_ablation.md §8.10: 0.15 by val GAME3
    [switch]$SkipH0,                  # H0 already scored
    [switch]$SkipDense,               # D2 dense alone already scored (it does not depend on the threshold)
    [switch]$DryRun                   # print the commands and exit
)

# Paper protocol (evaluation/evaluate_paper_protocol.py: every frame of the split, 5 random 5-camera draws per frame)
# for the homography dense-first hybrid (doc/head_ablation.md §8.6 / §8.8) and its references on the same fusion:
#   H0      the neck-ablation run ablation_v2/homography_256 (baseline decoder, EMA weights, precise BN)
#   D2      the dense expert alone (final EMA weights), per seed
#   hybrid  D2 + S4D2 under the density rule (dzone<thr>p: dense where its own density >= -Threshold p/m^2, sparse
#           peaks elsewhere; §8.9 used 0.25, §8.10 fixed 0.15)
# Results JSON next to each checkpoint (eval_<split>_<checkpoint>[_<routing>].json); logs in logs\.
# Sequential, detached; stops at the first failure.

$workDir = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path "$workDir\logs")) { New-Item -ItemType Directory -Force -Path "$workDir\logs" | Out-Null }
$pp = "uv run python -u .\evaluation\evaluate_paper_protocol.py --split $Split --num_workers 8"

$steps = @()
if (-not $SkipH0) {
    $steps += "$pp --model ablation --weights runs\neck_ablation\ablation_v2\homography_256\last.pth " +
              "> logs\paper_protocol_${Split}_hom_H0_log.txt 2>&1"
}
foreach ($s in @($Seeds -split '[,\s]+' | Where-Object { $_ })) {
    $dense = "multiscene_final_stage_2_abl_homography_256_dense_D2_s${s}_ema.pth"
    $hybrid = "multiscene_final_stage_2_abl_homography_256_dense_S4D2_s${s}_ema.pth"
    foreach ($w in @($dense, $hybrid)) {
        if (-not (Test-Path "$workDir\$w")) { Write-Host "Missing $w" -ForegroundColor Red; exit 1 }
    }
    if (-not $SkipDense) {
        $steps += "$pp --model proposed --routing dense --weights $dense > logs\paper_protocol_${Split}_hom_D2_s${s}_log.txt 2>&1"
    }
    $dz = "dz" + ($Threshold -replace '^0\.', '')   # 0.25 -> dz25, 0.15 -> dz15
    $steps += "$pp --model proposed --routing dzone${Threshold}p --weights $hybrid > logs\paper_protocol_${Split}_hom_S4D2_s${s}_${dz}p_log.txt 2>&1"
}

if ($DryRun) { $steps | ForEach-Object { Write-Host $_ }; exit 0 }
$chain = ($steps | ForEach-Object { "($_)" }) -join " && "
$cmd = "conhost.exe --headless cmd.exe /c `"cd /d `"$workDir`" && $chain`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
if ($result.ReturnValue -eq 0) {
    Write-Host "Started $($steps.Count) steps -- PID $($result.ProcessId)" -ForegroundColor Green
    Write-Host "  Logs: logs\paper_protocol_${Split}_hom_*_log.txt"
    Write-Host "  Stop: taskkill /PID $($result.ProcessId) /T /F"
} else {
    Write-Host "Failed to start. Error code: $($result.ReturnValue)" -ForegroundColor Red
}
