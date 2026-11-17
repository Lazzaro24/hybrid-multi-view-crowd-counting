param(
    [string]$Only = "",           # comma-separated step names to run (default: all, in the order below)
    [string]$Split = "test",
    [string]$OutDir = "runs\final_evals",
    [string]$Script = ".\evaluation\evaluate_paper_protocol.py",  # another copy of the script, e.g. next to a running chain
    [string]$Thresholds = "0.1,0.2,0.3,0.4,0.5,0.6,0.8",
    [int]$MaxAttempts = 3,
    [switch]$DryRun,
    [switch]$Foreground           # internal: run the steps in this process (the default relaunches itself detached)
)

# Final evaluations, E1/E4 (doc/final_evaluations.md): the paper protocol on the final models with one row per sample
# (--dump_samples -> <name>.samples.csv) and a detection threshold sweep (--cls_thres_list). No --per_scene and
# --num_workers 8, seed 0: the same frames and camera draws as the published results, so every learned/dzone step must
# reproduce its published MAE / MODA (MATCH in the log; another GPU: MAE within ~0.1, MODA within ~0.5). The oracle
# steps use the Stage 4 weights; their reference is the Stage 3 oracle result (same experts, Stage 4 trains the router
# only), so a MISMATCH there is a finding, not an error.
#
# A failed step is retried up to $MaxAttempts times, then left for later and the chain goes on; finished steps
# (<name>.json and <name>.samples.csv present) are skipped. Rerun the same command to retry. Each eval < 1 h and
# < 4 GB GPU memory, so it can run next to a training chain (check nvidia-smi first).
#
# Output: src\$OutDir\<name>.json, .samples.csv, _log.txt, _error.txt, chain_log.txt.

$workDir = Split-Path $PSScriptRoot -Parent
Set-Location $workDir
$chainLog = "$OutDir\chain_log.txt"

# name, model, routing, weights (remote layout), published MAE, MODA (null = no reference)
$h1hom = "full_length_dense_mask_homography\4_router_anchor\multiscene_final_stage_4_bud200_homography_256_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth"
$h1casa = "full_length_dense_mask_final\4_router_anchor\multiscene_final_stage_4_bud200_ca_sa_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth"
$all = @(
    @{ name = "baseline_ot";    model = "baseline"; routing = "learned";   weights = "runs\baseline\ot\best.pth";    mae = 46.2219; moda = 43.5248 },
    @{ name = "h1_hom";         model = "proposed"; routing = "learned";   weights = $h1hom;                         mae = 64.3507; moda = -164.3143 },
    @{ name = "h1_hom_oracle";  model = "proposed"; routing = "oracle";    weights = $h1hom;                         mae = 52.7424; moda = -96.1278 },
    @{ name = "h1_casa";        model = "proposed"; routing = "learned";   weights = $h1casa;                        mae = 74.1347; moda = -436.3199 },
    @{ name = "h1_casa_oracle"; model = "proposed"; routing = "oracle";    weights = $h1casa;                        mae = 65.1463; moda = -137.1332 },
    @{ name = "baseline_mse";   model = "baseline"; routing = "learned";   weights = "runs\baseline\run_1\best.pth"; mae = 66.9676; moda = -118.3072 },
    @{ name = "hybrid_hom";     model = "proposed"; routing = "dzone0.1p"; weights = "full_length_hybrid\3_sparse_s42\multiscene_final_stage_2_bud200_homography_256_S4D2_s42.pth"; mae = 77.9074; moda = -246.9178 }
)
$onlyList = @($Only -split '[,\s]+' | Where-Object { $_ })
$modelList = @($all | Where-Object { -not $onlyList -or $onlyList -contains $_.name })
if (-not $modelList) { Write-Host "No step matches -Only $Only" -ForegroundColor Red; exit 1 }

if (-not $Foreground -and -not $DryRun) {
    $need = @($Script, "runs\train_2d\new_best\resnet18_2d_pretrained_ep4.pth", "utils\weather_index.json") + @($modelList | ForEach-Object { $_.weights })
    $missing = @($need | Where-Object { -not (Test-Path "$workDir\$_") })
    if ($missing) { $missing | ForEach-Object { Write-Host "Missing $workDir\$_" -ForegroundColor Red }; exit 1 }
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Foreground -Split $Split -OutDir `"$OutDir`" " +
               "-Script `"$Script`" -Thresholds $Thresholds -MaxAttempts $MaxAttempts" + $(if ($Only) { " -Only $($onlyList -join ',')" } else { "" })
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = "conhost.exe --headless powershell.exe $argList"}
    if ($r.ReturnValue -eq 0) {
        Write-Host "Started detached final evaluations -- PID $($r.ProcessId)" -ForegroundColor Green
        Write-Host "  Progress: Get-Content `"$workDir\$chainLog`" -Tail 20 -Wait"
        Write-Host "  Stop:     taskkill /PID $($r.ProcessId) /T /F"
    } else { Write-Host "Failed to start. Error code: $($r.ReturnValue)" -ForegroundColor Red }
    exit 0
}

if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }
function Log($msg) { $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"; if (-not $DryRun) { Add-Content -Path $chainLog -Value $line }; Write-Host $line }
$pp = "uv run python -u $Script --num_workers 8 --seed 0 --split $Split --dump_samples --cls_thres_list $Thresholds"

Log "=== final evaluations start | split $Split | script $Script | output $workDir\$OutDir ==="
for ($i = 0; $i -lt $modelList.Count; $i++) { Log ("  {0}. {1}" -f ($i + 1), $modelList[$i].name) }
$failed = @()
for ($i = 0; $i -lt $modelList.Count; $i++) {
    $m = $modelList[$i]
    $out = "$OutDir\$($m.name)"
    $cmd = "$pp --model $($m.model) --routing $($m.routing) --weights $($m.weights) --out $out.json >> ${out}_log.txt 2>> ${out}_error.txt"
    if ($DryRun) { Write-Host "`n# $($m.name)`n$cmd"; continue }
    if ((Test-Path "$out.json") -and (Test-Path "$out.samples.csv")) { Log "step $($i + 1): $($m.name) already done"; continue }
    $ok = $false
    for ($a = 1; $a -le $MaxAttempts -and -not $ok; $a++) {
        Log ("step {0}/{1}: {2} (attempt {3}/{4})" -f ($i + 1), $modelList.Count, $m.name, $a, $MaxAttempts)
        cmd.exe /c $cmd
        $ok = $LASTEXITCODE -eq 0
        if (-not $ok) { Log "  failed (exit $LASTEXITCODE)"; Start-Sleep -Seconds 60 }
    }
    if (-not $ok) { $failed += $m.name; Log "  giving up on $($m.name) for now, continuing with the next step"; continue }
    $res = Get-Content "$out.json" -Raw | ConvertFrom-Json
    $dMae = [math]::Abs($res.MAE - $m.mae); $dModa = [math]::Abs($res.MODA - $m.moda)
    $verdict = if ($dMae -lt 0.0001 -and $dModa -lt 0.0001) { "MATCH (exact)" } elseif ($dMae -le 0.1 -and $dModa -le 0.5) { "MATCH (within GPU tolerance)" } else { "MISMATCH" }
    Log ("  {0}: MAE {1:N4} (published {2:N4}), MODA {3:N4} (published {4:N4}), {5} s -> {6}" -f $m.name, $res.MAE, $m.mae, $res.MODA, $m.moda, $res.seconds, $verdict)
}
if ($failed) { Log "=== done with failures: $($failed -join '; '); rerun the same command to retry them ===" }
else { Log "=== final evaluations done ===" }
