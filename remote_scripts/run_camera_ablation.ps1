param(
    [string]$Models = "",         # instead of the default four: "name=model:routing:weights;..." (routing 'learned' for baseline)
    [string]$Views = "3,7,10",    # 5 views = the existing paper-protocol results, not rerun
    [string]$Modes = "vnorm,raw", # vnorm = --view_norm (trained per-view magnitude restored), raw = as trained
    [string]$Split = "test",
    [string]$OutDir = "camera_ablation",
    [int]$StartStep = 1,
    [int]$MaxAttempts = 3,
    [switch]$DryRun,
    [switch]$Foreground           # internal: run the steps in this process (the default relaunches itself detached)
)

# Camera ablation, tier 1 (RQ3): the 5-view models scored with draws of N views (paper protocol otherwise: all
# frames, 5 seeded draws per frame), no retraining. evaluate_paper_protocol.py --num_views N; ca_sa's per-camera
# cross-attention slots are reused cyclically (utils/camera_slots.py).
# Both fusions scale with the camera count by construction: the view softmax (baseline, homography_256) gives each
# view 1/N, ca_sa sums its cameras. 'raw' shows that; 'vnorm' rescales to the trained 5-view magnitude and is the
# measurement of what more / fewer views give. vnorm runs first (all models and counts), then raw.
#
# Every eval runs with --per_scene: saved after each test scene (<out>.json.partial.json), so a crash costs at most one
# scene. A failed step is retried up to $MaxAttempts times (continuing from its last saved scene), then left for
# later and the chain goes on; finished steps (<out>.json present) are skipped. Rerun the same command to retry.
#
# Output: src\$OutDir\<name>_<N>v[_vnorm].json + _log.txt / _error.txt, chain_log.txt.

$workDir = Split-Path $PSScriptRoot -Parent
Set-Location $workDir
$chainLog = "$OutDir\chain_log.txt"

# Weight paths as laid out on the remote machine (the training chains' output folders)
$default = "baseline_ot=baseline:learned:runs\baseline\ot\best.pth;" +
           "hybrid_hom=proposed:dzone0.1p:full_length_hybrid\3_sparse_s42\multiscene_final_stage_2_bud200_homography_256_S4D2_s42.pth;" +
           "h1_hom=proposed:learned:full_length_dense_mask_homography\4_router_anchor\multiscene_final_stage_4_bud200_homography_256_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth;" +
           "h1_casa=proposed:learned:full_length_dense_mask_final\4_router_anchor\multiscene_final_stage_4_bud200_ca_sa_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth"
$spec = if ($Models) { $Models } else { $default }
$modelList = foreach ($m in @($spec -split ';' | Where-Object { $_ })) {
    $name, $rest = $m -split '=', 2
    $model, $routing, $weights = $rest -split ':', 3
    @{ name = $name; model = $model; routing = $routing; weights = $weights }
}
$viewList = @($Views -split '[,\s]+' | Where-Object { $_ })
$modeList = @($Modes -split '[,\s]+' | Where-Object { $_ })

if (-not $Foreground -and -not $DryRun) {
    foreach ($m in $modelList) { if (-not (Test-Path "$workDir\$($m.weights)")) { Write-Host "Missing $workDir\$($m.weights)" -ForegroundColor Red; exit 1 } }
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Foreground -Models `"$spec`" -Views $($viewList -join ',') " +
               "-Modes $($modeList -join ',') -Split $Split -OutDir $OutDir -StartStep $StartStep -MaxAttempts $MaxAttempts"
    $cmd = "conhost.exe --headless powershell.exe $argList"
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd}
    if ($r.ReturnValue -eq 0) {
        Write-Host "Started detached camera ablation -- PID $($r.ProcessId)" -ForegroundColor Green
        Write-Host "  Progress: Get-Content `"$workDir\$chainLog`" -Tail 20 -Wait"
        Write-Host "  Stop:     taskkill /PID $($r.ProcessId) /T /F"
    } else { Write-Host "Failed to start. Error code: $($r.ReturnValue)" -ForegroundColor Red }
    exit 0
}

if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }
function Log($msg) { $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"; if (-not $DryRun) { Add-Content -Path $chainLog -Value $line }; Write-Host $line }
$pp = "uv run python -u .\evaluation\evaluate_paper_protocol.py --num_workers 8 --split $Split --per_scene"

$steps = @()
foreach ($mode in $modeList) {
    foreach ($n in $viewList) {
        foreach ($m in $modelList) {
            $out = "$OutDir\$($m.name)_${n}v$(if ($mode -eq 'vnorm') { '_vnorm' })"
            $steps += @{ name = "$($m.name), $n views, $mode"; done = "$out.json"
                         cmd = "$pp --model $($m.model) --routing $($m.routing) --weights $($m.weights) --num_views $n " +
                               "$(if ($mode -eq 'vnorm') { '--view_norm' }) --out $out.json >> ${out}_log.txt 2>> ${out}_error.txt" }
        }
    }
}

Log "=== camera ablation start (steps $StartStep..$($steps.Count)) | split $Split | output $workDir\$OutDir ==="
for ($i = 0; $i -lt $steps.Count; $i++) { Log ("  {0,2}. {1}" -f ($i + 1), $steps[$i].name) }
if ($DryRun) { foreach ($s in $steps) { Write-Host "`n# $($s.name)`n$($s.cmd)" }; exit 0 }
$failed = @()
for ($i = $StartStep - 1; $i -lt $steps.Count; $i++) {
    $s = $steps[$i]
    if (Test-Path $s.done) { Log "step $($i + 1): $($s.name) already done"; continue }
    $ok = $false
    for ($a = 1; $a -le $MaxAttempts -and -not $ok; $a++) {
        Log ("step {0}/{1}: {2} (attempt {3}/{4})" -f ($i + 1), $steps.Count, $s.name, $a, $MaxAttempts)
        cmd.exe /c $s.cmd
        $ok = $LASTEXITCODE -eq 0
        if (-not $ok) { Log "  failed (exit $LASTEXITCODE)"; Start-Sleep -Seconds 60 }
    }
    if (-not $ok) { $failed += $s.name; Log "  giving up on $($s.name) for now, continuing with the next step" }
}
if ($failed) { Log "=== done with failures: $($failed -join '; '); rerun the same command to retry them ===" }
else { Log "=== camera ablation done ===" }
