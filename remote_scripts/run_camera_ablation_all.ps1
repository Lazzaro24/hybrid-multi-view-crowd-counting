param(
    [switch]$Foreground   # internal: run the chain in this process (the default relaunches itself detached)
)

# The whole camera ablation (doc/camera_ablation.md) on one machine, one part after the other (never two GPU jobs at once):
#   1. tier B vnorm   baseline_ot, h1_hom, h1_casa at 3/7/10 views     ~6.5 h
#   2. tier A seed 1  homography_256 + ca_sa trained at 3/5/7/10 views  ~8 h  (--val_every 5; one seed only)
#   3. tier B raw     the same three models                            ~6.5 h
#   4. tier B         hybrid_hom, vnorm and raw                        ~6 h
# Each part skips what is already finished, so rerunning this script after a crash or reboot continues where it stopped.
# Progress: camera_ablation\all_log.txt (parts), camera_ablation\chain_log.txt (evals),
#           runs\camera_ablation\<run>\chain_log.txt (training).

$workDir = Split-Path $PSScriptRoot -Parent
Set-Location $workDir

$baseline = "baseline_ot=baseline:learned:runs\baseline\ot\best.pth"
$h1hom = "h1_hom=proposed:learned:full_length_dense_mask_homography\4_router_anchor\multiscene_final_stage_4_bud200_homography_256_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth"
$h1casa = "h1_casa=proposed:learned:full_length_dense_mask_final\4_router_anchor\multiscene_final_stage_4_bud200_ca_sa_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth"
$hybrid = "hybrid_hom=proposed:dzone0.1p:full_length_hybrid\3_sparse_s42\multiscene_final_stage_2_bud200_homography_256_S4D2_s42.pth"
$three = "$baseline;$h1hom;$h1casa"

if (-not $Foreground) {
    $need = @("runs\train_2d\new_best\resnet18_2d_pretrained_ep4.pth", "datasets\SynMVCrowd\optimal_cameras_filtered.json") +
            @(foreach ($m in "$three;$hybrid" -split ';') { ($m -split ':', 3)[2] })
    $missing = @($need | Where-Object { -not (Test-Path "$workDir\$_") })
    if ($missing) { $missing | ForEach-Object { Write-Host "Missing $workDir\$_" -ForegroundColor Red }; exit 1 }
    $cmd = "conhost.exe --headless powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Foreground"
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd}
    if ($r.ReturnValue -eq 0) {
        Write-Host "Started the camera ablation chain -- PID $($r.ProcessId)" -ForegroundColor Green
        Write-Host "  Parts:    Get-Content `"$workDir\camera_ablation\all_log.txt`" -Tail 10 -Wait"
        Write-Host "  Evals:    Get-Content `"$workDir\camera_ablation\chain_log.txt`" -Tail 20 -Wait"
        Write-Host "  Training: Get-Content `"$workDir\runs\camera_ablation\v1\chain_log.txt`" -Tail 20 -Wait"
        Write-Host "  Stop:     taskkill /PID $($r.ProcessId) /T /F"
    } else { Write-Host "Failed to start. Error code: $($r.ReturnValue)" -ForegroundColor Red }
    exit 0
}

New-Item -ItemType Directory -Force -Path "camera_ablation" | Out-Null
function Log($msg) { Add-Content -Path "camera_ablation\all_log.txt" -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg" }

Log "=== camera ablation chain start ==="
Log "part 1/4: evals vnorm (baseline_ot, h1_hom, h1_casa)"
& "$PSScriptRoot\run_camera_ablation.ps1" -Foreground -Models $three -Views "3,7,10" -Modes vnorm
Log "part 2/4: training seed 1 (v1)"
& "$PSScriptRoot\run_camera_ablation_train.ps1" -Foreground -RunName v1 -ExtraArgs "--val_every 5"
Log "part 3/4: evals raw (baseline_ot, h1_hom, h1_casa)"
& "$PSScriptRoot\run_camera_ablation.ps1" -Foreground -Models $three -Views "3,7,10" -Modes raw
Log "part 4/4: evals hybrid_hom (vnorm, raw)"
& "$PSScriptRoot\run_camera_ablation.ps1" -Foreground -Models $hybrid -Views "3,7,10" -Modes "vnorm,raw"
Set-Location $workDir
Log "=== camera ablation chain done (check the part logs for steps that gave up) ==="
