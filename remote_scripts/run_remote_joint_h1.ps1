param(
    [int]$Epochs = 200,           # 200 x 1400 it = 280k iterations = the curriculum's (and the baseline's) budget
    [int]$FramesPerScene = 50,    # 28 train scenes x 50 frames = 1400 iterations/epoch
    [int]$ValEvery = 5,           # validation for the log only (final weights are the result)
    [double]$RouterWeight = 1.0,  # w_r (doc/joint_training.md §2: fixed from the smoke run's first-epoch terms)
    [double]$FusedWeight = 1.0,   # w_f
    [double]$SparseScale = 0.01,  # the OT step's sparse loss x s, mix kept (smoke 2026-10-31: sparse 32.9 vs dense 0.30,
                                  # router 0.92, fused 0.13 per patch; x0.01 brings all four within ~7x)
    [string]$OutDir = "full_length_joint_homography",   # folder under src\ for every weight file and log (no spaces)
    [int]$ResumeEpoch = 0,        # after a crash: restart step 2 from multiscene_checkpoint_..._epoch_N.pth (weights only,
                                  # LR schedule fast-forwarded, Adam fresh -- approximate until WP2's exact resume)
    [int]$StartStep = 1,
    [int]$StopAfter = 4,          # -StopAfter 1: only the smoke run
    [switch]$DryRun,
    [switch]$Foreground           # internal: run the steps in this process (the default relaunches itself detached)
)

# Joint run J of doc/joint_training.md: the H1 model on a fresh homography_256 fusion, neck + both experts + router
# trained together in one loop (no curriculum), compared with the curriculum's h1_hom (test 64.4 learned / 52.7 oracle).
# Steps:
#   1  smoke: 2 epochs x 1 frame/scene (finite terms, the four loss terms' balance in 0_smoke\val_logs\train_joint_*.csv)
#   2  joint training, $Epochs epochs x $FramesPerScene frames, validation every $ValEvery epochs
#   3  paper protocol test: final weights, learned routing      4  final weights, oracle routing
# Everything goes to src\$OutDir\ (chain_log.txt, 0_smoke\, 1_joint\, eval\).

$workDir = Split-Path $PSScriptRoot -Parent
Set-Location $workDir
$inv = [System.Globalization.CultureInfo]::InvariantCulture
$wr, $wf, $ss = $RouterWeight.ToString($inv), $FusedWeight.ToString($inv), $SparseScale.ToString($inv)
# sparse loss of the OT step: OT 1 + bg 0.05 + count 1.0, plus the trainer's own --count_loss_weight 0.01 on the same
# count term; x s keeps that mix: OT s, bg 0.05 s, genloss count 1.01 s - 0.01 (the 0.01 is added by the trainer)
if ($SparseScale -lt 0.0099) { Write-Host "-SparseScale must be >= 0.0099 (count weight 1.01 s - 0.01 >= 0)" -ForegroundColor Red; exit 1 }
$gBg = [Math]::Round(0.05 * $SparseScale, 8).ToString($inv)
$gCount = [Math]::Round(1.01 * $SparseScale - 0.01, 8).ToString($inv)
$dirs = [ordered]@{ smoke = "$OutDir\0_smoke"; joint = "$OutDir\1_joint"; eval = "$OutDir\eval" }
$chainLog = "$OutDir\chain_log.txt"
$tag = "_hom_s${ss}_wr${wr}_wf${wf}"
$final = "$($dirs.joint)\multiscene_final_stage_joint$tag.pth"
$resume = "$($dirs.joint)\multiscene_checkpoint_stage_joint${tag}_epoch_$ResumeEpoch.pth"
if ($ResumeEpoch -gt 0 -and -not $DryRun -and -not (Test-Path "$workDir\$resume")) {
    Write-Host "Missing $workDir\$resume" -ForegroundColor Red; exit 1
}

if (-not $Foreground -and -not $DryRun) {
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Foreground -Epochs $Epochs " +
               "-FramesPerScene $FramesPerScene -ValEvery $ValEvery -RouterWeight $wr -FusedWeight $wf -SparseScale $ss -OutDir $OutDir " +
               "-ResumeEpoch $ResumeEpoch -StartStep $StartStep -StopAfter $StopAfter"
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = "conhost.exe --headless powershell.exe $argList"}
    if ($r.ReturnValue -eq 0) {
        Write-Host "Started detached chain -- PID $($r.ProcessId)" -ForegroundColor Green
        Write-Host "  Progress: Get-Content `"$workDir\$chainLog`" -Tail 20 -Wait"
        Write-Host "  Stop:     taskkill /PID $($r.ProcessId) /T /F"
    } else { Write-Host "Failed to start. Error code: $($r.ReturnValue)" -ForegroundColor Red }
    exit 0
}

if (-not $DryRun) { foreach ($d in $dirs.Values) { New-Item -ItemType Directory -Force -Path $d | Out-Null } }
function Log($msg) { $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"; if (-not $DryRun) { Add-Content -Path $chainLog -Value $line }; Write-Host $line }

$py = "uv run python -u"
$train = "$py .\training_scripts\train_proposed_multiscene.py --joint --neck ablation --neck_variant homography_256 " +
         "--dense_out_weight 0.7 --count_loss_weight 0.01 --sparse_loss genloss --sparse_genloss_weight $ss " +
         "--sparse_genloss_topk 1000 --sparse_genloss_bg_weight $gBg --sparse_genloss_count_weight $gCount " +
         "--sparse_nms_mult 1 --sparse_nms_kernel 7 --tversky_alpha 0.7 --no_router_contrast --precise_bn_frames 1 " +
         "--skip_camera_validation --joint_router_weight $wr --joint_fused_weight $wf --ckpt_tag $tag"
function Redir($dir, $name, $op = ">") { "$op $dir\${name}_log.txt 2$op $dir\${name}_error.txt" }   # '>>': a resumed run appends
function StageOut($dir, $name, $op = ">") { "--ckpt_dir $dir --val_log_dir $dir\val_logs --log_file \\.\NUL $(Redir $dir $name $op)" }

$steps = @()
$steps += @{ name = "smoke: 2 epochs x 1 frame/scene"
             cmd = "$train --epochs 2 --frames_per_scene 1 --val_frames_per_scene 1 $(StageOut $dirs.smoke 'smoke')" }
$from = if ($ResumeEpoch -gt 0) { "--resume $resume --start_epoch $ResumeEpoch " } else { "" }
$steps += @{ name = "joint training ($Epochs epochs x $FramesPerScene frames$(if ($ResumeEpoch) { ", from epoch $ResumeEpoch" }))"
             cmd = "$train --epochs $Epochs --frames_per_scene $FramesPerScene --val_frames_per_scene 10 --val_every $ValEvery " +
                   "--full_scene_val_frames 10 --full_scene_val_draws 2 $from$(StageOut $dirs.joint 'joint' '>>')" }
foreach ($routing in "learned", "oracle") {
    $base = [System.IO.Path]::GetFileNameWithoutExtension($final)
    $steps += @{ name = "paper protocol test: $base ($routing routing)"
                 cmd = "$py .\evaluation\evaluate_paper_protocol.py --model proposed --weights $final --split test " +
                       "--routing $routing --out $($dirs.eval)\eval_test_${base}_$routing.json $(Redir $dirs.eval "eval_test_${base}_$routing")" }
}

Log "=== chain start (steps $StartStep..$StopAfter) | lineage $tag | output $workDir\$OutDir ==="
for ($i = 0; $i -lt $steps.Count; $i++) { Log ("  {0}. {1}" -f ($i + 1), $steps[$i].name) }
if ($DryRun) { foreach ($s in $steps) { Write-Host "`n# $($s.name)`n$($s.cmd)" }; exit 0 }
for ($i = $StartStep - 1; $i -lt [Math]::Min($StopAfter, $steps.Count); $i++) {
    Log ("step {0}/{1}: {2}" -f ($i + 1), $steps.Count, $steps[$i].name)
    cmd.exe /c $steps[$i].cmd
    if ($LASTEXITCODE -ne 0) { Log "  failed (exit $LASTEXITCODE)"; Log "=== chain stopped at step $($i + 1) ==="; exit 1 }
}
Log "=== chain done ==="
