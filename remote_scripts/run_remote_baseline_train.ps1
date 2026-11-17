# Launches the SynMVCrowd baseline training (frozen backbone, or trainable with -TrainBackbone) as a detached process.
# The defaults are the Baseline (OT) fine-tune: OT loss, Adam 1e-5 at a constant rate (released code),
# initialised from the trained MSE baseline, at most 100 epochs with early stopping after 25 epochs
# without a new best val MAE. Usage (from anywhere):
#   .\remote_scripts\run_remote_baseline_train.ps1 -Backbone runs\train_2d\<timestamp>\resnet18_2d_pretrained_ep4.pth
#   .\remote_scripts\run_remote_baseline_train.ps1 -Backbone <...> -Resume     # continue runs\baseline\<RunName>\last.pth
# Baseline (MSE) as trained for run_1 (paper text: SGD 1e-4, warmup + cosine over 200 epochs, from scratch):
#   .\remote_scripts\run_remote_baseline_train.ps1 -Backbone <...> -Loss mse -Optimizer sgd -Lr 1e-4 `
#       -LrSchedule cosine -InitFrom "" -Epochs 200 -Patience 0 -RunName mse_frozen
# Baseline (MSE) with the backbone trained jointly, as in the paper (weights + BatchNorm stats, same lr):
#   .\remote_scripts\run_remote_baseline_train.ps1 -Backbone <...> -Loss mse -Optimizer sgd -Lr 1e-4 `
#       -LrSchedule cosine -InitFrom "" -Epochs 200 -Patience 0 -TrainBackbone -RunName mse_unfrozen
param(
    # 2D-pretrained backbone -- must be a STRIDE-8 checkpoint from train_2d_pretrain.py
    [string]$Backbone = "runs\train_2d\SET_STRIDE8_RUN\resnet18_2d_pretrained_ep4.pth",
    [string]$RunName = "ot_frozen",
    [ValidateSet("mse", "ot")]
    [string]$Loss = "ot",
    # sgd = paper text (lr 1e-4); adam = released code (lr 1e-5) -- the OT loss diverges with sgd
    [ValidateSet("sgd", "adam")]
    [string]$Optimizer = "adam",
    [double]$Lr = 1e-5,
    # cosine = warmup + cosine over -Epochs (paper); constant = released code, pairs with -Patience
    [ValidateSet("cosine", "constant")]
    [string]$LrSchedule = "constant",
    # Checkpoint whose full model weights start the run ("" = train the decoder from scratch)
    [string]$InitFrom = "runs\baseline\run_1\best.pth",
    [int]$Epochs = 100,
    # Stop after this many epochs without a new best val MAE (0 = off); checked at validation epochs,
    # so keep it a multiple of -ValEvery (25 = 5 validations, the MSE run's val MAE swung by +-10 between checks)
    [int]$Patience = 25,
    # Full-scene validation over all val frames x 5 view draws is ~10k samples, so validate every
    # 5 epochs instead of every epoch (the final epoch is always validated)
    [int]$ValEvery = 5,
    # Train the backbone jointly (weights + BatchNorm stats, same lr) like the released main.py
    [switch]$TrainBackbone,
    [switch]$Resume
)

# Use the parent of this script's directory (src\) as the working directory
$workDir = Split-Path $PSScriptRoot -Parent

$logFile = "logs\baseline_${RunName}_log.txt"
$errFile = "logs\baseline_${RunName}_error.txt"

# Fail here instead of inside a detached process nobody is watching
if (-not (Test-Path (Join-Path $workDir $Backbone))) {
    Write-Host "Backbone checkpoint not found: $workDir\$Backbone" -ForegroundColor Red
    Write-Host "Pass the stride-8 pretraining checkpoint with -Backbone runs\train_2d\<timestamp>\resnet18_2d_pretrained_epX.pth"
    exit 1
}

$lrText = $Lr.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$extraArgs = "--backbone $Backbone --run_name $RunName --epochs $Epochs --val_every $ValEvery " +
             "--loss $Loss --optimizer $Optimizer --lr $lrText --lr_schedule $LrSchedule --patience $Patience"
if ($TrainBackbone) {
    $extraArgs += " --train_backbone"
}
if ($Patience -gt 0 -and ($Patience % $ValEvery) -ne 0) {
    Write-Host "Note: -Patience $Patience is not a multiple of -ValEvery $ValEvery; it is checked only at validation epochs." -ForegroundColor Yellow
}
if ($InitFrom -ne "" -and -not $Resume) {
    if (-not (Test-Path (Join-Path $workDir $InitFrom))) {
        Write-Host "Init checkpoint not found: $workDir\$InitFrom" -ForegroundColor Red
        exit 1
    }
    $extraArgs += " --init_from $InitFrom"
}
$redirect = ">"
if ($Resume) {
    $lastCkpt = "runs\baseline\$RunName\last.pth"
    if (-not (Test-Path (Join-Path $workDir $lastCkpt))) {
        Write-Host "Nothing to resume: $workDir\$lastCkpt does not exist" -ForegroundColor Red
        exit 1
    }
    $extraArgs += " --resume $lastCkpt"
    $redirect = ">>"  # append to the existing logs instead of overwriting them
}

Write-Host "Starting detached baseline training job..." -ForegroundColor Cyan
Write-Host "Working Directory: $workDir"
Write-Host "Arguments: $extraArgs"

# Make sure the log directory exists, otherwise the redirection fails silently
New-Item -ItemType Directory -Force -Path (Join-Path $workDir "logs") | Out-Null

# Construct the command
$cmd = "cmd.exe /c cd /d `"$workDir`" && uv run python .\training_scripts\train_baseline.py $extraArgs $redirect $logFile 2$redirect $errFile"

# Spawn the detached process
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}

if ($result.ReturnValue -eq 0) {
    $jobPid = $result.ProcessId
    Write-Host "Successfully started process!" -ForegroundColor Green
    Write-Host "Process ID (PID): $jobPid" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You can now safely close your SSH connection or terminal. The job will keep running."
    Write-Host "------------------------------------------------------"
    Write-Host "Quick Command Reference (Copy/Paste these if needed):"
    Write-Host "Check if running: Get-Process -Id $jobPid"
    Write-Host "View live logs:   Get-Content `"$workDir\$logFile`" -Tail 20 -Wait"
    Write-Host "View errors:      Get-Content `"$workDir\$errFile`" -Tail 20 -Wait"
    Write-Host "Stop training:    Stop-Process -Id $jobPid -Force"
    Write-Host "Resume later:     .\remote_scripts\run_remote_baseline_train.ps1 -Backbone $Backbone -RunName $RunName -Loss $Loss -Optimizer $Optimizer -Lr $lrText -LrSchedule $LrSchedule -Epochs $Epochs -Patience $Patience$(if ($TrainBackbone) { ' -TrainBackbone' }) -Resume"
    Write-Host "Early stop:       look for '>>> EARLY STOP' in the log; best.pth holds the best val MAE"
    Write-Host "Checkpoints:      $workDir\runs\baseline\$RunName\ (best.pth, last.pth)"
    Write-Host "------------------------------------------------------"
} else {
    Write-Host "Failed to start process. Error code: $($result.ReturnValue)" -ForegroundColor Red
}
