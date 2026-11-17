param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$Views = "3,7,10",                      # besides 5, which always runs first (the transfer source)
    [string]$Variants = "homography_256,ca_sa",
    [string]$ExtraArgs = "",                        # passed to every camera_ablation.py call, e.g. "--val_every 5 --seed 2"
    [int]$MaxAttempts = 5,
    [int]$StartStep = 1,
    [switch]$DryRun,
    [switch]$Foreground                             # internal: run the steps in this process (the default relaunches itself detached)
)

# Camera ablation, tier A (doc/camera_ablation.md §2): training_scripts/camera_ablation.py split into one process per
# step, so a crash costs at most the current epoch of the current step:
#   1..   train 5 views, one step per variant (the transfer needs them)
#   next  transfer 5 -> N (raw / vnorm), minutes
#   then  train N views per N and variant, smallest N first (10 views, the most likely to run out of memory, last)
#   last  summary table
# Every step saves its own result (result.json per (N, variant), one JSON per transfer); finished steps are skipped.
# A failed step is retried up to $MaxAttempts times (each attempt resumes from its last.pth; a driver-level CUDA
# out-of-memory leaves the process unusable, so it has to restart) and then left for later: the chain goes on with
# the next step. Rerun the same command to retry what is missing. Keep the PC awake (system sleep kills the run).
#
# Output: runs\camera_ablation\<RunName>\ (camera_ablation.py layout) + logs\<step>_log.txt / _error.txt, chain_log.txt.

$workDir = Split-Path $PSScriptRoot -Parent
Set-Location $workDir
$root = "runs\camera_ablation\$RunName"
$chainLog = "$root\chain_log.txt"
$variantList = @($Variants -split '[,\s]+' | Where-Object { $_ })
$viewList = @($Views -split '[,\s]+' | Where-Object { $_ -and $_ -ne '5' })

if (-not $Foreground -and -not $DryRun) {
    $argList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Foreground -RunName $RunName " +
               "-Views $($viewList -join ',') -Variants $($variantList -join ',') -ExtraArgs `"$ExtraArgs`" " +
               "-MaxAttempts $MaxAttempts -StartStep $StartStep"
    $cmd = "conhost.exe --headless powershell.exe $argList"
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd}
    if ($r.ReturnValue -eq 0) {
        Write-Host "Started detached camera ablation '$RunName' -- PID $($r.ProcessId)" -ForegroundColor Green
        Write-Host "  Steps:    Get-Content `"$workDir\$chainLog`" -Tail 20 -Wait"
        Write-Host "  Epochs:   Get-Content `"$workDir\$root\logs\<step>_log.txt`" -Tail 5"
        Write-Host "  Summary:  uv run python training_scripts/camera_ablation.py --mode summarize --run_name $RunName"
        Write-Host "  Stop:     taskkill /PID $($r.ProcessId) /T /F"
    } else { Write-Host "Failed to start. Error code: $($r.ReturnValue)" -ForegroundColor Red }
    exit 0
}

if (-not $DryRun) { New-Item -ItemType Directory -Force -Path "$root\logs" | Out-Null }
function Log($msg) { $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"; if (-not $DryRun) { Add-Content -Path $chainLog -Value $line }; Write-Host $line }
$py = "uv run python -u .\training_scripts\camera_ablation.py --run_name $RunName $ExtraArgs"

# done = the file a finished step leaves behind
$steps = @()
foreach ($v in $variantList) {
    $steps += @{ name = "train_5v_$v"; done = @("$root\5v\$v\result.json"); cmd = "$py --mode train --views 5 --variants $v" }
}
if ($viewList) {
    $steps += @{ name = "transfer"; cmd = "$py --mode transfer --views $($viewList -join ',') --variants $($variantList -join ',')"
                 done = @(foreach ($n in $viewList) { foreach ($v in $variantList) { "$root\transfer\${v}_5to${n}v.json" } }) }
}
foreach ($n in ($viewList | Sort-Object { [int]$_ })) {
    foreach ($v in $variantList) {
        $steps += @{ name = "train_${n}v_$v"; done = @("$root\${n}v\$v\result.json"); cmd = "$py --mode train --views $n --variants $v" }
    }
}
$steps += @{ name = "summary"; done = @(); cmd = "$py --mode summarize --views 5 --variants $($variantList -join ',')" }

Log "=== camera ablation '$RunName' (steps $StartStep..$($steps.Count)) | extra args: '$ExtraArgs' ==="
for ($i = 0; $i -lt $steps.Count; $i++) { Log ("  {0,2}. {1}" -f ($i + 1), $steps[$i].name) }
if ($DryRun) { foreach ($s in $steps) { Write-Host "`n# $($s.name)`n$($s.cmd)" }; exit 0 }

$failed = @()
for ($i = $StartStep - 1; $i -lt $steps.Count; $i++) {
    $s = $steps[$i]
    if ($s.done.Count -and -not ($s.done | Where-Object { -not (Test-Path $_) })) { Log "step $($i + 1): $($s.name) already done"; continue }
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
else { Log "=== camera ablation '$RunName' done ===" }
