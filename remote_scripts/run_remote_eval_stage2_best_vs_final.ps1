param(
    [string]$OutDir = "full_length_dense_mask",
    [string]$Tag = "_bud200_ca_sa_dout0.7_cw0.01",
    [switch]$DryRun
)

# Stage 2 (masked MSE) of the budget-matched chain (run_remote_budget_ca_sa_h1.ps1): is the checkpoint the 3-epoch-mean
# rule kept (epochs 2-4, saved at epoch 4) better or worse on test than the final epoch-50 weights? Decides whether the
# chain should pass on final weights. Paper protocol on the test split, every frame x 5 random view draws, for:
#   dense   the dense expert alone over the whole scene -- the clean comparison: the neck + dense expert are what
#           Stage 2 hands on (the OT step re-initialises the sparse expert)
#   oracle  GT dense zone -> dense expert, sparse expert elsewhere -- what Stage 2 was selected on; it also carries
#           Stage 2's MSE sparse expert, which got worse over training (train MAE 52 -> 102)
# Results: <OutDir>\eval\stage2_compare\eval_test_<ckpt>_<routing>.json (+ log / error). Sequential, detached;
# a failed eval doesn't stop the others.

$workDir = Split-Path $PSScriptRoot -Parent
Set-Location $workDir
$ckDir = "$OutDir\2_dense_mask"
$evalDir = "$OutDir\eval\stage2_compare"
$ckpts = @("multiscene_best_stage_2$Tag", "multiscene_final_stage_2$Tag")
foreach ($c in $ckpts) { if (-not $DryRun -and -not (Test-Path "$ckDir\$c.pth")) { Write-Host "Missing $ckDir\$c.pth" -ForegroundColor Red; exit 1 } }

$steps = @()
foreach ($r in @("dense", "oracle")) {
    foreach ($c in $ckpts) {
        $name = "eval_test_${c}_$r"
        $steps += "uv run python -u .\evaluation\evaluate_paper_protocol.py --model proposed --weights $ckDir\$c.pth " +
                  "--split test --routing $r --out $evalDir\$name.json > $evalDir\${name}_log.txt 2> $evalDir\${name}_error.txt"
    }
}
if ($DryRun) { $steps | ForEach-Object { Write-Host "$_`n" }; exit 0 }

New-Item -ItemType Directory -Force -Path $evalDir | Out-Null
$chain = ($steps | ForEach-Object { "($_)" }) -join " & "
$cmd = "conhost.exe --headless cmd.exe /c `"cd /d `"$workDir`" && $chain`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
if ($result.ReturnValue -eq 0) {
    Write-Host "Started $($steps.Count) evals -- PID $($result.ProcessId)" -ForegroundColor Green
    Write-Host "  Results: $workDir\$evalDir\"
    Write-Host "  Stop:    taskkill /PID $($result.ProcessId) /T /F"
} else {
    Write-Host "Failed to start. Error code: $($result.ReturnValue)" -ForegroundColor Red
}
