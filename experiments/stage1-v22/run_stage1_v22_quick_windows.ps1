$ErrorActionPreference = "Stop"
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
$env:PYTHONPATH = "$PSScriptRoot\src"
$run = "stage1_multiwindow_ortho_v22_quick"
& $Py -m pohang_stage1_v22.cli --run-name $run prepare --mode quick
& $Py -m pohang_stage1_v22.cli gpu-check
& $Py -m pohang_stage1_v22.cli --run-name $run baseline --backend CPU --mode quick
& $Py -m pohang_stage1_v22.cli --run-name $run baseline --backend GPU --mode quick
& $Py -m pohang_stage1_v22.cli --run-name $run audit --backend CPU --mode quick
& $Py -m pohang_stage1_v22.cli --run-name $run audit --backend GPU --mode quick
Write-Host "QUICK completed. Full Gate is intentionally not finalized from quick-only folds."
