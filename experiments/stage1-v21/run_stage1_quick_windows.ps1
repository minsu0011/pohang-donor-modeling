$ErrorActionPreference = "Stop"
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
$env:PYTHONPATH = "$PSScriptRoot\src"
& $Py -m pohang_stage1_v21.cli --run-name stage1_dynamic_v21 prepare
& $Py -m pohang_stage1_v21.cli --run-name stage1_dynamic_v21 baseline --backend CPU --mode quick
& $Py -m pohang_stage1_v21.cli --run-name stage1_dynamic_v21 audit --backend CPU --mode quick
Write-Host "QUICK complete. FULL hybrid is required for the 7-gate final decision."
