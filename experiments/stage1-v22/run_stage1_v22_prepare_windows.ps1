$ErrorActionPreference = "Stop"
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
$env:PYTHONPATH = "$PSScriptRoot\src"
& $Py -m pohang_stage1_v22.cli --run-name stage1_multiwindow_ortho_v22 prepare --mode full
