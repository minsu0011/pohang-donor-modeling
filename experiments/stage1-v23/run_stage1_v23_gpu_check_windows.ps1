$ErrorActionPreference = "Stop"
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
$env:PYTHONPATH = "$PSScriptRoot\src"
& $Py -m pohang_stage1_v23.cli gpu-check
if ($LASTEXITCODE -ne 0) { throw "RTX GPU preflight failed" }
nvidia-smi
