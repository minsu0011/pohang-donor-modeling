$ErrorActionPreference = "Stop"
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
$env:PYTHONPATH = "$PSScriptRoot\src"
Write-Host "=== NVIDIA status ==="
nvidia-smi
& $Py -m pohang_stage1_v22.cli gpu-check
