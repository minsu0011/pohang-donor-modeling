param([string]$PythonExe="python")
$ErrorActionPreference="Stop"
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -e .
Write-Host "Installed pohang-stage2-bias-structure-v4"
