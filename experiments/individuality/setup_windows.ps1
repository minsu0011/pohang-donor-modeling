param([string]$PythonExe="python")
$ErrorActionPreference = "Stop"
& $PythonExe -m pip install -r "$PSScriptRoot\requirements.txt"
& $PythonExe -m pip install -r "$PSScriptRoot\requirements-dev.txt"
