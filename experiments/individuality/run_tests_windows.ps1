$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$PSScriptRoot\src"
python -m pytest -q "$PSScriptRoot\tests"
