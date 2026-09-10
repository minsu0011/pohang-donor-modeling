$ErrorActionPreference="Stop"
$env:PYTHONPATH=(Join-Path $PSScriptRoot "src")
python -m pytest -q
