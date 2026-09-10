$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$PSScriptRoot\src"
python -m pohang_stage1_individuality.cli --config "$PSScriptRoot\config\default.yaml"
