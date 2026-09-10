param([string]$RunName = "stage1_feasibility_stage2_ablation_v23")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Run setup_windows.ps1 first." }
$env:PYTHONPATH = "$Root\src"
& $Py -m pohang_main_v2.cli --run-name $RunName verify
if ($LASTEXITCODE -ne 0) { throw "Input verification failed" }
