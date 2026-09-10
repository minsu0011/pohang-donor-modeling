$ErrorActionPreference = "Stop"
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
$env:PYTHONPATH = "$PSScriptRoot\src"
$run = "stage1_feasibility_stage2_ablation_v23_quick"
& $Py -m pohang_stage1_v23.cli --run-name $run prepare --mode quick
if ($LASTEXITCODE -ne 0) { throw "V2.3 QUICK prepare failed" }
& $Py -m pohang_stage1_v23.cli --run-name $run baseline --backend CPU --mode quick
if ($LASTEXITCODE -ne 0) { throw "V2.3 QUICK baseline failed" }
& $Py -m pohang_stage1_v23.cli --run-name $run ablation --backend CPU --kind group --mode quick
if ($LASTEXITCODE -ne 0) { throw "V2.3 QUICK group failed" }
& $Py -m pohang_stage1_v23.cli --run-name $run ablation --backend CPU --kind individual --mode quick
if ($LASTEXITCODE -ne 0) { throw "V2.3 QUICK individual failed" }
Write-Host "QUICK PASS. FULL still required."
