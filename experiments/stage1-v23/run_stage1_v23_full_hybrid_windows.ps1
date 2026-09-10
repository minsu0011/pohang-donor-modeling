$ErrorActionPreference = "Stop"
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
$env:PYTHONPATH = "$PSScriptRoot\src"
$run = "stage1_feasibility_stage2_ablation_v23"

Write-Host "[1/6] Feasibility-first Stage1: ALL 15 candidates x all inner windows (CPU ~24 threads)"
& $Py -m pohang_stage1_v23.cli --run-name $run prepare --mode full
if ($LASTEXITCODE -ne 0) { throw "V2.3 prepare failed" }

Write-Host "[2/6] RTX 5080 preflight"
& $Py -m pohang_stage1_v23.cli gpu-check
if ($LASTEXITCODE -ne 0) { throw "GPU preflight failed" }
Get-Process python* -ErrorAction SilentlyContinue | Format-Table Id,ProcessName,CPU,StartTime -AutoSize
nvidia-smi

Write-Host "[3/6] CPU + GPU diagnostic outer baselines in parallel"
$cpuLog = Join-Path $PSScriptRoot "v23_cpu_baseline.log"
$gpuLog = Join-Path $PSScriptRoot "v23_gpu_baseline.log"
$cpu = Start-Process $Py -ArgumentList @('-m','pohang_stage1_v23.cli','--run-name',$run,'baseline','--backend','CPU','--mode','full') -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $cpuLog -RedirectStandardError ($cpuLog+'.err') -PassThru
$gpu = Start-Process $Py -ArgumentList @('-m','pohang_stage1_v23.cli','--run-name',$run,'baseline','--backend','GPU','--mode','full') -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $gpuLog -RedirectStandardError ($gpuLog+'.err') -PassThru
$cpu.WaitForExit(); $gpu.WaitForExit(); $cpu.Refresh(); $gpu.Refresh()
if ($cpu.ExitCode -ne 0) { throw "CPU baseline failed. See $cpuLog.err" }
if ($gpu.ExitCode -ne 0) { throw "GPU baseline failed. See $gpuLog.err" }

Write-Host "[4/6] Stage2 GROUP ablation CPU + GPU in parallel"
$cpuGLog = Join-Path $PSScriptRoot "v23_cpu_group.log"
$gpuGLog = Join-Path $PSScriptRoot "v23_gpu_group.log"
$cpuG = Start-Process $Py -ArgumentList @('-m','pohang_stage1_v23.cli','--run-name',$run,'ablation','--backend','CPU','--kind','group','--mode','full') -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $cpuGLog -RedirectStandardError ($cpuGLog+'.err') -PassThru
$gpuG = Start-Process $Py -ArgumentList @('-m','pohang_stage1_v23.cli','--run-name',$run,'ablation','--backend','GPU','--kind','group','--mode','full') -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $gpuGLog -RedirectStandardError ($gpuGLog+'.err') -PassThru
$cpuG.WaitForExit(); $gpuG.WaitForExit(); $cpuG.Refresh(); $gpuG.Refresh()
if ($cpuG.ExitCode -ne 0) { throw "CPU group ablation failed. See $cpuGLog.err" }
if ($gpuG.ExitCode -ne 0) { throw "GPU group ablation failed. See $gpuGLog.err" }

Write-Host "[5/6] Stage2 INDIVIDUAL 78-feature ablation CPU + GPU in parallel"
$cpuILog = Join-Path $PSScriptRoot "v23_cpu_individual.log"
$gpuILog = Join-Path $PSScriptRoot "v23_gpu_individual.log"
$cpuI = Start-Process $Py -ArgumentList @('-m','pohang_stage1_v23.cli','--run-name',$run,'ablation','--backend','CPU','--kind','individual','--mode','full') -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $cpuILog -RedirectStandardError ($cpuILog+'.err') -PassThru
$gpuI = Start-Process $Py -ArgumentList @('-m','pohang_stage1_v23.cli','--run-name',$run,'ablation','--backend','GPU','--kind','individual','--mode','full') -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $gpuILog -RedirectStandardError ($gpuILog+'.err') -PassThru
$cpuI.WaitForExit(); $gpuI.WaitForExit(); $cpuI.Refresh(); $gpuI.Refresh()
if ($cpuI.ExitCode -ne 0) { throw "CPU individual ablation failed. See $cpuILog.err" }
if ($gpuI.ExitCode -ne 0) { throw "GPU individual ablation failed. See $gpuILog.err" }

Write-Host "[6/6] Aggregate + Fold2 diagnostics + gate"
& $Py -m pohang_stage1_v23.cli --run-name $run finalize
if ($LASTEXITCODE -ne 0) { throw "V2.3 finalize failed" }
Write-Host "Done: outputs\$run\stage1_v23\V23_REVIEW_KO.md"
