$ErrorActionPreference = "Stop"
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
$env:PYTHONPATH = "$PSScriptRoot\src"
$run = "stage1_multiwindow_ortho_v22"

Write-Host "[1/5] Multi-window Stage-1 tournament (CPU, ~24 logical threads)"
& $Py -m pohang_stage1_v22.cli --run-name $run prepare --mode full
if ($LASTEXITCODE -ne 0) { throw "V2.2 prepare failed" }

Write-Host "[2/5] RTX 5080 preflight"
& $Py -m pohang_stage1_v22.cli gpu-check
if ($LASTEXITCODE -ne 0) { throw "GPU preflight failed" }
Write-Host "Review existing Python processes; script does not blindly kill them."
Get-Process python* -ErrorAction SilentlyContinue | Format-Table Id,ProcessName,CPU,StartTime -AutoSize
nvidia-smi

Write-Host "[3/5] CPU + GPU outer baselines in parallel"
$cpuLog = Join-Path $PSScriptRoot "stage1_v22_cpu_baseline.log"
$gpuLog = Join-Path $PSScriptRoot "stage1_v22_gpu_baseline.log"
$cpu = Start-Process $Py -ArgumentList @('-m','pohang_stage1_v22.cli','--run-name',$run,'baseline','--backend','CPU','--mode','full') -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $cpuLog -RedirectStandardError ($cpuLog+'.err') -PassThru
$gpu = Start-Process $Py -ArgumentList @('-m','pohang_stage1_v22.cli','--run-name',$run,'baseline','--backend','GPU','--mode','full') -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $gpuLog -RedirectStandardError ($gpuLog+'.err') -PassThru
$cpu.WaitForExit(); $gpu.WaitForExit()
if ($cpu.ExitCode -ne 0) { throw "CPU baseline failed. See $cpuLog.err" }
if ($gpu.ExitCode -ne 0) { throw "GPU baseline failed. See $gpuLog.err" }

Write-Host "[4/5] CPU + GPU category/proxy/Age×Industry audits in parallel"
$cpuAuditLog = Join-Path $PSScriptRoot "stage1_v22_cpu_audit.log"
$gpuAuditLog = Join-Path $PSScriptRoot "stage1_v22_gpu_audit.log"
$cpuAudit = Start-Process $Py -ArgumentList @('-m','pohang_stage1_v22.cli','--run-name',$run,'audit','--backend','CPU','--mode','full') -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $cpuAuditLog -RedirectStandardError ($cpuAuditLog+'.err') -PassThru
$gpuAudit = Start-Process $Py -ArgumentList @('-m','pohang_stage1_v22.cli','--run-name',$run,'audit','--backend','GPU','--mode','full') -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $gpuAuditLog -RedirectStandardError ($gpuAuditLog+'.err') -PassThru
$cpuAudit.WaitForExit(); $gpuAudit.WaitForExit()
if ($cpuAudit.ExitCode -ne 0) { throw "CPU audit failed. See $cpuAuditLog.err" }
if ($gpuAudit.ExitCode -ne 0) { throw "GPU audit failed. See $gpuAuditLog.err" }

Write-Host "[5/5] Gate + report"
& $Py -m pohang_stage1_v22.cli --run-name $run finalize
if ($LASTEXITCODE -ne 0) { throw "V2.2 finalize failed" }
Write-Host "Done: outputs\$run\stage1_v22\STAGE1_V22_REVIEW_KO.md"
