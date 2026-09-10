$ErrorActionPreference="Stop"
$env:PYTHONPATH = (Join-Path $PSScriptRoot "src")
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
pohang-stage2-adaptive-v3 prepare --mode quick
pohang-stage2-adaptive-v3 select --mode quick
pohang-stage2-adaptive-v3 evaluate --mode quick
pohang-stage2-adaptive-v3 finalize --mode quick
Write-Host "QUICK complete: outputs/stage2_adaptive_v3"
