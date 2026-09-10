$ErrorActionPreference="Stop"
$env:PYTHONPATH=(Join-Path $PSScriptRoot "src")
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:NUMEXPR_NUM_THREADS="1"
pohang-stage2-bias-structure-v4 prepare --mode quick
pohang-stage2-bias-structure-v4 select --mode quick
pohang-stage2-bias-structure-v4 evaluate --mode quick
pohang-stage2-bias-structure-v4 finalize --mode quick
