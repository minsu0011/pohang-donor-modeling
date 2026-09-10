$ErrorActionPreference="Stop"
$env:PYTHONPATH=(Join-Path $PSScriptRoot "src")
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:NUMEXPR_NUM_THREADS="1"
pohang-stage2-bias-structure-v4 prepare --mode full
pohang-stage2-bias-structure-v4 gpu-check
pohang-stage2-bias-structure-v4 select --mode full
pohang-stage2-bias-structure-v4 evaluate --mode full
pohang-stage2-bias-structure-v4 finalize --mode full
python scripts/package_stage2_v4_results.py
Write-Host "FULL complete. Upload POHANG_STAGE2_BIAS_STRUCTURE_V4_FULL_RESULTS_*.zip for review."
