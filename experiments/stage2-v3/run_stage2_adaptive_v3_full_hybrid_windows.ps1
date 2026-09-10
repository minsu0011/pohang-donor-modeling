$ErrorActionPreference="Stop"
$env:PYTHONPATH = (Join-Path $PSScriptRoot "src")
# Prevent BLAS oversubscription while 12 linear workers / 3 CatBoost CPU workers share the 7950X3D.
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"

pohang-stage2-adaptive-v3 prepare --mode full
pohang-stage2-adaptive-v3 gpu-check
pohang-stage2-adaptive-v3 select --mode full
pohang-stage2-adaptive-v3 evaluate --mode full
pohang-stage2-adaptive-v3 finalize --mode full
python scripts/package_stage2_v3_results.py
Write-Host "FULL complete. Upload the generated POHANG_STAGE2_ADAPTIVE_V3_FULL_RESULTS_*.zip for review."
