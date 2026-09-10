param(
    [string]$CoreZip = "data/raw/POHANG_HANDOFF_V2_CORE_20260812.zip",
    [string]$SeoulZip = "data/raw/POHANG_HANDOFF_V2_SEOUL_DONOR_20260812.zip",
    [switch]$Quick,
    [ValidateSet("CPU", "GPU")]
    [string]$TaskType = "CPU"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& .\.venv\Scripts\Activate.ps1

$ArgsList = @(
    "-m", "pohang_donor_models.cli", "all",
    "--core-zip", $CoreZip,
    "--seoul-zip", $SeoulZip,
    "--task-type", $TaskType
)
if ($Quick) { $ArgsList += "--quick" }

python @ArgsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "완료. outputs/LATEST_RUN.txt를 확인하십시오."
