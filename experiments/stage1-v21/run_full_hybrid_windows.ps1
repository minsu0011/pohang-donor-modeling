param([string]$RunName = "residual_hybrid_v2")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Run setup_windows.ps1 first." }

# Prevent accidental duplicate CatBoost jobs from exhausting the RTX 5060 Ti.
try {
    $existing = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(\.exe)?$' -and
        $_.CommandLine -match 'pohang_main_v2'
    }
    if ($existing) {
        $detail = ($existing | ForEach-Object { "PID=$($_.ProcessId) $($_.CommandLine)" }) -join "`n"
        throw "Another Pohang V2 model process is already running. Stop only that process before FULL:`n$detail"
    }
} catch {
    if ($_.Exception.Message -like "Another Pohang V2*") { throw }
    Write-Warning "Process precheck skipped: $($_.Exception.Message)"
}

$LogDir = Join-Path $Root "outputs\$RunName\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Start-ModelProcess {
    param([string[]]$Arguments, [string]$Name)
    $stdout = Join-Path $LogDir "$Name.out.log"
    $stderr = Join-Path $LogDir "$Name.err.log"
    $process = Start-Process -FilePath $Py -ArgumentList $Arguments -PassThru -NoNewWindow `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    try { $process.PriorityClass = "AboveNormal" } catch { }
    return $process
}

function Wait-Checked {
    param($Process, [string]$Name)
    $Process.WaitForExit()
    if ($Process.ExitCode -ne 0) {
        $stderr = Join-Path $LogDir "$Name.err.log"
        $tail = if (Test-Path $stderr) { (Get-Content $stderr -Tail 100) -join "`n" } else { "" }
        throw "$Name failed with exit code $($Process.ExitCode).`n$tail"
    }
}

& $Py -m pytest -q
& $Py -m pohang_main_v2.cli --run-name $RunName gpu-check
& $Py -m pohang_main_v2.cli --run-name $RunName verify
& $Py -m pohang_main_v2.cli --run-name $RunName prepare
& $Py -m pohang_main_v2.cli --run-name $RunName correlate

Write-Host "[1/3] CPU/GPU rolling baselines..."
$cpuBase = Start-ModelProcess @("-m","pohang_main_v2.cli","--run-name",$RunName,"baseline","--backend","CPU","--mode","full") "baseline_cpu"
$gpuBase = Start-ModelProcess @("-m","pohang_main_v2.cli","--run-name",$RunName,"baseline","--backend","GPU","--mode","full") "baseline_gpu"
Wait-Checked $cpuBase "baseline_cpu"
Wait-Checked $gpuBase "baseline_gpu"

Write-Host "[2/3] Organic ablation branches. Groups/redundancy are cross-checked on both backends; 93 individual units are split 50:50."
$cpuAbl = Start-ModelProcess @("-m","pohang_main_v2.cli","--run-name",$RunName,"branch","--backend","CPU","--mode","full","--feature-shard-count","2","--feature-shard-index","0") "ablation_cpu_branch"
$gpuAbl = Start-ModelProcess @("-m","pohang_main_v2.cli","--run-name",$RunName,"branch","--backend","GPU","--mode","full","--feature-shard-count","2","--feature-shard-index","1") "ablation_gpu_branch"
Wait-Checked $cpuAbl "ablation_cpu_branch"
Wait-Checked $gpuAbl "ablation_gpu_branch"

Write-Host "[3/3] Category re-entry/proxy and recent-fold challenger diagnostics..."
$cpuDiag = Start-ModelProcess @("-m","pohang_main_v2.cli","--run-name",$RunName,"diagnose","--backend","CPU","--mode","full") "diagnose_cpu"
$gpuDiag = Start-ModelProcess @("-m","pohang_main_v2.cli","--run-name",$RunName,"diagnose","--backend","GPU","--mode","full") "diagnose_gpu"
Wait-Checked $cpuDiag "diagnose_cpu"
Wait-Checked $gpuDiag "diagnose_gpu"

& $Py -m pohang_main_v2.cli --run-name $RunName finalize
Write-Host "FULL HYBRID completed: outputs\$RunName"
