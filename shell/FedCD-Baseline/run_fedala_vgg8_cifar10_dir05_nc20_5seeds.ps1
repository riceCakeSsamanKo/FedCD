param(
    [string]$PythonBin = $(if ($env:FEDCD_PYTHON) { $env:FEDCD_PYTHON } else { "python" }),
    [string]$FlDataRoot = $env:FL_DATA_ROOT,
    [string[]]$Seeds = @("0", "1", "2", "3", "4"),
    [string]$CudaVisibleDevices = $(if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }),
    [string]$DeviceId = $(if ($env:DEVICE_ID) { $env:DEVICE_ID } else { "0" }),
    [int]$GlobalRounds = $(if ($env:GLOBAL_ROUNDS) { [int]$env:GLOBAL_ROUNDS } else { 100 }),
    [double]$LearningRate = $(if ($env:LR) { [double]$env:LR } else { 0.005 }),
    [int]$BatchSize = $(if ($env:LBS) { [int]$env:LBS } else { 128 }),
    [int]$LocalEpochs = $(if ($env:LOCAL_EPOCHS) { [int]$env:LOCAL_EPOCHS } else { 2 }),
    [double]$JoinRatio = $(if ($env:JOIN_RATIO) { [double]$env:JOIN_RATIO } else { 1.0 }),
    [double]$Eta = $(if ($env:ETA) { [double]$env:ETA } else { 1.0 }),
    [int]$RandPercent = $(if ($env:RAND_PERCENT) { [int]$env:RAND_PERCENT } else { 80 }),
    [int]$LayerIdx = $(if ($env:LAYER_IDX) { [int]$env:LAYER_IDX } else { 2 })
)

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$SystemDir = Join-Path $RepoRoot "FedCD-Baseline\system"
$MainPy = Join-Path $SystemDir "main.py"

$Dataset = "Cifar10_dir0.5_nc20"
$Scenario = "dir0.5_nc20"
$Model = "VGG8"
$Device = "cuda"
$NumClasses = "10"
$NumClients = "20"
$Times = "1"

if (-not (Get-Command $PythonBin -ErrorAction SilentlyContinue)) {
    Write-Error "Python interpreter not found: $PythonBin. Set FEDCD_PYTHON or pass -PythonBin."
}

if (-not (Test-Path $MainPy -PathType Leaf)) {
    Write-Error "main.py not found: $MainPy"
}

if ([string]::IsNullOrWhiteSpace($FlDataRoot)) {
    $SiblingRoot = Join-Path (Split-Path -Parent $RepoRoot) "fl_data"
    $RepoDataRoot = Join-Path $RepoRoot "fl_data"
    if (Test-Path (Join-Path $SiblingRoot $Dataset) -PathType Container) {
        $FlDataRoot = $SiblingRoot
    }
    elseif (Test-Path (Join-Path $RepoDataRoot $Dataset) -PathType Container) {
        $FlDataRoot = $RepoDataRoot
    }
    else {
        $FlDataRoot = $SiblingRoot
    }
}

if (-not (Test-Path (Join-Path $FlDataRoot "$Dataset\train") -PathType Container) -or
    -not (Test-Path (Join-Path $FlDataRoot "$Dataset\test") -PathType Container)) {
    Write-Error "Required dataset not found: $(Join-Path $FlDataRoot $Dataset). Generate the shared FedCCM/FedCD data first, or pass -FlDataRoot."
}

$ResolvedFlDataRoot = (Resolve-Path $FlDataRoot).Path
$env:FL_DATA_ROOT = $ResolvedFlDataRoot
$env:CUDA_VISIBLE_DEVICES = $CudaVisibleDevices
if (-not $env:MPLCONFIGDIR) {
    $env:MPLCONFIGDIR = Join-Path $env:TEMP "mpl"
}
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

$DateStr = (Get-Date).ToUniversalTime().ToString("yyyyMMdd")
$TimeStr = (Get-Date).ToUniversalTime().ToString("HHmmss")
$QueueRoot = Join-Path $ScriptDir "batch_runs\fedala_vgg8_cifar10_dir05_nc20_5seeds\date_$DateStr\time_$TimeStr"
New-Item -ItemType Directory -Force -Path $QueueRoot | Out-Null

$StatusCsv = Join-Path $QueueRoot "status.csv"
"algorithm,scenario,dataset,seed,status,exit_code,start_utc,end_utc,fl_data_root,goal" |
    Set-Content -Path $StatusCsv -Encoding utf8

Write-Host "[INFO] Queue root: $QueueRoot"
Write-Host "[INFO] Using python: $PythonBin"
Write-Host "[INFO] FL_DATA_ROOT: $ResolvedFlDataRoot"
Write-Host "[INFO] Dataset: $Dataset"
Write-Host "[INFO] Seeds: $($Seeds -join ' ')"
Write-Host ""

foreach ($Seed in $Seeds) {
    $Goal = "FedALA_${Scenario}_seed${Seed}_${DateStr}_${TimeStr}"
    $StartUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

    Write-Host "=========================================================="
    Write-Host "[START] algo=FedALA seed=$Seed dataset=$Dataset"
    Write-Host "[CONFIG] model=$Model rounds=$GlobalRounds lr=$LearningRate lbs=$BatchSize ls=$LocalEpochs jr=$JoinRatio"
    Write-Host "[EXTRA] eta=$Eta rand_percent=$RandPercent layer_idx=$LayerIdx"
    Write-Host "=========================================================="

    $RunArgs = @(
        "-u", $MainPy,
        "-data", $Dataset,
        "-ncl", $NumClasses,
        "-m", $Model,
        "-algo", "FedALA",
        "-gr", "$GlobalRounds",
        "-lr", "$LearningRate",
        "-lbs", "$BatchSize",
        "-ls", "$LocalEpochs",
        "-nc", $NumClients,
        "-jr", "$JoinRatio",
        "-t", $Times,
        "-go", $Goal,
        "-dev", $Device,
        "-did", $DeviceId,
        "--seed", "$Seed",
        "-et", "$Eta",
        "-s", "$RandPercent",
        "-p", "$LayerIdx"
    )

    & $PythonBin @RunArgs
    $ExitCode = $LASTEXITCODE
    $EndUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

    if ($ExitCode -eq 0) {
        $Status = "ok"
        Write-Host "[DONE] algo=FedALA seed=$Seed"
    }
    else {
        $Status = "failed"
        Write-Host "[FAIL] algo=FedALA seed=$Seed exit_code=$ExitCode"
    }

    "FedALA,$Scenario,$Dataset,$Seed,$Status,$ExitCode,$StartUtc,$EndUtc,$ResolvedFlDataRoot,$Goal" |
        Add-Content -Path $StatusCsv -Encoding utf8
    Write-Host ""

    if ($ExitCode -ne 0) {
        Write-Host "[INFO] Stopping after failed seed. Status CSV: $StatusCsv"
        exit $ExitCode
    }
}

Write-Host "[INFO] FedALA CIFAR-10 dir0.5 NC20 5-seed queue finished."
Write-Host "[INFO] Status CSV: $StatusCsv"
