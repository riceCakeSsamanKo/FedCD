[CmdletBinding()]
param(
    [string[]]$Datasets = @("Cifar10", "FashionMNIST"),
    [string[]]$Algorithms = @("FedAvg", "FedProx", "FedCross", "FedBN", "FedALA", "FedAS", "pFedMe", "cwFedAvg", "FedDST", "PMOE_FedPer", "FedCP", "DualFed"),
    [string]$TrainRho = "0.0",
    [string]$EvalRhos = "0.0,0.2,0.4,0.6,0.8",
    [string]$DatasetNameOverride = "",
    [bool]$EnableMultiRhoEval = $true,
    [string]$PythonBin = $(if ($env:FEDCD_PYTHON) { $env:FEDCD_PYTHON } else { "python" }),
    [string]$FlDataRoot = $(if ($env:FL_DATA_ROOT) { $env:FL_DATA_ROOT } else { "" }),
    [ValidateSet("VGG8", "ResNet10")]
    [string]$Model = "ResNet10",
    [string]$Device = "cuda",
    [string]$DeviceId = $(if ($env:DEVICE_ID) { $env:DEVICE_ID } else { "0" }),
    [int]$GlobalRounds = 100,
    [double]$LearningRate = 0.005,
    [int]$BatchSize = 128,
    [int]$LocalEpochs = 2,
    [double]$JoinRatio = 1.0,
    [int]$Times = 1,
    [int]$Seed = $(if ($env:SEED) { [int]$env:SEED } else { 1 }),
    [int]$NumClients = 50,
    [int]$NumClasses = 10,
    [int]$EvalGap = 1,
    [int]$CommonEvalBatchSize = 256,
    [int]$MaxParallel = 1,
    [bool]$DynamicClientEnabled = $false,
    [int]$DynamicClientJoinRound = 51,
    [string]$DynamicClientOldClasses = "0,1,2,3,4,5",
    [string]$DynamicClientNewClasses = "6,7,8,9",
    [double]$FedDstSparsity = 0.3,
    [double]$FedDstFinalSparsity = 0.3,
    [double]$FedDstReadjustmentRatio = 0.5,
    [int]$FedDstReadjustmentInterval = 10,
    [string]$FedDstSparsityDistribution = "erk",
    [string]$FedDstRateDecayMethod = "cosine",
    [int]$PmoeTopK = 8,
    [int]$PmoeFinetuneEpochs = 50,
    [double]$PmoeLr = 0.5,
    [int]$PmoeLockExperts = 0,
    [double]$FedCpLamda = 1.0,
    [double]$DualFedConLambda = 0.1,
    [double]$DualFedConTemp = 0.5
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaselineDir = Split-Path -Parent $ScriptDir
$SystemDir = Join-Path $BaselineDir "system"

function Get-FullPathIfPossible {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
    return [System.IO.Path]::GetFullPath($Path)
}

function Resolve-FlDataRoot {
    param([string]$RequestedRoot, [string]$ProbeDataset)
    $candidates = @(
        $RequestedRoot,
        (Join-Path $BaselineDir "..\fl_data"),
        (Join-Path $BaselineDir "..\..\fl_data"),
        (Join-Path $BaselineDir "fl_data")
    )
    $seen = @{}
    $searched = New-Object System.Collections.Generic.List[string]
    foreach ($candidate in $candidates) {
        $fullPath = Get-FullPathIfPossible $candidate
        if ([string]::IsNullOrWhiteSpace($fullPath) -or $seen.ContainsKey($fullPath)) { continue }
        $seen[$fullPath] = $true
        $searched.Add($fullPath) | Out-Null
        if (Test-Path -LiteralPath (Join-Path $fullPath $ProbeDataset) -PathType Container) { return $fullPath }
    }
    throw "SplitGP dataset '$ProbeDataset' was not found. Searched FL data roots: $($searched -join ', ')"
}

function Add-StatusRow {
    param(
        [int]$Idx,
        [int]$Total,
        [string]$DatasetBase,
        [string]$Algorithm,
        [string]$Dataset,
        [string]$Status,
        [int]$ExitCode,
        [string]$StartUtc,
        [string]$EndUtc,
        [string]$RunLog
    )
    $row = [pscustomobject]@{
        idx = $Idx
        total = $Total
        dataset_base = $DatasetBase
        algorithm = $Algorithm
        train_rho = $TrainRho
        eval_rhos = $EvalRhos
        num_clients = $NumClients
        seed = $Seed
        dynamic_clients = $DynamicClientEnabled
        dynamic_join_round = $DynamicClientJoinRound
        dataset = $Dataset
        status = $Status
        exit_code = $ExitCode
        start_utc = $StartUtc
        end_utc = $EndUtc
        run_log = $RunLog
    }
    $row | Export-Csv -LiteralPath $StatusCsv -NoTypeInformation -Append -Encoding UTF8
}

function Start-ExperimentJob {
    param(
        [int]$Idx,
        [int]$Total,
        [string]$DatasetBase,
        [string]$Algorithm,
        [string]$Dataset,
        [string]$RunLog,
        [string]$StartUtc,
        [string[]]$PythonArgs
    )
    $job = Start-Job -Name ("{0:D3}_{1}_{2}" -f $Idx, $DatasetBase.ToLowerInvariant(), $Algorithm) -ScriptBlock {
        param($SystemDir, $PythonBin, $PythonArgs, $RunLog, $FlDataRoot, $CudaVisibleDevices)
        $env:FL_DATA_ROOT = $FlDataRoot
        $env:CUDA_VISIBLE_DEVICES = $CudaVisibleDevices
        if (-not $env:MPLCONFIGDIR) {
            $env:MPLCONFIGDIR = Join-Path $env:TEMP "mpl"
        }
        New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null
        Set-Location $SystemDir
        & $PythonBin @PythonArgs *> $RunLog
        [pscustomobject]@{ ExitCode = $LASTEXITCODE }
    } -ArgumentList $SystemDir, $PythonBin, $PythonArgs, $RunLog, $ResolvedFlDataRoot, $DeviceId

    [pscustomobject]@{
        Job = $job
        Idx = $Idx
        Total = $Total
        DatasetBase = $DatasetBase
        Algorithm = $Algorithm
        Dataset = $Dataset
        RunLog = $RunLog
        StartUtc = $StartUtc
    }
}

function Receive-FinishedExperimentJobs {
    param([switch]$WaitForAny)
    if ($script:RunningJobs.Count -eq 0) { return }
    if ($WaitForAny) {
        Wait-Job -Job ($script:RunningJobs | ForEach-Object { $_.Job }) -Any | Out-Null
    }
    $finishedStates = @("Completed", "Failed", "Stopped")
    $finished = @($script:RunningJobs | Where-Object { $finishedStates -contains $_.Job.State })
    foreach ($entry in $finished) {
        $result = Receive-Job -Job $entry.Job -ErrorAction SilentlyContinue
        $exitCode = 1
        if ($null -ne $result -and $null -ne $result.PSObject.Properties["ExitCode"]) {
            $exitCode = [int]$result.ExitCode
        } elseif ($entry.Job.State -eq "Completed") {
            $exitCode = 0
        }
        $endUtc = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
        if ($exitCode -eq 0) {
            $status = "ok"
            Write-Host "[DONE $($entry.Idx)/$($entry.Total)] dataset=$($entry.Dataset) algo=$($entry.Algorithm) seed=$Seed"
        } else {
            $status = "failed"
            $script:FailedCount += 1
            Write-Host "[FAIL $($entry.Idx)/$($entry.Total)] dataset=$($entry.Dataset) algo=$($entry.Algorithm) seed=$Seed exit_code=$exitCode"
            Get-Content -LiteralPath $entry.RunLog -Tail 40 -ErrorAction SilentlyContinue
        }
        Add-StatusRow -Idx $entry.Idx -Total $entry.Total -DatasetBase $entry.DatasetBase -Algorithm $entry.Algorithm -Dataset $entry.Dataset -Status $status -ExitCode $exitCode -StartUtc $entry.StartUtc -EndUtc $endUtc -RunLog $entry.RunLog
        Remove-Job -Job $entry.Job -Force -ErrorAction SilentlyContinue
        Write-Host ""
    }
    $script:RunningJobs = @($script:RunningJobs | Where-Object { $finishedStates -notcontains $_.Job.State })
}

if (-not (Get-Command -Name $PythonBin -ErrorAction SilentlyContinue)) {
    throw "Python interpreter not found: $PythonBin. Set FEDCD_PYTHON or pass -PythonBin."
}
if (-not (Test-Path -LiteralPath $SystemDir -PathType Container)) {
    throw "System directory not found: $SystemDir"
}
if ($MaxParallel -lt 1) {
    throw "MaxParallel must be at least 1."
}

$probeDataset = $(if ($DatasetNameOverride) {
    $DatasetNameOverride
} else {
    "{0}_splitgp_pat_rho{1}_nc{2}" -f $Datasets[0], $TrainRho, $NumClients
})
$ResolvedFlDataRoot = Resolve-FlDataRoot -RequestedRoot $FlDataRoot -ProbeDataset $probeDataset
$env:FL_DATA_ROOT = $ResolvedFlDataRoot
$env:CUDA_VISIBLE_DEVICES = $DeviceId

$evalRhoItems = @($EvalRhos -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($datasetBase in $Datasets) {
    $trainDataset = $(if ($DatasetNameOverride) {
        $DatasetNameOverride
    } else {
        "{0}_splitgp_pat_rho{1}_nc{2}" -f $datasetBase, $TrainRho, $NumClients
    })
    if (-not (Test-Path -LiteralPath (Join-Path $ResolvedFlDataRoot $trainDataset) -PathType Container)) {
        throw "Missing train dataset: $(Join-Path $ResolvedFlDataRoot $trainDataset)"
    }
    foreach ($rho in $(if ($EnableMultiRhoEval) { $evalRhoItems } else { @() })) {
        $evalDataset = "{0}_splitgp_pat_rho{1}_nc{2}" -f $datasetBase, $rho, $NumClients
        if (-not (Test-Path -LiteralPath (Join-Path $ResolvedFlDataRoot $evalDataset) -PathType Container)) {
            throw "Missing eval dataset: $(Join-Path $ResolvedFlDataRoot $evalDataset)"
        }
    }
}

$now = [DateTime]::UtcNow
$DateStr = $now.ToString("yyyyMMdd")
$TimeStr = $now.ToString("HHmmss")
$ModelTag = $Model.ToLowerInvariant()
$ModeTag = $(if ($DynamicClientEnabled) { "dynamic_clients" } else { "standard" })
$RunTag = "splitgp_multirho_{0}_{1}_trainrho{2}_{3}_{4}" -f $ModelTag, $ModeTag, $TrainRho, $DateStr, $TimeStr
$QueueRoot = Join-Path $BaselineDir ("batch_runs\splitgp_multirho_{0}_{1}_windows\date_{2}\time_{3}" -f $ModelTag, $ModeTag, $DateStr, $TimeStr)
$RunLogDir = Join-Path $QueueRoot "run_logs"
New-Item -ItemType Directory -Force -Path $RunLogDir | Out-Null
$StatusCsv = Join-Path $QueueRoot "status.csv"
"idx,total,dataset_base,algorithm,train_rho,eval_rhos,num_clients,seed,dynamic_clients,dynamic_join_round,dataset,status,exit_code,start_utc,end_utc,run_log" | Set-Content -LiteralPath $StatusCsv -Encoding UTF8

$total = $Datasets.Count * $Algorithms.Count
$idx = 0
$script:FailedCount = 0
$script:RunningJobs = @()

Write-Host "[INFO] $Model SplitGP train-once multi-rho queue root: $QueueRoot"
Write-Host "[INFO] Python: $PythonBin"
Write-Host "[INFO] FL_DATA_ROOT: $ResolvedFlDataRoot"
Write-Host "[INFO] Datasets: $($Datasets -join ',')"
Write-Host "[INFO] Algorithms: $($Algorithms -join ',')"
Write-Host "[INFO] Train rho: $TrainRho"
Write-Host "[INFO] Eval rhos: $EvalRhos"
Write-Host "[INFO] Dynamic clients: $DynamicClientEnabled (join round: $DynamicClientJoinRound)"
Write-Host "[INFO] Max parallel jobs: $MaxParallel"
Write-Host ""

foreach ($datasetBase in $Datasets) {
    foreach ($algorithm in $Algorithms) {
        $idx += 1
        while ($script:RunningJobs.Count -ge $MaxParallel) {
            Receive-FinishedExperimentJobs -WaitForAny
        }

        $dataset = $(if ($DatasetNameOverride) {
            $DatasetNameOverride
        } else {
            "{0}_splitgp_pat_rho{1}_nc{2}" -f $datasetBase, $TrainRho, $NumClients
        })
        $safeTrainRho = $TrainRho -replace "\.", "p"
        $runLogName = "{0:D3}_{1}_{2}_trainrho{3}_seed{4}.log" -f $idx, $datasetBase.ToLowerInvariant(), $algorithm, $safeTrainRho, $Seed
        $runLog = Join-Path $RunLogDir $runLogName
        $goal = "{0}_{1}_trainrho{2}_multirho_nc{3}_{4}_seed{5}" -f $algorithm, $ModelTag, $TrainRho, $NumClients, $RunTag, $Seed
        $startUtc = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
        $extraArgs = @()

        switch ($algorithm) {
            "FedProx" { $extraArgs += @("-mu", "1.0") }
            "FedALA" { $extraArgs += @("-et", "1.0", "-s", "80", "-p", "2") }
            "FedCross" { $extraArgs += @("-fsb", "0", "-ca", "0.99", "-cmss", "1") }
            "cwFedAvg" { $extraArgs += @("-cw", "-wdr", "-plt", "-ncw", "1", "-wd", "10") }
            "FedDST" {
                $extraArgs += @(
                    "--feddst_sparsity", "$FedDstSparsity",
                    "--feddst_final_sparsity", "$FedDstFinalSparsity",
                    "--feddst_readjustment_ratio", "$FedDstReadjustmentRatio",
                    "--feddst_rounds_between_readjustments", "$FedDstReadjustmentInterval",
                    "--feddst_sparsity_distribution", $FedDstSparsityDistribution,
                    "--feddst_rate_decay_method", $FedDstRateDecayMethod
                )
            }
            "PMOE_FedPer" { $extraArgs += @("-tk", "$PmoeTopK", "-mfte", "$PmoeFinetuneEpochs", "-moelr", "$PmoeLr", "-le", "$PmoeLockExperts") }
            "FedCP" { $extraArgs += @("-lam", "$FedCpLamda") }
            "DualFed" { $extraArgs += @("--dualfed_con_lambda", "$DualFedConLambda", "--dualfed_con_temp", "$DualFedConTemp") }
        }

        Write-Host "=========================================================="
        Write-Host "[LAUNCH $idx/$total] dataset=$dataset algo=$algorithm model=$Model seed=$Seed"
        Write-Host "[CONFIG] train_rho=$TrainRho eval_rhos=$EvalRhos rounds=$GlobalRounds lr=$LearningRate lbs=$BatchSize ls=$LocalEpochs jr=$JoinRatio nc=$NumClients"
        Write-Host "[LOG] $runLog"
        Write-Host "=========================================================="

        $pythonArgs = @(
            "-u", "main.py",
            "-data", $dataset,
            "-ncl", "$NumClasses",
            "-m", $Model,
            "-algo", $algorithm,
            "-gr", "$GlobalRounds",
            "-lr", "$LearningRate",
            "-lbs", "$BatchSize",
            "-ls", "$LocalEpochs",
            "-nc", "$NumClients",
            "-jr", "$JoinRatio",
            "-t", "$Times",
            "--seed", "$Seed",
            "-eg", "$EvalGap",
            "--common_eval_batch_size", "$CommonEvalBatchSize",
            "-go", $goal,
            "-dev", $Device,
            "-did", "$DeviceId"
        )
        if ($EnableMultiRhoEval) {
            $pythonArgs += @("--eval-rhos", $EvalRhos)
        }
        if ($DynamicClientEnabled) {
            $pythonArgs += @(
                "--dynamic_client_enabled", "True",
                "--dynamic_client_join_round", "$DynamicClientJoinRound",
                "--dynamic_client_old_classes", $DynamicClientOldClasses,
                "--dynamic_client_new_classes", $DynamicClientNewClasses,
                "--dynamic_client_expected_existing_clients", "30",
                "--dynamic_client_expected_newcomer_clients", "20",
                "--dynamic_client_require_contiguous_ids", "True",
                "--eval_common_global", "False"
            )
        }
        $pythonArgs += $extraArgs
        $script:RunningJobs += Start-ExperimentJob -Idx $idx -Total $total -DatasetBase $datasetBase -Algorithm $algorithm -Dataset $dataset -RunLog $runLog -StartUtc $startUtc -PythonArgs $pythonArgs
    }
}

while ($script:RunningJobs.Count -gt 0) {
    Receive-FinishedExperimentJobs -WaitForAny
}

Write-Host "[INFO] Finished. Status CSV: $StatusCsv"
if ($script:FailedCount -gt 0) {
    Write-Host "[WARN] Failed jobs: $script:FailedCount"
    exit 1
}
exit 0
