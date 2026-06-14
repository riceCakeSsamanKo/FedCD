[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Cifar10", "FashionMNIST")]
    [string]$DatasetBase,

    [string]$PythonBin = $(if ($env:FEDCD_PYTHON) { $env:FEDCD_PYTHON } else { "python" }),
    [string]$FlDataRoot = $(if ($env:FL_DATA_ROOT) { $env:FL_DATA_ROOT } else { "" }),
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
    [int]$CompleteRound = 101,
    [int]$MaxParallel = 4,
    [string[]]$Algorithms = @("cwFedAvg", "FedALA", "FedAS", "FedAvg", "FedBN", "FedCross", "pFedMe", "FedProx"),
    [string[]]$Rhos = @("0.0", "0.2", "0.4", "0.6", "0.8")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaselineDir = Split-Path -Parent $ScriptDir
$SystemDir = Join-Path $BaselineDir "system"
$Model = "VGG8"
$NumClasses = 10
$DatasetKey = $DatasetBase.ToLowerInvariant()
if ($DatasetBase -eq "Cifar10") {
    $LogDataset = "cifar10"
} else {
    $LogDataset = $DatasetBase
}

function Get-FullPathIfPossible {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return ""
    }

    return [System.IO.Path]::GetFullPath($Path)
}

function Resolve-FlDataRoot {
    param(
        [string]$RequestedRoot,
        [string]$ProbeDataset
    )

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
        if ([string]::IsNullOrWhiteSpace($fullPath) -or $seen.ContainsKey($fullPath)) {
            continue
        }

        $seen[$fullPath] = $true
        $searched.Add($fullPath) | Out-Null

        $datasetPath = Join-Path $fullPath $ProbeDataset
        if (Test-Path -LiteralPath $datasetPath -PathType Container) {
            return $fullPath
        }
    }

    throw "SplitGP rho dataset '$ProbeDataset' was not found. Searched FL data roots: $($searched -join ', ')"
}

function Test-AccCsvComplete {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }

    try {
        $rows = Import-Csv -LiteralPath $Path
        $count = 0
        $maxRound = 0

        foreach ($row in $rows) {
            $round = 0
            if ([int]::TryParse([string]$row.round, [ref]$round)) {
                $count += 1
                if ($round -gt $maxRound) {
                    $maxRound = $round
                }
            }
        }

        return ($count -ge $CompleteRound -and $maxRound -ge $CompleteRound)
    } catch {
        return $false
    }
}

function Find-CompleteAccCsv {
    param(
        [string]$Algorithm,
        [string]$Rho
    )

    $searchRoot = Join-Path $BaselineDir ("logs\{0}\{1}\GM_{2}\splitgp_rho{3}\NC_{4}" -f $LogDataset, $Algorithm, $Model, $Rho, $NumClients)
    if (-not (Test-Path -LiteralPath $searchRoot -PathType Container)) {
        return $null
    }

    $candidates = Get-ChildItem -LiteralPath $searchRoot -Recurse -Filter "acc.csv" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending

    foreach ($candidate in $candidates) {
        if (Test-AccCsvComplete $candidate.FullName) {
            return $candidate.FullName
        }
    }

    return $null
}

function Add-StatusRow {
    param(
        [int]$Idx,
        [int]$Total,
        [string]$Algorithm,
        [string]$Rho,
        [string]$Dataset,
        [string]$Status,
        [int]$ExitCode,
        [string]$StartUtc,
        [string]$EndUtc,
        [string]$RunLog,
        [string]$ExistingAccCsv
    )

    $row = [pscustomobject]@{
        idx              = $Idx
        total            = $Total
        dataset_base     = $DatasetBase
        algorithm        = $Algorithm
        rho              = $Rho
        num_clients      = $NumClients
        seed             = $Seed
        dataset          = $Dataset
        status           = $Status
        exit_code        = $ExitCode
        start_utc        = $StartUtc
        end_utc          = $EndUtc
        run_log          = $RunLog
        existing_acc_csv = $ExistingAccCsv
    }

    $csvLine = $row | ConvertTo-Csv -NoTypeInformation | Select-Object -Skip 1
    Add-Content -LiteralPath $StatusCsv -Value $csvLine -Encoding UTF8
}

function Start-ExperimentJob {
    param(
        [int]$Idx,
        [int]$Total,
        [string]$Algorithm,
        [string]$Rho,
        [string]$Dataset,
        [string]$RunLog,
        [string]$StartUtc,
        [string[]]$PythonArgs
    )

    $job = Start-Job -Name ("{0:D3}_{1}_{2}_rho{3}" -f $Idx, $DatasetKey, $Algorithm, ($Rho -replace "\.", "p")) -ScriptBlock {
        param(
            [string]$PythonBin,
            [string[]]$PythonArgs,
            [string]$WorkingDirectory,
            [string]$RunLog,
            [string]$ResolvedFlDataRoot,
            [string]$MplConfigDir,
            [string]$CudaVisibleDevices
        )

        $exitCode = 1
        try {
            $env:FL_DATA_ROOT = $ResolvedFlDataRoot
            $env:MPLCONFIGDIR = $MplConfigDir
            $env:CUDA_VISIBLE_DEVICES = $CudaVisibleDevices
            Set-Location -LiteralPath $WorkingDirectory
            & $PythonBin @PythonArgs *> $RunLog
            if ($null -eq $LASTEXITCODE) {
                $exitCode = 0
            } else {
                $exitCode = $LASTEXITCODE
            }
        } catch {
            $_ | Out-File -LiteralPath $RunLog -Append -Encoding UTF8
            $exitCode = 1
        }

        [pscustomobject]@{
            ExitCode = $exitCode
        }
    } -ArgumentList $PythonBin, $PythonArgs, $SystemDir, $RunLog, $ResolvedFlDataRoot, $env:MPLCONFIGDIR, $env:CUDA_VISIBLE_DEVICES

    return [pscustomobject]@{
        Job       = $job
        Idx       = $Idx
        Total     = $Total
        Algorithm = $Algorithm
        Rho       = $Rho
        Dataset   = $Dataset
        RunLog    = $RunLog
        StartUtc  = $StartUtc
    }
}

function Receive-FinishedExperimentJobs {
    param([switch]$WaitForAny)

    if ($WaitForAny -and $script:RunningJobs.Count -gt 0) {
        $jobs = @($script:RunningJobs | ForEach-Object { $_.Job })
        Wait-Job -Job $jobs -Any | Out-Null
    }

    $finishedStates = @("Completed", "Failed", "Stopped")
    $finishedEntries = @($script:RunningJobs | Where-Object { $finishedStates -contains $_.Job.State })
    foreach ($entry in $finishedEntries) {
        $exitCode = 1
        $result = $null

        try {
            $received = @(Receive-Job -Job $entry.Job -ErrorAction SilentlyContinue)
            if ($received.Count -gt 0) {
                $result = $received | Select-Object -Last 1
            }
        } catch {
            $result = $null
        }

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
            Write-Host "[FAIL] Last 40 log lines:"
            Get-Content -LiteralPath $entry.RunLog -Tail 40 -ErrorAction SilentlyContinue
        }

        Add-StatusRow -Idx $entry.Idx -Total $entry.Total -Algorithm $entry.Algorithm -Rho $entry.Rho -Dataset $entry.Dataset -Status $status -ExitCode $exitCode -StartUtc $entry.StartUtc -EndUtc $endUtc -RunLog $entry.RunLog -ExistingAccCsv ""
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

$probeDataset = "{0}_splitgp_pat_rho0.0_nc{1}" -f $DatasetBase, $NumClients
$ResolvedFlDataRoot = Resolve-FlDataRoot -RequestedRoot $FlDataRoot -ProbeDataset $probeDataset
$env:FL_DATA_ROOT = $ResolvedFlDataRoot

if (-not $env:MPLCONFIGDIR) {
    $env:MPLCONFIGDIR = Join-Path $env:TEMP "mpl"
}
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

if (-not $env:CUDA_VISIBLE_DEVICES) {
    $env:CUDA_VISIBLE_DEVICES = $DeviceId
}

$now = [DateTime]::UtcNow
$DateStr = $now.ToString("yyyyMMdd")
$TimeStr = $now.ToString("HHmmss")
$RunTag = "splitgp_rho_{0}_{1}_{2}" -f $DatasetKey, $DateStr, $TimeStr
$QueueRoot = Join-Path $BaselineDir ("batch_runs\splitgp_rho_baselines_vgg8_windows\{0}\date_{1}\time_{2}" -f $DatasetKey, $DateStr, $TimeStr)
$RunLogDir = Join-Path $QueueRoot "run_logs"
New-Item -ItemType Directory -Force -Path $RunLogDir | Out-Null

$StatusCsv = Join-Path $QueueRoot "status.csv"
"idx,total,dataset_base,algorithm,rho,num_clients,seed,dataset,status,exit_code,start_utc,end_utc,run_log,existing_acc_csv" |
    Set-Content -LiteralPath $StatusCsv -Encoding UTF8

$total = $Algorithms.Count * $Rhos.Count
$idx = 0
$script:FailedCount = 0
$script:RunningJobs = @()

Write-Host "[INFO] Dataset base: $DatasetBase"
Write-Host "[INFO] Queue root: $QueueRoot"
Write-Host "[INFO] Status CSV: $StatusCsv"
Write-Host "[INFO] Run logs: $RunLogDir"
Write-Host "[INFO] Python: $PythonBin"
Write-Host "[INFO] FL_DATA_ROOT: $ResolvedFlDataRoot"
Write-Host "[INFO] Device: ${Device}:${DeviceId}"
Write-Host "[INFO] Total runs: $total"
Write-Host "[INFO] Max parallel runs: $MaxParallel"
Write-Host "[INFO] Resume policy: skip runs with an existing acc.csv completed through round $CompleteRound."
Write-Host ""

foreach ($algorithm in $Algorithms) {
    foreach ($rho in $Rhos) {
        $dataset = "{0}_splitgp_pat_rho{1}_nc{2}" -f $DatasetBase, $rho, $NumClients
        $datasetPath = Join-Path $ResolvedFlDataRoot $dataset
        if (-not (Test-Path -LiteralPath $datasetPath -PathType Container)) {
            throw "Missing dataset: $datasetPath"
        }

        $idx += 1
        $existingAccCsv = Find-CompleteAccCsv -Algorithm $algorithm -Rho $rho
        if ($existingAccCsv) {
            $startUtc = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
            $endUtc = $startUtc
            Write-Host "[SKIP $idx/$total] complete through round ${CompleteRound}: dataset=$dataset algo=$algorithm seed=$Seed"
            Write-Host "[SKIP] existing acc.csv: $existingAccCsv"
            Add-StatusRow -Idx $idx -Total $total -Algorithm $algorithm -Rho $rho -Dataset $dataset -Status "skipped_completed" -ExitCode 0 -StartUtc $startUtc -EndUtc $endUtc -RunLog "" -ExistingAccCsv $existingAccCsv
            Write-Host ""
            continue
        }

        while ($script:RunningJobs.Count -ge $MaxParallel) {
            Receive-FinishedExperimentJobs -WaitForAny
        }

        $goal = "{0}_splitgp_rho{1}_nc{2}_{3}_seed{4}" -f $algorithm, $rho, $NumClients, $RunTag, $Seed
        $safeRho = $rho -replace "\.", "p"
        $runLogName = "{0:D3}_{1}_{2}_rho{3}_nc{4}_seed{5}.log" -f $idx, $DatasetKey, $algorithm, $safeRho, $NumClients, $Seed
        $runLog = Join-Path $RunLogDir $runLogName
        $startUtc = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
        $extraArgs = @()

        switch ($algorithm) {
            "FedProx" {
                $extraArgs += @("-mu", "1.0")
            }
            "FedALA" {
                $extraArgs += @("-et", "1.0", "-s", "80", "-p", "2")
            }
            "FedCross" {
                $extraArgs += @("-fsb", "0", "-ca", "0.99", "-cmss", "1")
            }
            "cwFedAvg" {
                $extraArgs += @("-cw", "-wdr", "-plt", "-ncw", "1", "-wd", "10")
            }
        }

        Write-Host "=========================================================="
        Write-Host "[LAUNCH $idx/$total] dataset=$dataset algo=$algorithm seed=$Seed"
        Write-Host "[CONFIG] model=$Model rounds=$GlobalRounds lr=$LearningRate lbs=$BatchSize ls=$LocalEpochs jr=$JoinRatio nc=$NumClients ncl=$NumClasses"
        Write-Host "[LOG] $runLog"
        Write-Host "[RUNNING] $($script:RunningJobs.Count + 1)/$MaxParallel"
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
            "-go", $goal,
            "-dev", $Device,
            "-did", "$DeviceId"
        )
        $pythonArgs += $extraArgs

        $script:RunningJobs += Start-ExperimentJob -Idx $idx -Total $total -Algorithm $algorithm -Rho $rho -Dataset $dataset -RunLog $runLog -StartUtc $startUtc -PythonArgs $pythonArgs
    }
}

while ($script:RunningJobs.Count -gt 0) {
    Receive-FinishedExperimentJobs -WaitForAny
}

Write-Host "[INFO] SplitGP rho baseline queue finished for $DatasetBase."
Write-Host "[INFO] Status CSV: $StatusCsv"

if ($script:FailedCount -gt 0) {
    Write-Host "[WARN] Failed runs: $script:FailedCount"
    exit 1
}

exit 0
