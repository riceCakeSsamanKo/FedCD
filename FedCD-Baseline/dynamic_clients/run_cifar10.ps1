[CmdletBinding()]
param(
    [int]$MaxParallel = 1,
    [int]$Seed = 1,
    [string[]]$Algorithms = @(
        "FedAvg", "FedProx", "FedCross", "FedBN", "FedALA", "FedAS",
        "pFedMe", "cwFedAvg", "FedDST", "FedCP", "DualFed"
    )
)

$Runner = Join-Path (Split-Path -Parent $PSScriptRoot) "rho_experiment_windows\run_splitgp_multirho_resnet10.ps1"

& $Runner `
    -Datasets @("Cifar10") `
    -Algorithms $Algorithms `
    -Model "VGG8" `
    -DatasetNameOverride "Cifar10_dynamic_clients_nc50" `
    -EnableMultiRhoEval $false `
    -MaxParallel $MaxParallel `
    -Seed $Seed `
    -DynamicClientEnabled $true `
    -DynamicClientJoinRound 51 `
    -DynamicClientOldClasses "0,1,2,3,4,5" `
    -DynamicClientNewClasses "6,7,8,9"

exit $LASTEXITCODE
