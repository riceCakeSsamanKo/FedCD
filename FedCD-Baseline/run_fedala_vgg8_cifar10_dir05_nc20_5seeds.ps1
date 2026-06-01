param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ScriptPath = Join-Path $PSScriptRoot "..\shell\FedCD-Baseline\run_fedala_vgg8_cifar10_dir05_nc20_5seeds.ps1"
& $ScriptPath @RemainingArgs
exit $LASTEXITCODE
