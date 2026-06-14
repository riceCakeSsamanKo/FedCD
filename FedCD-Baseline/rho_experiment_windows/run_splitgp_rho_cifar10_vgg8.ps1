$ScriptPath = Join-Path $PSScriptRoot "run_splitgp_rho_dataset_vgg8.ps1"
& $ScriptPath -DatasetBase "Cifar10" @args
exit $LASTEXITCODE
