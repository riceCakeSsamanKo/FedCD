$ScriptPath = Join-Path $PSScriptRoot "run_splitgp_rho_dataset_vgg8.ps1"
& $ScriptPath -DatasetBase "FashionMNIST" @args
exit $LASTEXITCODE
