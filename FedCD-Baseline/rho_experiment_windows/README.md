# SplitGP rho VGG8 Windows runners

These PowerShell runners are Windows equivalents of `FedCD-Baseline/run_splitgp_rho_baselines_vgg8.sh`, split by dataset.

Run from PowerShell:

```powershell
Set-Location C:\Users\mulso\Documents\GitHub\FedCD\FedCD-Baseline\rho_experiment_windows
.\run_splitgp_rho_cifar10_vgg8.ps1
.\run_splitgp_rho_fashionmnist_vgg8.ps1
```

Optional environment overrides:

```powershell
$env:FEDCD_PYTHON = "C:\path\to\python.exe"
$env:FL_DATA_ROOT = "C:\Users\mulso\Documents\GitHub\fl_data"
$env:DEVICE_ID = "0"
```

Optional parameter overrides:

```powershell
.\run_splitgp_rho_cifar10_vgg8.ps1 -PythonBin "C:\path\to\python.exe" -Device cuda -DeviceId 0 -Seed 1 -MaxParallel 4
.\run_splitgp_rho_fashionmnist_vgg8.ps1 -Device cpu
```

Each dataset runner executes 8 algorithms by 5 rho values, for 40 runs:

- algorithms: `cwFedAvg`, `FedALA`, `FedAS`, `FedAvg`, `FedBN`, `FedCross`, `pFedMe`, `FedProx`
- rho values: `0.0`, `0.2`, `0.4`, `0.6`, `0.8`
- datasets: `{Dataset}_splitgp_pat_rho{rho}_nc50`

The default concurrency is 4 experiments at a time. Override it with `-MaxParallel`:

```powershell
.\run_splitgp_rho_cifar10_vgg8.ps1 -MaxParallel 2
.\run_splitgp_rho_fashionmnist_vgg8.ps1 -MaxParallel 4
```

Queue status and per-run logs are written under:

```text
FedCD-Baseline\batch_runs\splitgp_rho_baselines_vgg8_windows\{dataset}\date_YYYYMMDD\time_HHMMSS\
```

The resume policy matches the Bash script: skip a run when a completed `acc.csv` exists through round 101.
