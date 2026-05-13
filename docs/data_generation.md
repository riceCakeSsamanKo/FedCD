# Reproducible FL Data Generation

This repository now has one canonical entry point for the FL splits and one
entry point for OOD proxy data. The target layout is the sibling `../fl_data`
directory unless another path is passed explicitly.

## Canonical Setting

| Dataset | Generator style | Output names | Split rule |
| --- | --- | --- | --- |
| CIFAR-10 | FedCD baseline | `Cifar10_{pat,dir0.1,dir0.5,dir1.0}_nc{20,50}` | keep original torchvision train/test split, then partition train and test across clients |
| CIFAR-100 | FedCD baseline | `Cifar100_{pat,dir0.1,dir0.5,dir1.0}_nc{20,50}` | keep original torchvision train/test split, then partition train and test across clients |
| FashionMNIST | FedCCM style | `FashionMNIST_{pat,dir0.1,dir0.5,dir1.0}_nc{20,50}` | merge original train/test, partition across clients, then split each client 75/25 |

Common FL settings:

- `seed=1`
- `non_iid=true`
- `balance=true`
- scenarios: `pat`, `dir0.1`, `dir0.5`, `dir1.0`
- client counts: `20`, `50`
- CIFAR-10 `class_per_client=2`
- CIFAR-100 `class_per_client=10`
- FashionMNIST `class_per_client=2`

## Generate FL Splits

Activate the same Python environment used for FedCD/FedCCM experiments first.
It must be able to import `torch`, `torchvision`, `sklearn`, `numpy`, and
`ujson`.

From the FedCD repository root:

```powershell
python tools\regenerate_fedcd_fl_data.py --fl-data-root ..\fl_data --delete-existing
```

The script stores raw torchvision downloads in `..\fl_data\_raw` by default.
If raw files are already present and the machine has no network access, use:

```powershell
python tools\regenerate_fedcd_fl_data.py --fl-data-root ..\fl_data --delete-existing --no-download-raw
```

Selective examples:

```powershell
python tools\regenerate_fedcd_fl_data.py --fl-data-root ..\fl_data --datasets Cifar10 --num-clients 20 --scenarios dir0.1 --delete-existing
python tools\regenerate_fedcd_fl_data.py --fl-data-root ..\fl_data --datasets FashionMNIST --delete-existing
```

For byte-for-byte comparable random streams, run the full default command on an
empty or deleted target directory.

## Generate OOD Proxy Data

The OOD proxy builder reuses the FedCCM implementation. Place the FedCCM repo
next to FedCD as `..\FedCCM`, or pass `--fedccm-root <path>`.

```powershell
python tools\prepare_ood_proxy.py --fl-data-root ..\fl_data --fedccm-root ..\FedCCM --proxy-source current
```

`--proxy-source current` builds the current proxy family:

- MNIST proxy for FashionMNIST and RGB MNIST proxy for CIFAR-10
- TinyImageNet non-overlap proxies for CIFAR-10, CIFAR-100, FashionMNIST
- COCO non-overlap proxies for CIFAR-10, CIFAR-100, FashionMNIST
- combined TinyImageNet+COCO proxies for CIFAR-10, CIFAR-100, FashionMNIST

To rebuild proxy directories from scratch:

```powershell
python tools\prepare_ood_proxy.py --fl-data-root ..\fl_data --fedccm-root ..\FedCCM --proxy-source current --delete-existing
```

The proxy script intentionally does not regenerate or overwrite FL split
directories.

## Verify

Check FL splits only:

```powershell
python tools\verify_fl_data_setup.py --fl-data-root ..\fl_data
```

Check FL splits and expected OOD proxy directories:

```powershell
python tools\verify_fl_data_setup.py --fl-data-root ..\fl_data --check-ood-proxy
```

Expected sample counts:

- CIFAR-10 and CIFAR-100: train `50000`, test `10000` for every split.
- FashionMNIST: total `70000`, with each generated split around train `52500`
  and test `17500` because each client is split 75/25 after partitioning.
