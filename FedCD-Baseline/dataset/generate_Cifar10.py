import numpy as np
import os
import sys
import random
import glob
import json
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import separate_data, save_file


random.seed(1)
np.random.seed(1)
dir_path = "Cifar10/"


def _config_matches(config, num_clients, niid, balance, partition, alpha):
    return (
        config.get("num_clients") == num_clients
        and config.get("non_iid") == niid
        and config.get("balance") == balance
        and config.get("partition") == partition
        and float(config.get("alpha", -1.0)) == float(alpha)
        and bool(config.get("use_original_test_split", False))
    )


def _clear_npz_files(path):
    for file_path in glob.glob(os.path.join(path, "*.npz")):
        os.remove(file_path)


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition, alpha=0.1):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if _config_matches(config, num_clients, niid, balance, partition, alpha):
                print("\nDataset already generated (original CIFAR-10 split: 50k/10k).\n")
                return
        except Exception:
            pass
        
    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    train_image = np.array(trainset.data.cpu().detach().numpy())
    train_label = np.array(trainset.targets.cpu().detach().numpy())
    test_image = np.array(testset.data.cpu().detach().numpy())
    test_label = np.array(testset.targets.cpu().detach().numpy())

    num_classes = len(set(train_label))
    print(f'Number of classes: {num_classes}')

    # Partition original train split (50,000) across clients.
    X_train, y_train, statistic = separate_data(
        (train_image, train_label),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        class_per_client=2,
        alpha=alpha,
    )

    # Partition original test split (10,000) across clients.
    X_test, y_test, _ = separate_data(
        (test_image, test_label),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        class_per_client=2,
        alpha=alpha,
    )

    train_data = [{'x': X_train[i], 'y': y_train[i]} for i in range(num_clients)]
    test_data = [{'x': X_test[i], 'y': y_test[i]} for i in range(num_clients)]

    _clear_npz_files(train_path)
    _clear_npz_files(test_path)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition, alpha=alpha)

    # Mark this dataset as using the original CIFAR-10 split.
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["use_original_test_split"] = True
    config["original_train_samples"] = int(len(train_label))
    config["original_test_samples"] = int(len(test_label))
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    num_clients = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    alpha = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
    if len(sys.argv) > 6:
        dir_path = sys.argv[6]
        if not dir_path.endswith("/"):
            dir_path += "/"

    generate_dataset(dir_path, num_clients, niid, balance, partition, alpha)
