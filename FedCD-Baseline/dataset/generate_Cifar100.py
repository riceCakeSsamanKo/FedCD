import numpy as np
import os
import sys
import random
import glob
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "Cifar100/"
rawdata_root = None


def _clear_npz_files(path):
    for file_path in glob.glob(os.path.join(path, "*.npz")):
        os.remove(file_path)


# Allocate data to users
def generate_dataset(
    dir_path,
    num_clients,
    niid,
    balance,
    partition,
    alpha=0.5,
    rawdata_root=None,
    class_per_client=10,
):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition, alpha=alpha):
        return
        
    # Get Cifar100 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    raw_root = rawdata_root if rawdata_root is not None else dir_path + "rawdata"
    trainset = torchvision.datasets.CIFAR100(
        root=raw_root, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=raw_root, train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data(
        (dataset_image, dataset_label),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        class_per_client=class_per_client,
        alpha=alpha,
    )
    train_data, test_data = split_data(X, y)
    _clear_npz_files(train_path)
    _clear_npz_files(test_path)
    save_file(
        config_path,
        train_path,
        test_path,
        train_data,
        test_data,
        num_clients,
        num_classes,
        statistic,
        niid,
        balance,
        partition,
        alpha,
    )


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    alpha = 0.5
    class_per_client = 10

    if len(sys.argv) > 4:
        num_clients = int(sys.argv[4])
    if len(sys.argv) > 5:
        alpha = float(sys.argv[5])
    if len(sys.argv) > 6:
        dir_path = sys.argv[6]
        if not dir_path.endswith("/"):
            dir_path += "/"
    if len(sys.argv) > 7:
        rawdata_root = sys.argv[7]
    if len(sys.argv) > 8:
        class_per_client = int(sys.argv[8])

    generate_dataset(dir_path, num_clients, niid, balance, partition, alpha, rawdata_root, class_per_client)
