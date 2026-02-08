import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
import requests
import zipfile
from utils.dataset_utils import check, separate_data, split_data, save_file
from torchvision.datasets import ImageFolder, DatasetFolder

random.seed(1)
np.random.seed(1)
num_clients = 20

# [Fix] Use absolute path based on this script's location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_PATH = os.path.join(CURRENT_DIR, "TinyImagenet")

# https://github.com/QinbinLi/MOON/blob/6c7a4ed1b1a8c0724fa2976292a667a828e3ff5d/datasets.py#L148
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Directory not found: {self.root}")

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)

def download_url(url, save_path):
    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    else:
        raise Exception(f"Failed to download. Status code: {response.status_code}")

def unzip_file(zip_path, extract_to):
    print(f"Unzipping {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train")
    test_path = os.path.join(dir_path, "test")

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Get data
    rawdata_dir = os.path.join(dir_path, "rawdata")
    if not os.path.exists(rawdata_dir):
        os.makedirs(rawdata_dir)

    tiny_root = os.path.join(rawdata_dir, "tiny-imagenet-200")
    if not os.path.exists(tiny_root):
        zip_path = os.path.join(rawdata_dir, "tiny-imagenet-200.zip")
        if not os.path.exists(zip_path):
            download_url('http://cs231n.stanford.edu/tiny-imagenet-200.zip', zip_path)
        unzip_file(zip_path, rawdata_dir)
    else:
        print('rawdata already exists.\n')

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # [Fix] Final absolute path for ImageFolder
    train_img_path = os.path.join(tiny_root, "train")
    print(f"Loading images from: {train_img_path}")
    
    trainset = ImageFolder_custom(root=train_img_path, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=20)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    # Use the absolute DIR_PATH
    generate_dataset(DIR_PATH, num_clients, niid, balance, partition)