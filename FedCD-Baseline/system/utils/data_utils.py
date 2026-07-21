import numpy as np
import os
import torch
from collections import defaultdict
from functools import lru_cache
from torch.utils.data import Dataset


FEDPRISM_SCENARIOS = ('id', 'ood', 'mix')


class FedPrismScenarioDataset(Dataset):
    # Keep the 8,000-image pool shared instead of materializing it per client.
    def __init__(self, pool_x, pool_y, pool_indices):
        self.pool_x = pool_x
        self.pool_y = pool_y
        self.pool_indices = torch.as_tensor(pool_indices, dtype=torch.int64)

    def __len__(self):
        return int(self.pool_indices.numel())

    def __getitem__(self, index):
        pool_index = int(self.pool_indices[index])
        return self.pool_x[pool_index], self.pool_y[pool_index]


def _get_fl_data_root(dataset=None):
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    env_root = os.environ.get("FL_DATA_ROOT", "").strip()
    candidates = [
        os.path.abspath(os.path.expanduser(env_root)) if env_root else "",
        os.path.abspath(os.path.join(repo_root, "..", "data", "fl_data")),
        os.path.join(repo_root, "fl_data"),
        os.path.abspath(os.path.join(repo_root, "..", "fl_data")),
    ]
    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if os.path.isdir(path) and (dataset is None or os.path.isdir(os.path.join(path, dataset))):
            return path
    fallback = next(path for path in candidates if path)
    if dataset is not None:
        searched = ", ".join(path for path in candidates if path)
        raise FileNotFoundError(
            f"Dataset '{dataset}' was not found under FL data roots: {searched}"
        )
    return fallback


def _dataset_root(dataset):
    return os.path.join(_get_fl_data_root(dataset), dataset)


@lru_cache(maxsize=256)
def _read_npz_data_dict(path):
    with open(path, 'rb') as file_obj:
        with np.load(file_obj, allow_pickle=True) as archive:
            if archive.files != ['data']:
                raise ValueError(f'Unexpected NPZ members in {path}: {archive.files}')
            data = archive['data'].tolist()
    if not isinstance(data, dict):
        raise TypeError(f'Expected a data dictionary in {path}')
    return data


@lru_cache(maxsize=4)
def _fedprism_pool_tensors(dataset):
    pool_path = os.path.join(_dataset_root(dataset), 'test', 'pool.npz')
    pool = _read_npz_data_dict(pool_path)
    x = torch.from_numpy(np.asarray(pool['x'], dtype=np.float32))
    y = torch.from_numpy(np.asarray(pool['y'], dtype=np.int64))
    return x, y


def is_fedprism_scenario_dataset(dataset):
    dataset_root = _dataset_root(dataset)
    return (
        os.path.isfile(os.path.join(dataset_root, 'test', 'pool.npz'))
        and all(
            os.path.isdir(os.path.join(dataset_root, 'test', scenario))
            for scenario in FEDPRISM_SCENARIOS
        )
    )


def read_fedprism_scenario_data(dataset, idx, scenario):
    scenario = str(scenario).strip().lower()
    if scenario not in FEDPRISM_SCENARIOS:
        raise ValueError(
            f'scenario must be one of {FEDPRISM_SCENARIOS}; got {scenario!r}'
        )

    selection_path = os.path.join(
        _dataset_root(dataset), 'test', scenario, f'{idx}.npz'
    )
    selection = _read_npz_data_dict(selection_path)
    pool_indices = np.asarray(selection['pool_indices'], dtype=np.int64)
    pool_x, pool_y = _fedprism_pool_tensors(dataset)

    if pool_indices.ndim != 1:
        raise ValueError(f'pool_indices must be one-dimensional in {selection_path}')
    if np.any(pool_indices < 0) or np.any(pool_indices >= len(pool_y)):
        raise IndexError(f'Out-of-range pool index in {selection_path}')

    expected_y = np.asarray(selection.get('y', []), dtype=np.int64)
    actual_y = pool_y[torch.from_numpy(pool_indices)].numpy()
    if expected_y.shape != actual_y.shape or not np.array_equal(expected_y, actual_y):
        raise ValueError(
            f'FedPRISM {scenario} labels for client {idx} do not match test/pool.npz'
        )
    return FedPrismScenarioDataset(pool_x, pool_y, pool_indices)


def read_data(dataset, idx, is_train=True, scenario=None):
    fl_data_root = _get_fl_data_root(dataset)
    if is_train:
        data_dir = os.path.join(fl_data_root, dataset, "train")
    else:
        if scenario is not None:
            return read_fedprism_scenario_data(dataset, idx, scenario)
        data_dir = os.path.join(fl_data_root, dataset, "test")

    file = os.path.join(data_dir, f"{idx}.npz")
    if not is_train and not os.path.isfile(file) and is_fedprism_scenario_dataset(dataset):
        return read_fedprism_scenario_data(dataset, idx, 'mix')
    with open(file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data


def read_client_data(dataset, idx, is_train=True, few_shot=0, scenario=None):
    data = read_data(dataset, idx, is_train, scenario=scenario)
    if isinstance(data, Dataset):
        return data
    if "News" in dataset:
        data_list = process_text(data)
    elif "Shakespeare" in dataset:
        data_list = process_Shakespeare(data)
    else:
        data_list = process_image(data)

    if is_train and few_shot > 0:
        shot_cnt_dict = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new
    return data_list


def has_reserved_data(dataset, name='fext_train'):
    fl_data_root = _get_fl_data_root(dataset)
    path = os.path.join(fl_data_root, dataset, 'reserved', f'{name}.npz')
    if not os.path.isfile(path):
        return False
    with open(path, 'rb') as file_obj:
        data = np.load(file_obj, allow_pickle=True)['data'].tolist()
    return len(data.get('y', [])) > 0

def process_image(data):
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]


def process_text(data):
    X, X_lens = list(zip(*data['x']))
    y = data['y']
    X = torch.Tensor(X).type(torch.int64)
    X_lens = torch.Tensor(X_lens).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]


def process_Shakespeare(data):
    X = torch.Tensor(data['x']).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]
