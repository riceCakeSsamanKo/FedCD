import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SYSTEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SYSTEM_DIR not in sys.path:
    sys.path.insert(0, SYSTEM_DIR)

from utils import data_utils


class DataUtilsRootTest(unittest.TestCase):
    @staticmethod
    def _write_npz(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, data=payload)

    def test_reads_fedprism_scenarios_from_shared_pool(self):
        original_env = os.environ.get('FL_DATA_ROOT')
        data_utils._read_npz_data_dict.cache_clear()
        data_utils._fedprism_pool_tensors.cache_clear()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.environ['FL_DATA_ROOT'] = tmpdir
                dataset = 'Toy_fedprism_idoodmix_nc1'
                root = Path(tmpdir) / dataset

                pool = {
                    'x': np.arange(8, dtype=np.float32).reshape(4, 2),
                    'y': np.array([0, 1, 2, 3], dtype=np.int64),
                }
                self._write_npz(root / 'test' / 'pool.npz', pool)
                selections = {
                    'id': np.array([0, 1], dtype=np.int64),
                    'ood': np.array([2, 3], dtype=np.int64),
                    'mix': np.array([0, 2], dtype=np.int64),
                }
                for scenario, indices in selections.items():
                    self._write_npz(
                        root / 'test' / scenario / '0.npz',
                        {
                            'pool_indices': indices,
                            'y': pool['y'][indices],
                        },
                    )

                default_test = data_utils.read_client_data(dataset, 0, is_train=False)
                ood_test = data_utils.read_client_data(
                    dataset, 0, is_train=False, scenario='ood'
                )

                self.assertIsInstance(default_test, data_utils.FedPrismScenarioDataset)
                self.assertEqual(len(default_test), 2)
                self.assertEqual([int(default_test[i][1]) for i in range(2)], [0, 2])
                self.assertEqual([int(ood_test[i][1]) for i in range(2)], [2, 3])
                self.assertIs(default_test.pool_x, ood_test.pool_x)
        finally:
            data_utils._read_npz_data_dict.cache_clear()
            data_utils._fedprism_pool_tensors.cache_clear()
            if original_env is None:
                os.environ.pop('FL_DATA_ROOT', None)
            else:
                os.environ['FL_DATA_ROOT'] = original_env

    def test_get_fl_data_root_finds_shared_data_fl_data(self):
        original_file = data_utils.__file__
        original_env = os.environ.pop("FL_DATA_ROOT", None)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                fake_utils = (
                    Path(tmpdir)
                    / "FedCD"
                    / "FedCD-Baseline"
                    / "system"
                    / "utils"
                )
                fl_data_root = Path(tmpdir) / "data" / "fl_data"
                dataset_root = fl_data_root / "Demo_splitgp_pat_rho0.8_nc50"
                dataset_root.mkdir(parents=True)
                data_utils.__file__ = str(fake_utils / "data_utils.py")

                resolved = data_utils._get_fl_data_root(
                    "Demo_splitgp_pat_rho0.8_nc50"
                )

                self.assertEqual(os.path.abspath(resolved), str(fl_data_root))
        finally:
            data_utils.__file__ = original_file
            if original_env is not None:
                os.environ["FL_DATA_ROOT"] = original_env


if __name__ == "__main__":
    unittest.main()
