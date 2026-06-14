import os
import sys
import tempfile
import unittest
from pathlib import Path


SYSTEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SYSTEM_DIR not in sys.path:
    sys.path.insert(0, SYSTEM_DIR)

from utils import data_utils


class DataUtilsRootTest(unittest.TestCase):
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
