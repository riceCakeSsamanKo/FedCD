import os
import sys
import unittest
from types import SimpleNamespace

import torch


SYSTEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SYSTEM_DIR not in sys.path:
    sys.path.insert(0, SYSTEM_DIR)

from flcore.clients.clientpFedMe import clientpFedMe


class PFedMePersonalizedStateTest(unittest.TestCase):
    def test_set_parameters_preserves_personalized_params(self):
        base_model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            base_model.weight.fill_(1.0)

        args = SimpleNamespace(
            model=base_model,
            algorithm="pFedMe",
            dataset="dummy",
            device="cpu",
            save_folder_name="models",
            num_classes=2,
            batch_size=4,
            local_learning_rate=0.01,
            local_epochs=1,
            few_shot=0,
            learning_rate_decay_gamma=0.99,
            learning_rate_decay=False,
            lamda=1.0,
            K=5,
            p_learning_rate=0.01,
            seed=0,
        )
        client = clientpFedMe(
            args,
            id=0,
            train_samples=1,
            test_samples=1,
            train_slow=False,
            send_slow=False,
        )

        preserved = [p.detach().clone() + 3.0 for p in client.personalized_params]
        client.personalized_params = [p.clone() for p in preserved]

        new_model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            new_model.weight.fill_(2.0)

        client.set_parameters(new_model)

        for current, expected in zip(client.personalized_params, preserved):
            self.assertTrue(torch.equal(current, expected))
        for current, expected in zip(client.local_params, new_model.parameters()):
            self.assertTrue(torch.equal(current, expected.detach()))
        for current, expected in zip(client.model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(current.detach(), expected.detach()))


if __name__ == "__main__":
    unittest.main()
