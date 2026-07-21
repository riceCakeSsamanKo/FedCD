import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from utils.model_state import average_module_states, copy_module_state
from flcore.servers.serverbase import Server


def make_model():
    return nn.Sequential(
        nn.Linear(2, 2, bias=False),
        nn.BatchNorm1d(2),
    )


class ModelStateTest(unittest.TestCase):
    def test_copy_module_state_includes_batch_norm_buffers(self):
        source = make_model()
        target = make_model()
        with torch.no_grad():
            source[0].weight.fill_(3.0)
            source[1].running_mean.copy_(torch.tensor([1.0, 2.0]))
            source[1].running_var.copy_(torch.tensor([4.0, 5.0]))
            source[1].num_batches_tracked.fill_(7)

        copy_module_state(source, target)

        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)

    def test_average_module_states_includes_batch_norm_buffers(self):
        first = make_model()
        second = make_model()
        with torch.no_grad():
            first[0].weight.fill_(2.0)
            second[0].weight.fill_(6.0)
            first[1].running_mean.copy_(torch.tensor([2.0, 4.0]))
            second[1].running_mean.copy_(torch.tensor([6.0, 8.0]))
            first[1].running_var.fill_(2.0)
            second[1].running_var.fill_(10.0)
            first[1].num_batches_tracked.fill_(2)
            second[1].num_batches_tracked.fill_(6)

        averaged = average_module_states([first, second], [0.25, 0.75])

        self.assertTrue(torch.allclose(averaged[0].weight, torch.full_like(averaged[0].weight, 5.0)))
        self.assertTrue(torch.allclose(averaged[1].running_mean, torch.tensor([5.0, 7.0])))
        self.assertTrue(torch.allclose(averaged[1].running_var, torch.full_like(averaged[1].running_var, 8.0)))
        self.assertEqual(int(averaged[1].num_batches_tracked.item()), 5)


class FedPrismScenarioProtocolTest(unittest.TestCase):
    def test_parses_and_deduplicates_scenarios(self):
        self.assertEqual(
            Server._parse_eval_scenarios('id, ood mix,id'),
            ['id', 'ood', 'mix'],
        )

    def test_rejects_unknown_scenario(self):
        with self.assertRaises(ValueError):
            Server._parse_eval_scenarios('id,unknown')


class FedPrismEvalProtocolTest(unittest.TestCase):
    @staticmethod
    def _fake_client_data(dataset, client_id, is_train=True, few_shot=0):
        del dataset, is_train, few_shot
        return [
            (torch.tensor([float(client_id), float(local_idx)]), torch.tensor(local_idx))
            for local_idx in range(5)
        ]

    def _server(self):
        server = Server.__new__(Server)
        server.num_clients = 2
        server.dataset = 'Toy_splitgp_pat_rho0.8_nc2'
        server.few_shot = 0
        server.fedprism_eval_reserved_fraction = 0.2
        server.fedprism_eval_reserved_seed = 0
        server.global_test_samples = 0
        server._fedprism_eval_positions = None
        server._fedprism_eval_data_cache = {}
        return server

    def test_reuses_same_held_out_indices_for_every_rho(self):
        server = self._server()
        with patch('flcore.servers.serverbase.read_client_data', side_effect=self._fake_client_data), \
                patch('flcore.servers.serverbase.has_reserved_data', return_value=False):
            initial = server._fedprism_eval_data_by_client(server.dataset)
            other_rho = server._fedprism_eval_data_by_client('Toy_splitgp_pat_rho0.2_nc2')

        self.assertEqual(sum(map(len, initial.values())), 8)
        initial_indices = [[int(label.item()) for _, label in initial[idx]] for idx in range(2)]
        other_indices = [[int(label.item()) for _, label in other_rho[idx]] for idx in range(2)]
        self.assertEqual(initial_indices, other_indices)


if __name__ == '__main__':
    unittest.main()
