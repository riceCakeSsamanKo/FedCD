import copy
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientcp import *
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from utils.model_state import average_module_states


class FedCP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        in_dim = list(args.model.head.parameters())[0].shape[1]
        cs = ConditionalSelection(in_dim, in_dim).to(args.device)

        self.global_modules = copy.deepcopy(args.model.base)
        self.base_size_MB = self._module_size_mb(args.model.base)
        self.head_size_MB = self._module_size_mb(args.model.head)
        self.cs_size_MB = self._module_size_mb(cs)
        self.model_size_MB = self.base_size_MB + self.head_size_MB + self.cs_size_MB

        self.set_slow_clients()
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientCP(self.args,
                            id=i,
                            train_samples=len(train_data),
                            test_samples=len(test_data),
                            train_slow=train_slow,
                            send_slow=send_slow,
                            ConditionalSelection=cs)
            self.clients.append(client)

        self._ensure_dynamic_client_groups()
        self._assign_fedprism_eval_data(self.dataset)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(
            "FedCP communication payloads: "
            f"base={self.base_size_MB:.4f} MB, "
            f"head={self.head_size_MB:.4f} MB, "
            f"conditional_selector={self.cs_size_MB:.4f} MB"
        )
        print("Finished creating server and clients.")

        self.Budget = []
        self.head = None
        self.cs = None

    @staticmethod
    def _module_size_mb(module):
        return sum(param.numel() for param in module.parameters()) * 4 / (1024 * 1024)

    def _on_dynamic_clients_activated(self, new_clients):
        for client in new_clients:
            client.set_parameters(self.global_modules)
            if self.head is not None:
                client.set_head_g(self.head)
            if self.cs is not None:
                client.set_cs(self.cs)

    def send_models(self):
        assert (len(self.clients) > 0)
        clients = self._dynamic_client_active_clients()
        for client in clients:
            start_time = time.time()
            client.set_parameters(self.global_modules)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
        self.downlink_MB += len(clients) * self.base_size_MB

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_modules.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_modules = average_module_states(
            self.uploaded_models,
            self.uploaded_weights,
        )

    def evaluate(self, acc=None):
        return super().evaluate(acc=acc)

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate before local training")
                self.evaluate()

            for client in self.selected_clients:
                client.train_cs_model()
                client.generate_upload_head()

            self.receive_models()
            self.aggregate_parameters()
            self.send_models()
            self.global_head()
            self.global_cs()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:])/len(self.Budget[1:]))
        else:
            print(0.0)

        self.save_results()
        self.save_global_model()

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model.model.base)
        self.uplink_MB += len(self.selected_clients) * self.base_size_MB

    def global_head(self):
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_model_gs.append(client.model.head_g)
        self.uplink_MB += len(self.selected_clients) * self.head_size_MB

        self.head = average_module_states(
            self.uploaded_model_gs,
            self.uploaded_weights,
        )

        for client in self.selected_clients:
            client.set_head_g(self.head)
        self.downlink_MB += len(self.selected_clients) * self.head_size_MB

    def add_head(self, w, head):
        for server_param, client_param in zip(self.head.parameters(), head.parameters()):
            server_param.data += client_param.data.clone() * w

    def global_cs(self):
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_model_gs.append(client.model.gate.cs)
        self.uplink_MB += len(self.selected_clients) * self.cs_size_MB

        self.cs = average_module_states(
            self.uploaded_model_gs,
            self.uploaded_weights,
        )

        for client in self.selected_clients:
            client.set_cs(self.cs)
        self.downlink_MB += len(self.selected_clients) * self.cs_size_MB

    def add_cs(self, w, cs):
        for server_param, client_param in zip(self.cs.parameters(), cs.parameters()):
            server_param.data += client_param.data.clone() * w


class ConditionalSelection(nn.Module):
    def __init__(self, in_dim, h_dim):
        super(ConditionalSelection, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dim, h_dim*2),
            nn.LayerNorm([h_dim*2]),
            nn.ReLU(),
        )

    def forward(self, x, tau=1, hard=False):
        shape = x.shape
        x = self.fc(x)
        x = x.view(shape[0], 2, -1)
        x = F.gumbel_softmax(x, dim=1, tau=tau, hard=hard)
        return x[:, 0, :], x[:, 1, :]
