import copy
import random
import time

from tqdm import tqdm

from flcore.clients.clientdualfed import clientDualFed
from flcore.servers.serverbase import Server
from utils.model_state import average_module_states


class DualFed(Server):
    """DualFed-style baseline adapted to the PFLLIB split-model interface."""

    def __init__(self, args, times):
        super().__init__(args, times)
        if not hasattr(args.model, "base") or not hasattr(args.model, "head"):
            raise ValueError("DualFed expects a BaseHeadSplit model.")

        self.global_model = copy.deepcopy(args.model.base)
        self.global_head = copy.deepcopy(args.model.head)
        self.base_size_MB = self._module_size_mb(self.global_model)
        self.global_head_size_MB = self._module_size_mb(self.global_head)
        self.model_size_MB = self.base_size_MB + self.global_head_size_MB

        self.set_slow_clients()
        self.set_clients(clientDualFed)
        self.Budget = []

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(
            "DualFed communication payloads: "
            f"base={self.base_size_MB:.4f} MB, "
            f"global_head={self.global_head_size_MB:.4f} MB, "
            f"shared={self.model_size_MB:.4f} MB"
        )
        print(
            "DualFed configuration: "
            f"contrastive_lambda={getattr(args, 'dualfed_con_lambda', 0.1)}, "
            f"contrastive_temperature={getattr(args, 'dualfed_con_temp', 0.5)}"
        )
        print("Finished creating DualFed server and clients.")

    @staticmethod
    def _module_size_mb(module):
        return sum(param.numel() for param in module.parameters()) * 4 / (1024 * 1024)

    @staticmethod
    def _zero_module(module):
        for param in module.parameters():
            param.data.zero_()

    def _on_dynamic_clients_activated(self, new_clients):
        for client in new_clients:
            client.set_shared_parameters(self.global_model, self.global_head)

    @staticmethod
    def _add_module_parameters(target, source, weight):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data += source_param.data.clone().to(target_param.device) * weight

    def send_models(self, clients=None):
        assert self.clients
        recipients = clients
        if recipients is None:
            recipients = (
                self.selected_clients
                if self.selected_clients
                else self._dynamic_client_active_clients()
            )
        for client in tqdm(recipients, desc="Distributing DualFed shared modules", leave=False):
            start_time = time.time()
            client.set_shared_parameters(self.global_model, self.global_head)
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)
        self.downlink_MB += len(recipients) * self.model_size_MB

    def receive_models(self):
        assert self.selected_clients
        keep = int((1 - self.client_drop_rate) * self.current_num_join_clients)
        keep = max(1, min(keep, len(self.selected_clients)))
        active_clients = random.sample(self.selected_clients, keep)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_bases = []
        self.uploaded_heads = []
        total_samples = 0
        for client in tqdm(active_clients, desc="Collecting DualFed shared modules", leave=False):
            try:
                client_time_cost = (
                    client.train_time_cost["total_cost"]
                    / client.train_time_cost["num_rounds"]
                    + client.send_time_cost["total_cost"]
                    / client.send_time_cost["num_rounds"]
                )
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost > self.time_threthold:
                continue
            total_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_bases.append(copy.deepcopy(client.model.base).cpu())
            self.uploaded_heads.append(copy.deepcopy(client.global_head).cpu())

        if total_samples == 0:
            self.uploaded_weights = []
            return
        self.uploaded_weights = [weight / total_samples for weight in self.uploaded_weights]
        self.uplink_MB += len(self.uploaded_bases) * self.model_size_MB

    def aggregate_parameters(self):
        if not self.uploaded_weights:
            return
        self.global_model = average_module_states(
            self.uploaded_bases,
            self.uploaded_weights,
        )
        self.global_head = average_module_states(
            self.uploaded_heads,
            self.uploaded_weights,
        )

    def train(self):
        for round_idx in tqdm(range(self.global_rounds + 1), desc="Global Rounds"):
            start_time = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if round_idx % self.eval_gap == 0:
                print(f"\n-------------Round number: {round_idx}-------------")
                print("\nEvaluate DualFed personalized models")
                self.evaluate()

            for client in tqdm(self.selected_clients, desc="Training DualFed clients", leave=False):
                client.train()

            self.receive_models()
            if self.uploaded_weights:
                if self.dlg_eval and round_idx % self.dlg_gap == 0:
                    self.call_dlg(round_idx)
                self.aggregate_parameters()

            self.Budget.append(time.time() - start_time)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc) if self.rs_test_acc else 0.0)
        print("\nAverage time cost per round.")
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        else:
            print(0.0)

        self.save_results()
        self.save_global_model()
