import copy
import random
import time

from tqdm import tqdm

from flcore.clients.clientpmoefedper import clientPMOEFedPer
from flcore.servers.serverbase import Server


class PMOEFedPer(Server):
    """FedPer pre-training followed by PM-MOE personal-head mixing."""

    def __init__(self, args, times):
        super().__init__(args, times)
        if not hasattr(args.model, "base") or not hasattr(args.model, "head"):
            raise ValueError("PMOE_FedPer expects a BaseHeadSplit model.")

        self.global_model = copy.deepcopy(args.model.base)
        self.model_size_MB = self._module_size_mb(self.global_model)
        self.personal_head_size_MB = self._module_size_mb(args.model.head)

        self.set_slow_clients()
        self.set_clients(clientPMOEFedPer)
        self.Budget = []

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(
            "PMOE_FedPer communication payloads: "
            f"base={self.model_size_MB:.4f} MB, "
            f"personal_head={self.personal_head_size_MB:.4f} MB"
        )
        print(
            f"PMOE configuration: experts={self.num_clients}, topk={args.topk}, "
            f"fine_tuning_epochs={args.moe_fine_tuning_epochs}, "
            f"experts_unlocked={bool(args.lock_experts)}"
        )
        print("Finished creating PMOE_FedPer server and clients.")

    @staticmethod
    def _module_size_mb(module):
        return sum(param.numel() for param in module.parameters()) * 4 / (1024 * 1024)

    def send_models(self, clients=None):
        assert self.clients
        recipients = clients
        if recipients is None:
            recipients = self.selected_clients if self.selected_clients else self.clients
        for client in tqdm(recipients, desc="Distributing PMOE_FedPer base", leave=False):
            start_time = time.time()
            client.set_parameters(self.global_model)
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
        self.uploaded_models = []
        total_samples = 0
        for client in tqdm(active_clients, desc="Collecting PMOE_FedPer bases", leave=False):
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
            self.uploaded_models.append(copy.deepcopy(client.model.base).cpu())

        if total_samples == 0:
            return
        self.uploaded_weights = [weight / total_samples for weight in self.uploaded_weights]
        self.uplink_MB += len(self.uploaded_models) * self.model_size_MB

    def _run_pmoe_finetuning(self):
        # Every client contributes one converged personal head. Each client then
        # receives the complete pool and learns only its local top-k gate unless
        # --lock_experts 1 explicitly unlocks the experts.
        personal_heads = [copy.deepcopy(client.model.head).cpu() for client in self.clients]
        expert_count = len(personal_heads)
        self.uplink_MB += expert_count * self.personal_head_size_MB
        self.downlink_MB += expert_count * expert_count * self.personal_head_size_MB

        print("\n-------------PM-MOE fine-tuning-------------")
        print(
            f"Collected {expert_count} personal heads; distributing "
            f"{expert_count} experts to each of {len(self.clients)} clients."
        )
        for client in tqdm(self.clients, desc="Fine-tuning PMOE gates"):
            client.set_moe_experts(personal_heads)
            client.moe_finetune()

    def train(self):
        for round_idx in tqdm(range(self.global_rounds + 1), desc="Global Rounds"):
            start_time = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if round_idx % self.eval_gap == 0:
                print(f"\n-------------Round number: {round_idx}-------------")
                print("\nEvaluate personalized FedPer models")
                self.evaluate()

            for client in tqdm(
                self.selected_clients, desc="Training PMOE_FedPer clients", leave=False
            ):
                client.train()

            self.receive_models()
            if self.uploaded_models:
                if self.dlg_eval and round_idx % self.dlg_gap == 0:
                    self.call_dlg(round_idx)
                self.aggregate_parameters()

            self.Budget.append(time.time() - start_time)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])
            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        # Align every client's base with the final aggregate before collecting
        # and mixing the converged personal heads.
        self.send_models(clients=self.clients)
        print("\n-------------Before PM-MOE-------------")
        self.evaluate()
        self._run_pmoe_finetuning()
        print("\n-------------After PM-MOE-------------")
        self.evaluate()

        print("\nBest accuracy.")
        print(max(self.rs_test_acc) if self.rs_test_acc else 0.0)
        print("\nAverage pre-training time cost per round.")
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        else:
            print(0.0)
        self.save_results()
        self.save_global_model()
