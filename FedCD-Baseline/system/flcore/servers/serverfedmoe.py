import copy
import random
import time

from tqdm import tqdm

from flcore.clients.clientfedmoe import clientFedMoE
from flcore.servers.serverbase import Server


class FedMoE(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        if not hasattr(args.model, "base"):
            raise ValueError("FedMoE expects args.model to be split into BaseHeadSplit before server construction.")
        self.global_model = copy.deepcopy(args.model.base)
        self.model_size_MB = sum(p.numel() for p in self.global_model.parameters()) * 4 / (1024 * 1024)

        self.set_slow_clients()
        self.set_clients(clientFedMoE)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"FedMoE communication payload: base module only ({self.model_size_MB:.4f} MB)")
        print("Finished creating FedMoE server and clients.")
        self.Budget = []

    def send_models(self):
        assert len(self.clients) > 0
        clients = self.selected_clients if len(self.selected_clients) > 0 else self.clients
        for client in tqdm(clients, desc="Distributing FedMoE base", leave=False):
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)
        self.downlink_MB += len(clients) * self.model_size_MB

    def receive_models(self):
        assert len(self.selected_clients) > 0
        keep = int((1 - self.client_drop_rate) * self.current_num_join_clients)
        active_clients = random.sample(self.selected_clients, keep)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        total_samples = 0
        for client in tqdm(active_clients, desc="Collecting FedMoE bases", leave=False):
            try:
                client_time_cost = (
                    client.train_time_cost["total_cost"] / client.train_time_cost["num_rounds"]
                    + client.send_time_cost["total_cost"] / client.send_time_cost["num_rounds"]
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

    def train(self):
        for i in tqdm(range(self.global_rounds + 1), desc="Global Rounds"):
            start_time = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized FedMoE models")
                self.evaluate()

            for client in tqdm(self.selected_clients, desc="Training FedMoE clients", leave=False):
                client.train()

            self.receive_models()
            if self.uploaded_models:
                if self.dlg_eval and i % self.dlg_gap == 0:
                    self.call_dlg(i)
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

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFedMoE)
            print("\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
