import copy
import random
import time
import torch.nn as nn
from flcore.clients.clientbn import clientBN
from flcore.servers.serverbase import Server
from tqdm import tqdm


_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def _bn_parameter_names(model):
    names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, _BN_TYPES):
            for param_name, _ in module.named_parameters(recurse=False):
                names.add(f"{module_name}.{param_name}" if module_name else param_name)
    return names


class FedBN(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.bn_parameter_names = _bn_parameter_names(self.global_model)
        self.transmitted_parameter_names = [
            name for name, _ in self.global_model.named_parameters()
            if name not in self.bn_parameter_names
        ]
        self.full_model_size_MB = self.model_size_MB
        self.model_size_MB = self._transmitted_model_size_mb()

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientBN)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(
            "FedBN communication payload: non-BN parameters only "
            f"({self.model_size_MB:.4f} MB/client, full model {self.full_model_size_MB:.4f} MB/client)"
        )
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def _transmitted_model_size_mb(self):
        params = dict(self.global_model.named_parameters())
        return sum(params[name].numel() for name in self.transmitted_parameter_names) * 4 / (1024 * 1024)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in tqdm(active_clients, desc="Collecting models", leave=False):
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        self.uplink_MB += len(self.uploaded_models) * self.model_size_MB

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        global_params = dict(self.global_model.named_parameters())
        for name in self.transmitted_parameter_names:
            global_params[name].data.zero_()

        for w, client_model in zip(self.uploaded_weights, tqdm(self.uploaded_models, desc="Aggregating models", leave=False)):
            client_params = dict(client_model.named_parameters())
            for name in self.transmitted_parameter_names:
                global_params[name].data += client_params[name].data.clone() * w

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientBN)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
