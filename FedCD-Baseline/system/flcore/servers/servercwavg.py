import copy
import random
import time
from collections import defaultdict

import numpy as np
import torch

from flcore.clients.clientcwavg import cwclientAVG
from flcore.servers.serverbase import Server


def _module_size_mb(module):
    return sum(p.numel() for p in module.parameters()) * 4 / (1024 * 1024)


class cwFedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(cwclientAVG)

        self.Budget = []
        self.uploaded_weights_cw = []
        self.uploaded_models_head = []

        if isinstance(args.model, list):
            raise NotImplementedError("cwFedAvg baseline integration expects a single model.")

        self.cw_global_model = []
        if self.args.add_cw:
            for _ in range(self.num_classes):
                if self.args.partial_layer_train:
                    self.layer_groups = args.layer_groups
                    cw_layers = {}
                    for name, module in args.model.named_children():
                        if name in self.layer_groups["cw"]:
                            cw_layers[name] = copy.deepcopy(module)
                    self.cw_global_model.append(cw_layers)
                elif self.args.decision_layer_only:
                    self.cw_global_model.append(copy.deepcopy(args.model.head))
                else:
                    self.cw_global_model.append(copy.deepcopy(args.model))

        if not self.args.decision_layer_only and not self.args.partial_layer_train:
            self.global_model = None

        if self.args.add_proto:
            self.global_protos = [None for _ in range(args.num_classes)]

        if self.args.partial_layer_train:
            self.cw_model_size_MB = sum(
                _module_size_mb(module) for name, module in args.model.named_children() if name in self.layer_groups["cw"]
            )
        elif self.args.decision_layer_only:
            self.cw_model_size_MB = _module_size_mb(args.model.head)
        else:
            self.cw_model_size_MB = _module_size_mb(args.model)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def _cw_downlink_per_client_mb(self):
        base_mb = self.model_size_MB if self.global_model is not None else 0.0
        if self.args.add_cw:
            return base_mb + self.num_classes * self.cw_model_size_MB
        return base_mb

    def train(self):
        for round_idx in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            assert len(self.clients) > 0

            for client in self.clients:
                start_time = time.time()
                if self.args.add_cw:
                    client.local_initializtion_cw(self.cw_global_model, self.global_model)
                else:
                    client.set_parameters(self.global_model)

                client.send_time_cost["num_rounds"] += 1
                client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)
            self.downlink_MB += len(self.clients) * self._cw_downlink_per_client_mb()

            if round_idx % self.eval_gap == 0:
                print(f"\n-------------Round number: {round_idx}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            if self.args.add_proto:
                self.receive_protos()
                self.global_protos = proto_aggregation(self.uploaded_protos)
                self.send_protos()

            if self.args.add_cw:
                self.receive_models_cw()
            else:
                self.receive_models()

            if self.dlg_eval and round_idx % self.dlg_gap == 0:
                self.call_dlg(round_idx)

            if self.args.add_cw:
                if self.args.decision_layer_only or self.args.partial_layer_train:
                    self.aggregate_parameters()
                    self.aggregate_parameters_cw()
                else:
                    self.aggregate_parameters_cw()
            else:
                self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        self.save_results()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(cwclientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

        print("Final value of regularizer(||p-p^||)")
        sum_wdr = []
        sum_zero_class = []
        for client in self.clients:
            fc_weight_norm = torch.tensor(client.aggregate_weight_calc(), dtype=torch.float32)
            if self.args.clip_weight:
                fc_weight_norm[client.mask.cpu()] = 0
            fc_weight = fc_weight_norm / torch.sum(fc_weight_norm).clamp_min(1e-12)
            wd_regularizer = torch.norm(fc_weight - client.gt.detach().cpu(), p=2)
            gt_np = client.gt.detach().cpu().numpy()
            fc_weight_np = fc_weight.detach().cpu().numpy()
            sum_wdr.append(float(wd_regularizer.detach().cpu().item()))
            sum_zero_class.append(float(np.sum(fc_weight_np[gt_np == 0])))

        if sum_wdr:
            print(f"Average value of regularizer: {sum(sum_wdr) / len(sum_wdr)}")
            print(f"Average value of sum of zero class: {sum(sum_zero_class) / len(sum_zero_class)}")

    def send_protos(self):
        assert len(self.clients) > 0
        for client in self.clients:
            start_time = time.time()
            client.set_protos(self.global_protos)
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert len(self.selected_clients) > 0
        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)

    def receive_models_cw(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients)
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_weights_cw = []
        self.uploaded_models = []
        self.uploaded_models_head = []
        tot_samples = 0

        for client in active_clients:
            try:
                client_time_cost = (
                    client.train_time_cost["total_cost"] / client.train_time_cost["num_rounds"]
                    + client.send_time_cost["total_cost"] / client.send_time_cost["num_rounds"]
                )
            except ZeroDivisionError:
                client_time_cost = 0

            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_models.append(client.model)

                if self.args.partial_layer_train:
                    cw_layers = {}
                    for name, module in client.model.named_children():
                        if name in self.layer_groups["cw"]:
                            cw_layers[name] = copy.deepcopy(module)
                    self.uploaded_models_head.append(cw_layers)
                elif self.args.decision_layer_only:
                    self.uploaded_models_head.append(copy.deepcopy(client.model.head))

                self.uploaded_weights.append(client.train_samples)
                if self.args.use_true_dist:
                    weight_list = client.data_dist
                else:
                    weight_list = client.aggregate_weight_calc()
                denom = float(sum(weight_list)) if sum(weight_list) != 0 else 1.0
                self.uploaded_weights_cw.append([x / denom for x in weight_list])

        if tot_samples > 0:
            for idx, weight in enumerate(self.uploaded_weights):
                self.uploaded_weights[idx] = weight / tot_samples

        # Communication accounting:
        # one full local model upload per accepted client; head/cw layers are extracted server-side.
        self.uplink_MB += len(self.uploaded_models) * self.model_size_MB

    def aggregate_parameters_cw(self):
        assert len(self.uploaded_models) > 0

        for class_idx in range(len(self.cw_global_model)):
            if self.args.decision_layer_only:
                self.cw_global_model[class_idx] = copy.deepcopy(self.uploaded_models_head[0])
                for param in self.cw_global_model[class_idx].parameters():
                    param.data.zero_()
            elif self.args.partial_layer_train:
                for _, module in self.cw_global_model[class_idx].items():
                    for param in module.parameters():
                        param.data.zero_()
            else:
                for param in self.cw_global_model[class_idx].parameters():
                    param.data.zero_()

        fedavg_weight = np.array(self.uploaded_weights, dtype=np.float64).reshape(len(self.uploaded_weights), 1)
        fedavg_weight = np.repeat(fedavg_weight, self.num_classes, axis=1)
        uploaded_weight_np = np.array(self.uploaded_weights_cw, dtype=np.float64) * fedavg_weight
        marginal_weight_np = np.tile(
            np.sum(uploaded_weight_np, axis=0, keepdims=True), (len(self.uploaded_weights), 1)
        )
        marginal_weight_np[marginal_weight_np == 0] = 1.0
        normalized_weight = np.divide(uploaded_weight_np, marginal_weight_np).tolist()

        for idx, class_weights in enumerate(normalized_weight):
            if self.args.partial_layer_train:
                local_model = self.uploaded_models_head[idx]
                for class_idx in range(len(self.cw_global_model)):
                    for name, module in self.cw_global_model[class_idx].items():
                        for target, source in zip(module.parameters(), local_model[name].parameters()):
                            target.data += source.data.clone() * class_weights[class_idx]
            elif self.args.decision_layer_only:
                local_model = self.uploaded_models_head[idx]
                for class_idx in range(len(self.cw_global_model)):
                    for target, source in zip(self.cw_global_model[class_idx].parameters(), local_model.parameters()):
                        target.data += source.data.clone() * class_weights[class_idx]
            else:
                local_model = self.uploaded_models[idx]
                for class_idx in range(len(self.cw_global_model)):
                    for target, source in zip(self.cw_global_model[class_idx].parameters(), local_model.parameters()):
                        target.data += source.data.clone() * class_weights[class_idx]


def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for label, proto_list in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for item in proto_list:
                proto += item.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label
