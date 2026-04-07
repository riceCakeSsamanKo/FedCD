import copy
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from flcore.clients.clientbase import Client


def _last_linear(module):
    last = None
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            last = submodule
    if last is None:
        raise RuntimeError("cwFedAvg requires at least one Linear layer in the target module.")
    return last


class cwclientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args

        if self.args.partial_layer_train:
            self.layer_groups = args.layer_groups
            self.aggregate_global = {}
            for name, module in args.model.named_children():
                if name in self.layer_groups["cw"]:
                    self.aggregate_global[name] = copy.deepcopy(module)
        else:
            self.aggregate_global = copy.deepcopy(args.model)

        self.weight_decay = float(self.args.weight_decay)
        self.data_dist = list(args.data_dist[id])
        total = float(sum(self.data_dist))
        gt = [element / total if total > 0 else 0.0 for element in self.data_dist]
        self.gt = torch.tensor(gt, device=self.device, dtype=torch.float32)
        self.mask = self.gt == 0

        if self.args.split_train:
            if self.args.partial_layer_train:
                common_params = []
                if self.args.cw_layer_num != 0:
                    for name in self.layer_groups["common"]:
                        common_params.extend(list(getattr(self.model, name).parameters()))
                self.optimizer_common = torch.optim.SGD(common_params, lr=self.learning_rate)
                cw_params = []
                for name in self.layer_groups["cw"]:
                    cw_params.extend(list(getattr(self.model, name).parameters()))
                self.optimizer_cw = torch.optim.SGD(cw_params, lr=self.args.head_lr)
            else:
                self.optimizer_common = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
                self.optimizer_cw = torch.optim.SGD(self.model.head.parameters(), lr=self.args.head_lr)

        if self.args.add_proto:
            self.protos = None
            self.global_protos = None
            self.loss_mse = nn.MSELoss()
            self.lamda = 1.0

    def _weight_target_module(self):
        if self.args.partial_layer_train:
            return self.model
        if hasattr(self.model, "head"):
            return self.model.head
        return self.model

    def _decision_weight_norm(self):
        linear = _last_linear(self._weight_target_module())
        return torch.norm(linear.weight, dim=1).unsqueeze(0)

    def train(self):
        self.model.to(self.device)
        self.model.train()

        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2 + 1))

        if self.args.split_train:
            if self.args.partial_layer_train:
                for name, layer in self.model.named_children():
                    requires_grad = name in self.layer_groups["cw"]
                    for param in layer.parameters():
                        param.requires_grad = requires_grad
            elif self.args.decision_layer_only:
                for param in self.model.base.parameters():
                    param.requires_grad = False
                for param in self.model.head.parameters():
                    param.requires_grad = True

            trainloader = self.load_train_data(batch_size=self.args.head_bs)
            for _ in range(max_local_epochs):
                for x, y in trainloader:
                    if isinstance(x, list):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    output = self.model(x)
                    loss = self.loss(output, y)
                    if self.args.add_wdr:
                        fc_weight_norm = self._decision_weight_norm().view(-1)
                        fc_weight = fc_weight_norm / torch.sum(fc_weight_norm).clamp_min(1e-12)
                        wd_regularizer = torch.norm(self.gt - fc_weight, p=2)
                        loss += 0.5 * self.weight_decay * wd_regularizer

                    self.optimizer_cw.zero_grad()
                    loss.backward()
                    self.optimizer_cw.step()

            if self.args.partial_layer_train:
                for name, layer in self.model.named_children():
                    requires_grad = name not in self.layer_groups["cw"]
                    for param in layer.parameters():
                        param.requires_grad = requires_grad
            elif self.args.decision_layer_only:
                for param in self.model.base.parameters():
                    param.requires_grad = True
                for param in self.model.head.parameters():
                    param.requires_grad = False

            trainloader = self.load_train_data()
            for _ in range(max_local_epochs):
                for x, y in trainloader:
                    if isinstance(x, list):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    output = self.model(x)
                    loss = self.loss(output, y)
                    if self.args.add_proto and self.global_protos is not None:
                        rep = self.model.base(x)
                        proto_new = copy.deepcopy(rep.detach())
                        for idx, yy in enumerate(y):
                            label = yy.item()
                            if not isinstance(self.global_protos[label], list):
                                proto_new[idx, :] = self.global_protos[label].data
                        loss += self.loss_mse(proto_new, rep) * self.lamda

                    self.optimizer_common.zero_grad()
                    loss.backward()
                    self.optimizer_common.step()
        else:
            trainloader = self.load_train_data()
            for _ in range(max_local_epochs):
                for x, y in trainloader:
                    if isinstance(x, list):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))

                    output = self.model(x)
                    loss = self.loss(output, y)
                    if self.args.add_wdr:
                        fc_weight_norm = self._decision_weight_norm().view(-1)
                        fc_weight = fc_weight_norm / torch.sum(fc_weight_norm).clamp_min(1e-12)
                        wd_regularizer = torch.norm(self.gt - fc_weight, p=2)
                        loss += 0.5 * self.weight_decay * wd_regularizer

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        if self.args.add_proto:
            self.collect_protos()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time
        self.model.to("cpu")

    def aggregate_weight_calc(self):
        fc_weight_norm = self._decision_weight_norm().view(-1)
        return fc_weight_norm.detach().cpu().numpy().tolist()

    def local_initializtion_cw(self, received_cw_global_models, global_model=None):
        if self.args.use_true_dist:
            weight_list = self.data_dist
        else:
            weight_list = self.aggregate_weight_calc()
        denom = float(sum(weight_list)) if sum(weight_list) != 0 else 1.0
        weight_list = [element / denom for element in weight_list]

        if self.args.partial_layer_train:
            for key, module in self.aggregate_global.items():
                for param in module.parameters():
                    param.data = torch.zeros_like(param.data)

            for w, cw_g_model in zip(weight_list, received_cw_global_models):
                for name, module in cw_g_model.items():
                    for agg_param, global_param in zip(self.aggregate_global[name].parameters(), module.parameters()):
                        agg_param.data += global_param.data.clone() * w
        else:
            self.aggregate_global = copy.deepcopy(received_cw_global_models[0])
            for param in self.aggregate_global.parameters():
                param.data = torch.zeros_like(param.data)

            for w, cw_g_model in zip(weight_list, received_cw_global_models):
                for agg_param, global_param in zip(self.aggregate_global.parameters(), cw_g_model.parameters()):
                    agg_param.data += global_param.data.clone() * w

        if self.args.decision_layer_only:
            for new_param, old_param in zip(global_model.base.parameters(), self.model.base.parameters()):
                old_param.data = new_param.data.clone()
            for new_param, old_param in zip(self.aggregate_global.parameters(), self.model.head.parameters()):
                old_param.data = new_param.data.clone()
        elif self.args.partial_layer_train:
            for name, module in self.model.named_children():
                if name in self.layer_groups["cw"]:
                    for new_param, old_param in zip(self.aggregate_global[name].parameters(), module.parameters()):
                        old_param.data = new_param.data.clone()
                else:
                    for g_name, g_module in global_model.named_children():
                        if g_name in name:
                            for new_param, old_param in zip(g_module.parameters(), module.parameters()):
                                old_param.data = new_param.data.clone()
        else:
            for new_param, old_param in zip(self.aggregate_global.parameters(), self.model.parameters()):
                old_param.data = new_param.data.clone()

    def set_protos(self, global_protos):
        self.global_protos = copy.deepcopy(global_protos)

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)

                for idx, yy in enumerate(y):
                    protos[yy.item()].append(rep[idx, :].detach().data)

        self.protos = agg_func(protos)
        self.model.to("cpu")


def agg_func(protos):
    for label, proto_list in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for item in proto_list:
                proto += item.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]
    return protos
