import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from flcore.clients.clientbase import Client
from flcore.trainmodel.moe.gate import Gating
from utils.model_state import copy_module_state


class PersonalHeadMoE(nn.Module):
    """Top-k mixture over the converged personal classification heads."""

    def __init__(self, experts, input_dim, topk):
        super().__init__()
        if not experts:
            raise ValueError("PMOE requires at least one personal-head expert.")
        self.experts = nn.ModuleList(experts)
        self.gating = Gating(input_dim, len(experts))
        self.topk = max(1, min(int(topk), len(experts)))

    def forward(self, representation):
        flat_representation = representation.flatten(1)
        gate_weights = self.gating(flat_representation)
        top_weights, top_indices = torch.topk(gate_weights, self.topk, dim=1)

        # Preserve the original PM-MOE top-k weighting semantics while using
        # tensor gathering instead of the source code's expert-index bug.
        expert_logits = torch.stack(
            [expert(representation) for expert in self.experts], dim=1
        )
        gather_index = top_indices.unsqueeze(-1).expand(
            -1, -1, expert_logits.size(-1)
        )
        selected_logits = torch.gather(expert_logits, 1, gather_index)
        return torch.sum(selected_logits * top_weights.unsqueeze(-1), dim=1)


class clientPMOEFedPer(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        if not hasattr(self.model, "base") or not hasattr(self.model, "head"):
            raise ValueError("PMOE_FedPer requires a BaseHeadSplit model.")
        if not hasattr(self.model.head, "in_features"):
            raise ValueError("PMOE_FedPer currently expects a linear personal head.")

        self.args = args
        self.moe_fine_tuning_epochs = int(args.moe_fine_tuning_epochs)
        self.moe_learning_rate = float(args.moe_lr)
        self.lock_experts = int(args.lock_experts)
        self.topk = int(args.topk)
        self.trained_experts = None
        self.is_moe_finetune = False

    def _move_batch(self, x, y):
        if isinstance(x, list):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        return x, y.to(self.device)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow and max_local_epochs > 1:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2 + 1)

        for _ in range(max_local_epochs):
            for x, y in trainloader:
                x, y = self._move_batch(x, y)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.model.cpu()
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def set_parameters(self, model):
        source = model.base if hasattr(model, "base") else model
        copy_module_state(source, self.model.base)

    def set_moe_experts(self, personal_heads):
        self.trained_experts = copy.deepcopy(personal_heads)

    def moe_finetune(self):
        if not self.trained_experts:
            raise ValueError("The PMOE expert pool was not distributed to the client.")
        if self.topk > len(self.trained_experts):
            raise ValueError(
                f"PMOE topk={self.topk} exceeds the expert count={len(self.trained_experts)}."
            )

        start_time = time.time()
        self.model.moe = PersonalHeadMoE(
            experts=self.trained_experts,
            input_dim=self.model.head.in_features,
            topk=self.topk,
        )
        self.model.to(self.device)
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.moe.gating.parameters():
            param.requires_grad = True
        if self.lock_experts == 1:  # Original PM-MOE convention: 1 unlocks experts.
            for param in self.model.moe.experts.parameters():
                param.requires_grad = True

        trainable = [param for param in self.model.parameters() if param.requires_grad]
        optimizer = torch.optim.SGD(trainable, lr=self.moe_learning_rate)
        trainloader = self.load_train_data()

        for _ in range(self.moe_fine_tuning_epochs):
            for x, y in trainloader:
                x, y = self._move_batch(x, y)
                representation = self.model.base(x)
                output = self.model.moe(representation)
                loss = self.loss(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.is_moe_finetune = True
        self.model.cpu()
        self.train_time_cost["total_cost"] += time.time() - start_time

    def _forward(self, x):
        representation = self.model.base(x)
        if self.is_moe_finetune:
            return self.model.moe(representation)
        return self.model.head(representation)

    def _move_eval_batch(self, x, y):
        return self._move_batch(x, y)

    def _eval_forward(self, x):
        return torch.nan_to_num(self._forward(x), nan=0.0, posinf=1e6, neginf=-1e6)

    def test_metrics(self):
        testloader = self.load_test_data()
        self.model.to(self.device)
        self.model.eval()
        test_acc = 0
        test_num = 0
        probabilities = []
        targets = []

        with torch.no_grad():
            for x, y in testloader:
                x, y = self._move_batch(x, y)
                output = torch.nan_to_num(
                    self._forward(x), nan=0.0, posinf=1e6, neginf=-1e6
                )
                test_acc += torch.sum(torch.argmax(output, dim=1) == y).item()
                test_num += y.shape[0]
                probabilities.append(torch.softmax(output, dim=1).cpu().numpy())
                class_count = self.num_classes + (1 if self.num_classes == 2 else 0)
                binary = label_binarize(
                    y.cpu().numpy(), classes=np.arange(class_count)
                )
                targets.append(binary[:, :2] if self.num_classes == 2 else binary)

        self.model.cpu()
        if not probabilities:
            return test_acc, test_num, 0.0
        probabilities = np.concatenate(probabilities, axis=0)
        targets = np.concatenate(targets, axis=0)
        try:
            auc = metrics.roc_auc_score(targets, probabilities, average="micro")
        except ValueError:
            auc = 0.0
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()
        train_num = 0
        losses = 0.0
        with torch.no_grad():
            for x, y in trainloader:
                x, y = self._move_batch(x, y)
                output = self._forward(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        self.model.cpu()
        return losses, train_num
