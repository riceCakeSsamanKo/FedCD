import copy
import time

import numpy as np
import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from flcore.clients.clientbase import Client
from flcore.trainmodel.moe.moe import ExtractorToPMoE


class clientFedMoE(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args
        if not hasattr(self.model, "base") or not hasattr(self.model, "head"):
            raise ValueError("FedMoE expects a BaseHeadSplit model with .base and .head modules.")
        self.model.local_extra = copy.deepcopy(self.model.base)

    def _move_batch_to_device(self, x):
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
            if torch.is_floating_point(x[0]) and not torch.isfinite(x[0]).all():
                x[0] = torch.nan_to_num(x[0], nan=0.0, posinf=1.0, neginf=0.0)
            return x
        x = x.to(self.device)
        if torch.is_floating_point(x) and not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        return x

    def _gate_input_dim(self, trainloader):
        sample_x, _ = trainloader.dataset[0]
        if type(sample_x) == type([]):
            return 32
        return int(sample_x.reshape(-1).numel())

    def train(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()

        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2 + 1))

        trained_experts = [self.model.local_extra, self.model.base]
        self.model.moe = ExtractorToPMoE(
            trained_experts=trained_experts,
            gate_input_dim=self._gate_input_dim(trainloader),
            args=self.args,
        ).to(self.device)
        self.moe_opt = torch.optim.SGD(self.model.parameters(), lr=self.args.local_learning_rate)

        for _ in range(max_local_epochs):
            for x, y in tqdm(trainloader, desc=f"Client {self.id} FedMoE Training", leave=False):
                x = self._move_batch_to_device(x)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                if self.dataset == "AGNews":
                    text, _ = x
                    emb = self.model.base.embedding(text)
                    rep = self.model.moe(emb.mean(1))
                else:
                    rep = self.model.moe(x)

                output = self.model.head(rep)
                if not torch.isfinite(output).all():
                    continue
                loss = self.loss(output, y)
                if not torch.isfinite(loss):
                    continue

                self.moe_opt.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                grad_is_finite = True
                for param in self.model.parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        grad_is_finite = False
                        break
                if not grad_is_finite:
                    self.moe_opt.zero_grad()
                    continue
                self.moe_opt.step()
                for param in self.model.parameters():
                    if not torch.isfinite(param.data).all():
                        param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1e4, neginf=-1e4)

        self.model.cpu()
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def _forward_fedmoe(self, x):
        if hasattr(self.model, "moe") and self.model.moe is not None:
            if self.dataset == "AGNews":
                text, _ = x
                emb = self.model.base.embedding(text)
                rep = self.model.moe(emb.mean(1))
            else:
                rep = self.model.moe(x)
            return self.model.head(rep)
        return self.model(x)

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        invalid_values_found = False

        with torch.no_grad():
            for x, y in testloaderfull:
                x = self._move_batch_to_device(x)
                y = y.to(self.device)
                output = self._forward_fedmoe(x)
                if not torch.isfinite(output).all():
                    output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
                    invalid_values_found = True

                test_acc += torch.sum(torch.argmax(output, dim=1) == y).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        self.model.cpu()
        if len(y_prob) == 0 or len(y_true) == 0:
            return test_acc, test_num, 0.0

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1e6, neginf=-1e6)
        if y_prob.ndim == 2 and y_prob.shape[1] > 1:
            y_prob = y_prob - np.max(y_prob, axis=1, keepdims=True)
            y_prob = np.exp(y_prob)
            denom = np.sum(y_prob, axis=1, keepdims=True)
            denom[denom == 0] = 1.0
            y_prob = y_prob / denom

        try:
            auc = metrics.roc_auc_score(y_true, y_prob, average="micro")
        except ValueError:
            auc = 0.0

        if invalid_values_found:
            print(f"Warning: non-finite values detected during FedMoE evaluation on client {self.id}; sanitized for AUC.")
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        invalid_values_found = False
        with torch.no_grad():
            for x, y in trainloader:
                x = self._move_batch_to_device(x)
                y = y.to(self.device)
                output = self._forward_fedmoe(x)
                if not torch.isfinite(output).all():
                    output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
                    invalid_values_found = True
                loss = self.loss(output, y)
                if not torch.isfinite(loss):
                    invalid_values_found = True
                    continue
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        self.model.cpu()
        if invalid_values_found:
            print(f"Warning: non-finite values detected during FedMoE train-metric eval on client {self.id}; invalid batches skipped.")
        return losses, train_num
    def set_parameters(self, model):
        source = model.base if hasattr(model, "base") else model
        for new_param, old_param in zip(source.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()


