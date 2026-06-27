import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from flcore.clients.clientbase import Client


class clientDualFed(Client):
    """DualFed-style client with a personal head and a shared global head."""

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        if not hasattr(self.model, "base") or not hasattr(self.model, "head"):
            raise ValueError("DualFed expects a BaseHeadSplit model.")

        self.contrastive_lambda = float(getattr(args, "dualfed_con_lambda", 0.1))
        self.contrastive_temperature = float(getattr(args, "dualfed_con_temp", 0.5))
        self.global_head = copy.deepcopy(self.model.head)

        self.local_optimizer = torch.optim.SGD(
            list(self.model.base.parameters()) + list(self.model.head.parameters()),
            lr=self.learning_rate,
        )
        self.global_head_optimizer = torch.optim.SGD(
            self.global_head.parameters(),
            lr=self.learning_rate,
        )

    def _move_batch(self, x, y):
        if isinstance(x, list):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        return x, y.to(self.device)

    def _supervised_contrastive_loss(self, features, labels):
        if features.size(0) <= 1:
            return features.new_tensor(0.0)

        features = F.normalize(features.flatten(1), p=2, dim=1)
        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(features.device)
        logits_mask = torch.ones_like(positive_mask) - torch.eye(
            positive_mask.size(0), device=features.device
        )
        positive_mask = positive_mask * logits_mask
        positives_per_anchor = positive_mask.sum(dim=1)
        valid_anchor = positives_per_anchor > 0
        if not torch.any(valid_anchor):
            return features.new_tensor(0.0)

        logits = torch.matmul(features, features.T) / max(self.contrastive_temperature, 1e-8)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positives_per_anchor.clamp_min(1.0)
        return -mean_log_prob_pos[valid_anchor].mean()

    def set_shared_parameters(self, base, global_head):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()
        for new_param, old_param in zip(global_head.parameters(), self.global_head.parameters()):
            old_param.data = new_param.data.clone()

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.to(self.device)
        self.global_head.to(self.device)
        self.model.train()
        self.global_head.train()

        max_local_epochs = self.local_epochs
        if self.train_slow and max_local_epochs > 1:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2 + 1)

        for _ in range(max_local_epochs):
            for x, y in trainloader:
                x, y = self._move_batch(x, y)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                features = self.model.base(x)
                local_logits = self.model.head(features)
                loss = self.loss(local_logits, y)
                if self.contrastive_lambda > 0:
                    loss = loss + self.contrastive_lambda * self._supervised_contrastive_loss(features, y)

                self.local_optimizer.zero_grad()
                loss.backward()
                self.local_optimizer.step()

        for _ in range(max_local_epochs):
            for x, y in trainloader:
                x, y = self._move_batch(x, y)
                with torch.no_grad():
                    features = self.model.base(x)
                global_logits = self.global_head(features.detach())
                loss = self.loss(global_logits, y)

                self.global_head_optimizer.zero_grad()
                loss.backward()
                self.global_head_optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.model.cpu()
        self.global_head.cpu()
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def _forward_logits(self, x):
        features = self.model.base(x)
        return self.model.head(features) + self.global_head(features)

    def _move_eval_batch(self, x, y):
        return self._move_batch(x, y)

    def _prepare_eval_model(self):
        self.model.to(self.device)
        self.global_head.to(self.device)
        self.model.eval()
        self.global_head.eval()

    def _cleanup_eval_model(self):
        self.model.cpu()
        self.global_head.cpu()

    def _eval_forward(self, x):
        return torch.nan_to_num(self._forward_logits(x), nan=0.0, posinf=1e6, neginf=-1e6)

    def test_metrics(self):
        testloader = self.load_test_data()
        self.model.to(self.device)
        self.global_head.to(self.device)
        self.model.eval()
        self.global_head.eval()

        test_acc = 0
        test_num = 0
        probabilities = []
        targets = []

        with torch.no_grad():
            for x, y in testloader:
                x, y = self._move_batch(x, y)
                output = torch.nan_to_num(
                    self._forward_logits(x), nan=0.0, posinf=1e6, neginf=-1e6
                )
                test_acc += torch.sum(torch.argmax(output, dim=1) == y).item()
                test_num += y.shape[0]
                probabilities.append(torch.softmax(output, dim=1).cpu().numpy())
                class_count = self.num_classes + (1 if self.num_classes == 2 else 0)
                binary = label_binarize(y.cpu().numpy(), classes=np.arange(class_count))
                targets.append(binary[:, :2] if self.num_classes == 2 else binary)

        self.model.cpu()
        self.global_head.cpu()
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
        self.global_head.to(self.device)
        self.model.eval()
        self.global_head.eval()

        train_num = 0
        losses = 0.0
        with torch.no_grad():
            for x, y in trainloader:
                x, y = self._move_batch(x, y)
                output = self._forward_logits(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        self.model.cpu()
        self.global_head.cpu()
        return losses, train_num
