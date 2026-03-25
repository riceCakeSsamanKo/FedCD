import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from flcore.clients.clientbase import Client
from flcore.trainmodel.fedcd2 import extract_gm_state, extract_pm_state


class clientFedCD2(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.optimizer_gm = torch.optim.SGD(
            list(self.model.gm_base.parameters()) + list(self.model.gm_head.parameters()),
            lr=self.learning_rate,
        )
        self.optimizer_pm = torch.optim.SGD(
            list(self.model.pm_base.parameters()) + list(self.model.pm_head.parameters()),
            lr=self.learning_rate,
        )
        self.learning_rate_scheduler_gm = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_gm,
            gamma=args.learning_rate_decay_gamma,
        )
        self.learning_rate_scheduler_pm = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_pm,
            gamma=args.learning_rate_decay_gamma,
        )

        self.fedcd2_alpha = float(getattr(args, "fedcd2_alpha", 0.5))
        self.fedcd2_beta = float(getattr(args, "fedcd2_beta", 0.5))
        self.fedcd2_fnc_weight = float(getattr(args, "fedcd2_fnc_weight", 0.05))
        self.fedcd2_local_pm_weight = float(getattr(args, "fedcd2_local_pm_weight", 0.5))
        self.fedcd2_cluster_pm_weight = float(getattr(args, "fedcd2_cluster_pm_weight", 0.5))
        self.plocal_epochs = max(1, int(getattr(args, "plocal_epochs", 1)))

        self.cluster_id = 0
        self.pending_cluster_pm_state = None
        self.pm_signature = None

    def _move_batch(self, x, y):
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
            if torch.is_floating_point(x[0]) and not torch.isfinite(x[0]).all():
                x[0] = torch.nan_to_num(x[0], nan=0.0, posinf=1.0, neginf=0.0)
        else:
            x = x.to(self.device)
            if torch.is_floating_point(x) and not torch.isfinite(x).all():
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        y = y.to(self.device)
        return x, y

    def _set_requires_grad(self, modules, flag):
        for module in modules:
            for param in module.parameters():
                param.requires_grad = flag

    def _module_state(self, module):
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    def _flatten_state(self, state_dict):
        return torch.cat([v.reshape(-1).float() for v in state_dict.values()], dim=0)

    def set_cluster_id(self, cluster_id):
        self.cluster_id = int(cluster_id)

    def set_gm_state(self, gm_state):
        self.model.gm_base.load_state_dict(gm_state["base"], strict=True)
        self.model.gm_head.load_state_dict(gm_state["head"], strict=True)

    def set_cluster_pm_state(self, pm_state):
        self.pending_cluster_pm_state = pm_state

    def sync_pm_state(self):
        if self.pending_cluster_pm_state is None:
            return
        local_w = float(self.fedcd2_local_pm_weight)
        cluster_w = float(self.fedcd2_cluster_pm_weight)
        total = max(local_w + cluster_w, 1e-8)

        current_base = self.model.pm_base.state_dict()
        mixed_base = {}
        for key, value in current_base.items():
            cluster_value = self.pending_cluster_pm_state["base"][key].to(value.device)
            mixed_base[key] = torch.nan_to_num(
                value * (local_w / total) + cluster_value * (cluster_w / total),
                nan=0.0, posinf=1e4, neginf=-1e4
            )
        self.model.pm_base.load_state_dict(mixed_base, strict=True)

        current_head = self.model.pm_head.state_dict()
        mixed_head = {}
        for key, value in current_head.items():
            cluster_value = self.pending_cluster_pm_state["head"][key].to(value.device)
            mixed_head[key] = torch.nan_to_num(
                value * (local_w / total) + cluster_value * (cluster_w / total),
                nan=0.0, posinf=1e4, neginf=-1e4
            )
        self.model.pm_head.load_state_dict(mixed_head, strict=True)
        self.pending_cluster_pm_state = None

    def get_gm_state(self):
        return extract_gm_state(self.model)

    def get_pm_state(self):
        return extract_pm_state(self.model)

    def get_pm_signature(self):
        if self.pm_signature is None:
            pm_state = self.get_pm_state()
            base_sig = self._flatten_state(pm_state["base"])
            return base_sig.numpy()
        return self.pm_signature

    def train(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()
        start_time = time.time()

        pre_pm_state = self.get_pm_state()

        max_pm_epochs = self.plocal_epochs
        max_gm_epochs = self.local_epochs
        if self.train_slow:
            max_pm_epochs = np.random.randint(1, max_pm_epochs + 1)
            max_gm_epochs = np.random.randint(1, max_gm_epochs + 1)

        # Stage 1: PM update, following FedRep/Ditto-style staged personalization.
        self._set_requires_grad([self.model.gm_base, self.model.gm_head], False)
        self._set_requires_grad([self.model.pm_base, self.model.pm_head, self.model.pm_proj, self.model.gm_proj], True)
        for _ in range(max_pm_epochs):
            for x, y in trainloader:
                x, y = self._move_batch(x, y)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                outputs = self.model.forward_all(x)
                if not torch.isfinite(outputs["fused_logits"]).all():
                    continue
                loss_fuse = self.loss(outputs["fused_logits"], y)
                loss_pm = self.loss(outputs["pm_logits"], y)
                loss_fnc = ((outputs["gm_proj"] * outputs["pm_proj"]).sum(dim=1) ** 2).mean()
                loss = loss_fuse + self.fedcd2_beta * loss_pm + self.fedcd2_fnc_weight * loss_fnc
                if not torch.isfinite(loss):
                    continue

                self.optimizer_pm.zero_grad()
                loss.backward()
                clip_grad_norm_(
                    list(self.model.pm_base.parameters()) + list(self.model.pm_head.parameters()) +
                    list(self.model.pm_proj.parameters()) + list(self.model.gm_proj.parameters()),
                    max_norm=10.0,
                )
                self.optimizer_pm.step()

        # Stage 2: GM update, following FedRep-style shared backbone optimization.
        self._set_requires_grad([self.model.gm_base, self.model.gm_head], True)
        self._set_requires_grad([self.model.pm_base, self.model.pm_head], False)
        for _ in range(max_gm_epochs):
            for x, y in trainloader:
                x, y = self._move_batch(x, y)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                outputs = self.model.forward_all(x)
                if not torch.isfinite(outputs["gm_logits"]).all():
                    continue
                loss = self.loss(outputs["gm_logits"], y)
                if not torch.isfinite(loss):
                    continue

                self.optimizer_gm.zero_grad()
                loss.backward()
                clip_grad_norm_(
                    list(self.model.gm_base.parameters()) + list(self.model.gm_head.parameters()),
                    max_norm=10.0,
                )
                self.optimizer_gm.step()

        self._set_requires_grad([self.model.pm_base, self.model.pm_head], True)
        self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler_gm.step()
            self.learning_rate_scheduler_pm.step()

        post_pm_state = self.get_pm_state()
        delta_vec = []
        for section in ["base", "head"]:
            for key in pre_pm_state[section].keys():
                delta = post_pm_state[section][key] - pre_pm_state[section][key]
                delta_vec.append(delta.reshape(-1).float())
        delta_vec = torch.cat(delta_vec, dim=0)
        norm = torch.norm(delta_vec, p=2).clamp_min(1e-12)
        self.pm_signature = (delta_vec / norm).cpu().numpy()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def _eval_metrics_on_loader(self, loader):
        self.model.to(self.device)
        self.model.eval()
        fused_correct = 0
        gm_correct = 0
        pm_correct = 0
        total = 0
        loss_sum = 0.0
        y_true = []
        y_prob = []
        local_agree_pm = 0
        local_agree_gm = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = self._move_batch(x, y)
                outputs = self.model.forward_all(x)
                fused_logits = torch.nan_to_num(outputs["fused_logits"], nan=0.0, posinf=1e4, neginf=-1e4)
                gm_logits = torch.nan_to_num(outputs["gm_logits"], nan=0.0, posinf=1e4, neginf=-1e4)
                pm_logits = torch.nan_to_num(outputs["pm_logits"], nan=0.0, posinf=1e4, neginf=-1e4)
                batch_loss = self.loss(fused_logits, y)

                fused_pred = fused_logits.argmax(dim=1)
                gm_pred = gm_logits.argmax(dim=1)
                pm_pred = pm_logits.argmax(dim=1)
                fused_correct += torch.sum(fused_pred == y).item()
                gm_correct += torch.sum(gm_pred == y).item()
                pm_correct += torch.sum(pm_pred == y).item()
                local_agree_pm += torch.sum(fused_pred == pm_pred).item()
                local_agree_gm += torch.sum(fused_pred == gm_pred).item()
                total += y.shape[0]
                loss_sum += batch_loss.item() * y.shape[0]

                prob = F.softmax(fused_logits, dim=1).detach().cpu().numpy()
                y_prob.append(prob)
                nc = self.num_classes + 1 if self.num_classes == 2 else self.num_classes
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        self.model.cpu()
        auc = 0.5
        if total > 0:
            try:
                auc = metrics.roc_auc_score(np.concatenate(y_true, axis=0), np.concatenate(y_prob, axis=0), average="micro")
            except ValueError:
                auc = 0.5

        return {
            "fused_correct": fused_correct,
            "gm_correct": gm_correct,
            "pm_correct": pm_correct,
            "num_samples": total,
            "loss": loss_sum,
            "auc_weighted": auc * total,
            "agree_pm": local_agree_pm,
            "agree_gm": local_agree_gm,
        }

    def local_eval_metrics(self):
        return self._eval_metrics_on_loader(self.load_test_data())
