import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from flcore.clients.clientbase import Client
from flcore.servers.feddst_utils import (
    apply_masks,
    bits_to_mb,
    clone_masks,
    mask_gradients,
    readjust_masks,
    sparse_payload_bits,
)


class clientDST(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.masks = None
        self._last_received_masks = None
        self.feddst_fp16 = bool(getattr(args, "feddst_fp16", False))
        self.feddst_sparsity_distribution = str(getattr(args, "feddst_sparsity_distribution", "erk"))
        self.mu = float(getattr(args, "mu", 0.0))

    def set_sparse_parameters(self, model, masks):
        self.model.load_state_dict(model.state_dict())
        self.masks = clone_masks(masks)
        apply_masks(self.model, self.masks)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self._last_received_masks = clone_masks(masks)

    def sparse_upload_mb(self, include_mask_bits=False):
        return bits_to_mb(sparse_payload_bits(self.model, self.masks, include_mask_bits=include_mask_bits, fp16=self.feddst_fp16))

    def train(self, readjust=False, readjustment_ratio=0.0, target_sparsity=0.8):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        if self.masks is not None:
            apply_masks(self.model, self.masks)
        self.model.train()

        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2 + 1))

        global_params = None
        if self.mu > 0:
            global_params = [param.detach().clone() for param in self.model.parameters()]

        last_loss = None
        for _ in range(max_local_epochs):
            for x, y in tqdm(trainloader, desc=f"Client {self.id} FedDST Training", leave=False):
                if type(x) == type([]):
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)
                if torch.is_floating_point(x) and not torch.isfinite(x).all():
                    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

                output = self.model(x)
                if not torch.isfinite(output).all():
                    continue
                loss = self.loss(output, y)
                if self.mu > 0 and global_params is not None:
                    prox = 0.0
                    for param, global_param in zip(self.model.parameters(), global_params):
                        prox = prox + torch.sum((param - global_param.to(param.device)) ** 2)
                    loss = loss + 0.5 * self.mu * prox
                if not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                if self.masks is not None:
                    mask_gradients(self.model, self.masks)
                clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
                if self.masks is not None:
                    apply_masks(self.model, self.masks)
                last_loss = float(loss.detach().item())

                for param in self.model.parameters():
                    if not torch.isfinite(param.data).all():
                        param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1e4, neginf=-1e4)

        if readjust and self.masks is not None:
            self.masks = readjust_masks(
                self.model,
                self.masks,
                ratio=float(readjustment_ratio),
                target_sparsity=float(target_sparsity),
                distribution=self.feddst_sparsity_distribution,
            )

        self.model.cpu()
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time
        return {"loss": last_loss, "masks": clone_masks(self.masks) if self.masks is not None else None}
