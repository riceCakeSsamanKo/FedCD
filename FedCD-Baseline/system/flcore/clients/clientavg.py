import copy
import torch
import numpy as np
import time
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from flcore.clients.clientbase import Client


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(tqdm(trainloader, desc=f"Client {self.id} Training", leave=False)):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                    if torch.is_floating_point(x[0]) and not torch.isfinite(x[0]).all():
                        x[0] = torch.nan_to_num(x[0], nan=0.0, posinf=1.0, neginf=0.0)
                else:
                    x = x.to(self.device)
                    if torch.is_floating_point(x) and not torch.isfinite(x).all():
                        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                if not torch.isfinite(output).all():
                    continue
                loss = self.loss(output, y)
                if not torch.isfinite(loss):
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                grad_is_finite = True
                for p in self.model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grad_is_finite = False
                        break
                if not grad_is_finite:
                    self.optimizer.zero_grad()
                    continue
                self.optimizer.step()
                for p in self.model.parameters():
                    if not torch.isfinite(p.data).all():
                        p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1e4, neginf=-1e4)

        self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
