import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from flcore.clients.clientbase import Client
from torch.nn.utils import clip_grad_norm_


class clientProx(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu

        self.global_params = copy.deepcopy(list(self.model.parameters()))

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = PerturbedGradientDescent(
            self.model.parameters(), lr=self.learning_rate, mu=self.mu)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.to(self.device)
        self.global_params = [p.to(self.device) for p in self.global_params]
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for x, y in trainloader:
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
                self.optimizer.step(self.global_params, self.device)
                for p in self.model.parameters():
                    if not torch.isfinite(p.data).all():
                        p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1e4, neginf=-1e4)

        self.model.cpu()
        self.global_params = [p.cpu() for p in self.global_params]

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.global_params = [p.to(self.device) for p in self.global_params]
        self.model.eval()

        train_num = 0
        losses = 0
        invalid_values_found = False
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                    if torch.is_floating_point(x[0]) and not torch.isfinite(x[0]).all():
                        x[0] = torch.nan_to_num(x[0], nan=0.0, posinf=1.0, neginf=0.0)
                        invalid_values_found = True
                else:
                    x = x.to(self.device)
                    if torch.is_floating_point(x) and not torch.isfinite(x).all():
                        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
                        invalid_values_found = True
                y = y.to(self.device)
                output = self.model(x)
                if not torch.isfinite(output).all():
                    output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
                    invalid_values_found = True
                loss = self.loss(output, y)
                if not torch.isfinite(loss):
                    invalid_values_found = True
                    continue

                gm = torch.cat([p.data.view(-1) for p in self.global_params], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                loss += 0.5 * self.mu * torch.norm(gm-pm, p=2)
                if not torch.isfinite(loss):
                    invalid_values_found = True
                    continue

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        self.model.cpu()
        self.global_params = [p.cpu() for p in self.global_params]
        # self.save_model(self.model, 'model')

        if invalid_values_found:
            print(f"Warning: non-finite values detected during train-metric eval on client {self.id}; invalid batches skipped.")

        return losses, train_num
