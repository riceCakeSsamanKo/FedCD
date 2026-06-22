import numpy as np
import time
import torch.nn as nn
from flcore.clients.clientbase import Client


_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def _bn_parameter_names(model):
    names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, _BN_TYPES):
            for param_name, _ in module.named_parameters(recurse=False):
                names.add(f"{module_name}.{param_name}" if module_name else param_name)
    return names


class clientBN(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model):
        bn_names = _bn_parameter_names(model)
        own_params = dict(self.model.named_parameters())
        for name, param in model.named_parameters():
            if name not in bn_names:
                own_params[name].data = param.data.clone()
