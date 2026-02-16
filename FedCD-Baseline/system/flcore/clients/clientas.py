import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from torch.autograd import grad
from torch.nn.utils import clip_grad_norm_




class clientAS(Client):


    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.fim_trace_history = []

    def train(self, is_selected):
        self.model.to(self.device)
        if is_selected:
            trainloader = self.load_train_data()
            self.model.train()

        
            start_time = time.time()

            max_local_epochs = self.local_epochs
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)

            for step in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
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

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time


            # set model to eval mode
            self.model.eval()
            # print(f'client{self.id}, start cal fim.')
            # Compute FIM and its trace after training
            fim_trace_sum = 0
            for i, (x, y) in enumerate(self.load_train_data()):
                # Forward pass
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                if not torch.isfinite(outputs).all():
                    continue
                # Negative log likelihood as our loss
                nll = -torch.nn.functional.log_softmax(outputs, dim=1)[range(len(y)), y].mean()
                if not torch.isfinite(nll):
                    continue

                # Compute gradient of the negative log likelihood w.r.t. model parameters
                grads = grad(nll, self.model.parameters())

                # Compute and accumulate the trace of the Fisher Information Matrix
                for g in grads:
                    fim_trace_sum += torch.sum(g ** 2).detach()

            # add the fisher log
            self.fim_trace_history.append(fim_trace_sum.item())

        else:
            trainloader = self.load_train_data()
            self.model.eval()
            # Compute FIM and its trace after training
            fim_trace_sum = 0
            for i, (x, y) in enumerate(trainloader):
                # Forward pass
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                if not torch.isfinite(outputs).all():
                    continue
                # Negative log likelihood as our loss
                nll = -torch.nn.functional.log_softmax(outputs, dim=1)[range(len(y)), y].mean()
                if not torch.isfinite(nll):
                    continue

                # Compute gradient of the negative log likelihood w.r.t. model parameters
                grads = grad(nll, self.model.parameters())

                # Compute and accumulate the trace of the Fisher Information Matrix
                for g in grads:
                    fim_trace_sum += torch.sum(g ** 2).detach()

            # add the fisher log
            self.fim_trace_history.append(fim_trace_sum.item())

        self.model.cpu()

    def evaluate(self):
        testloader = self.load_test_data()
        self.model.to(self.device)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        accuracy = 100. * correct / total
        self.model.cpu()
        return accuracy
    
    # def set_parameters(self, model, progress):
        # # Substitute the parameters of the base, enabling personalization
        # for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
        #     old_param.data = new_param.data.clone()

    def set_parameters(self, model, progress):

        # Get class-specific prototypes from the local model
        local_prototypes = [[] for _ in range(self.num_classes)]
        batch_size = 16  # or any other suitable value
        trainloader = self.load_train_data(batch_size=batch_size)

        self.model.to(self.device)
        model.to(self.device)

        # print(f'client{id}')
        for x_batch, y_batch in trainloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                proto_batch = self.model.base(x_batch)

            # Scatter the prototypes based on their labels
            for proto, y in zip(proto_batch, y_batch):
                local_prototypes[y.item()].append(proto)

        mean_prototypes = []

        # print(f'client{self.id}')
        for class_prototypes in local_prototypes:

            if not class_prototypes == []:
                # Stack the tensors for the current class
                stacked_protos = torch.stack(class_prototypes)

                # Compute the mean tensor for the current class
                mean_proto = torch.mean(stacked_protos, dim=0)
                mean_prototypes.append(mean_proto)
            else:
                mean_prototypes.append(None)

        # Align global model's prototype with the local prototype
        alignment_optimizer = torch.optim.SGD(model.base.parameters(), lr=0.01)  # Adjust learning rate and optimizer as needed
        alignment_loss_fn = torch.nn.MSELoss()

        # print(f'client{self.id}')
        for _ in range(1):  # Iterate for 1 epochs; adjust as needed
            for x_batch, y_batch in trainloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                global_proto_batch = model.base(x_batch)
                if not torch.isfinite(global_proto_batch).all():
                    continue
                loss = 0
                total_samples = len(y_batch)
                for label in y_batch.unique():
                    if mean_prototypes[label.item()] is not None:
                        class_samples = (y_batch == label).sum()
                        class_loss = alignment_loss_fn(global_proto_batch[y_batch == label], mean_prototypes[label.item()])
                        loss += class_loss * (class_samples / total_samples)
                alignment_optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.base.parameters(), max_norm=10.0)
                alignment_optimizer.step()

        # Substitute the parameters of the base, enabling personalization
        for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

        self.model.cpu()
        model.cpu()


        # end
