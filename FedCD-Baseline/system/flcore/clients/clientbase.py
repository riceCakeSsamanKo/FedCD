import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from utils.model_state import copy_module_state


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(args.seed + id)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        self.eval_class_filter = None
        self.eval_test_data = None


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        if self.eval_test_data is None:
            test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        else:
            test_data = self.eval_test_data
        test_data = self.filter_eval_data(test_data)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def set_eval_test_data(self, eval_test_data):
        if eval_test_data is None or isinstance(eval_test_data, Dataset):
            self.eval_test_data = eval_test_data
        else:
            self.eval_test_data = list(eval_test_data)
        if self.eval_test_data is not None:
            self.test_samples = len(self.eval_test_data)

    def set_eval_class_filter(self, class_indices=None):
        if class_indices is None:
            self.eval_class_filter = None
        else:
            self.eval_class_filter = {int(class_idx) for class_idx in class_indices}

    @staticmethod
    def _label_to_int(label):
        if torch.is_tensor(label):
            return int(label.detach().cpu().item())
        if hasattr(label, "item"):
            return int(label.item())
        return int(label)

    def filter_eval_data(self, data):
        if self.eval_class_filter is None:
            return data
        return [
            sample
            for sample in data
            if self._label_to_int(sample[1]) in self.eval_class_filter
        ]

    def _train_label_set(self):
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        return {self._label_to_int(label) for _, label in train_data}

    def _move_eval_batch(self, x, y):
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
            if torch.is_floating_point(x[0]) and not torch.isfinite(x[0]).all():
                x[0] = torch.nan_to_num(x[0], nan=0.0, posinf=1.0, neginf=0.0)
        else:
            x = x.to(self.device)
            if torch.is_floating_point(x) and not torch.isfinite(x).all():
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        return x, y.to(self.device)

    def _prepare_eval_model(self):
        self.model.to(self.device)
        self.model.eval()

    def _cleanup_eval_model(self):
        self.model.cpu()

    def _eval_forward(self, x):
        return self.model(x)

    def test_label_split_metrics(self):
        """Return per-client ID/OOD correct counts using train labels as ID labels."""
        train_labels = self._train_label_set()
        testloader = self.load_test_data()
        self._prepare_eval_model()

        id_correct = 0
        id_total = 0
        ood_correct = 0
        ood_total = 0
        invalid_values_found = False

        with torch.no_grad():
            for x, y in testloader:
                x, y = self._move_eval_batch(x, y)
                output = self._eval_forward(x)
                if not torch.isfinite(output).all():
                    output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
                    invalid_values_found = True

                pred = torch.argmax(output, dim=1)
                correct = pred.eq(y)
                y_cpu = y.detach().cpu().tolist()
                id_mask = torch.tensor(
                    [int(label) in train_labels for label in y_cpu],
                    dtype=torch.bool,
                    device=y.device,
                )
                ood_mask = ~id_mask

                id_correct += correct[id_mask].sum().item()
                id_total += id_mask.sum().item()
                ood_correct += correct[ood_mask].sum().item()
                ood_total += ood_mask.sum().item()

        self._cleanup_eval_model()
        if invalid_values_found:
            print(f"Warning: non-finite values detected during ID/OOD eval on client {self.id}; sanitized.")

        return {
            "id_correct": id_correct,
            "id_total": id_total,
            "ood_correct": ood_correct,
            "ood_total": ood_total,
        }
        
    def set_parameters(self, model):
        copy_module_state(model, self.model)

    def clone_model(self, model, target):
        copy_module_state(model, target)

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        invalid_values_found = False
        
        with torch.no_grad():
            for x, y in testloaderfull:
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

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
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
        # self.save_model(self.model, 'model')

        if len(y_prob) == 0 or len(y_true) == 0:
            return test_acc, test_num, 0.0

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        # AUC expects finite scores; convert non-finite values and normalize logits.
        y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1e6, neginf=-1e6)
        if y_prob.ndim == 2 and y_prob.shape[1] > 1:
            y_prob = y_prob - np.max(y_prob, axis=1, keepdims=True)
            y_prob = np.exp(y_prob)
            denom = np.sum(y_prob, axis=1, keepdims=True)
            denom[denom == 0] = 1.0
            y_prob = y_prob / denom

        try:
            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        except ValueError:
            auc = 0.0

        if invalid_values_found:
            print(f"Warning: non-finite values detected during evaluation on client {self.id}; sanitized for AUC.")
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
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
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        self.model.cpu()
        # self.save_model(self.model, 'model')

        if invalid_values_found:
            print(f"Warning: non-finite values detected during train-metric eval on client {self.id}; invalid batches skipped.")

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
