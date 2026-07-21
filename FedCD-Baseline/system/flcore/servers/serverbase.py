import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import re
from tqdm import tqdm
from utils.data_utils import (
    FEDPRISM_SCENARIOS,
    has_reserved_data,
    is_fedprism_scenario_dataset,
    read_client_data,
    read_fedprism_scenario_data,
)
from utils.dlg import DLG
from utils.dynamic_clients import DynamicClientExperimentMixin
from utils.model_state import average_module_states


class Server(DynamicClientExperimentMixin, object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break
        self._init_dynamic_client_experiment(args)

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

        self.model_size_MB = sum(p.numel() for p in self.global_model.parameters()) * 4 / (1024 * 1024)
        self.uplink_MB = 0
        self.downlink_MB = 0
        self.fedprism_eval_match = bool(getattr(args, 'fedprism_eval_match', True))
        self.fedprism_eval_reserved_fraction = float(
            getattr(args, 'fedprism_eval_reserved_fraction', 0.2)
        )
        self.fedprism_eval_reserved_seed = int(
            getattr(args, 'fedprism_eval_reserved_seed', 0)
        )
        self._fedprism_eval_positions = None
        self._fedprism_eval_data_cache = {}
        # Track last logged cumulative values so CSV records per-eval communication
        # (FedCD-style) instead of lifetime cumulative totals.
        self._last_logged_uplink_MB = 0.0
        self._last_logged_downlink_MB = 0.0
        self.eval_common_global = bool(getattr(args, "eval_common_global", True))
        self.global_test_samples = int(
            getattr(args, "global_test_samples", getattr(args, "common_test_samples", 0))
        )
        self.common_eval_batch_size = int(getattr(args, "common_eval_batch_size", 256))
        self.global_test_loader = self._build_global_test_loader() if self.eval_common_global else None
        # Backward-compatible alias
        self.common_test_loader = self.global_test_loader
        self.rs_global_test_acc = []
        # Backward-compatible alias
        self.rs_common_test_acc = self.rs_global_test_acc
        self.rs_id_test_acc = []
        self.rs_ood_test_acc = []
        self.rs_id_client_std = []
        self.rs_ood_client_std = []
        self.rs_id_test_count = []
        self.rs_ood_test_count = []
        self.eval_rhos = self._parse_eval_rhos(getattr(args, "eval_rhos", ""))
        self.multi_rho_eval = bool(self.eval_rhos) and self._is_splitgp_rho_dataset(self.dataset)
        self.eval_rho_items = self._build_eval_rho_items(self.eval_rhos) if self.multi_rho_eval else []
        self.eval_rho_log_paths = {}
        self._last_eval_round_comm = (0.0, 0.0)
        if self.multi_rho_eval:
            exp_dir = getattr(args, "exp_dir", None)
            if exp_dir:
                for item in self.eval_rho_items:
                    log_dir = os.path.join(exp_dir, item["label"])
                    os.makedirs(log_dir, exist_ok=True)
                    self.eval_rho_log_paths[item["label"]] = os.path.join(log_dir, "acc.csv")
            print(
                "[Multi-Rho Eval] One training run will evaluate test rho(s): "
                + ", ".join(f"{item['rho']:.1f}" for item in self.eval_rho_items)
            )

        self.eval_scenarios = self._parse_eval_scenarios(
            getattr(args, 'eval_scenarios', '')
        )
        self.multi_scenario_eval = (
            bool(self.eval_scenarios)
            and is_fedprism_scenario_dataset(self.dataset)
        )
        self.eval_scenario_log_paths = {}
        if self.eval_scenarios and not self.multi_scenario_eval:
            raise ValueError(
                f'--eval-scenarios requires a FedPRISM ID/OOD/Mix dataset: {self.dataset}'
            )
        if self.multi_scenario_eval:
            exp_dir = getattr(args, 'exp_dir', None)
            if exp_dir:
                for scenario in self.eval_scenarios:
                    label = f'eval_{scenario}'
                    log_dir = os.path.join(exp_dir, label)
                    os.makedirs(log_dir, exist_ok=True)
                    self.eval_scenario_log_paths[scenario] = os.path.join(
                        log_dir, 'acc.csv'
                    )
            print(
                '[ID/OOD/Mix Eval] One training run will evaluate: '
                + ', '.join(self.eval_scenarios)
            )

    @staticmethod
    def _parse_eval_scenarios(value):
        if value is None:
            return []
        scenarios = []
        for token in re.split(r'[\s,]+', str(value).strip().lower()):
            if not token or token in scenarios:
                continue
            if token not in FEDPRISM_SCENARIOS:
                raise ValueError(
                    f'Unknown evaluation scenario {token!r}; '
                    f'choose from {FEDPRISM_SCENARIOS}'
                )
            scenarios.append(token)
        return scenarios

    @staticmethod
    def _parse_eval_rhos(value):
        if value is None:
            return []
        tokens = re.split(r"[\s,]+", str(value).strip())
        rhos = []
        seen = set()
        for token in tokens:
            if not token:
                continue
            rho = float(token)
            if not np.isfinite(rho):
                continue
            key = round(rho, 12)
            if key in seen:
                continue
            seen.add(key)
            rhos.append(rho)
        return rhos

    @staticmethod
    def _is_splitgp_rho_dataset(dataset):
        return bool(re.search(r"rho[0-9]+(?:\.[0-9]+)?", str(dataset), flags=re.IGNORECASE))

    @staticmethod
    def _format_eval_rho_label(rho):
        return f"eval_rho_{float(rho):.1f}"

    def _dataset_for_eval_rho(self, rho):
        return re.sub(
            r"rho[0-9]+(?:\.[0-9]+)?",
            f"rho{float(rho):.1f}",
            str(self.dataset),
            count=1,
            flags=re.IGNORECASE,
        )

    def _build_eval_rho_items(self, rhos):
        return [
            {
                "rho": float(rho),
                "label": self._format_eval_rho_label(rho),
                "dataset": self._dataset_for_eval_rho(rho),
            }
            for rho in rhos
        ]

    def _build_fedprism_eval_positions(self):
        positions = []
        for client_id in range(self.num_clients):
            client_data = read_client_data(
                self.dataset,
                client_id,
                is_train=False,
                few_shot=self.few_shot,
            )
            positions.extend((client_id, local_idx) for local_idx in range(len(client_data)))

        if not positions:
            self._fedprism_eval_positions = []
            return

        eval_positions = list(positions)
        reserved_count = 0
        if not has_reserved_data(self.dataset):
            fraction = min(max(self.fedprism_eval_reserved_fraction, 0.0), 0.95)
            shuffled = list(range(len(positions)))
            rng = random.Random(self.fedprism_eval_reserved_seed)
            rng.shuffle(shuffled)
            reserved_count = int(round(len(shuffled) * fraction))
            if fraction > 0.0 and reserved_count <= 0:
                reserved_count = 1
            if reserved_count >= len(shuffled):
                reserved_count = max(0, len(shuffled) - 1)
            reserved_indices = set(shuffled[:reserved_count])
            eval_positions = [position for idx, position in enumerate(positions) if idx not in reserved_indices]

        if 0 < self.global_test_samples < len(eval_positions):
            rng = random.Random(self.fedprism_eval_reserved_seed + 1)
            eval_positions = rng.sample(eval_positions, self.global_test_samples)

        self._fedprism_eval_positions = eval_positions
        print(
            f'[FedPRISM Eval Match] total={len(positions)}, '
            f'excluded={reserved_count}, evaluated={len(eval_positions)}'
        )

    def _fedprism_eval_data_by_client(self, dataset):
        dataset = str(dataset)
        if dataset in self._fedprism_eval_data_cache:
            return self._fedprism_eval_data_cache[dataset]
        if self._fedprism_eval_positions is None:
            self._build_fedprism_eval_positions()

        by_client = {client_id: [] for client_id in range(self.num_clients)}
        client_cache = {}
        for client_id, local_idx in self._fedprism_eval_positions:
            if client_id not in client_cache:
                client_cache[client_id] = read_client_data(
                    dataset,
                    client_id,
                    is_train=False,
                    few_shot=self.few_shot,
                )
            client_data = client_cache[client_id]
            if 0 <= local_idx < len(client_data):
                by_client[client_id].append(client_data[local_idx])
        self._fedprism_eval_data_cache[dataset] = by_client
        return by_client

    def _assign_fedprism_eval_data(self, dataset):
        if not self.fedprism_eval_match or not self._is_splitgp_rho_dataset(dataset):
            return
        eval_by_client = self._fedprism_eval_data_by_client(dataset)
        for client in self.clients:
            client.set_eval_test_data(eval_by_client.get(int(client.id), []))

    def _assign_fedprism_scenario_data(self, scenario):
        if not is_fedprism_scenario_dataset(self.dataset):
            raise ValueError(f'Not a FedPRISM ID/OOD/Mix dataset: {self.dataset}')
        for client in self.clients:
            client.set_eval_test_data(
                read_fedprism_scenario_data(self.dataset, int(client.id), scenario)
            )

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)
        self._ensure_dynamic_client_groups()
        self._assign_fedprism_eval_data(self.dataset)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        self._advance_dynamic_client_round()
        candidate_clients = self._dynamic_client_active_clients()
        candidate_count = len(candidate_clients)
        if candidate_count <= 0:
            raise RuntimeError("No active clients are available for this round.")
        base_join_count = max(1, int(candidate_count * self.join_ratio))
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(base_join_count, candidate_count + 1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = base_join_count
        selected_clients = list(
            np.random.choice(candidate_clients, self.current_num_join_clients, replace=False)
        )

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        clients = (
            self.selected_clients
            if len(self.selected_clients) > 0
            else self._dynamic_client_active_clients()
        )
        for client in tqdm(clients, desc="Distributing models", leave=False):
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
        
        self.downlink_MB += len(clients) * self.model_size_MB

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in tqdm(active_clients, desc="Collecting models", leave=False):
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        self.uplink_MB += len(self.uploaded_models) * self.model_size_MB

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = average_module_states(
            self.uploaded_models,
            self.uploaded_weights,
        )

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_local_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                if len(self.rs_global_test_acc) > 0:
                    hf.create_dataset('rs_global_test_acc', data=self.rs_global_test_acc)
                    hf.create_dataset('rs_common_test_acc', data=self.rs_global_test_acc)
                if len(self.rs_id_test_acc) > 0:
                    hf.create_dataset('rs_id_test_acc', data=self.rs_id_test_acc)
                    hf.create_dataset('rs_ood_test_acc', data=self.rs_ood_test_acc)
                    hf.create_dataset('rs_id_client_std', data=self.rs_id_client_std)
                    hf.create_dataset('rs_ood_client_std', data=self.rs_ood_client_std)
                    hf.create_dataset('rs_id_test_count', data=self.rs_id_test_count)
                    hf.create_dataset('rs_ood_test_count', data=self.rs_ood_test_count)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        eval_clients = self._evaluation_clients()
        for c in tqdm(eval_clients, desc="Testing clients", leave=False):
            ct, ns, auc = c.test_metrics()
            if not bool(getattr(self, "eval_common_global", True)) and ns > 0:
                tot_correct.append(ct * 1.0 / ns)
                tot_auc.append(auc)
                num_samples.append(1)
            else:
                tot_correct.append(ct*1.0)
                tot_auc.append(auc*ns)
                num_samples.append(ns)

        ids = [c.id for c in eval_clients]

        return ids, num_samples, tot_correct, tot_auc

    @staticmethod
    def _metric_or_nan(value):
        if value is None:
            return float("nan")
        return float(value)

    @staticmethod
    def _format_optional_metric(value):
        if value is None:
            return ""
        try:
            if np.isnan(value):
                return ""
        except TypeError:
            pass
        return f"{float(value):.4f}"

    @staticmethod
    def _format_optional_count(value):
        if value is None:
            return ""
        return str(int(value))

    def evaluate_label_split_metrics(self, metric_method="test_label_split_metrics"):
        if not self._is_splitgp_rho_dataset(self.dataset):
            return None

        id_accs = []
        ood_accs = []
        id_total = 0
        ood_total = 0
        for client in tqdm(self._evaluation_clients(), desc='Testing ID/OOD clients', leave=False):
            metrics_fn = getattr(client, metric_method, None)
            if metrics_fn is None:
                continue
            metrics = metrics_fn()
            client_id_total = int(metrics.get("id_total", 0))
            client_ood_total = int(metrics.get("ood_total", 0))
            id_total += client_id_total
            ood_total += client_ood_total
            if client_id_total > 0:
                id_accs.append(float(metrics.get("id_correct", 0)) / client_id_total)
            if client_ood_total > 0:
                ood_accs.append(float(metrics.get("ood_correct", 0)) / client_ood_total)

        return {
            "id_test_acc": float(np.mean(id_accs)) if id_accs else None,
            "ood_test_acc": float(np.mean(ood_accs)) if ood_accs else None,
            "id_client_std": float(np.std(id_accs)) if id_accs else None,
            "ood_client_std": float(np.std(ood_accs)) if ood_accs else None,
            "id_test_count": id_total,
            "ood_test_count": ood_total,
        }

    def _build_global_test_loader(self):
        shared_test_data = []
        for client_id in range(self.num_clients):
            shared_test_data.extend(
                read_client_data(self.dataset, client_id, is_train=False, few_shot=self.few_shot)
            )

        if len(shared_test_data) == 0:
            print("[Baseline] Global test set is empty. Skipping shared evaluation.")
            return None

        if 0 < self.global_test_samples < len(shared_test_data):
            rng = random.Random(0)
            sampled_idx = rng.sample(range(len(shared_test_data)), self.global_test_samples)
            shared_test_data = [shared_test_data[idx] for idx in sampled_idx]

        print(f"[Baseline] Global Test Set Size: {len(shared_test_data)}")
        return torch.utils.data.DataLoader(
            shared_test_data,
            batch_size=self.common_eval_batch_size,
            shuffle=False,
            drop_last=False,
        )

    # Backward-compatible alias
    def _build_common_test_loader(self):
        return self._build_global_test_loader()

    def evaluate_global_test_acc(self):
        if not self.eval_common_global or self.global_test_loader is None:
            return None

        acc_sum = 0.0
        valid_clients = 0

        for client in self._evaluation_clients():
            client.model.to(self.device)
            client.model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in self.global_test_loader:
                    if type(x) == type([]):
                        x = x[0]
                    x = x.to(self.device)
                    y = y.to(self.device)
                    output = client.model(x)
                    correct += (torch.argmax(output, dim=1) == y).sum().item()
                    total += y.size(0)

            client.model.to("cpu")
            if total > 0:
                acc_sum += correct / total
                valid_clients += 1

        if self.device == "cuda":
            torch.cuda.empty_cache()

        if valid_clients == 0:
            return None
        return acc_sum / valid_clients

    # Backward-compatible alias
    def evaluate_common_test_acc(self):
        return self.evaluate_global_test_acc()

    def avg_generalization_metrics(self):
        # Compatibility helper for algorithms (e.g., FedAS) that expect this API.
        return self.evaluate_global_test_acc()

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        eval_clients = self._evaluation_clients()
        for c in tqdm(eval_clients, desc="Calculating train metrics", leave=False):
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in eval_clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        total_test_samples = sum(stats[1])
        total_train_samples = sum(stats_train[1])

        if total_test_samples > 0:
            local_test_acc = sum(stats[2]) * 1.0 / total_test_samples
            test_auc = sum(stats[3]) * 1.0 / total_test_samples
        else:
            local_test_acc = 0.0
            test_auc = 0.0

        if total_train_samples > 0:
            train_loss = sum(stats_train[2]) * 1.0 / total_train_samples
        else:
            train_loss = 0.0
        global_test_acc = self.evaluate_global_test_acc()
        split_metrics = self.evaluate_label_split_metrics()

        accs = [a / n for a, n in zip(stats[2], stats[1]) if n > 0]
        aucs = [a / n for a, n in zip(stats[3], stats[1]) if n > 0]
        std_acc = float(np.std(accs)) if len(accs) > 0 else 0.0
        std_auc = float(np.std(aucs)) if len(aucs) > 0 else 0.0
        
        if acc == None:
            self.rs_test_acc.append(local_test_acc)
            if split_metrics is not None:
                self.rs_id_test_acc.append(self._metric_or_nan(split_metrics.get("id_test_acc")))
                self.rs_ood_test_acc.append(self._metric_or_nan(split_metrics.get("ood_test_acc")))
                self.rs_id_client_std.append(self._metric_or_nan(split_metrics.get("id_client_std")))
                self.rs_ood_client_std.append(self._metric_or_nan(split_metrics.get("ood_client_std")))
                self.rs_id_test_count.append(int(split_metrics.get("id_test_count", 0)))
                self.rs_ood_test_count.append(int(split_metrics.get("ood_test_count", 0)))
        else:
            acc.append(local_test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Local Test Accuracy: {:.4f}".format(local_test_acc))
        if global_test_acc is not None:
            print("Averaged Global Test Accuracy: {:.4f}".format(global_test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(std_acc))
        print("Std Test AUC: {:.4f}".format(std_auc))
        if split_metrics is not None:
            id_text = self._format_optional_metric(split_metrics.get("id_test_acc")) or "N/A"
            ood_text = self._format_optional_metric(split_metrics.get("ood_test_acc")) or "N/A"
            print(f"Averaged ID Test Accuracy: {id_text}")
            print(f"Averaged OOD Test Accuracy: {ood_text}")
            print(
                "ID/OOD Test Counts: "
                f"{split_metrics.get('id_test_count', 0)}/"
                f"{split_metrics.get('ood_test_count', 0)}"
            )

        if acc == None and global_test_acc is not None:
            self.rs_global_test_acc.append(global_test_acc)

        self.log_usage(local_test_acc, train_loss, global_test_acc, split_metrics=split_metrics)
        if acc is None:
            self.log_multi_scenario_eval(train_loss=train_loss)
            self.log_multi_rho_eval(train_loss=train_loss)

    def log_usage(self, local_test_acc, train_loss, global_test_acc=None, split_metrics=None):
        file_path = getattr(self.args, "log_path", "usage.csv")
        round_num = len(self.rs_test_acc)
        round_uplink = max(0.0, self.uplink_MB - self._last_logged_uplink_MB)
        round_downlink = max(0.0, self.downlink_MB - self._last_logged_downlink_MB)
        self._last_logged_uplink_MB = self.uplink_MB
        self._last_logged_downlink_MB = self.downlink_MB
        self._last_eval_round_comm = (round_uplink, round_downlink)
        self._write_usage_row(
            file_path,
            round_num,
            local_test_acc,
            train_loss,
            global_test_acc,
            round_uplink,
            round_downlink,
            split_metrics=split_metrics,
        )

    def _write_usage_row(
        self,
        file_path,
        round_num,
        local_test_acc,
        train_loss,
        global_test_acc,
        round_uplink,
        round_downlink,
        split_metrics=None,
    ):
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(
                    "round,local_test_acc,global_test_acc,train_loss,"
                    "id_test_acc,ood_test_acc,id_test_count,ood_test_count,"
                    "id_client_std,ood_client_std,uplink_mb,downlink_mb,total_mb\n"
                )

        total_mb = round_uplink + round_downlink
        global_str = f"{global_test_acc:.4f}" if global_test_acc is not None else ""
        id_acc = self._format_optional_metric(split_metrics.get("id_test_acc") if split_metrics else None)
        ood_acc = self._format_optional_metric(split_metrics.get("ood_test_acc") if split_metrics else None)
        id_count = self._format_optional_count(split_metrics.get("id_test_count") if split_metrics else None)
        ood_count = self._format_optional_count(split_metrics.get("ood_test_count") if split_metrics else None)
        id_std = self._format_optional_metric(split_metrics.get("id_client_std") if split_metrics else None)
        ood_std = self._format_optional_metric(split_metrics.get("ood_client_std") if split_metrics else None)
        with open(file_path, "a") as f:
            f.write(
                f"{round_num},{local_test_acc:.4f},{global_str},{train_loss:.4f},"
                f"{id_acc},{ood_acc},{id_count},{ood_count},{id_std},{ood_std},"
                f"{round_uplink:.2f},{round_downlink:.2f},{total_mb:.2f}\n"
            )
        self._maybe_log_dynamic_client_metrics()

    def _evaluate_local_acc_on_scenario(self, scenario):
        original_eval_data = [client.eval_test_data for client in self.clients]
        original_test_samples = [client.test_samples for client in self.clients]
        try:
            self._assign_fedprism_scenario_data(scenario)
            stats = self.test_metrics()
            total_test_samples = sum(stats[1])
            if total_test_samples <= 0:
                return 0.0
            return sum(stats[2]) * 1.0 / total_test_samples
        finally:
            for client, eval_data, test_samples in zip(
                self.clients,
                original_eval_data,
                original_test_samples,
            ):
                client.eval_test_data = eval_data
                client.test_samples = test_samples

    def _evaluate_local_metrics_on_dataset(self, dataset):
        original_dataset = self.dataset
        original_client_datasets = [client.dataset for client in self.clients]
        original_eval_data = [client.eval_test_data for client in self.clients]
        original_test_samples = [client.test_samples for client in self.clients]
        try:
            self.dataset = dataset
            for client in self.clients:
                client.dataset = dataset
            self._assign_fedprism_eval_data(dataset)
            stats = self.test_metrics()
            total_test_samples = sum(stats[1])
            if total_test_samples <= 0:
                local_acc = 0.0
            else:
                local_acc = sum(stats[2]) * 1.0 / total_test_samples
            split_metrics = self.evaluate_label_split_metrics()
            return local_acc, split_metrics
        finally:
            self.dataset = original_dataset
            for client, client_dataset, eval_data, test_samples in zip(
                self.clients,
                original_client_datasets,
                original_eval_data,
                original_test_samples,
            ):
                client.dataset = client_dataset
                client.eval_test_data = eval_data
                client.test_samples = test_samples

    def _evaluate_local_acc_on_dataset(self, dataset):
        local_acc, _ = self._evaluate_local_metrics_on_dataset(dataset)
        return local_acc

    def log_multi_scenario_eval(self, train_loss=0.0):
        if not self.multi_scenario_eval:
            return
        round_num = len(self.rs_test_acc)
        round_uplink, round_downlink = self._last_eval_round_comm
        for scenario in self.eval_scenarios:
            acc = self._evaluate_local_acc_on_scenario(scenario)
            print(
                f'[ID/OOD/Mix Eval] scenario={scenario} '
                f'Local Test Accuracy: {acc:.4f}'
            )
            file_path = self.eval_scenario_log_paths.get(scenario)
            if file_path:
                self._write_usage_row(
                    file_path,
                    round_num,
                    acc,
                    train_loss,
                    None,
                    round_uplink,
                    round_downlink,
                )

    def log_multi_rho_eval(self, train_loss=0.0):
        if not self.multi_rho_eval:
            return
        round_num = len(self.rs_test_acc)
        round_uplink, round_downlink = self._last_eval_round_comm
        for item in self.eval_rho_items:
            acc, split_metrics = self._evaluate_local_metrics_on_dataset(item["dataset"])
            print(f"[Multi-Rho Eval] rho={item['rho']:.1f} Local Test Accuracy: {acc:.4f}")
            if split_metrics is not None:
                id_text = self._format_optional_metric(split_metrics.get("id_test_acc")) or "N/A"
                ood_text = self._format_optional_metric(split_metrics.get("ood_test_acc")) or "N/A"
                print(f"[Multi-Rho Eval] rho={item['rho']:.1f} ID/OOD Accuracy: {id_text}/{ood_text}")
            file_path = self.eval_rho_log_paths.get(item["label"])
            if file_path:
                self._write_usage_row(
                    file_path,
                    round_num,
                    acc,
                    train_loss,
                    None,
                    round_uplink,
                    round_downlink,
                    split_metrics=split_metrics,
                )

    def print_(self, local_test_acc, test_auc, train_loss):
        print("Average Local Test Accuracy: {:.4f}".format(local_test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            if not bool(getattr(self, "eval_common_global", True)) and ns > 0:
                tot_correct.append(ct * 1.0 / ns)
                tot_auc.append(auc)
                num_samples.append(1)
            else:
                tot_correct.append(ct*1.0)
                tot_auc.append(auc*ns)
                num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
