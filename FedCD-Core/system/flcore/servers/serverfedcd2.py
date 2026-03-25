import copy
import os
import random
import time
from collections import defaultdict

import h5py
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import ConcatDataset, DataLoader

from flcore.clients.clientfedcd2 import clientFedCD2
from flcore.servers.serverbase import Server
from flcore.trainmodel.fedcd2 import (
    average_nested_states,
    build_fedcd2_model,
    extract_gm_state,
    extract_pm_state,
    slice_pm_from_gm,
)
from utils.data_utils import read_client_data


class FedCD2(Server):
    def __init__(self, args, times):
        args = copy.copy(args)
        args.model = build_fedcd2_model(
            args.model,
            args.dataset,
            args.num_classes,
            pm_feature_dim=int(getattr(args, "fedcd2_pm_feature_dim", 128)),
            fnc_dim=int(getattr(args, "fedcd2_fnc_dim", 128)),
            pm_vgg_width_ratio=float(getattr(args, "fedcd2_pm_vgg_width_ratio", 0.25)),
        ).to(args.device)

        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientFedCD2)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating FedCD2 server and clients.")

        self.num_clusters = max(1, int(getattr(args, "num_clusters", 5)))
        self.cluster_period = max(1, int(getattr(args, "cluster_period", 5)))
        self.gm_period = max(1, int(getattr(args, "global_period", 1)))
        self.cluster_warmup_rounds = max(0, int(getattr(args, "fedcd2_cluster_warmup_rounds", 5)))
        self.cluster_assignments = {c.id: (c.id % self.num_clusters) for c in self.clients}
        self.global_gm_state = extract_gm_state(self.global_model)
        default_pm_state = extract_pm_state(self.global_model)
        self.cluster_pm_states = {cluster_id: copy.deepcopy(default_pm_state) for cluster_id in range(self.num_clusters)}
        self.global_test_loader = self._build_global_test_loader()

        self.uploaded_client_payloads = []
        self.last_round_comm = {
            "total_uplink_bytes": 0.0,
            "total_broadcast_bytes": 0.0,
            "gm_broadcast_bytes_per_client": 0.0,
            "cluster_pm_broadcast_bytes_per_client": 0.0,
        }
        self.Budget = []
        self.rs_global_test_acc = []

    def _build_global_test_loader(self):
        datasets = []
        for cid in range(self.num_clients):
            data = read_client_data(self.dataset, cid, is_train=False, few_shot=self.few_shot)
            if len(data) > 0:
                datasets.append(data)
        if not datasets:
            return None
        return DataLoader(ConcatDataset(datasets), batch_size=256, shuffle=False, drop_last=False)

    @staticmethod
    def _state_bytes(nested_state):
        total = 0
        for section in nested_state.values():
            for tensor in section.values():
                total += tensor.numel() * tensor.element_size()
        return float(total)

    def send_models(self):
        recipients = self.selected_clients if len(self.selected_clients) > 0 else self.clients
        gm_bytes = self._state_bytes(self.global_gm_state)
        cluster_pm_ref = next(iter(self.cluster_pm_states.values()))
        cluster_pm_bytes = self._state_bytes(cluster_pm_ref)
        total_broadcast = 0.0

        for client in recipients:
            start_time = time.time()
            cluster_id = self.cluster_assignments[client.id]
            client.set_cluster_id(cluster_id)
            client.set_gm_state(self.global_gm_state)
            client.set_cluster_pm_state(self.cluster_pm_states[cluster_id])
            client.sync_pm_state()
            total_broadcast += gm_bytes + cluster_pm_bytes
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

        self.last_round_comm["gm_broadcast_bytes_per_client"] = gm_bytes
        self.last_round_comm["cluster_pm_broadcast_bytes_per_client"] = cluster_pm_bytes
        self.last_round_comm["total_broadcast_bytes"] = total_broadcast

    def receive_models(self):
        assert len(self.selected_clients) > 0
        active_count = max(1, int((1 - self.client_drop_rate) * self.current_num_join_clients))
        active_clients = random.sample(self.selected_clients, active_count)

        self.uploaded_client_payloads = []
        total_samples = 0
        total_uplink = 0.0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost["total_cost"] / client.train_time_cost["num_rounds"] + \
                    client.send_time_cost["total_cost"] / client.send_time_cost["num_rounds"]
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost > self.time_threthold:
                continue

            gm_state = client.get_gm_state()
            pm_state = client.get_pm_state()
            signature = client.get_pm_signature()
            total_samples += client.train_samples
            total_uplink += self._state_bytes(gm_state) + self._state_bytes(pm_state)
            self.uploaded_client_payloads.append({
                "id": client.id,
                "weight_raw": client.train_samples,
                "gm_state": gm_state,
                "pm_state": pm_state,
                "signature": signature,
            })

        for payload in self.uploaded_client_payloads:
            payload["weight"] = payload["weight_raw"] / max(total_samples, 1)
        self.last_round_comm["total_uplink_bytes"] = total_uplink

    def _aggregate_global_gm(self):
        weighted = [(payload["weight"], payload["gm_state"]) for payload in self.uploaded_client_payloads]
        if weighted:
            self.global_gm_state = average_nested_states(weighted)

    def _aggregate_cluster_pm(self):
        grouped = defaultdict(list)
        for payload in self.uploaded_client_payloads:
            cluster_id = self.cluster_assignments[payload["id"]]
            grouped[cluster_id].append((payload["weight"], payload["pm_state"]))

        if not grouped:
            return

        refreshed = {}
        for cluster_id, states in grouped.items():
            cluster_pm_avg = average_nested_states(states)
            cluster_clients = [p for p in self.uploaded_client_payloads if self.cluster_assignments[p["id"]] == cluster_id]
            cluster_gm_avg = average_nested_states([(p["weight"], p["gm_state"]) for p in cluster_clients])
            extracted_pm = slice_pm_from_gm(cluster_gm_avg, cluster_pm_avg)
            blended = {"base": {}, "head": {}}
            for section in ["base", "head"]:
                for key, tensor in cluster_pm_avg[section].items():
                    blended[section][key] = 0.8 * tensor + 0.2 * extracted_pm[section][key]
            refreshed[cluster_id] = blended
        for cluster_id in range(self.num_clusters):
            if cluster_id in refreshed:
                self.cluster_pm_states[cluster_id] = refreshed[cluster_id]

    def _update_clusters(self, round_idx):
        if round_idx < self.cluster_warmup_rounds:
            return
        if round_idx == 0 or round_idx % self.cluster_period != 0:
            return
        if len(self.uploaded_client_payloads) < self.num_clusters:
            return

        ids = [p["id"] for p in self.uploaded_client_payloads]
        signatures = np.stack([p["signature"] for p in self.uploaded_client_payloads], axis=0)
        try:
            clusterer = AgglomerativeClustering(n_clusters=self.num_clusters, metric="cosine", linkage="average")
        except TypeError:
            clusterer = AgglomerativeClustering(n_clusters=self.num_clusters, affinity="cosine", linkage="average")
        labels = clusterer.fit_predict(signatures)
        for cid, label in zip(ids, labels):
            self.cluster_assignments[cid] = int(label)
        print(f"[FedCD2] Round {round_idx}: cluster assignment update done (clusters={len(set(labels))})")

    def _collect_eval_stats(self, use_global_loader=False):
        fused_correct = 0
        gm_correct = 0
        pm_correct = 0
        num_samples = 0
        loss_sum = 0.0
        auc_weighted = 0.0
        agree_pm = 0
        agree_gm = 0

        for client in self.clients:
            if use_global_loader and self.global_test_loader is not None:
                metric = client._eval_metrics_on_loader(self.global_test_loader)
            else:
                metric = client.local_eval_metrics()
            fused_correct += metric["fused_correct"]
            gm_correct += metric["gm_correct"]
            pm_correct += metric["pm_correct"]
            num_samples += metric["num_samples"]
            loss_sum += metric["loss"]
            auc_weighted += metric["auc_weighted"]
            agree_pm += metric["agree_pm"]
            agree_gm += metric["agree_gm"]

        return {
            "fused_acc": fused_correct / max(num_samples, 1),
            "gm_acc": gm_correct / max(num_samples, 1),
            "pm_acc": pm_correct / max(num_samples, 1),
            "train_loss": loss_sum / max(num_samples, 1),
            "auc": auc_weighted / max(num_samples, 1),
            "agree_pm": agree_pm / max(num_samples, 1),
            "agree_gm": agree_gm / max(num_samples, 1),
            "num_samples": num_samples,
        }

    def evaluate(self, round_idx=0):
        local_stats = self._collect_eval_stats(use_global_loader=False)
        global_stats = self._collect_eval_stats(use_global_loader=True) if self.global_test_loader is not None else local_stats

        self.rs_test_acc.append(local_stats["fused_acc"])
        self.rs_global_test_acc.append(global_stats["fused_acc"])
        self.rs_test_auc.append(local_stats["auc"])
        self.rs_train_loss.append(local_stats["train_loss"])

        print(f"Server: Overall Averaged Local Test Accuracy: {local_stats['fused_acc']:.4f}")
        print(f"Server: PM-only Local Test Accuracy: {local_stats['pm_acc']:.4f}")
        print(f"Server: GM-only Local Test Accuracy: {local_stats['gm_acc']:.4f}")
        print(f"Server: Local Fused Argmax Agreement (with PM/GM): {local_stats['agree_pm']:.4f}/{local_stats['agree_gm']:.4f}")
        print(f"Server: Overall Averaged Global Test Accuracy: {global_stats['fused_acc']:.4f}")
        print(f"Server: Fused Argmax Agreement (with PM/GM): {global_stats['agree_pm']:.4f}/{global_stats['agree_gm']:.4f}")
        print(f"Server: GM-only Global Test Accuracy: {global_stats['gm_acc']:.4f}")
        print(f"Server: PM-only Global Test Accuracy: {global_stats['pm_acc']:.4f}")
        print(f"Server: Overall Averaged Test AUC: {local_stats['auc']:.4f}")
        print(f"Server: Overall Averaged Train Loss: {local_stats['train_loss']:.4f}")
        self._print_cluster_detail()

    def _print_cluster_detail(self):
        grouped_ids = defaultdict(list)
        for cid, cluster_id in self.cluster_assignments.items():
            grouped_ids[cluster_id].append(cid)
        print("Server: Cluster-wise Accuracy Detail:")
        for cluster_id in sorted(grouped_ids.keys()):
            sample_count = 0
            correct = 0.0
            for client_id in grouped_ids[cluster_id]:
                client = self.clients[client_id]
                metric = client.local_eval_metrics()
                sample_count += metric["num_samples"]
                correct += metric["fused_correct"]
            cluster_acc = correct / max(sample_count, 1)
            print(f"  Cluster {cluster_id}: {cluster_acc:.4f} (samples: {sample_count})")

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                self.evaluate(round_idx=i)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if i % self.gm_period == 0:
                self._aggregate_global_gm()
            self._update_clusters(i)
            self._aggregate_cluster_pm()

            self.Budget.append(time.time() - s_t)
            total_uplink_mb = self.last_round_comm["total_uplink_bytes"] / (1024 * 1024)
            avg_uplink_mb = total_uplink_mb / max(len(self.selected_clients), 1)
            print(f"[FedCD2] Round {i} Total Uplink Size: {total_uplink_mb:.2f} MB (Avg: {avg_uplink_mb:.2f} MB/client)")
            print(f"[FedCD2] Round {i}: cluster PM aggregation done")
            if i % self.gm_period == 0:
                print(f"[FedCD2] Round {i}: global GM aggregation done")
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

        print("\nBest local accuracy.")
        print(max(self.rs_test_acc) if self.rs_test_acc else 0.0)
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]) if len(self.Budget) > 1 else 0.0)
        self.save_results()

    def save_results(self):
        if hasattr(self.args, "log_usage_path") and self.args.log_usage_path:
            result_path = os.path.dirname(self.args.log_usage_path)
        else:
            result_path = "../results/"
        os.makedirs(result_path, exist_ok=True)

        if len(self.rs_test_acc) == 0:
            return

        file_name = f"{self.dataset}_{self.algorithm}_{self.goal}_{self.times}.h5"
        file_path = os.path.join(result_path, file_name)
        print("File path: " + file_path)
        with h5py.File(file_path, "w") as hf:
            hf.create_dataset("rs_test_acc", data=self.rs_test_acc)
            hf.create_dataset("rs_local_test_acc", data=self.rs_test_acc)
            hf.create_dataset("rs_global_test_acc", data=self.rs_global_test_acc)
            hf.create_dataset("rs_test_auc", data=self.rs_test_auc)
            hf.create_dataset("rs_train_loss", data=self.rs_train_loss)
