import copy
import time
import subprocess
import shutil
import os
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torchvision
from flcore.trainmodel.models import SmallFExt
from flcore.clients.clientfedcd import clientFedCD
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data 

class FedCD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()

        # 클라이언트 클래스 연결
        self.set_clients(clientFedCD)
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        # [중요] 서버용 Proxy Data 로드 (Knowledge Distillation용)
        # 일단 테스트 데이터셋의 일부를 Proxy로 쓴다고 가정 (실제 연구에선 별도 데이터 권장)
        self.proxy_data_loader = self.load_proxy_data() 

        # FedCD clustering config
        self.num_clusters = max(1, int(getattr(args, "num_clusters", 1)))
        self.cluster_period = max(1, int(getattr(args, "cluster_period", 1)))
        self.pm_period = max(1, int(getattr(args, "pm_period", 1)))
        self.global_period = max(1, int(getattr(args, "global_period", 1)))
        self.cluster_sample_size = int(getattr(args, "cluster_sample_size", 512))
        self.cluster_map = {c.id: (c.id % self.num_clusters) for c in self.clients}
        self.log_usage = bool(getattr(args, "log_usage", False))
        self.log_usage_every = max(1, int(getattr(args, "log_usage_every", 1)))
        self.log_usage_path = str(getattr(args, "log_usage_path", "logs/result.csv"))
        # Set cluster_acc.csv in the same directory as usage.csv
        self.log_cluster_path = os.path.join(os.path.dirname(self.log_usage_path), "cluster_acc.csv")
        self._usage_header_written = False
        self._cluster_header_written = False
        self.eval_common_global = bool(getattr(args, "eval_common_global", True))
        self.common_test_samples = int(getattr(args, "common_test_samples", 2000))
        self.common_eval_batch_size = int(getattr(args, "common_eval_batch_size", 256))
        self.common_test_loader = self._build_common_test_loader() if self.eval_common_global else None
        self.f_ext = self._build_f_ext(args)
        self.f_ext_dim = getattr(self.f_ext, "out_dim", None)
        self.gm_head, self.gm_final = self._split_classifier(self.global_model)
        self.gm_adapter = self._build_adapter(self.gm_head, self.gm_final)
        pm_model = getattr(args, "pm_model", None)
        if pm_model is not None:
            self.pm_head, self.pm_final = self._split_classifier(pm_model)
            self.pm_adapter = self._build_adapter(self.pm_head, self.pm_final)
        else:
            self.pm_head = None
            self.pm_final = None
            self.pm_adapter = None

        # [ACT] Adaptive Clustering Threshold Initialization
        self.adaptive_threshold = bool(getattr(args, "adaptive_threshold", False))
        self.current_threshold = float(getattr(args, "cluster_threshold", 0.0))
        # If ACT is enabled but initial threshold is 0, start with a small value
        if self.adaptive_threshold and self.current_threshold <= 0:
            self.current_threshold = 0.05
            
        self.threshold_step = float(getattr(args, "threshold_step", 0.05))
        self.threshold_decay = float(getattr(args, "threshold_decay", 0.9))
        self.threshold_max = float(getattr(args, "threshold_max", 0.95))
        
        # [ACT] Zig-Zag Convergence State
        self.act_direction = 1 # 1: increase, -1: decrease
        self.acc_history = [] # Stores mean accuracy for regression
        self.window_size = int(getattr(args, "act_window_size", 5))

    def _build_f_ext(self, args):
        model_name = str(getattr(args, "fext_model", "SmallFExt"))
        if model_name == "VGG16":
            # Load Pretrained VGG16 features
            base_model = torchvision.models.vgg16(pretrained=True)
            f_ext = nn.Sequential(base_model.features, base_model.avgpool, nn.Flatten())
            f_ext.out_dim = 512 * 7 * 7 # VGG16 final feature map size
        elif model_name == "SmallFExt":
            in_channels = 1 if "MNIST" in args.dataset else 3
            fext_dim = int(getattr(args, "fext_dim", 512))
            f_ext = SmallFExt(in_channels=in_channels, out_dim=fext_dim)
        else:
            raise NotImplementedError(f"Unknown fext_model: {model_name}")
            
        for param in f_ext.parameters():
            param.requires_grad = False
        f_ext.eval()
        return f_ext

    def _split_classifier(self, model):
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
            if isinstance(model.classifier, nn.Sequential) and len(model.classifier) > 1:
                head = nn.Sequential(*list(model.classifier.children())[:-1])
                final = list(model.classifier.children())[-1]
                return head, final
            return nn.Identity(), model.classifier
        if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
            return nn.Identity(), model.fc
        return nn.Identity(), nn.Identity()

    @staticmethod
    def _first_linear_in_features(module):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                return layer.in_features
        return None

    def _build_adapter(self, head, final):
        f_ext_dim = self.f_ext_dim
        pm_in_dim = self._first_linear_in_features(head)
        if pm_in_dim is None:
            pm_in_dim = self._first_linear_in_features(final)
        if f_ext_dim is None or pm_in_dim is None:
            return None
        if f_ext_dim == pm_in_dim:
            return None
        return nn.Linear(f_ext_dim, pm_in_dim)

    def _read_gpu_util(self):
        smi = shutil.which("nvidia-smi")
        if not smi:
            return None
        cmd = [
            smi,
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(cmd, encoding="utf-8").strip()
        except Exception:
            return None
        if not out:
            return None
        first_line = out.splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) != 3:
            return None
        util, mem_used, mem_total = parts
        return util, mem_used, mem_total

    def _log_usage(
        self,
        round_idx,
        stage,
        wall_start,
        cpu_start,
        test_acc=None,
        train_loss=None,
        uplink=0,
        downlink=0,
        common_test_acc=None,
    ):
        wall_delta = time.time() - wall_start
        # cpu_delta = time.process_time() - cpu_start
        # cpu_pct = (cpu_delta / wall_delta * 100.0) if wall_delta > 0 else 0.0

        acc_str = f"acc={test_acc:.4f}" if test_acc is not None else ""
        common_acc_str = f"common_acc={common_test_acc:.4f}" if common_test_acc is not None else ""
        loss_str = f"loss={train_loss:.4f}" if train_loss is not None else ""
        msg = (
            f"[FedCD] Round {round_idx} | {stage} | "
            f"wall={wall_delta:.2f}s "
            + f"{acc_str} {common_acc_str} {loss_str}"
        )
        # print(msg)
        self._append_usage_csv(round_idx, test_acc, common_test_acc, train_loss, uplink, downlink)

    def _append_usage_csv(self, round_idx, test_acc, common_test_acc, train_loss, uplink, downlink):
        path = self.log_usage_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        header = "round,test_acc,common_test_acc,train_loss,uplink_mb,downlink_mb,total_mb\n"
        
        # Only log rows that have metrics (evaluation stage)
        if test_acc is None and common_test_acc is None and train_loss is None:
            return

        t_acc = f"{test_acc:.4f}" if test_acc is not None else ""
        common_acc = f"{common_test_acc:.4f}" if common_test_acc is not None else ""
        t_loss = f"{train_loss:.4f}" if train_loss is not None else ""
        uplink_mb = uplink / (1024**2)
        downlink_mb = downlink / (1024**2)
        total_mb = uplink_mb + downlink_mb
        line = f"{round_idx},{t_acc},{common_acc},{t_loss},{uplink_mb:.4f},{downlink_mb:.4f},{total_mb:.4f}\n"
        
        if not self._usage_header_written:
            need_header = not os.path.exists(path)
            with open(path, "a", encoding="utf-8") as f:
                if need_header:
                    f.write(header)
                f.write(line)
            self._usage_header_written = True
        else:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)

    def _log_cluster_acc(self, round_idx, cluster_id, accuracy, samples):
        path = self.log_cluster_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        header = "round,cluster_id,accuracy,samples\n"
        line = f"{round_idx},{cluster_id},{accuracy:.4f},{samples}\n"
        
        if not self._cluster_header_written:
            need_header = not os.path.exists(path)
            with open(path, "a", encoding="utf-8") as f:
                if need_header:
                    f.write(header)
                f.write(line)
            self._cluster_header_written = True
        else:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)

    def send_models(self):
        assert (len(self.clients) > 0)
        gm_state = {k: v.detach().cpu() for k, v in self.global_model.state_dict().items()}
        gm_adapter_state = None
        if self.gm_adapter is not None:
            gm_adapter_state = {k: v.detach().cpu() for k, v in self.gm_adapter.state_dict().items()}
        payload = {"gm_state": gm_state, "gm_adapter": gm_adapter_state}
        
        # [Info] Calculate and print Broadcast GM Size
        total_bytes = 0
        if payload["gm_state"]:
             total_bytes += sum(v.numel() * v.element_size() for v in payload["gm_state"].values())
        if payload["gm_adapter"]:
             total_bytes += sum(v.numel() * v.element_size() for v in payload["gm_adapter"].values())
        
        print(f"[FedCD] Broadcast GM Size: {total_bytes / (1024**2):.2f} MB per client")
        
        broadcast_bytes = total_bytes * len(self.clients)

        for client in self.clients:
            start_time = time.time()
            client.set_parameters(payload)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            
        return broadcast_bytes

    def train(self):
        for i in range(self.global_rounds + 1):
            wall_start = time.time()
            cpu_start = time.process_time()
            self.selected_clients = self.select_clients()
            
            # Init communication cost for this round
            round_uplink = 0
            round_downlink = 0

            # 1. 로컬 학습 수행 (Local Training)
            print(f"\n[FedCD] Round {i}: training {len(self.selected_clients)} clients")
            for client in tqdm(self.selected_clients, desc=f"Round {i} Local Training", leave=False):
                client.train()
            
            if self.log_usage and i % self.log_usage_every == 0:
                self._log_usage(i, "post_local", wall_start, cpu_start)

            # 2. PM 수집 (Receive PMs)
            received_pms = []
            total_uplink_bytes = 0
            for client in tqdm(self.selected_clients, desc=f"Round {i} Collecting PMs", leave=False):
                pm_state = client.upload_parameters()
                received_pms.append((client.id, pm_state))
                total_uplink_bytes += sum(v.numel() * v.element_size() for v in pm_state.values())
            
            round_uplink += total_uplink_bytes

            if len(self.selected_clients) > 0:
                avg_uplink = total_uplink_bytes / len(self.selected_clients)
                print(f"[FedCD] Round {i} Total Uplink Size: {total_uplink_bytes / (1024**2):.2f} MB (Avg: {avg_uplink / (1024**2):.2f} MB/client)")

            # 2.5 클러스터링 갱신 및 상세 로깅
            if i % self.cluster_period == 0:
                self.cluster_map = self.cluster_clients_by_distribution()
                
                # 상세 클러스터링 현황 로깅
                cluster_groups = {}
                for cid, clust_id in self.cluster_map.items():
                    if clust_id not in cluster_groups:
                        cluster_groups[clust_id] = []
                    cluster_groups[clust_id].append(cid)
                
                print(f"\n[FedCD] Round {i}: Clustering Result:")
                for c_id in sorted(cluster_groups.keys()):
                    clients_in_cluster = sorted(cluster_groups[c_id])
                    print(f"  Cluster {c_id} ({len(clients_in_cluster)} clients): {clients_in_cluster}")

            # 3. 클러스터 내 PM 집계 및 배포
            cluster_pms = self.aggregate_cluster_pms(received_pms)
            if i % self.pm_period == 0 and cluster_pms:
                downlink_bytes = self.send_cluster_pms(cluster_pms)
                round_downlink += downlink_bytes
                print(f"[FedCD] Round {i}: PM aggregation and cluster update done")
                if self.log_usage and i % self.log_usage_every == 0:
                    self._log_usage(i, "post_pm", wall_start, cpu_start)

            # 4. 서버 측 앙상블 증류 (Server-side Ensemble Distillation)
            if i % self.global_period == 0 and cluster_pms:
                self.aggregate_and_distill(list(cluster_pms.values()))
                # 업데이트된 GM을 모든 클라이언트에게 배포 (Downlink)
                downlink_bytes = self.send_models()
                round_downlink += downlink_bytes
                print(f"[FedCD] Round {i}: Server-side distillation and GM update done")
                # Warm-up classifier after GM update
                if getattr(self.args, "fedcd_warmup_epochs", 0) > 0:
                    for client in tqdm(self.selected_clients, desc=f"Round {i} Classifier Warm-up", leave=False):
                        client.warmup_classifier()
                if self.log_usage and i % self.log_usage_every == 0:
                    self._log_usage(i, "post_gm", wall_start, cpu_start)
            
            if i % self.eval_gap == 0:
                print(f"\n------------- Round number: {i} -------------")
                # 평가 실행 및 클러스터별 정확도 로깅
                self.evaluate_with_clusters(i, wall_start, cpu_start, round_uplink, round_downlink)

        print("\nTraining finished. Saving results...")
        self.save_results()

    # 기존 evaluate 함수 대신 클러스터 정보 포함하여 평가
    def evaluate_with_clusters(self, round_idx, wall_start, cpu_start, uplink=0, downlink=0):
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        # stats: ids, num_samples, tot_correct, tot_auc
        ids, num_samples, tot_correct, tot_auc = stats
        # stats_train: ids, num_samples, losses
        _, num_samples_train, losses = stats_train
        
        # 전체 평균 정확도 및 손실 계산 (개별 클라이언트 로컬 테스트셋)
        total_samples = sum(num_samples)
        total_samples_train = sum(num_samples_train)
        
        avg_acc = sum(tot_correct) / total_samples if total_samples > 0 else 0.0
        avg_auc = sum(tot_auc) / total_samples if total_samples > 0 else 0.0
        avg_loss = sum(losses) / total_samples_train if total_samples_train > 0 else 0.0
        common_test_acc = self.evaluate_common_test_acc()
            
        print(f"Server: Overall Averaged Personalized Test Accuracy: {avg_acc:.4f}")
        if common_test_acc is not None:
            print(f"Server: Common Global Test Accuracy: {common_test_acc:.4f}")
        print(f"Server: Overall Averaged Test AUC: {avg_auc:.4f}")
        print(f"Server: Overall Averaged Train Loss: {avg_loss:.4f}")

        # [ACT] Trigger adaptive threshold adjustment
        # Only adjust threshold at the end of a clustering period to synchronize with structure updates
        if self.adaptive_threshold and (round_idx % self.cluster_period == 0):
            # Calculate per-client accuracy
            current_accs = [correct / n_samples if n_samples > 0 else 0.0 for correct, n_samples in zip(tot_correct, num_samples)]
            self.adjust_dynamic_threshold(ids, current_accs)

        # PFLlib 내부 변수에 저장 (h5 파일 저장용)
        self.rs_test_acc.append(avg_acc)
        self.rs_test_auc.append(avg_auc)
        self.rs_train_loss.append(avg_loss)

        # CSV 로그에 기록
        if self.log_usage:
            self._log_usage(
                round_idx,
                "evaluation",
                wall_start,
                cpu_start,
                test_acc=avg_acc,
                common_test_acc=common_test_acc,
                train_loss=avg_loss,
                uplink=uplink,
                downlink=downlink,
            )

        # 클러스터별 정확도 계산 및 출력, 로깅
        cluster_stats = {}
        for i, cid in enumerate(ids):
            c_id = self.cluster_map.get(cid, -1) # -1 if not clustered yet
            if c_id not in cluster_stats:
                cluster_stats[c_id] = {"correct": 0, "samples": 0}
            cluster_stats[c_id]["correct"] += tot_correct[i]
            cluster_stats[c_id]["samples"] += num_samples[i]
            
        print("Server: Cluster-wise Accuracy Detail:")
        for c_id in sorted(cluster_stats.keys()):
            s = cluster_stats[c_id]
            if s["samples"] > 0:
                c_acc = s["correct"] / s["samples"]
                print(f"  Cluster {c_id}: {c_acc:.4f} (samples: {s['samples']})")
                if self.log_usage:
                    self._log_cluster_acc(round_idx, c_id, c_acc, s["samples"])
            else:
                print(f"  Cluster {c_id}: N/A (no samples)")

    def aggregate_cluster_pms(self, received_pms):
        # Streamed aggregation to avoid holding all PMs in memory
        cluster_sums = {}
        cluster_counts = {}
        with torch.no_grad():
            for client_id, state in received_pms:
                cluster_id = self.cluster_map.get(client_id, 0)
                if cluster_id not in cluster_sums:
                    cluster_sums[cluster_id] = {k: v.clone() for k, v in state.items()}
                    cluster_counts[cluster_id] = 1
                else:
                    for key, value in state.items():
                        cluster_sums[cluster_id][key] += value
                    cluster_counts[cluster_id] += 1

        cluster_avg = {}
        for cluster_id, sum_state in cluster_sums.items():
            count = cluster_counts.get(cluster_id, 1)
            cluster_avg[cluster_id] = {k: v / count for k, v in sum_state.items()}
        return cluster_avg

    def send_cluster_pms(self, cluster_pms):
        # [Info] Calculate and print Cluster PM Size (Assuming all clusters have same model size)
        total_broadcast_bytes = 0
        
        if cluster_pms:
            # Assuming all clusters have roughly same PM structure/size
            first_pm = next(iter(cluster_pms.values()))
            pm_bytes = sum(v.numel() * v.element_size() for v in first_pm.values())
            print(f"[FedCD] Broadcast Cluster PM Size: {pm_bytes / (1024**2):.2f} MB per client (in cluster)")

        for client in self.clients:
            cluster_id = self.cluster_map.get(client.id, 0)
            if cluster_id in cluster_pms:
                client.set_personalized_parameters(cluster_pms[cluster_id])
                # Add size for this client
                current_pm = cluster_pms[cluster_id]
                current_bytes = sum(v.numel() * v.element_size() for v in current_pm.values())
                total_broadcast_bytes += current_bytes
        
        return total_broadcast_bytes

    def adjust_dynamic_threshold(self, ids, current_accs):
        """
        [ACT] Adjust clustering threshold using Relative Trend Convergence.
        Strategy:
        1. Calculate current improvement: current_acc - prev_acc.
        2. Calculate established trend (Slope) from recent history.
        3. If current improvement is LESS than the trend slope (slowdown), 
           reverse direction and decay step.
        """
        if not self.adaptive_threshold:
            return

        mean_acc = sum(current_accs) / len(current_accs) if current_accs else 0.0
        
        if len(self.acc_history) >= 2:
            # 1. Calculate established trend (Slope) from history
            x = np.arange(len(self.acc_history))
            y = np.array(self.acc_history)
            trend_slope, _ = np.polyfit(x, y, 1)
            
            # 2. Calculate current improvement (latest step)
            current_diff = mean_acc - self.acc_history[-1]
            
            print(f"[ACT] Mean Acc: {mean_acc:.4f} | Current Diff: {current_diff:.6f} | Trend Slope: {trend_slope:.6f}")

            # 3. Decision: If current growth is slower than the established trend
            # We also add act_min_slope as a safety floor to avoid reversing on tiny noise 
            # when the trend is near zero.
            min_slope = getattr(self.args, "act_min_slope", 0.0001)
            
            if current_diff < max(min_slope, trend_slope):
                reason = "Slowdown" if current_diff >= 0 else "Drop"
                print(f"[ACT] {reason} detected (Diff < Trend)! Reversing direction and decaying step.")
                self.act_direction *= -1
                self.threshold_step *= self.threshold_decay
                # Reset history to establish a new trend in the new direction
                self.acc_history = []
            else:
                print(f"[ACT] Growth Accelerating (Diff >= Trend). Continuing direction.")
        else:
            print(f"[ACT] Mean Acc: {mean_acc:.4f} | Collecting history (n={len(self.acc_history)})...")

        # Update History
        self.acc_history.append(mean_acc)
        if len(self.acc_history) > self.window_size:
            self.acc_history.pop(0)

        # Update Threshold
        old_th = self.current_threshold
        self.current_threshold += self.act_direction * self.threshold_step
        self.current_threshold = max(0.01, min(self.threshold_max, self.current_threshold))
        
        print(f"[ACT] Updated Threshold: {old_th:.4f} -> {self.current_threshold:.4f} (Dir: {self.act_direction}, Step: {self.threshold_step:.4f})")

    def cluster_clients_by_distribution(self):
        # Cluster clients by f_ext feature distribution stats
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.preprocessing import normalize

        features = []
        client_ids = []
        for client in self.clients:
            mean, var = client.get_feature_stats(self.cluster_sample_size)
            # Flatten to ensure 1D vectors before concatenation
            feat = torch.cat([mean.flatten(), var.flatten()], dim=0).cpu().numpy()
            features.append(feat)
            client_ids.append(client.id)

        X = np.stack(features, axis=0)
        # L2 Normalization (Cosine-like distance)
        X = normalize(X, axis=1)

        # [ACT] Use self.current_threshold
        threshold = self.current_threshold
        
        if threshold > 0:
            print(f"[FedCD] Using Agglomerative Clustering (L2 Normalized) with threshold={threshold:.4f}")
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                linkage='ward'
            )
            labels = clustering.fit_predict(X)
            # Update num_clusters for logging/monitoring
            self.num_clusters = len(set(labels))
        else:
            n_clusters = min(self.num_clusters, len(client_ids))
            if n_clusters <= 1:
                return {cid: 0 for cid in client_ids}
            print(f"[FedCD] Using K-Means Clustering with n_clusters={n_clusters}")
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            labels = kmeans.fit_predict(X)

        return {cid: int(label) for cid, label in zip(client_ids, labels)}

    def _build_common_test_loader(self):
        # Build one shared test subset so all clients are evaluated on exactly the same data.
        shared_test_data = []
        for client_id in range(self.num_clients):
            shared_test_data.extend(read_client_data(self.dataset, client_id, is_train=False))

        if len(shared_test_data) == 0:
            print("[FedCD] Common global test set is empty. Skipping shared evaluation.")
            return None

        if self.common_test_samples > 0 and self.common_test_samples < len(shared_test_data):
            rng = random.Random(0)
            sample_indices = rng.sample(range(len(shared_test_data)), self.common_test_samples)
            shared_test_data = [shared_test_data[idx] for idx in sample_indices]

        num_workers = int(getattr(self.args, "num_workers", 0))
        pin_memory = bool(getattr(self.args, "pin_memory", False)) and self.device == "cuda"
        loader_kwargs = {
            "batch_size": self.common_eval_batch_size,
            "shuffle": False,
            "drop_last": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = int(getattr(self.args, "prefetch_factor", 2))

        print(f"[FedCD] Common Global Test Set Size: {len(shared_test_data)}")
        return torch.utils.data.DataLoader(shared_test_data, **loader_kwargs)

    def evaluate_common_test_acc(self):
        if not self.eval_common_global or self.common_test_loader is None:
            return None

        acc_sum = 0.0
        valid_clients = 0
        device = self.device
        use_non_blocking = device == "cuda" and bool(getattr(self.args, "pin_memory", False))

        for client in self.clients:
            client.model.to(device)
            client.model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in self.common_test_loader:
                    if type(x) == type([]):
                        x = x[0]
                    x = x.to(device, non_blocking=use_non_blocking)
                    y = y.to(device, non_blocking=use_non_blocking)
                    output = client.model(x)
                    correct += (torch.argmax(output, dim=1) == y).sum().item()
                    total += y.size(0)

            client.model.to("cpu")
            if total > 0:
                acc_sum += correct / total
                valid_clients += 1

        if device == "cuda":
            torch.cuda.empty_cache()

        if valid_clients == 0:
            return None
        return acc_sum / valid_clients

    @staticmethod
    def average_state_dicts(states):
        avg_state = {}
        for key in states[0].keys():
            avg_state[key] = sum(state[key] for state in states) / len(states)
        return avg_state

    def aggregate_and_distill(self, received_pm_states):
        print("Server: Distilling Knowledge from PM Ensemble to GM...")

        def _distill_once(device):
            # GM (Student) 학습 준비 (f_ext는 frozen)
            self.global_model.to(device)
            self.f_ext.to(device)
            self.gm_head.to(device)
            self.gm_final.to(device)
            self.f_ext.eval()
            self.gm_head.train()
            self.gm_final.train()
            if self.gm_adapter is not None:
                self.gm_adapter.to(device)
            if self.pm_head is not None:
                self.pm_head.to(device)
            if self.pm_final is not None:
                self.pm_final.to(device)
            if self.pm_adapter is not None:
                self.pm_adapter.to(device)
            gm_params = list(self.gm_head.parameters()) + list(self.gm_final.parameters())
            if self.gm_adapter is not None:
                gm_params += list(self.gm_adapter.parameters())
            optimizer = torch.optim.SGD(gm_params, lr=0.01)
            kl_loss = nn.KLDivLoss(reduction='batchmean')
            use_amp = device == "cuda" and bool(getattr(self.args, "amp", False))
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            # Proxy Data로 증류 (예: 1 Epoch)
            for x, _ in tqdm(self.proxy_data_loader, desc="Ensemble Distillation", leave=False):
                x = x.to(device, non_blocking=(device == "cuda"))

                with torch.no_grad():
                    z = self.f_ext(x)
                    if z.dim() > 2:
                        z = torch.flatten(z, 1)
                    z_gm = self.gm_adapter(z) if self.gm_adapter is not None else z

                # Teacher Ensemble Logits (sequential to save memory)
                with torch.no_grad():
                    ensemble_logits = None
                    for state in received_pm_states:
                        head_state = {k.replace("head.", ""): v.to(device) for k, v in state.items() if k.startswith("head.")}
                        final_state = {k.replace("final.", ""): v.to(device) for k, v in state.items() if k.startswith("final.")}
                        adapter_state = {k.replace("adapter.", ""): v.to(device) for k, v in state.items() if k.startswith("adapter.")}

                        if self.pm_head is not None and head_state:
                            self.pm_head.load_state_dict(head_state, strict=True)
                        if self.pm_final is not None and final_state:
                            self.pm_final.load_state_dict(final_state, strict=True)
                        if self.pm_adapter is not None and adapter_state:
                            self.pm_adapter.load_state_dict(adapter_state, strict=True)

                        z_pm = self.pm_adapter(z) if self.pm_adapter is not None else z
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            logits = self.pm_final(self.pm_head(z_pm))
                        if ensemble_logits is None:
                            ensemble_logits = logits
                        else:
                            ensemble_logits = ensemble_logits + logits
                    ensemble_logits = ensemble_logits / len(received_pm_states)
                    target_prob = torch.softmax(ensemble_logits, dim=1)
                
                # Student (GM) Logits
                with torch.cuda.amp.autocast(enabled=use_amp):
                    student_logits = self.gm_final(self.gm_head(z_gm))
                    student_log_prob = torch.log_softmax(student_logits, dim=1)
                    loss = kl_loss(student_log_prob, target_prob)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Move back to CPU to reduce GPU memory pressure
            self.global_model.to("cpu")
            self.f_ext.to("cpu")
            self.gm_head.to("cpu")
            self.gm_final.to("cpu")
            if self.gm_adapter is not None:
                self.gm_adapter.to("cpu")
            if self.pm_head is not None:
                self.pm_head.to("cpu")
            if self.pm_final is not None:
                self.pm_final.to("cpu")
            if self.pm_adapter is not None:
                self.pm_adapter.to("cpu")
            if device == "cuda" and self.args.avoid_oom:
                torch.cuda.empty_cache()

        try:
            _distill_once(self.device)
        except RuntimeError as err:
            if self.device == "cuda" and "out of memory" in str(err).lower():
                print("[Warn] OOM during server distillation. Falling back to CPU.")
                torch.cuda.empty_cache()
                _distill_once("cpu")
                return
            raise
            
    def load_proxy_data(self):
        # [수정] Proxy Data로 TinyImageNet을 사용하거나 N개의 임의 샘플을 추출하도록 개선
        proxy_dataset_name = getattr(self.args, "proxy_dataset", "TinyImagenet")
        proxy_samples = int(getattr(self.args, "proxy_samples", 1000))
        
        print(f"[FedCD] Loading Proxy Data: {proxy_dataset_name} (Samples: {proxy_samples})")

        # TinyImageNet 경로 설정
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        tiny_path = os.path.join(base_dir, "dataset", "TinyImagenet", "rawdata", "tiny-imagenet-200", "train")

        if proxy_dataset_name == "TinyImagenet" and os.path.exists(tiny_path):
            from torchvision.datasets import ImageFolder
            import torchvision.transforms as transforms
            from torch.utils.data import Subset
            import random

            transform = transforms.Compose([
                transforms.Resize((64, 64)), # TinyImageNet default size
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # 흑백 이미지일 경우 3채널로 변환하는 transform 추가 (필요시)
            full_dataset = ImageFolder(root=tiny_path, transform=transform)
            
            # 전체 데이터셋 중 임의로 N장 선택 (라벨은 무시됨)
            if proxy_samples < len(full_dataset):
                indices = random.sample(range(len(full_dataset)), proxy_samples)
                proxy_dataset = Subset(full_dataset, indices)
            else:
                proxy_dataset = full_dataset
            
            print(f"[FedCD] Successfully loaded {len(proxy_dataset)} samples from TinyImageNet.")
        else:
            # Fallback: 기존 방식 (첫 번째 클라이언트의 테스트 데이터 사용)
            if proxy_dataset_name == "TinyImagenet":
                print(f"[Warn] TinyImageNet raw data not found at {tiny_path}. Fallback to client test data.")
            
            from utils.data_utils import read_client_data
            test_data = read_client_data(self.args.dataset, 0, is_train=False)
            
            # N개 샘플링
            if proxy_samples < len(test_data):
                import random
                proxy_dataset = random.sample(test_data, proxy_samples)
            else:
                proxy_dataset = test_data

        num_workers = int(getattr(self.args, "num_workers", 0))
        pin_memory = bool(getattr(self.args, "pin_memory", False)) and self.device == "cuda"
        loader_kwargs = {
            "batch_size": 32,
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = int(getattr(self.args, "prefetch_factor", 2))
        
        return torch.utils.data.DataLoader(proxy_dataset, **loader_kwargs)
