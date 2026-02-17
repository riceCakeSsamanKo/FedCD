import copy
import time
import subprocess
import shutil
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.max_dynamic_clusters = max(0, int(getattr(args, "max_dynamic_clusters", self.num_clusters)))
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
        self.global_test_samples = int(
            getattr(args, "global_test_samples", getattr(args, "common_test_samples", 0))
        )
        self.common_eval_batch_size = int(getattr(args, "common_eval_batch_size", 256))
        self.global_test_loader = self._build_global_test_loader() if self.eval_common_global else None
        # Backward-compatible alias
        self.common_test_loader = self.global_test_loader
        self.f_ext = self._build_f_ext(args)
        self.f_ext_dim = getattr(self.f_ext, "out_dim", None)
        self.generalized_module = self._extract_module(self.global_model)
        self.generalized_adapter = self._build_adapter(self.generalized_module)
        pm_model = getattr(args, "pm_model", None)
        if pm_model is None:
            pm_model = copy.deepcopy(self.global_model)
        self.personalized_module = self._extract_module(pm_model)
        self.personalized_adapter = self._build_adapter(self.personalized_module)
        # Cluster-wise GM states (each cluster keeps its own distilled GM).
        self.cluster_generalized_states = {}
        self._initialize_cluster_generalized_states()
        # Global combiner shared by server and broadcast to clients.
        # Initialize from the first client combiner for shape consistency.
        self.global_combiner = copy.deepcopy(self.clients[0].combiner)
        self.global_combiner.to("cpu")
        self.broadcast_global_combiner = bool(getattr(args, "broadcast_global_combiner", False))
        # Distillation hyperparameters
        self.distill_lr = float(getattr(args, "fedcd_distill_lr", 0.01))
        self.distill_temp = float(getattr(args, "fedcd_distill_temp", 2.0))
        self.distill_kl_weight = float(getattr(args, "fedcd_distill_kl_weight", 1.0))
        self.distill_ce_weight = float(getattr(args, "fedcd_distill_ce_weight", 0.2))
        self.prototype_samples = int(getattr(args, "fedcd_prototype_samples", 512))
        self.proto_weight = float(getattr(args, "fedcd_proto_weight", 0.3))
        self.relation_weight = float(getattr(args, "fedcd_relation_weight", 0.1))
        self.cluster_proto_info = {}
        self.global_proto_info = None
        self.cluster_teacher_confidence = {}

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

    def _extract_module(self, model):
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
            return model.classifier
        if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
            return model.fc
        return nn.Identity()

    @staticmethod
    def _first_linear_in_features(module):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                return layer.in_features
        return None

    def _build_adapter(self, module):
        f_ext_dim = self.f_ext_dim
        pm_in_dim = self._first_linear_in_features(module)
        if f_ext_dim is None or pm_in_dim is None:
            return None
        if f_ext_dim == pm_in_dim:
            return None
        return nn.Linear(f_ext_dim, pm_in_dim)

    @staticmethod
    def _merge_legacy_module_state(module, head_state, final_state):
        if not head_state and not final_state:
            return {}
        merged_state = dict(head_state)
        if final_state:
            if isinstance(module, nn.Sequential) and len(module) > 0:
                last_idx = str(len(module) - 1)
                for key, value in final_state.items():
                    merged_state[f"{last_idx}.{key}"] = value
            else:
                merged_state.update(final_state)
        return merged_state

    def _current_generalized_state(self):
        state = {
            f"generalized_module.{k}": v.detach().cpu()
            for k, v in self.generalized_module.state_dict().items()
        }
        if self.generalized_adapter is not None:
            state.update({
                f"generalized_adapter.{k}": v.detach().cpu()
                for k, v in self.generalized_adapter.state_dict().items()
            })
        return state

    def _initialize_cluster_generalized_states(self):
        base_state = self._current_generalized_state()
        for cluster_id in set(self.cluster_map.values()):
            self.cluster_generalized_states[cluster_id] = {
                k: v.clone() for k, v in base_state.items()
            }

    def _ensure_cluster_generalized_states(self, cluster_ids=None):
        if cluster_ids is None:
            cluster_ids = set(self.cluster_map.values())
        base_state = self._current_generalized_state()
        for cluster_id in cluster_ids:
            if cluster_id not in self.cluster_generalized_states:
                self.cluster_generalized_states[cluster_id] = {
                    k: v.clone() for k, v in base_state.items()
                }

    def _apply_generalized_state_to_server(self, generalized_state):
        module_state = {
            k.replace("generalized_module.", ""): v
            for k, v in generalized_state.items()
            if k.startswith("generalized_module.")
        }
        adapter_state = {
            k.replace("generalized_adapter.", ""): v
            for k, v in generalized_state.items()
            if k.startswith("generalized_adapter.")
        }
        if module_state:
            self.generalized_module.load_state_dict(module_state, strict=True)
        if self.generalized_adapter is not None and adapter_state:
            self.generalized_adapter.load_state_dict(adapter_state, strict=True)

    def _weighted_average_generalized_states(self, cluster_gm_states, cluster_counts):
        weighted_states = []
        weights = []
        for cluster_id, state in cluster_gm_states.items():
            w = float(cluster_counts.get(cluster_id, 1))
            if w <= 0:
                continue
            weighted_states.append(state)
            weights.append(w)
        if not weighted_states:
            return None

        total_weight = sum(weights)
        avg_state = {}
        template = weighted_states[0]
        for key in template.keys():
            acc = None
            for state, w in zip(weighted_states, weights):
                if key not in state:
                    continue
                contrib = state[key] * (w / total_weight)
                acc = contrib if acc is None else (acc + contrib)
            if acc is not None:
                avg_state[key] = acc
        return avg_state

    def _build_gm_broadcast_parts(self):
        # Broadcast GM components and optionally global combiner.
        parts = {}
        parts.update({
            f"generalized_module.{k}": v.detach().cpu()
            for k, v in self.generalized_module.state_dict().items()
        })
        if self.generalized_adapter is not None:
            parts.update({
                f"generalized_adapter.{k}": v.detach().cpu()
                for k, v in self.generalized_adapter.state_dict().items()
            })
        if self.broadcast_global_combiner:
            parts.update({
                f"global_combiner.{k}": v.detach().cpu()
                for k, v in self.global_combiner.state_dict().items()
            })
        return parts

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
        local_test_acc=None,
        train_loss=None,
        uplink=0,
        downlink=0,
        global_test_acc=None,
        gm_only_global_test_acc=None,
    ):
        wall_delta = time.time() - wall_start
        # cpu_delta = time.process_time() - cpu_start
        # cpu_pct = (cpu_delta / wall_delta * 100.0) if wall_delta > 0 else 0.0

        local_acc_str = f"local_acc={local_test_acc:.4f}" if local_test_acc is not None else ""
        global_acc_str = f"global_acc={global_test_acc:.4f}" if global_test_acc is not None else ""
        gm_only_acc_str = (
            f"gm_only_global_acc={gm_only_global_test_acc:.4f}"
            if gm_only_global_test_acc is not None
            else ""
        )
        loss_str = f"loss={train_loss:.4f}" if train_loss is not None else ""
        msg = (
            f"[FedCD] Round {round_idx} | {stage} | "
            f"wall={wall_delta:.2f}s "
            + f"{local_acc_str} {global_acc_str} {gm_only_acc_str} {loss_str}"
        )
        # print(msg)
        self._append_usage_csv(
            round_idx,
            local_test_acc,
            global_test_acc,
            gm_only_global_test_acc,
            train_loss,
            uplink,
            downlink,
        )

    def _append_usage_csv(
        self,
        round_idx,
        local_test_acc,
        global_test_acc,
        gm_only_global_test_acc,
        train_loss,
        uplink,
        downlink,
    ):
        path = self.log_usage_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        header = "round,local_test_acc,global_test_acc,gm_only_global_test_acc,train_loss,uplink_mb,downlink_mb,total_mb\n"
        
        # Only log rows that have metrics (evaluation stage)
        if (
            local_test_acc is None
            and global_test_acc is None
            and gm_only_global_test_acc is None
            and train_loss is None
        ):
            return

        local_acc = f"{local_test_acc:.4f}" if local_test_acc is not None else ""
        global_acc = f"{global_test_acc:.4f}" if global_test_acc is not None else ""
        gm_only_acc = f"{gm_only_global_test_acc:.4f}" if gm_only_global_test_acc is not None else ""
        t_loss = f"{train_loss:.4f}" if train_loss is not None else ""
        uplink_mb = uplink / (1024**2)
        downlink_mb = downlink / (1024**2)
        total_mb = uplink_mb + downlink_mb
        line = (
            f"{round_idx},{local_acc},{global_acc},{gm_only_acc},{t_loss},"
            f"{uplink_mb:.4f},{downlink_mb:.4f},{total_mb:.4f}\n"
        )
        
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
        gm_parts = self._build_gm_broadcast_parts()
        payload = {"gm_parts": gm_parts}
        
        # [Info] Calculate and print broadcast size
        total_bytes = sum(v.numel() * v.element_size() for v in gm_parts.values())
        if self.broadcast_global_combiner:
            print(f"[FedCD] Broadcast GM+Global Combiner Size: {total_bytes / (1024**2):.2f} MB per client")
        else:
            print(f"[FedCD] Broadcast GM-Only Size: {total_bytes / (1024**2):.2f} MB per client")
        
        broadcast_bytes = total_bytes * len(self.clients)

        for client in self.clients:
            start_time = time.time()
            client.set_parameters(payload)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            
        return broadcast_bytes

    def send_global_combiner(self):
        assert (len(self.clients) > 0)
        combiner_parts = {
            f"global_combiner.{k}": v.detach().cpu()
            for k, v in self.global_combiner.state_dict().items()
        }
        payload = {"gm_parts": combiner_parts}
        total_bytes = sum(v.numel() * v.element_size() for v in combiner_parts.values())
        print(f"[FedCD] Broadcast Global Combiner-Only Size: {total_bytes / (1024**2):.2f} MB per client")
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
            received_prototypes = []
            total_uplink_bytes = 0
            for client in tqdm(self.selected_clients, desc=f"Round {i} Collecting PMs", leave=False):
                pm_state = client.upload_parameters()
                proto_state = client.upload_class_prototypes(self.prototype_samples)
                received_pms.append((client.id, pm_state))
                received_prototypes.append((client.id, proto_state))
                total_uplink_bytes += sum(v.numel() * v.element_size() for v in pm_state.values())
                total_uplink_bytes += sum(v.numel() * v.element_size() for v in proto_state.values())
            
            round_uplink += total_uplink_bytes

            if len(self.selected_clients) > 0:
                avg_uplink = total_uplink_bytes / len(self.selected_clients)
                print(f"[FedCD] Round {i} Total Uplink Size: {total_uplink_bytes / (1024**2):.2f} MB (Avg: {avg_uplink / (1024**2):.2f} MB/client)")

            # 2.5 클러스터링 갱신 및 상세 로깅
            if i % self.cluster_period == 0:
                prev_cluster_map = dict(self.cluster_map)
                raw_cluster_map = self.cluster_clients_by_distribution()
                self.cluster_map = self._align_cluster_labels(prev_cluster_map, raw_cluster_map)
                self._ensure_cluster_generalized_states()
                
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

            (
                self.cluster_proto_info,
                self.global_proto_info,
                self.cluster_teacher_confidence,
            ) = self.aggregate_cluster_prototypes(received_prototypes)
            if self.cluster_teacher_confidence:
                conf_msg = ", ".join(
                    f"{cid}:{conf:.3f}" for cid, conf in sorted(self.cluster_teacher_confidence.items())
                )
                print(f"[FedCD] Prototype consensus confidence: {conf_msg}")

            # 3. 클러스터 내 PM 집계 및 배포
            cluster_pms = self.aggregate_cluster_pms(received_pms)
            if i % self.pm_period == 0 and cluster_pms:
                downlink_bytes = self.send_cluster_pms(cluster_pms)
                round_downlink += downlink_bytes
                print(f"[FedCD] Round {i}: PM aggregation and cluster update done")
                if self.log_usage and i % self.log_usage_every == 0:
                    self._log_usage(i, "post_pm", wall_start, cpu_start)

            # 4. 서버 측 앙상블 증류 (Server-side Ensemble Distillation)
            if i > 0 and i % self.global_period == 0 and cluster_pms:
                cluster_counts = self._get_cluster_client_counts(received_pms)
                cluster_gm_states = self.aggregate_and_distill(cluster_pms, cluster_counts)
                # Distilled cluster GM을 각 클러스터 클라이언트에 배포
                downlink_bytes = self.send_cluster_gms(cluster_gm_states)
                calib_epochs = int(getattr(self.args, "fedcd_combiner_calib_epochs", 1))
                if calib_epochs > 0:
                    print(f"[FedCD] Round {i}: Combiner-only calibration on selected clients (epochs={calib_epochs})")
                    for client in tqdm(self.selected_clients, desc=f"Round {i} Combiner Calibration", leave=False):
                        client.calibrate_combiner()
                # Optional: broadcast global combiner if enabled
                if self.broadcast_global_combiner:
                    self.update_global_combiner(cluster_pms, cluster_counts)
                    downlink_bytes += self.send_global_combiner()
                round_downlink += downlink_bytes
                print(f"[FedCD] Round {i}: Server-side cluster GM distillation/update done")
                # Warm-up Personalized Module after GM update
                if getattr(self.args, "fedcd_warmup_epochs", 0) > 0:
                    for client in tqdm(self.selected_clients, desc=f"Round {i} PM Warm-up", leave=False):
                        client.warmup_personalized_module()
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
        global_test_acc = self.evaluate_global_test_acc()
        gm_only_global_test_acc = self.evaluate_gm_only_global_test_acc()
            
        print(f"Server: Overall Averaged Local Test Accuracy: {avg_acc:.4f}")
        if global_test_acc is not None:
            print(f"Server: Overall Averaged Global Test Accuracy: {global_test_acc:.4f}")
        if gm_only_global_test_acc is not None:
            print(f"Server: GM-only Global Test Accuracy: {gm_only_global_test_acc:.4f}")
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
                local_test_acc=avg_acc,
                global_test_acc=global_test_acc,
                gm_only_global_test_acc=gm_only_global_test_acc,
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

    def aggregate_cluster_prototypes(self, received_prototypes):
        """
        Aggregate confidence-weighted PM-only class prototypes by cluster.
        Returns:
          cluster_proto_info: {
            cluster_id: {
              "prototypes": Tensor[num_classes, num_classes],
              "valid_mask": Tensor[num_classes] (bool),
              "confidence": float
            }
          }
          global_proto_info: same schema as cluster entry
          cluster_teacher_confidence: {cluster_id: float}
        """
        cluster_logit_conf_sum = {}
        cluster_conf_sum = {}
        cluster_counts_sum = {}

        global_logit_conf_sum = None
        global_conf_sum = None
        global_counts_sum = None

        for client_id, proto in received_prototypes:
            cluster_id = self.cluster_map.get(client_id, 0)
            lcs = proto.get("class_logit_conf_sum", None)
            cfs = proto.get("class_conf_sum", None)
            cnt = proto.get("class_counts", None)
            if lcs is None or cfs is None or cnt is None:
                continue

            lcs = lcs.float().cpu()
            cfs = cfs.float().cpu()
            cnt = cnt.float().cpu()

            if cluster_id not in cluster_logit_conf_sum:
                cluster_logit_conf_sum[cluster_id] = lcs.clone()
                cluster_conf_sum[cluster_id] = cfs.clone()
                cluster_counts_sum[cluster_id] = cnt.clone()
            else:
                cluster_logit_conf_sum[cluster_id] += lcs
                cluster_conf_sum[cluster_id] += cfs
                cluster_counts_sum[cluster_id] += cnt

            if global_logit_conf_sum is None:
                global_logit_conf_sum = lcs.clone()
                global_conf_sum = cfs.clone()
                global_counts_sum = cnt.clone()
            else:
                global_logit_conf_sum += lcs
                global_conf_sum += cfs
                global_counts_sum += cnt

        cluster_proto_info = {}
        cluster_teacher_confidence = {}

        for cluster_id in cluster_logit_conf_sum.keys():
            conf_sum = cluster_conf_sum[cluster_id]
            counts = cluster_counts_sum[cluster_id]
            valid = (counts > 0) & (conf_sum > 0)

            prototypes = torch.zeros_like(cluster_logit_conf_sum[cluster_id])
            if torch.any(valid):
                denom = conf_sum.unsqueeze(1).clamp_min(1e-12)
                prototypes = cluster_logit_conf_sum[cluster_id] / denom

            class_conf_mean = conf_sum / counts.clamp_min(1.0)
            confidence = float(class_conf_mean[valid].mean().item()) if torch.any(valid) else 1.0
            # Clamp to a safe range for teacher weighting.
            confidence = max(0.05, min(1.0, confidence))

            cluster_proto_info[cluster_id] = {
                "prototypes": prototypes,
                "valid_mask": valid,
                "confidence": confidence,
            }
            cluster_teacher_confidence[cluster_id] = confidence

        global_proto_info = None
        if global_logit_conf_sum is not None:
            global_valid = (global_counts_sum > 0) & (global_conf_sum > 0)
            global_proto = torch.zeros_like(global_logit_conf_sum)
            if torch.any(global_valid):
                global_proto = global_logit_conf_sum / global_conf_sum.unsqueeze(1).clamp_min(1e-12)
            global_conf_mean = global_conf_sum / global_counts_sum.clamp_min(1.0)
            global_confidence = float(global_conf_mean[global_valid].mean().item()) if torch.any(global_valid) else 1.0
            global_confidence = max(0.05, min(1.0, global_confidence))
            global_proto_info = {
                "prototypes": global_proto,
                "valid_mask": global_valid,
                "confidence": global_confidence,
            }

        return cluster_proto_info, global_proto_info, cluster_teacher_confidence

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

    def send_cluster_gms(self, cluster_gm_states):
        total_broadcast_bytes = 0
        if not cluster_gm_states:
            return total_broadcast_bytes

        self._ensure_cluster_generalized_states()

        first_state = next(iter(cluster_gm_states.values()))
        gm_bytes = sum(v.numel() * v.element_size() for v in first_state.values())
        print(f"[FedCD] Broadcast Cluster GM Size: {gm_bytes / (1024**2):.2f} MB per client (in cluster)")

        for client in self.clients:
            cluster_id = self.cluster_map.get(client.id, 0)
            gm_state = cluster_gm_states.get(cluster_id, None)
            if gm_state is None:
                gm_state = self.cluster_generalized_states.get(cluster_id, None)
            if gm_state is None:
                gm_state = self._current_generalized_state()
            start_time = time.time()
            client.set_parameters({"gm_parts": gm_state})
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            total_broadcast_bytes += sum(v.numel() * v.element_size() for v in gm_state.values())

        return total_broadcast_bytes

    def _get_cluster_client_counts(self, received_pms):
        counts = {}
        for client_id, _ in received_pms:
            cluster_id = self.cluster_map.get(client_id, 0)
            counts[cluster_id] = counts.get(cluster_id, 0) + 1
        return counts

    def update_global_combiner(self, cluster_pms, cluster_counts):
        weighted_states = []
        weights = []
        for cluster_id, state in cluster_pms.items():
            combiner_state = {
                k.replace("combiner.", ""): v
                for k, v in state.items()
                if k.startswith("combiner.")
            }
            if not combiner_state:
                continue
            weight = float(cluster_counts.get(cluster_id, 1))
            if weight <= 0:
                continue
            weighted_states.append(combiner_state)
            weights.append(weight)

        if not weighted_states:
            return

        total_weight = sum(weights)
        avg_state = {}
        template = self.global_combiner.state_dict()
        for key in template.keys():
            acc = None
            for state, w in zip(weighted_states, weights):
                if key not in state:
                    continue
                contrib = state[key] * (w / total_weight)
                acc = contrib if acc is None else (acc + contrib)
            if acc is not None:
                avg_state[key] = acc

        if avg_state:
            self.global_combiner.load_state_dict(avg_state, strict=False)

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
            found_clusters = len(set(labels))
            if self.max_dynamic_clusters > 0 and found_clusters > self.max_dynamic_clusters:
                n_clusters = min(self.max_dynamic_clusters, len(client_ids))
                print(
                    f"[FedCD] Dynamic clustering produced {found_clusters} clusters; "
                    f"capping to {n_clusters} via K-Means."
                )
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
                labels = kmeans.fit_predict(X)
                found_clusters = len(set(labels))
            # Update num_clusters for logging/monitoring
            self.num_clusters = found_clusters
        else:
            n_clusters = min(self.num_clusters, len(client_ids))
            if n_clusters <= 1:
                return {cid: 0 for cid in client_ids}
            print(f"[FedCD] Using K-Means Clustering with n_clusters={n_clusters}")
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            labels = kmeans.fit_predict(X)

        return {cid: int(label) for cid, label in zip(client_ids, labels)}

    def _align_cluster_labels(self, prev_cluster_map, new_cluster_map):
        """
        Keep cluster IDs temporally stable by matching new labels to previous labels
        using maximum client overlap (Hungarian if available, otherwise greedy).
        """
        if not prev_cluster_map or not new_cluster_map:
            return new_cluster_map

        prev_labels = sorted(set(prev_cluster_map.values()))
        new_labels = sorted(set(new_cluster_map.values()))
        if not prev_labels or not new_labels:
            return new_cluster_map

        prev_groups = {label: set() for label in prev_labels}
        new_groups = {label: set() for label in new_labels}
        for cid, label in prev_cluster_map.items():
            prev_groups[label].add(cid)
        for cid, label in new_cluster_map.items():
            new_groups[label].add(cid)

        overlap = np.zeros((len(prev_labels), len(new_labels)), dtype=np.int64)
        for i, old_label in enumerate(prev_labels):
            for j, new_label in enumerate(new_labels):
                overlap[i, j] = len(prev_groups[old_label] & new_groups[new_label])

        remap = {}
        used_old = set()
        used_new = set()
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-overlap)
            for r, c in zip(row_ind, col_ind):
                if overlap[r, c] <= 0:
                    continue
                old_label = prev_labels[r]
                new_label = new_labels[c]
                remap[new_label] = old_label
                used_old.add(old_label)
                used_new.add(new_label)
        except Exception:
            pairs = []
            for i, old_label in enumerate(prev_labels):
                for j, new_label in enumerate(new_labels):
                    score = overlap[i, j]
                    if score > 0:
                        pairs.append((score, old_label, new_label))
            pairs.sort(reverse=True)
            for _, old_label, new_label in pairs:
                if old_label in used_old or new_label in used_new:
                    continue
                remap[new_label] = old_label
                used_old.add(old_label)
                used_new.add(new_label)

        next_label = (max(prev_labels) + 1) if prev_labels else 0
        assigned_targets = set(remap.values())
        for new_label in new_labels:
            if new_label in remap:
                continue
            while next_label in assigned_targets:
                next_label += 1
            remap[new_label] = next_label
            assigned_targets.add(next_label)

        aligned_cluster_map = {
            cid: int(remap[new_label])
            for cid, new_label in new_cluster_map.items()
        }

        if any(remap[nl] != nl for nl in new_labels):
            remap_msg = ", ".join(f"{nl}->{remap[nl]}" for nl in sorted(remap.keys()))
            print(f"[FedCD] Aligned cluster labels: {remap_msg}")

        return aligned_cluster_map

    def _build_global_test_loader(self):
        # Build one shared test subset so all clients are evaluated on exactly the same data.
        shared_test_data = []
        for client_id in range(self.num_clients):
            shared_test_data.extend(read_client_data(self.dataset, client_id, is_train=False))

        if len(shared_test_data) == 0:
            print("[FedCD] Global test set is empty. Skipping shared evaluation.")
            return None

        if self.global_test_samples > 0 and self.global_test_samples < len(shared_test_data):
            rng = random.Random(0)
            sample_indices = rng.sample(range(len(shared_test_data)), self.global_test_samples)
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

        print(f"[FedCD] Global Test Set Size: {len(shared_test_data)}")
        return torch.utils.data.DataLoader(shared_test_data, **loader_kwargs)

    # Backward-compatible alias
    def _build_common_test_loader(self):
        return self._build_global_test_loader()

    def evaluate_global_test_acc(self):
        if not self.eval_common_global or self.global_test_loader is None:
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
                for x, y in self.global_test_loader:
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

    def evaluate_gm_only_global_test_acc(self):
        if not self.eval_common_global or self.global_test_loader is None:
            return None

        device = self.device
        use_non_blocking = device == "cuda" and bool(getattr(self.args, "pin_memory", False))
        acc_sum = 0.0
        valid_clients = 0

        for client in self.clients:
            client.f_ext.to(device)
            client.generalized_module.to(device)
            client.f_ext.eval()
            client.generalized_module.eval()
            if client.generalized_adapter is not None:
                client.generalized_adapter.to(device)
                client.generalized_adapter.eval()

            total = 0
            correct = 0
            with torch.no_grad():
                for x, y in self.global_test_loader:
                    if type(x) == type([]):
                        x = x[0]
                    x = x.to(device, non_blocking=use_non_blocking)
                    y = y.to(device, non_blocking=use_non_blocking)

                    z = client.f_ext(x)
                    if z.dim() > 2:
                        z = torch.flatten(z, 1)
                    z_gm = client.generalized_adapter(z) if client.generalized_adapter is not None else z
                    logits = client.generalized_module(z_gm)

                    correct += (torch.argmax(logits, dim=1) == y).sum().item()
                    total += y.size(0)

            client.f_ext.to("cpu")
            client.generalized_module.to("cpu")
            if client.generalized_adapter is not None:
                client.generalized_adapter.to("cpu")
            if total > 0:
                acc_sum += correct / total
                valid_clients += 1

        if device == "cuda":
            torch.cuda.empty_cache()

        if valid_clients == 0:
            return None
        return acc_sum / valid_clients

    # Backward-compatible alias
    def evaluate_common_test_acc(self):
        return self.evaluate_global_test_acc()

    @staticmethod
    def average_state_dicts(states):
        avg_state = {}
        for key in states[0].keys():
            avg_state[key] = sum(state[key] for state in states) / len(states)
        return avg_state

    def aggregate_and_distill(self, cluster_pm_states, cluster_counts):
        print("Server: Ensemble distillation (teacher: GM_i+PM_i+combiner_i, student: frozen PM_k+combiner_k + trainable GM_k)...")

        if not cluster_pm_states:
            return {}

        def _build_teacher_components(device):
            teacher_components = {}
            for cluster_id, state in cluster_pm_states.items():
                base_weight = float(cluster_counts.get(cluster_id, 1))
                conf_gate = float(self.cluster_teacher_confidence.get(cluster_id, 1.0))
                weight = base_weight * conf_gate
                if weight <= 0:
                    continue

                personalized_module_state = {
                    k.replace("personalized_module.", ""): v.to(device)
                    for k, v in state.items()
                    if k.startswith("personalized_module.")
                }
                personalized_adapter_state = {
                    k.replace("personalized_adapter.", ""): v.to(device)
                    for k, v in state.items()
                    if k.startswith("personalized_adapter.")
                }
                if not personalized_module_state:
                    head_state = {
                        k.replace("head.", ""): v.to(device)
                        for k, v in state.items()
                        if k.startswith("head.")
                    }
                    final_state = {
                        k.replace("final.", ""): v.to(device)
                        for k, v in state.items()
                        if k.startswith("final.")
                    }
                    personalized_module_state = self._merge_legacy_module_state(
                        self.personalized_module,
                        head_state,
                        final_state,
                    )
                    if not personalized_adapter_state:
                        personalized_adapter_state = {
                            k.replace("adapter.", ""): v.to(device)
                            for k, v in state.items()
                            if k.startswith("adapter.")
                        }

                cluster_combiner_state = {
                    k.replace("combiner.", ""): v.to(device)
                    for k, v in state.items()
                    if k.startswith("combiner.")
                }

                gm_state = self.cluster_generalized_states.get(cluster_id, self._current_generalized_state())
                generalized_module_state = {
                    k.replace("generalized_module.", ""): v.to(device)
                    for k, v in gm_state.items()
                    if k.startswith("generalized_module.")
                }
                generalized_adapter_state = {
                    k.replace("generalized_adapter.", ""): v.to(device)
                    for k, v in gm_state.items()
                    if k.startswith("generalized_adapter.")
                }

                if not personalized_module_state or not cluster_combiner_state or not generalized_module_state:
                    continue

                gm_module = copy.deepcopy(self.generalized_module).to(device)
                gm_module.load_state_dict(generalized_module_state, strict=True)
                gm_module.eval()
                for p in gm_module.parameters():
                    p.requires_grad = False

                gm_adapter = None
                if self.generalized_adapter is not None:
                    gm_adapter = copy.deepcopy(self.generalized_adapter).to(device)
                    if generalized_adapter_state:
                        gm_adapter.load_state_dict(generalized_adapter_state, strict=True)
                    gm_adapter.eval()
                    for p in gm_adapter.parameters():
                        p.requires_grad = False

                pm_module = copy.deepcopy(self.personalized_module).to(device)
                pm_module.load_state_dict(personalized_module_state, strict=True)
                pm_module.eval()
                for p in pm_module.parameters():
                    p.requires_grad = False

                pm_adapter = None
                if self.personalized_adapter is not None:
                    pm_adapter = copy.deepcopy(self.personalized_adapter).to(device)
                    if personalized_adapter_state:
                        pm_adapter.load_state_dict(personalized_adapter_state, strict=True)
                    pm_adapter.eval()
                    for p in pm_adapter.parameters():
                        p.requires_grad = False

                cluster_combiner = copy.deepcopy(self.global_combiner).to(device)
                cluster_combiner.load_state_dict(cluster_combiner_state, strict=True)
                cluster_combiner.eval()
                for p in cluster_combiner.parameters():
                    p.requires_grad = False

                teacher_components[cluster_id] = {
                    "weight": weight,
                    "gm_module": gm_module,
                    "gm_adapter": gm_adapter,
                    "pm_module": pm_module,
                    "pm_adapter": pm_adapter,
                    "combiner": cluster_combiner,
                }
            return teacher_components

        def _distill_once(device):
            def _relation_matrix(x):
                if x.size(0) < 2:
                    return None
                x = F.normalize(x, p=2, dim=1)
                return torch.matmul(x, x.t())

            self.f_ext.to(device)
            self.f_ext.eval()
            use_amp = device == "cuda" and bool(getattr(self.args, "amp", False))
            temp = max(1e-6, self.distill_temp)
            kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
            ce_loss_fn = nn.CrossEntropyLoss()

            self._ensure_cluster_generalized_states(set(cluster_pm_states.keys()))
            teacher_components = _build_teacher_components(device)
            if not teacher_components:
                print("[FedCD] No valid GM+PM+combiner teachers. Skipping distillation.")
                return {}

            self._ensure_cluster_generalized_states(set(teacher_components.keys()))
            updated_cluster_gm_states = {}

            for target_cluster_id, target in teacher_components.items():
                gm_init_state = self.cluster_generalized_states.get(
                    target_cluster_id, self._current_generalized_state()
                )

                gm_module = copy.deepcopy(self.generalized_module).to(device)
                gm_module_state = {
                    k.replace("generalized_module.", ""): v.to(device)
                    for k, v in gm_init_state.items()
                    if k.startswith("generalized_module.")
                }
                if gm_module_state:
                    gm_module.load_state_dict(gm_module_state, strict=True)
                gm_module.train()

                gm_adapter = None
                if self.generalized_adapter is not None:
                    gm_adapter = copy.deepcopy(self.generalized_adapter).to(device)
                    gm_adapter_state = {
                        k.replace("generalized_adapter.", ""): v.to(device)
                        for k, v in gm_init_state.items()
                        if k.startswith("generalized_adapter.")
                    }
                    if gm_adapter_state:
                        gm_adapter.load_state_dict(gm_adapter_state, strict=True)
                    gm_adapter.train()

                target_pm_module = target["pm_module"]
                target_pm_adapter = target["pm_adapter"]
                target_combiner = target["combiner"]
                target_proto_info = self.cluster_proto_info.get(target_cluster_id, None)
                if target_proto_info is None:
                    target_proto_info = self.global_proto_info
                proto_targets = None
                proto_valid = None
                if target_proto_info is not None:
                    proto_targets = target_proto_info["prototypes"].to(device)
                    proto_valid = target_proto_info["valid_mask"].to(device)

                student_params = list(gm_module.parameters())
                if gm_adapter is not None:
                    student_params += list(gm_adapter.parameters())
                optimizer = torch.optim.SGD(student_params, lr=self.distill_lr)
                scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

                for x, _ in tqdm(
                    self.proxy_data_loader,
                    desc=f"Distill GM_k (cluster {target_cluster_id})",
                    leave=False,
                ):
                    x = x.to(device, non_blocking=(device == "cuda"))

                    with torch.no_grad():
                        # Shared frozen feature extractor for both teacher and student.
                        z = self.f_ext(x)
                        if z.dim() > 2:
                            z = torch.flatten(z, 1)

                        # Teacher: ensemble over (GM_i + PM_i + combiner_i) with frozen branches.
                        teacher_prob_sum = None
                        total_weight = 0.0
                        for cluster_id, comp in teacher_components.items():
                            # Exclude self-teacher (i == k): GM_k is distilled only from other clusters.
                            if cluster_id == target_cluster_id:
                                continue
                            gm_adapter_i = comp["gm_adapter"]
                            gm_module_i = comp["gm_module"]
                            pm_adapter_i = comp["pm_adapter"]
                            pm_module_i = comp["pm_module"]
                            combiner_i = comp["combiner"]
                            weight_i = comp["weight"]

                            z_gm_i = gm_adapter_i(z) if gm_adapter_i is not None else z
                            gm_logits_i = gm_module_i(z_gm_i)
                            z_pm_i = pm_adapter_i(z) if pm_adapter_i is not None else z
                            pm_logits_i = pm_module_i(z_pm_i)

                            teacher_input_i = torch.cat([gm_logits_i, pm_logits_i], dim=1)
                            teacher_logits_i = combiner_i(teacher_input_i)
                            teacher_prob_i = torch.softmax(teacher_logits_i / temp, dim=1)

                            total_weight += weight_i
                            if teacher_prob_sum is None:
                                teacher_prob_sum = teacher_prob_i * weight_i
                            else:
                                teacher_prob_sum += teacher_prob_i * weight_i

                        if teacher_prob_sum is None or total_weight <= 0:
                            continue
                        teacher_prob_ensemble = teacher_prob_sum / total_weight
                        # Student fixed PM_k branch.
                        z_pm_k = target_pm_adapter(z) if target_pm_adapter is not None else z
                        target_pm_logits = target_pm_module(z_pm_k)

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        # Student: frozen (f_ext, PM_k, combiner_k) + trainable GM_k.
                        z_gm = gm_adapter(z) if gm_adapter is not None else z
                        gm_logits = gm_module(z_gm)
                        student_input = torch.cat([gm_logits, target_pm_logits.detach()], dim=1)
                        student_logits = target_combiner(student_input)

                        student_log_prob = torch.log_softmax(student_logits / temp, dim=1)
                        kd_loss = kl_loss_fn(student_log_prob, teacher_prob_ensemble) * (temp * temp)
                        ce_loss = student_logits.new_tensor(0.0)
                        pseudo_labels = torch.argmax(teacher_prob_ensemble, dim=1)
                        if self.distill_ce_weight > 0:
                            ce_loss = ce_loss_fn(student_logits, pseudo_labels)
                        proto_loss = student_logits.new_tensor(0.0)
                        relation_loss = student_logits.new_tensor(0.0)
                        if proto_targets is not None and proto_valid is not None and self.proto_weight > 0:
                            target_logits_proto = proto_targets.to(dtype=gm_logits.dtype)
                            cls_losses = []
                            cls_centers = []
                            proto_centers = []
                            for cls in range(self.num_classes):
                                if not bool(proto_valid[cls].item()):
                                    continue
                                cls_mask = (pseudo_labels == cls)
                                if not torch.any(cls_mask):
                                    continue
                                gm_center = gm_logits[cls_mask].mean(dim=0)
                                proto_center = target_logits_proto[cls]
                                cls_losses.append(F.mse_loss(gm_center, proto_center))
                                if self.relation_weight > 0:
                                    cls_centers.append(gm_center)
                                    proto_centers.append(proto_center)
                            if cls_losses:
                                proto_loss = torch.stack(cls_losses).mean()
                            if self.relation_weight > 0 and len(cls_centers) >= 2:
                                student_rel = _relation_matrix(torch.stack(cls_centers, dim=0))
                                target_rel = _relation_matrix(torch.stack(proto_centers, dim=0))
                                if student_rel is not None and target_rel is not None:
                                    relation_loss = F.mse_loss(student_rel, target_rel)

                        loss = (
                            self.distill_kl_weight * kd_loss
                            + self.distill_ce_weight * ce_loss
                            + self.proto_weight * proto_loss
                            + self.relation_weight * relation_loss
                        )

                    if not torch.isfinite(loss):
                        optimizer.zero_grad()
                        continue

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                distilled_state = {
                    f"generalized_module.{k}": v.detach().cpu()
                    for k, v in gm_module.state_dict().items()
                }
                if gm_adapter is not None:
                    distilled_state.update({
                        f"generalized_adapter.{k}": v.detach().cpu()
                        for k, v in gm_adapter.state_dict().items()
                    })
                    gm_adapter.to("cpu")
                gm_module.to("cpu")
                updated_cluster_gm_states[target_cluster_id] = distilled_state

            for comp in teacher_components.values():
                comp["gm_module"].to("cpu")
                if comp["gm_adapter"] is not None:
                    comp["gm_adapter"].to("cpu")
                comp["pm_module"].to("cpu")
                if comp["pm_adapter"] is not None:
                    comp["pm_adapter"].to("cpu")
                comp["combiner"].to("cpu")

            self.f_ext.to("cpu")
            if device == "cuda" and self.args.avoid_oom:
                torch.cuda.empty_cache()

            return updated_cluster_gm_states

        try:
            cluster_gm_states = _distill_once(self.device)
        except RuntimeError as err:
            if self.device == "cuda" and "out of memory" in str(err).lower():
                print("[Warn] OOM during cluster GM distillation. Falling back to CPU.")
                torch.cuda.empty_cache()
                cluster_gm_states = _distill_once("cpu")
            else:
                raise

        if not cluster_gm_states:
            return {}

        self.cluster_generalized_states.update(cluster_gm_states)
        avg_state = self._weighted_average_generalized_states(
            self.cluster_generalized_states,
            {cid: cluster_counts.get(cid, 1) for cid in self.cluster_generalized_states.keys()},
        )
        if avg_state:
            self._apply_generalized_state_to_server(avg_state)

        return cluster_gm_states
            
    def load_proxy_data(self):
        proxy_dataset_name = getattr(self.args, "proxy_dataset", "TinyImagenet")
        proxy_dataset_key = str(proxy_dataset_name).lower()
        proxy_samples = int(getattr(self.args, "proxy_samples", 1000))

        print(f"[FedCD] Loading Proxy Data: {proxy_dataset_name} (Samples: {proxy_samples})")

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        tiny_path = os.path.join(base_dir, "dataset", "TinyImagenet", "rawdata", "tiny-imagenet-200", "train")
        cifar100_root = os.path.join(base_dir, "dataset", "Cifar100", "rawdata")

        if proxy_dataset_key in {"tinyimagenet", "tiny-imagenet"} and os.path.exists(tiny_path):
            from torchvision.datasets import ImageFolder
            import torchvision.transforms as transforms
            from torch.utils.data import Subset
            import random

            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            full_dataset = ImageFolder(root=tiny_path, transform=transform)
            if proxy_samples > 0 and proxy_samples < len(full_dataset):
                indices = random.sample(range(len(full_dataset)), proxy_samples)
                proxy_dataset = Subset(full_dataset, indices)
            else:
                proxy_dataset = full_dataset
            print(f"[FedCD] Successfully loaded {len(proxy_dataset)} samples from TinyImageNet.")

        elif proxy_dataset_key in {"cifar100", "cifar-100"}:
            from torchvision.datasets import CIFAR100
            import torchvision.transforms as transforms
            from torch.utils.data import Subset
            import random

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            try:
                try:
                    full_dataset = CIFAR100(
                        root=cifar100_root, train=True, download=False, transform=transform
                    )
                except RuntimeError:
                    print(f"[Warn] CIFAR-100 cache not found at {cifar100_root}. Trying download...")
                    full_dataset = CIFAR100(
                        root=cifar100_root, train=True, download=True, transform=transform
                    )

                if proxy_samples > 0 and proxy_samples < len(full_dataset):
                    indices = random.sample(range(len(full_dataset)), proxy_samples)
                    proxy_dataset = Subset(full_dataset, indices)
                else:
                    proxy_dataset = full_dataset
                print(f"[FedCD] Successfully loaded {len(proxy_dataset)} samples from CIFAR-100.")
            except Exception as err:
                print(f"[Warn] Failed to load CIFAR-100 proxy ({err}). Fallback to union test data.")
                from utils.data_utils import read_client_data
                test_data = []
                for cid in range(self.num_clients):
                    test_data.extend(read_client_data(self.args.dataset, cid, is_train=False))
                if proxy_samples > 0 and proxy_samples < len(test_data):
                    proxy_dataset = random.sample(test_data, proxy_samples)
                else:
                    proxy_dataset = test_data
                print(f"[FedCD] Fallback proxy samples loaded: {len(proxy_dataset)}")

        else:
            if proxy_dataset_key in {"tinyimagenet", "tiny-imagenet"}:
                print(
                    f"[Warn] TinyImageNet raw data not found at {tiny_path}. "
                    "Fallback to union of all clients' test data."
                )

            from utils.data_utils import read_client_data
            import random
            test_data = []
            for cid in range(self.num_clients):
                test_data.extend(read_client_data(self.args.dataset, cid, is_train=False))
            if proxy_samples > 0 and proxy_samples < len(test_data):
                proxy_dataset = random.sample(test_data, proxy_samples)
            else:
                proxy_dataset = test_data
            print(f"[FedCD] Fallback proxy samples loaded: {len(proxy_dataset)}")

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
