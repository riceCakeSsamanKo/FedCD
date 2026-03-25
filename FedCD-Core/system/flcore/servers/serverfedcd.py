import copy
import time
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torchvision
from flcore.trainmodel.models import SmallFExt, TinyFExt
from flcore.clients.clientfedcd import clientFedCD
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data 

class FeatureSpaceGenerator(nn.Module):
    def __init__(self, feature_dim, noise_dim=128, hidden_dim=512):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.noise_dim = int(noise_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim + self.noise_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )

    def forward(self, noise, anchor=None):
        if anchor is None:
            anchor = torch.zeros(
                noise.size(0),
                self.feature_dim,
                device=noise.device,
                dtype=noise.dtype,
            )
        delta = self.net(torch.cat([noise, anchor], dim=1))
        return anchor + delta

class FedCD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()

        # 클라이언트 클래스 연결
        self.set_clients(clientFedCD)
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        # FedCD clustering config
        self.enable_clustering = bool(getattr(args, "fedcd_enable_clustering", True))
        self.enable_pm_aggregation = bool(getattr(args, "fedcd_enable_pm_aggregation", True))
        self.num_clusters = max(1, int(getattr(args, "num_clusters", 1)))
        if not self.enable_clustering:
            self.num_clusters = 1
        self.max_dynamic_clusters = max(0, int(getattr(args, "max_dynamic_clusters", self.num_clusters)))
        self.cluster_period = max(1, int(getattr(args, "cluster_period", 1)))
        self.pm_period = max(1, int(getattr(args, "pm_period", 1)))
        self.global_period = max(1, int(getattr(args, "global_period", 1)))
        self.cluster_sample_size = int(getattr(args, "cluster_sample_size", 512))
        self.cluster_map = {c.id: (c.id % self.num_clusters) for c in self.clients}
        self.client_distribution_stats = {}
        self.cluster_distribution_stats = {}
        if not self.enable_clustering:
            self.cluster_map = {c.id: 0 for c in self.clients}
            self.max_dynamic_clusters = 0
            print("[FedCD] Clustering disabled: all clients are treated as one cluster.")
        if not self.enable_pm_aggregation:
            print("[FedCD] PM aggregation disabled: PM stays local per client.")
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
        self.last_global_gate_stats = None
        self.last_local_gate_stats = None
        self.router_server_distill_enable = bool(
            getattr(args, "fedcd_router_server_distill_enable", True)
        )
        if self.router_server_distill_enable:
            any_router = any(getattr(c, "router", None) is not None for c in self.clients)
            if not any_router:
                self.router_server_distill_enable = False
                print("[FedCD] Router-context distillation disabled: no client router is enabled.")
        self.router_server_samples = int(getattr(args, "fedcd_router_server_samples", 256))
        self.router_server_samples = max(0, self.router_server_samples)
        self.router_server_period = int(getattr(args, "fedcd_router_server_period", 1))
        self.router_server_period = max(1, self.router_server_period)
        self.router_server_neg_mode = str(
            getattr(args, "fedcd_router_server_neg_mode", "all_other")
        ).strip().lower()
        if self.router_server_neg_mode not in {"all_other", "farthest_k"}:
            self.router_server_neg_mode = "all_other"
        self.router_server_neg_topk = int(getattr(args, "fedcd_router_server_neg_topk", 2))
        self.router_server_neg_topk = max(0, self.router_server_neg_topk)
        self.generalized_module = self._extract_module(self.global_model)
        pm_model = getattr(args, "pm_model", None)
        if pm_model is None:
            pm_model = copy.deepcopy(self.global_model)
        self.personalized_module = self._extract_module(pm_model)
        target_fext_dim = self._resolve_module_input_dim(
            self.generalized_module,
            self.personalized_module,
        )
        self.f_ext = self._build_f_ext(args, target_dim=target_fext_dim)
        # Adapter is intentionally disabled.
        self.generalized_adapter = None
        self.personalized_adapter = None
        # Cluster-wise GM states (each cluster keeps its own distilled GM).
        self.cluster_generalized_states = {}
        self._initialize_cluster_generalized_states()
        self.gm_update_mode = str(getattr(args, "fedcd_gm_update_mode", "local")).strip().lower()
        if self.gm_update_mode not in {
            "local",
            "server_pm_teacher",
            "server_pm_fedavg",
            "server_pm_subnet",
            "server_proto_teacher",
            "hybrid_local_proto",
        }:
            raise ValueError(f"Unknown fedcd_gm_update_mode: {self.gm_update_mode}")
        self.pm_to_gm_mask_enable = bool(
            getattr(args, "fedcd_pm_to_gm_mask_enable", False)
        )
        self.pm_to_gm_mask_unified = bool(
            getattr(args, "fedcd_pm_to_gm_mask_unified", True)
        )
        self._pm_to_gm_hidden_mask_cache = {}
        self.hybrid_proto_blend = float(getattr(args, "fedcd_hybrid_proto_blend", 0.35))
        self.hybrid_proto_blend = min(max(self.hybrid_proto_blend, 0.0), 1.0)
        self.pm_teacher_lr = float(getattr(args, "fedcd_pm_teacher_lr", 0.01))
        self.pm_teacher_temp = float(getattr(args, "fedcd_pm_teacher_temp", 2.0))
        self.pm_teacher_kl_weight = float(getattr(args, "fedcd_pm_teacher_kl_weight", 1.0))
        self.pm_teacher_ce_weight = float(getattr(args, "fedcd_pm_teacher_ce_weight", 0.2))
        self.pm_teacher_samples = int(getattr(args, "fedcd_pm_teacher_samples", 2000))
        self.pm_teacher_batch_size = int(getattr(args, "fedcd_pm_teacher_batch_size", 256))
        self.pm_teacher_epochs = max(1, int(getattr(args, "fedcd_pm_teacher_epochs", 1)))
        self.pm_teacher_proxy_dataset = str(getattr(args, "fedcd_pm_teacher_proxy_dataset", "Cifar100"))
        self.pm_teacher_proxy_root = str(getattr(args, "fedcd_pm_teacher_proxy_root", ""))
        self.pm_teacher_proxy_split = str(getattr(args, "fedcd_pm_teacher_proxy_split", "train"))
        self.pm_teacher_proxy_download = bool(getattr(args, "fedcd_pm_teacher_proxy_download", False))
        self.pm_teacher_allow_test_fallback = bool(getattr(args, "fedcd_pm_teacher_allow_test_fallback", False))
        self.pm_teacher_source = str(getattr(args, "fedcd_pm_teacher_source", "cluster")).strip().lower()
        if self.pm_teacher_source not in {"cluster", "client"}:
            raise ValueError(
                "fedcd_pm_teacher_source must be one of: cluster | client "
                f"(got: {self.pm_teacher_source})"
            )
        self.pm_teacher_confidence_weight = bool(getattr(args, "fedcd_pm_teacher_confidence_weight", True))
        self.pm_teacher_confidence_min = float(getattr(args, "fedcd_pm_teacher_confidence_min", 0.05))
        self.pm_teacher_confidence_min = min(max(self.pm_teacher_confidence_min, 0.0), 1.0)
        self.pm_teacher_confidence_power = float(getattr(args, "fedcd_pm_teacher_confidence_power", 1.0))
        self.pm_teacher_confidence_power = max(self.pm_teacher_confidence_power, 1e-6)
        self.pm_teacher_ensemble_confidence = bool(
            getattr(args, "fedcd_pm_teacher_ensemble_confidence", True)
        )
        self.pm_teacher_topk = int(getattr(args, "fedcd_pm_teacher_topk", 0))
        self.pm_teacher_topk = max(0, self.pm_teacher_topk)
        self.pm_teacher_abstain_threshold = float(
            getattr(args, "fedcd_pm_teacher_abstain_threshold", 0.0)
        )
        self.pm_teacher_abstain_threshold = min(max(self.pm_teacher_abstain_threshold, 0.0), 1.0)
        self.pm_teacher_teacher_abstain_threshold = float(
            getattr(args, "fedcd_pm_teacher_teacher_abstain_threshold", 0.0)
        )
        self.pm_teacher_teacher_abstain_threshold = min(
            max(self.pm_teacher_teacher_abstain_threshold, 0.0), 1.0
        )
        self.pm_teacher_min_active_teachers = int(
            getattr(args, "fedcd_pm_teacher_min_active_teachers", 1)
        )
        self.pm_teacher_min_active_teachers = max(1, self.pm_teacher_min_active_teachers)
        self.pm_teacher_consensus_min_ratio = float(
            getattr(args, "fedcd_pm_teacher_consensus_min_ratio", 0.0)
        )
        self.pm_teacher_consensus_min_ratio = min(max(self.pm_teacher_consensus_min_ratio, 0.0), 1.0)
        self.pm_teacher_correct_only = bool(
            getattr(args, "fedcd_pm_teacher_correct_only", False)
        )
        if self.pm_teacher_correct_only:
            print(
                "[FedCD] fedcd_pm_teacher_correct_only=True: "
                "use a label-compatible proxy dataset (e.g., Cifar10 for Cifar10 task)."
            )
        self.pm_teacher_rel_weight = float(getattr(args, "fedcd_pm_teacher_rel_weight", 0.2))
        self.pm_teacher_rel_batch = int(getattr(args, "fedcd_pm_teacher_rel_batch", 64))
        self.pm_teacher_competence_filter = bool(
            getattr(args, "fedcd_pm_teacher_competence_filter", False)
        )
        self.pm_teacher_competence_rel_weight = max(
            0.0, float(getattr(args, "fedcd_pm_teacher_competence_rel_weight", 0.5))
        )
        self.pm_teacher_competence_support_weight = max(
            0.0, float(getattr(args, "fedcd_pm_teacher_competence_support_weight", 0.2))
        )
        self.pm_teacher_competence_proto_weight = max(
            0.0, float(getattr(args, "fedcd_pm_teacher_competence_proto_weight", 0.3))
        )
        self.pm_teacher_competence_min = float(
            getattr(args, "fedcd_pm_teacher_competence_min", 0.35)
        )
        self.pm_teacher_competence_min = min(max(self.pm_teacher_competence_min, 0.0), 1.0)
        self.pm_teacher_cluster_dist_weighting = bool(
            getattr(args, "fedcd_pm_teacher_cluster_dist_weighting", False)
        )
        self.pm_teacher_cluster_dist_tau = max(
            1e-6, float(getattr(args, "fedcd_pm_teacher_cluster_dist_tau", 1.0))
        )
        self.pm_teacher_cluster_dist_metric = str(
            getattr(args, "fedcd_pm_teacher_cluster_dist_metric", "mahalanobis")
        ).strip().lower()
        if self.pm_teacher_cluster_dist_metric not in {"mahalanobis", "euclidean"}:
            self.pm_teacher_cluster_dist_metric = "mahalanobis"
        self.pm_teacher_datafree_enable = bool(
            getattr(args, "fedcd_pm_teacher_datafree_enable", False)
        )
        self.pm_teacher_datafree_batches = max(
            1, int(getattr(args, "fedcd_pm_teacher_datafree_batches", 32))
        )
        self.pm_teacher_datafree_steps = max(
            1, int(getattr(args, "fedcd_pm_teacher_datafree_steps", 20))
        )
        self.pm_teacher_datafree_lr = max(
            1e-6, float(getattr(args, "fedcd_pm_teacher_datafree_lr", 0.05))
        )
        self.pm_teacher_datafree_noise_scale = max(
            0.0, float(getattr(args, "fedcd_pm_teacher_datafree_noise_scale", 0.25))
        )
        self.pm_teacher_datafree_entropy_weight = max(
            0.0, float(getattr(args, "fedcd_pm_teacher_datafree_entropy_weight", 1.0))
        )
        self.pm_teacher_datafree_diversity_weight = max(
            0.0, float(getattr(args, "fedcd_pm_teacher_datafree_diversity_weight", 0.5))
        )
        self.pm_teacher_datafree_l2_weight = max(
            0.0, float(getattr(args, "fedcd_pm_teacher_datafree_l2_weight", 1e-4))
        )
        self.pm_teacher_datafree_init_from_proto = bool(
            getattr(args, "fedcd_pm_teacher_datafree_init_from_proto", True)
        )
        self.pm_teacher_datafree_generator_enable = bool(
            getattr(args, "fedcd_pm_teacher_datafree_generator_enable", False)
        )
        self.pm_teacher_datafree_generator_noise_dim = max(
            1, int(getattr(args, "fedcd_pm_teacher_datafree_generator_noise_dim", 128))
        )
        self.pm_teacher_datafree_generator_hidden_dim = max(
            1, int(getattr(args, "fedcd_pm_teacher_datafree_generator_hidden_dim", 512))
        )
        self.pm_teacher_datafree_generator_lr = max(
            1e-6, float(getattr(args, "fedcd_pm_teacher_datafree_generator_lr", 1e-3))
        )
        self.pm_teacher_datafree_generator_steps = max(
            1, int(getattr(args, "fedcd_pm_teacher_datafree_generator_steps", 1))
        )
        self.pm_teacher_datafree_generator_anchor_weight = max(
            0.0, float(getattr(args, "fedcd_pm_teacher_datafree_generator_anchor_weight", 0.1))
        )
        self.pm_teacher_feature_dim = int(target_fext_dim) if target_fext_dim is not None else None
        self.pm_teacher_feature_generator = None
        if self.gm_update_mode == "server_pm_teacher":
            print(f"[FedCD] PM-teacher source for GM distillation: {self.pm_teacher_source}")
            if self.pm_teacher_datafree_enable:
                print(
                    "[FedCD] PM-teacher data-free KD enabled: "
                    f"batches={self.pm_teacher_datafree_batches}, "
                    f"steps={self.pm_teacher_datafree_steps}, "
                    f"lr={self.pm_teacher_datafree_lr:.4f}"
                )
                if self.pm_teacher_datafree_generator_enable:
                    print(
                        "[FedCD] PM-teacher feature generator enabled: "
                        f"noise_dim={self.pm_teacher_datafree_generator_noise_dim}, "
                        f"hidden_dim={self.pm_teacher_datafree_generator_hidden_dim}, "
                        f"lr={self.pm_teacher_datafree_generator_lr:.4f}, "
                        f"steps={self.pm_teacher_datafree_generator_steps}"
                    )
                if self.pm_teacher_ce_weight > 0:
                    print(
                        "[FedCD] PM-teacher data-free KD ignores label CE loss "
                        "(no proxy labels are used)."
                    )
            if self.pm_teacher_source == "cluster" and self.pm_teacher_cluster_dist_weighting:
                print(
                    "[FedCD] PM-teacher cluster distance weighting enabled: "
                    f"metric={self.pm_teacher_cluster_dist_metric}, tau={self.pm_teacher_cluster_dist_tau:.4f}"
                )
        self.init_pretrain = bool(getattr(args, "fedcd_init_pretrain", True))
        self.init_epochs = max(0, int(getattr(args, "fedcd_init_epochs", 1)))
        self.init_lr = float(getattr(args, "fedcd_init_lr", 0.005))
        self.init_samples = int(getattr(args, "fedcd_init_samples", 2000))
        self.init_batch_size = int(getattr(args, "fedcd_init_batch_size", 256))
        self.init_ce_weight = float(getattr(args, "fedcd_init_ce_weight", 1.0))
        self.init_kd_weight = float(getattr(args, "fedcd_init_kd_weight", 1.0))
        self.init_entropy_weight = float(getattr(args, "fedcd_init_entropy_weight", 0.05))
        self.init_diversity_weight = float(getattr(args, "fedcd_init_diversity_weight", 0.05))
        self.proto_teacher_lr = float(getattr(args, "fedcd_proto_teacher_lr", 0.01))
        self.proto_teacher_steps = int(getattr(args, "fedcd_proto_teacher_steps", 200))
        self.proto_teacher_batch_size = int(getattr(args, "fedcd_proto_teacher_batch_size", 256))
        self.proto_teacher_temp = float(getattr(args, "fedcd_proto_teacher_temp", 2.0))
        self.proto_teacher_ce_weight = float(getattr(args, "fedcd_proto_teacher_ce_weight", 1.0))
        self.proto_teacher_kl_weight = float(getattr(args, "fedcd_proto_teacher_kl_weight", 0.5))
        self.proto_teacher_noise_scale = float(getattr(args, "fedcd_proto_teacher_noise_scale", 1.0))
        self.proto_teacher_min_count = float(getattr(args, "fedcd_proto_teacher_min_count", 1.0))
        self.proto_teacher_client_samples = int(getattr(args, "fedcd_proto_teacher_client_samples", 0))
        self.proto_teacher_confidence_weight = bool(getattr(args, "fedcd_proto_teacher_confidence_weight", True))
        self.proto_teacher_confidence_min = float(getattr(args, "fedcd_proto_teacher_confidence_min", 0.05))
        self.proto_teacher_confidence_min = min(max(self.proto_teacher_confidence_min, 0.0), 1.0)
        self.proto_teacher_confidence_power = float(getattr(args, "fedcd_proto_teacher_confidence_power", 1.0))
        self.proto_teacher_confidence_power = max(self.proto_teacher_confidence_power, 1e-6)
        self.pm_teacher_loader = None
        if self.gm_update_mode == "server_pm_teacher" and not self.pm_teacher_datafree_enable:
            self.pm_teacher_loader = self._build_pm_teacher_loader()
        if self.init_pretrain and self.init_epochs > 0:
            self._pretrain_and_broadcast_initial_components()

        # [ACT] Adaptive Clustering Threshold Initialization
        self.adaptive_threshold = bool(getattr(args, "adaptive_threshold", False)) and self.enable_clustering
        self.current_threshold = float(getattr(args, "cluster_threshold", 0.0))
        # If ACT is enabled but initial threshold is 0, start with a small value
        if self.adaptive_threshold and self.current_threshold <= 0:
            self.current_threshold = 0.05
            
        self.threshold_step = float(getattr(args, "threshold_step", 0.01))
        self.threshold_step_max = float(getattr(args, "threshold_step_max", 0.1))
        self.threshold_step_max = max(self.threshold_step_max, 1e-6)
        self.threshold_step = min(abs(self.threshold_step), self.threshold_step_max)
        self.threshold_decay = float(getattr(args, "threshold_decay", 0.9))
        self.threshold_max = float(getattr(args, "threshold_max", 0.95))
        
        # [ACT] Zig-Zag Convergence State
        self.act_direction = 1 # 1: increase, -1: decrease
        self.acc_history = [] # Stores mean accuracy for regression
        self.window_size = int(getattr(args, "act_window_size", 5))

        # Automated search-time early stop controls.
        self.search_enable = bool(getattr(args, "fedcd_search_enable", False))
        self.search_min_rounds = max(1, int(getattr(args, "fedcd_search_min_rounds", 8)))
        self.search_patience = max(1, int(getattr(args, "fedcd_search_patience", 6)))
        self.search_drop_patience = max(1, int(getattr(args, "fedcd_search_drop_patience", 3)))
        self.search_drop_delta = max(0.0, float(getattr(args, "fedcd_search_drop_delta", 0.003)))
        self.search_score_gm_weight = float(getattr(args, "fedcd_search_score_gm_weight", 0.75))
        self.search_score_pm_weight = float(getattr(args, "fedcd_search_score_pm_weight", 0.25))
        self.search_score_eps = max(0.0, float(getattr(args, "fedcd_search_score_eps", 1e-4)))
        self.search_min_pm_local_acc = float(getattr(args, "fedcd_search_min_pm_local_acc", 0.55))
        self.search_min_gm_global_acc = float(getattr(args, "fedcd_search_min_gm_global_acc", 0.18))
        self.search_best_score = float("-inf")
        self.search_no_improve_rounds = 0
        self.search_dual_drop_rounds = 0
        self.search_prev_pm_local = None
        self.search_prev_gm_global = None

    def _search_score(self, pm_local_test_acc, gm_only_global_test_acc):
        pm = 0.0 if pm_local_test_acc is None else float(pm_local_test_acc)
        gm = 0.0 if gm_only_global_test_acc is None else float(gm_only_global_test_acc)
        return self.search_score_pm_weight * pm + self.search_score_gm_weight * gm

    def _should_stop_for_search(self, round_idx, pm_local_test_acc, gm_only_global_test_acc):
        if not self.search_enable:
            return False
        if pm_local_test_acc is None or gm_only_global_test_acc is None:
            return False
        if round_idx < self.search_min_rounds:
            self.search_prev_pm_local = float(pm_local_test_acc)
            self.search_prev_gm_global = float(gm_only_global_test_acc)
            return False

        score = self._search_score(pm_local_test_acc, gm_only_global_test_acc)
        if score > (self.search_best_score + self.search_score_eps):
            self.search_best_score = score
            self.search_no_improve_rounds = 0
        else:
            self.search_no_improve_rounds += 1

        dual_drop = False
        if self.search_prev_pm_local is not None and self.search_prev_gm_global is not None:
            pm_drop = float(pm_local_test_acc) < (self.search_prev_pm_local - self.search_drop_delta)
            gm_drop = float(gm_only_global_test_acc) < (self.search_prev_gm_global - self.search_drop_delta)
            dual_drop = pm_drop and gm_drop
        if dual_drop:
            self.search_dual_drop_rounds += 1
        else:
            self.search_dual_drop_rounds = 0

        self.search_prev_pm_local = float(pm_local_test_acc)
        self.search_prev_gm_global = float(gm_only_global_test_acc)

        hard_fail = (
            float(pm_local_test_acc) < self.search_min_pm_local_acc
            and float(gm_only_global_test_acc) < self.search_min_gm_global_acc
        )
        if hard_fail:
            print(
                "[FedCD][Search] Early stop: both PM-local and GM-global are below floors "
                f"({pm_local_test_acc:.4f} < {self.search_min_pm_local_acc:.4f}, "
                f"{gm_only_global_test_acc:.4f} < {self.search_min_gm_global_acc:.4f})."
            )
            return True

        if self.search_no_improve_rounds >= self.search_patience:
            print(
                "[FedCD][Search] Early stop: no score improvement for "
                f"{self.search_no_improve_rounds} eval rounds."
            )
            return True

        if self.search_dual_drop_rounds >= self.search_drop_patience:
            print(
                "[FedCD][Search] Early stop: PM-local and GM-global dropped together for "
                f"{self.search_dual_drop_rounds} consecutive eval rounds."
            )
            return True

        return False

    def _build_f_ext(self, args, target_dim=None):
        model_name = str(getattr(args, "fext_model", "SmallFExt"))
        if model_name == "VGG16":
            # Load Pretrained VGG16 features
            base_model = torchvision.models.vgg16(pretrained=True)
            f_ext = nn.Sequential(base_model.features, base_model.avgpool, nn.Flatten())
            f_ext.out_dim = 512 * 7 * 7 # VGG16 final feature map size
        elif model_name == "SmallFExt":
            in_channels = 1 if "MNIST" in args.dataset else 3
            if target_dim is not None:
                fext_dim = int(target_dim)
            else:
                fext_dim = int(getattr(args, "fext_dim", 512))
            f_ext = SmallFExt(in_channels=in_channels, out_dim=fext_dim)
        elif model_name == "TinyFExt":
            in_channels = 1 if "MNIST" in args.dataset else 3
            if target_dim is not None:
                fext_dim = int(target_dim)
            else:
                fext_dim = int(getattr(args, "fext_dim", 256))
            f_ext = TinyFExt(in_channels=in_channels, out_dim=fext_dim)
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

    def _resolve_module_input_dim(self, *modules):
        dims = [self._first_linear_in_features(m) for m in modules]
        dims = [d for d in dims if d is not None]
        if not dims:
            return None
        unique_dims = sorted(set(int(d) for d in dims))
        if len(unique_dims) != 1:
            raise ValueError(
                "Adapter-free FedCD requires identical GM/PM classifier input dims, "
                f"but got {unique_dims}."
            )
        return unique_dims[0]

    def _ensure_pm_teacher_feature_generator(self, feature_dim):
        feature_dim = int(feature_dim)
        if (
            self.pm_teacher_feature_generator is None
            or getattr(self.pm_teacher_feature_generator, "feature_dim", None) != feature_dim
            or getattr(self.pm_teacher_feature_generator, "noise_dim", None) != self.pm_teacher_datafree_generator_noise_dim
            or getattr(self.pm_teacher_feature_generator, "hidden_dim", None) != self.pm_teacher_datafree_generator_hidden_dim
        ):
            self.pm_teacher_feature_generator = FeatureSpaceGenerator(
                feature_dim=feature_dim,
                noise_dim=self.pm_teacher_datafree_generator_noise_dim,
                hidden_dim=self.pm_teacher_datafree_generator_hidden_dim,
            )
        return self.pm_teacher_feature_generator

    def _current_generalized_state(self):
        state = {
            f"f_ext.{k}": v.detach().cpu()
            for k, v in self.f_ext.state_dict().items()
        }
        state.update({
            f"generalized_module.{k}": v.detach().cpu()
            for k, v in self.generalized_module.state_dict().items()
        })
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
        f_ext_state = {
            k.replace("f_ext.", ""): v
            for k, v in generalized_state.items()
            if k.startswith("f_ext.")
        }
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
        if f_ext_state:
            self.f_ext.load_state_dict(f_ext_state, strict=True)
        if module_state:
            self.generalized_module.load_state_dict(module_state, strict=True)
        if self.generalized_adapter is not None and adapter_state:
            self.generalized_adapter.load_state_dict(adapter_state, strict=True)

    def _build_gm_broadcast_parts(self):
        # Broadcast shared task encoder + GM components.
        parts = {}
        parts.update({
            f"f_ext.{k}": v.detach().cpu()
            for k, v in self.f_ext.state_dict().items()
        })
        parts.update({
            f"generalized_module.{k}": v.detach().cpu()
            for k, v in self.generalized_module.state_dict().items()
        })
        if self.generalized_adapter is not None:
            parts.update({
                f"generalized_adapter.{k}": v.detach().cpu()
                for k, v in self.generalized_adapter.state_dict().items()
            })
        return parts

    def _build_initial_parts(self):
        parts = {}
        parts.update({
            f"f_ext.{k}": v.detach().cpu()
            for k, v in self.f_ext.state_dict().items()
        })
        parts.update({
            f"generalized_module.{k}": v.detach().cpu()
            for k, v in self.generalized_module.state_dict().items()
        })
        if self.generalized_adapter is not None:
            parts.update({
                f"generalized_adapter.{k}": v.detach().cpu()
                for k, v in self.generalized_adapter.state_dict().items()
            })
        parts.update({
            f"personalized_module.{k}": v.detach().cpu()
            for k, v in self.personalized_module.state_dict().items()
        })
        if self.personalized_adapter is not None:
            parts.update({
                f"personalized_adapter.{k}": v.detach().cpu()
                for k, v in self.personalized_adapter.state_dict().items()
            })
        return parts

    def _build_init_proxy_loader(self):
        try:
            proxy_data, proxy_source = self._build_proxy_distill_data()
        except Exception as err:
            print(f"[FedCD] Init pretrain proxy load failed: {err}")
            return None, None

        if proxy_data is None or len(proxy_data) == 0:
            print("[FedCD] Init pretrain skipped: proxy data is empty.")
            return None, None

        if self.init_samples > 0 and self.init_samples < len(proxy_data):
            rng = random.Random(0)
            sample_indices = rng.sample(range(len(proxy_data)), self.init_samples)
            proxy_data = (
                [proxy_data[idx] for idx in sample_indices]
                if isinstance(proxy_data, list)
                else torch.utils.data.Subset(proxy_data, sample_indices)
            )

        num_workers = int(getattr(self.args, "num_workers", 0))
        pin_memory = bool(getattr(self.args, "pin_memory", False)) and self.device == "cuda"
        loader_kwargs = {
            "batch_size": max(1, int(self.init_batch_size)),
            "shuffle": True,
            "drop_last": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = int(getattr(self.args, "prefetch_factor", 2))

        loader = torch.utils.data.DataLoader(proxy_data, **loader_kwargs)
        return loader, proxy_source

    def _pretrain_and_broadcast_initial_components(self):
        loader, proxy_source = self._build_init_proxy_loader()
        if loader is None:
            return

        print(
            f"[FedCD] Initialization pretrain: source={proxy_source}, "
            f"epochs={self.init_epochs}, lr={self.init_lr}, samples={len(loader.dataset)}"
        )

        def _run_once(device):
            use_amp = device == "cuda" and bool(getattr(self.args, "amp", False))
            temp = max(1e-6, float(self.pm_teacher_temp))
            eps = 1e-12

            self.f_ext.to(device)
            self.generalized_module.to(device)
            self.personalized_module.to(device)
            self.f_ext.train()
            self.generalized_module.train()
            self.personalized_module.train()
            if self.generalized_adapter is not None:
                self.generalized_adapter.to(device)
                self.generalized_adapter.train()
            if self.personalized_adapter is not None:
                self.personalized_adapter.to(device)
                self.personalized_adapter.train()

            for p in self.f_ext.parameters():
                p.requires_grad = True
            for p in self.generalized_module.parameters():
                p.requires_grad = True
            for p in self.personalized_module.parameters():
                p.requires_grad = True
            if self.generalized_adapter is not None:
                for p in self.generalized_adapter.parameters():
                    p.requires_grad = True
            if self.personalized_adapter is not None:
                for p in self.personalized_adapter.parameters():
                    p.requires_grad = True

            params = list(self.f_ext.parameters())
            params += list(self.generalized_module.parameters())
            params += list(self.personalized_module.parameters())
            if self.generalized_adapter is not None:
                params += list(self.generalized_adapter.parameters())
            if self.personalized_adapter is not None:
                params += list(self.personalized_adapter.parameters())

            optimizer = torch.optim.SGD(params, lr=self.init_lr, momentum=0.9, weight_decay=1e-4)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            for _ in range(self.init_epochs):
                for x, y in tqdm(loader, desc="Init pretrain on proxy", leave=False):
                    if type(x) == type([]):
                        x = x[0]
                    x = x.to(device, non_blocking=(device == "cuda"))
                    y = y.to(device, non_blocking=(device == "cuda")).long()

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        z = self.f_ext(x)
                        if z.dim() > 2:
                            z = torch.flatten(z, 1)

                        z_gm = self.generalized_adapter(z) if self.generalized_adapter is not None else z
                        z_pm = self.personalized_adapter(z) if self.personalized_adapter is not None else z
                        logits_gm = self.generalized_module(z_gm)
                        logits_pm = self.personalized_module(z_pm)

                        prob_gm_t = torch.softmax(logits_gm / temp, dim=1)
                        prob_pm_t = torch.softmax(logits_pm / temp, dim=1)
                        log_prob_gm_t = torch.log_softmax(logits_gm / temp, dim=1)
                        log_prob_pm_t = torch.log_softmax(logits_pm / temp, dim=1)
                        kd_loss = 0.5 * (
                            F.kl_div(log_prob_gm_t, prob_pm_t, reduction="batchmean")
                            + F.kl_div(log_prob_pm_t, prob_gm_t, reduction="batchmean")
                        ) * (temp * temp)

                        prob_mix = 0.5 * (
                            torch.softmax(logits_gm, dim=1) + torch.softmax(logits_pm, dim=1)
                        )
                        entropy_loss = -(prob_mix * torch.log(prob_mix.clamp_min(eps))).sum(dim=1).mean()
                        mean_prob = prob_mix.mean(dim=0)
                        diversity_loss = torch.sum(mean_prob * torch.log(mean_prob.clamp_min(eps)))

                        total_loss = (
                            self.init_kd_weight * kd_loss
                            + self.init_entropy_weight * entropy_loss
                            + self.init_diversity_weight * diversity_loss
                        )

                        if self.init_ce_weight > 0:
                            valid = (y >= 0) & (y < self.num_classes)
                            if valid.any():
                                logits_fused = logits_gm + logits_pm
                                ce_loss = (
                                    F.cross_entropy(logits_gm[valid], y[valid])
                                    + F.cross_entropy(logits_pm[valid], y[valid])
                                    + F.cross_entropy(logits_fused[valid], y[valid])
                                ) / 3.0
                                total_loss = total_loss + self.init_ce_weight * ce_loss

                    if not torch.isfinite(total_loss):
                        continue
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            self.f_ext.to("cpu")
            self.generalized_module.to("cpu")
            self.personalized_module.to("cpu")
            if self.generalized_adapter is not None:
                self.generalized_adapter.to("cpu")
            if self.personalized_adapter is not None:
                self.personalized_adapter.to("cpu")
            self.f_ext.eval()
            self.generalized_module.eval()
            self.personalized_module.eval()
            if self.generalized_adapter is not None:
                self.generalized_adapter.eval()
            if self.personalized_adapter is not None:
                self.personalized_adapter.eval()

            for p in self.f_ext.parameters():
                p.requires_grad = False
            if device == "cuda" and self.args.avoid_oom:
                torch.cuda.empty_cache()

        try:
            _run_once(self.device)
        except RuntimeError as err:
            if self.device == "cuda" and "out of memory" in str(err).lower():
                print("[Warn] OOM during initialization pretrain. Falling back to CPU.")
                torch.cuda.empty_cache()
                _run_once("cpu")
            else:
                raise

        base_state = self._current_generalized_state()
        for cluster_id in set(self.cluster_map.values()):
            self.cluster_generalized_states[cluster_id] = {k: v.clone() for k, v in base_state.items()}

        init_parts = self._build_initial_parts()
        total_bytes = sum(v.numel() * v.element_size() for v in init_parts.values())
        print(
            f"[FedCD] Broadcast initial components (f_ext+GM+PM): "
            f"{total_bytes / (1024**2):.2f} MB per client"
        )
        payload = {"init_parts": init_parts}
        for client in self.clients:
            client.set_initial_parameters(payload["init_parts"])

    def _log_usage(
        self,
        round_idx,
        stage,
        wall_start,
        cpu_start,
        local_test_acc=None,
        pm_local_test_acc=None,
        gm_local_test_acc=None,
        train_loss=None,
        uplink=0,
        downlink=0,
        global_test_acc=None,
        gm_only_global_test_acc=None,
        pm_global_test_acc=None,
    ):
        wall_delta = time.time() - wall_start
        # cpu_delta = time.process_time() - cpu_start
        # cpu_pct = (cpu_delta / wall_delta * 100.0) if wall_delta > 0 else 0.0

        local_acc_str = f"local_acc={local_test_acc:.4f}" if local_test_acc is not None else ""
        pm_local_acc_str = (
            f"pm_local_acc={pm_local_test_acc:.4f}" if pm_local_test_acc is not None else ""
        )
        gm_local_acc_str = (
            f"gm_local_acc={gm_local_test_acc:.4f}" if gm_local_test_acc is not None else ""
        )
        global_acc_str = f"global_acc={global_test_acc:.4f}" if global_test_acc is not None else ""
        gm_only_acc_str = (
            f"gm_only_global_acc={gm_only_global_test_acc:.4f}"
            if gm_only_global_test_acc is not None
            else ""
        )
        pm_only_acc_str = (
            f"pm_global_acc={pm_global_test_acc:.4f}"
            if pm_global_test_acc is not None
            else ""
        )
        loss_str = f"loss={train_loss:.4f}" if train_loss is not None else ""
        msg = (
            f"[FedCD] Round {round_idx} | {stage} | "
            f"wall={wall_delta:.2f}s "
            + (
                f"{local_acc_str} {pm_local_acc_str} {gm_local_acc_str} "
                f"{global_acc_str} {gm_only_acc_str} {pm_only_acc_str} {loss_str}"
            )
        )
        # print(msg)
        self._append_usage_csv(
            round_idx,
            local_test_acc,
            pm_local_test_acc,
            gm_local_test_acc,
            global_test_acc,
            gm_only_global_test_acc,
            pm_global_test_acc,
            train_loss,
            uplink,
            downlink,
        )

    def _append_usage_csv(
        self,
        round_idx,
        local_test_acc,
        pm_local_test_acc,
        gm_local_test_acc,
        global_test_acc,
        gm_only_global_test_acc,
        pm_global_test_acc,
        train_loss,
        uplink,
        downlink,
    ):
        path = self.log_usage_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        header = (
            "round,local_test_acc,pm_local_test_acc,gm_local_test_acc,"
            "global_test_acc,gm_only_global_test_acc,pm_global_test_acc,"
            "local_pm_weight_mean,local_pm_weight_min,local_pm_weight_max,"
            "local_agree_pm,local_agree_gm,"
            "global_pm_weight_mean,global_pm_weight_min,global_pm_weight_max,"
            "global_agree_pm,global_agree_gm,"
            "train_loss,uplink_mb,downlink_mb,total_mb\n"
        )
        
        # Only log rows that have metrics (evaluation stage)
        if (
            local_test_acc is None
            and pm_local_test_acc is None
            and gm_local_test_acc is None
            and global_test_acc is None
            and gm_only_global_test_acc is None
            and pm_global_test_acc is None
            and train_loss is None
        ):
            return

        local_acc = f"{local_test_acc:.4f}" if local_test_acc is not None else ""
        pm_local_acc = f"{pm_local_test_acc:.4f}" if pm_local_test_acc is not None else ""
        gm_local_acc = f"{gm_local_test_acc:.4f}" if gm_local_test_acc is not None else ""
        global_acc = f"{global_test_acc:.4f}" if global_test_acc is not None else ""
        gm_only_acc = f"{gm_only_global_test_acc:.4f}" if gm_only_global_test_acc is not None else ""
        pm_only_acc = f"{pm_global_test_acc:.4f}" if pm_global_test_acc is not None else ""
        local_pm_weight_mean = ""
        local_pm_weight_min = ""
        local_pm_weight_max = ""
        local_agree_pm = ""
        local_agree_gm = ""
        if self.last_local_gate_stats is not None:
            local_pm_weight_mean = f"{self.last_local_gate_stats['pm_weight_mean']:.4f}"
            local_pm_weight_min = f"{self.last_local_gate_stats['pm_weight_min']:.4f}"
            local_pm_weight_max = f"{self.last_local_gate_stats['pm_weight_max']:.4f}"
            local_agree_pm = f"{self.last_local_gate_stats['agree_with_pm']:.4f}"
            local_agree_gm = f"{self.last_local_gate_stats['agree_with_gm']:.4f}"
        global_pm_weight_mean = ""
        global_pm_weight_min = ""
        global_pm_weight_max = ""
        global_agree_pm = ""
        global_agree_gm = ""
        if self.last_global_gate_stats is not None:
            global_pm_weight_mean = f"{self.last_global_gate_stats['pm_weight_mean']:.4f}"
            global_pm_weight_min = f"{self.last_global_gate_stats['pm_weight_min']:.4f}"
            global_pm_weight_max = f"{self.last_global_gate_stats['pm_weight_max']:.4f}"
            global_agree_pm = f"{self.last_global_gate_stats['agree_with_pm']:.4f}"
            global_agree_gm = f"{self.last_global_gate_stats['agree_with_gm']:.4f}"
        t_loss = f"{train_loss:.4f}" if train_loss is not None else ""
        uplink_mb = uplink / (1024**2)
        downlink_mb = downlink / (1024**2)
        total_mb = uplink_mb + downlink_mb
        line = (
            f"{round_idx},{local_acc},{pm_local_acc},{gm_local_acc},"
            f"{global_acc},{gm_only_acc},{pm_only_acc},"
            f"{local_pm_weight_mean},{local_pm_weight_min},{local_pm_weight_max},"
            f"{local_agree_pm},{local_agree_gm},"
            f"{global_pm_weight_mean},{global_pm_weight_min},{global_pm_weight_max},"
            f"{global_agree_pm},{global_agree_gm},{t_loss},"
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
        print(f"[FedCD] Broadcast Shared FExt+GM Size: {total_bytes / (1024**2):.2f} MB per client")
        
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
            received_gms = []
            received_protos = []
            received_router_stats = []
            total_uplink_bytes = 0
            for client in tqdm(self.selected_clients, desc=f"Round {i} Collecting PMs", leave=False):
                pm_state = client.upload_parameters()
                received_pms.append((client.id, pm_state))
                total_uplink_bytes += sum(v.numel() * v.element_size() for v in pm_state.values())
                if self.gm_update_mode in {"local", "hybrid_local_proto"}:
                    gm_state = client.upload_generalized_parameters()
                    if gm_state:
                        received_gms.append((client.id, gm_state, float(getattr(client, "train_samples", 1))))
                        total_uplink_bytes += sum(v.numel() * v.element_size() for v in gm_state.values())
                if self.gm_update_mode in {"server_proto_teacher", "hybrid_local_proto"} or (
                    self.gm_update_mode == "server_pm_teacher"
                    and (self.pm_teacher_competence_filter or self.pm_teacher_datafree_enable)
                ):
                    proto_state = client.upload_pm_prototypes(max_samples=self.proto_teacher_client_samples)
                    if proto_state:
                        received_protos.append((client.id, proto_state))
                        total_uplink_bytes += sum(v.numel() * v.element_size() for v in proto_state.values())
                if (
                    self.router_server_distill_enable
                    and i % self.router_server_period == 0
                ):
                    router_state = client.upload_router_feature_stats(max_samples=self.router_server_samples)
                    if router_state:
                        received_router_stats.append((client.id, router_state))
                        total_uplink_bytes += self._tensor_state_nbytes(router_state)
            
            round_uplink += total_uplink_bytes

            if len(self.selected_clients) > 0:
                avg_uplink = total_uplink_bytes / len(self.selected_clients)
                print(f"[FedCD] Round {i} Total Uplink Size: {total_uplink_bytes / (1024**2):.2f} MB (Avg: {avg_uplink / (1024**2):.2f} MB/client)")

            # 2.5 클러스터링 갱신 및 상세 로깅
            if self.enable_clustering and i % self.cluster_period == 0:
                prev_cluster_map = dict(self.cluster_map)
                raw_cluster_map = self.cluster_clients_by_distribution()
                self.cluster_map = self._align_cluster_labels(prev_cluster_map, raw_cluster_map)
                self._update_cluster_distribution_stats(self.cluster_map)
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

            if (
                self.router_server_distill_enable
                and i % self.router_server_period == 0
                and received_router_stats
            ):
                cluster_router_contexts = self._build_router_cluster_contexts(received_router_stats)
                if cluster_router_contexts:
                    router_downlink = self.send_router_distribution_context(cluster_router_contexts)
                    round_downlink += router_downlink

            # 3. 클러스터 내 PM 집계 및 배포
            need_cluster_pm = (
                self.enable_pm_aggregation
                or self.gm_update_mode == "server_pm_fedavg"
                or (self.gm_update_mode == "server_pm_teacher" and self.pm_teacher_source == "cluster")
            )
            cluster_pms = self.aggregate_cluster_pms(received_pms) if need_cluster_pm else {}
            cluster_counts = self._get_cluster_client_counts(received_pms) if need_cluster_pm else {}
            pm_client_weights = {
                int(c.id): float(max(1, getattr(c, "train_samples", 1)))
                for c in self.selected_clients
            }
            cluster_teacher_weights = {}
            for client in self.selected_clients:
                cluster_id = self.cluster_map.get(client.id, 0)
                cluster_teacher_weights[cluster_id] = cluster_teacher_weights.get(cluster_id, 0.0) + float(
                    max(1, getattr(client, "train_samples", 1))
                )
            if self.enable_pm_aggregation and i % self.pm_period == 0 and cluster_pms:
                downlink_bytes = self.send_cluster_pms(cluster_pms)
                round_downlink += downlink_bytes
                print(f"[FedCD] Round {i}: PM aggregation and cluster update done")
                if self.log_usage and i % self.log_usage_every == 0:
                    self._log_usage(i, "post_pm", wall_start, cpu_start)

            # 4. GM 전역 통합 (Global GM Aggregation across all clients)
            if i % self.global_period == 0:
                if self.gm_update_mode == "local":
                    global_gm_state = self.aggregate_global_gms(received_gms) if received_gms else None
                elif self.gm_update_mode == "server_pm_teacher":
                    teacher_proto_states = None
                    if self.pm_teacher_source == "cluster":
                        teacher_states = [(int(cluster_id), state) for cluster_id, state in cluster_pms.items()]
                        teacher_weights = {
                            int(cluster_id): float(max(1.0, cluster_teacher_weights.get(cluster_id, 1.0)))
                            for cluster_id in cluster_pms.keys()
                        }
                        if (self.pm_teacher_competence_filter or self.pm_teacher_datafree_enable) and received_protos:
                            teacher_proto_states = self._aggregate_cluster_proto_states(received_protos)
                        global_gm_state = self.distill_global_gm_from_pm_teachers(
                            teacher_states, teacher_weights, teacher_proto_states
                        )
                    else:
                        if (self.pm_teacher_competence_filter or self.pm_teacher_datafree_enable) and received_protos:
                            teacher_proto_states = {int(cid): state for cid, state in received_protos}
                        global_gm_state = self.distill_global_gm_from_pm_teachers(
                            received_pms, pm_client_weights, teacher_proto_states
                        )
                elif self.gm_update_mode == "server_pm_fedavg":
                    global_gm_state = self.update_gm_from_pm_fedavg(cluster_pms, cluster_counts)
                elif self.gm_update_mode == "server_pm_subnet":
                    global_gm_state = self.update_gm_from_pm_subnet(received_pms, pm_client_weights)
                elif self.gm_update_mode == "server_proto_teacher":
                    global_gm_state = self.update_gm_from_pm_prototypes(received_protos)
                else:
                    global_gm_state = self.update_gm_hybrid_local_proto(received_gms, received_protos)
                if global_gm_state:
                    if self.gm_update_mode == "local":
                        self._apply_generalized_state_to_server(global_gm_state)
                        for c_id in set(self.cluster_map.values()):
                            self.cluster_generalized_states[c_id] = {
                                k: v.clone() for k, v in global_gm_state.items()
                            }
                    downlink_bytes = self.send_models()
                else:
                    downlink_bytes = 0
                round_downlink += downlink_bytes
                if self.gm_update_mode == "local":
                    print(f"[FedCD] Round {i}: Server-side global GM aggregation/update done")
                elif self.gm_update_mode == "server_pm_teacher":
                    print(f"[FedCD] Round {i}: Server-side PM-teacher GM distillation/update done")
                elif self.gm_update_mode == "server_pm_fedavg":
                    print(f"[FedCD] Round {i}: Server-side PM->GM FedAvg update done")
                elif self.gm_update_mode == "server_pm_subnet":
                    print(f"[FedCD] Round {i}: Server-side PM FedAvg -> GM subnet update done")
                elif self.gm_update_mode == "server_proto_teacher":
                    print(f"[FedCD] Round {i}: Server-side prototype-based GM update done")
                else:
                    print(
                        "[FedCD] Round "
                        f"{i}: Hybrid GM update done (FedAvg local GM + prototype refine, "
                        f"blend={self.hybrid_proto_blend:.2f})"
                    )
                # Warm-up Personalized Module after GM update
                if getattr(self.args, "fedcd_warmup_epochs", 0) > 0:
                    for client in tqdm(self.selected_clients, desc=f"Round {i} PM Warm-up", leave=False):
                        client.warmup_personalized_module()
                if self.log_usage and i % self.log_usage_every == 0:
                    self._log_usage(i, "post_gm", wall_start, cpu_start)
            
            if i % self.eval_gap == 0:
                print(f"\n------------- Round number: {i} -------------")
                # 평가 실행 및 클러스터별 정확도 로깅
                metrics = self.evaluate_with_clusters(i, wall_start, cpu_start, round_uplink, round_downlink)
                if self._should_stop_for_search(
                    i,
                    metrics.get("pm_local_test_acc"),
                    metrics.get("gm_only_global_test_acc"),
                ):
                    print(f"[FedCD][Search] Stop round: {i}")
                    break

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
        gm_local_test_acc, pm_local_test_acc = self.evaluate_local_branch_test_accs()
        global_test_acc = self.evaluate_global_test_acc()
        gm_only_global_test_acc = self.evaluate_gm_only_global_test_acc()
        pm_global_test_acc = self.evaluate_pm_only_global_test_acc()
            
        print(f"Server: Overall Averaged Local Test Accuracy: {avg_acc:.4f}")
        if pm_local_test_acc is not None:
            print(f"Server: PM-only Local Test Accuracy: {pm_local_test_acc:.4f}")
        if gm_local_test_acc is not None:
            print(f"Server: GM-only Local Test Accuracy: {gm_local_test_acc:.4f}")
        print(f"Server: Fusion Mode: {getattr(self.args, 'fedcd_fusion_mode', 'unknown')}")
        if self.last_local_gate_stats is not None:
            local_gate = self.last_local_gate_stats
            print(
                "Server: Local Fusion PM Weight (mean/min/max): "
                f"{local_gate['pm_weight_mean']:.4f}/{local_gate['pm_weight_min']:.4f}/{local_gate['pm_weight_max']:.4f}"
            )
            print(
                "Server: Local Fusion Argmax Agreement (with PM/GM): "
                f"{local_gate['agree_with_pm']:.4f}/{local_gate['agree_with_gm']:.4f}"
            )
        if global_test_acc is not None:
            print(f"Server: Overall Averaged Global Test Accuracy: {global_test_acc:.4f}")
            if self.last_global_gate_stats is not None:
                gate = self.last_global_gate_stats
                print(
                    "Server: Fusion PM Weight (mean/min/max): "
                    f"{gate['pm_weight_mean']:.4f}/{gate['pm_weight_min']:.4f}/{gate['pm_weight_max']:.4f}"
                )
                print(
                    "Server: Fusion Argmax Agreement (with PM/GM): "
                    f"{gate['agree_with_pm']:.4f}/{gate['agree_with_gm']:.4f}"
                )
        if gm_only_global_test_acc is not None:
            print(f"Server: GM-only Global Test Accuracy: {gm_only_global_test_acc:.4f}")
        if pm_global_test_acc is not None:
            print(f"Server: PM-only Global Test Accuracy: {pm_global_test_acc:.4f}")
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
                pm_local_test_acc=pm_local_test_acc,
                gm_local_test_acc=gm_local_test_acc,
                global_test_acc=global_test_acc,
                gm_only_global_test_acc=gm_only_global_test_acc,
                pm_global_test_acc=pm_global_test_acc,
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

        return {
            "local_test_acc": avg_acc,
            "pm_local_test_acc": pm_local_test_acc,
            "gm_local_test_acc": gm_local_test_acc,
            "global_test_acc": global_test_acc,
            "gm_only_global_test_acc": gm_only_global_test_acc,
            "pm_global_test_acc": pm_global_test_acc,
            "train_loss": avg_loss,
        }

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

    def _get_cluster_client_counts(self, received_pms):
        counts = {}
        for client_id, _ in received_pms:
            cluster_id = self.cluster_map.get(client_id, 0)
            counts[cluster_id] = counts.get(cluster_id, 0) + 1
        return counts

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

    @staticmethod
    def _tensor_state_nbytes(state):
        total = 0
        if not isinstance(state, dict):
            return total
        for v in state.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
        return total

    @staticmethod
    def _mean_var_from_sums(sum_tensor, sq_sum_tensor, count_tensor):
        if count_tensor <= 0:
            return None, None
        denom = float(max(count_tensor, 1e-12))
        mean = sum_tensor / denom
        var = (sq_sum_tensor / denom) - mean.pow(2)
        return mean.float(), var.float().clamp_min(1e-6)

    @staticmethod
    def _class_mean_var_from_sums(counts, feat_sum, feat_sq_sum, fallback_mean=None, fallback_var=None):
        C = counts.size(0)
        D = feat_sum.size(1)
        mean = torch.zeros((C, D), dtype=torch.float32)
        var = torch.ones((C, D), dtype=torch.float32)
        valid = counts > 0
        if valid.any():
            idx = valid.nonzero(as_tuple=False).view(-1)
            cnt = counts[idx].unsqueeze(1).clamp_min(1e-12)
            cls_mean = feat_sum[idx] / cnt
            cls_var = (feat_sq_sum[idx] / cnt) - cls_mean.pow(2)
            mean[idx] = cls_mean.float()
            var[idx] = cls_var.float().clamp_min(1e-6)
        if fallback_mean is not None and fallback_var is not None:
            invalid = ~valid
            if invalid.any():
                mean[invalid] = fallback_mean[invalid].float()
                var[invalid] = fallback_var[invalid].float().clamp_min(1e-6)
        return mean, var, valid.float()

    def _build_router_cluster_contexts(self, received_router_stats):
        if not received_router_stats:
            return {}

        valid_payloads = []
        num_classes = None
        feat_dim = None
        for client_id, state in received_router_stats:
            if not state:
                continue
            if not {"counts", "feat_sum", "feat_sq_sum"}.issubset(set(state.keys())):
                continue
            counts = state["counts"].detach().cpu().float()
            feat_sum = state["feat_sum"].detach().cpu().float()
            feat_sq_sum = state["feat_sq_sum"].detach().cpu().float()
            if counts.dim() != 1 or feat_sum.dim() != 2 or feat_sq_sum.dim() != 2:
                continue
            if feat_sum.shape != feat_sq_sum.shape:
                continue
            if feat_sum.size(0) != counts.size(0):
                continue

            if num_classes is None:
                num_classes = counts.size(0)
                feat_dim = feat_sum.size(1)
            elif num_classes != counts.size(0) or feat_dim != feat_sum.size(1):
                continue

            valid_payloads.append((int(client_id), counts, feat_sum, feat_sq_sum))

        if not valid_payloads or num_classes is None or feat_dim is None:
            return {}

        global_counts = torch.zeros((num_classes,), dtype=torch.float32)
        global_feat_sum = torch.zeros((num_classes, feat_dim), dtype=torch.float32)
        global_feat_sq_sum = torch.zeros((num_classes, feat_dim), dtype=torch.float32)

        cluster_counts = {}
        cluster_feat_sum = {}
        cluster_feat_sq_sum = {}

        for client_id, counts, feat_sum, feat_sq_sum in valid_payloads:
            cluster_id = int(self.cluster_map.get(client_id, 0))
            global_counts += counts
            global_feat_sum += feat_sum
            global_feat_sq_sum += feat_sq_sum

            if cluster_id not in cluster_counts:
                cluster_counts[cluster_id] = counts.clone()
                cluster_feat_sum[cluster_id] = feat_sum.clone()
                cluster_feat_sq_sum[cluster_id] = feat_sq_sum.clone()
            else:
                cluster_counts[cluster_id] += counts
                cluster_feat_sum[cluster_id] += feat_sum
                cluster_feat_sq_sum[cluster_id] += feat_sq_sum

        global_total = float(global_counts.sum().item())
        if global_total <= 0:
            return {}

        global_sum_vec = global_feat_sum.sum(dim=0)
        global_sq_sum_vec = global_feat_sq_sum.sum(dim=0)
        global_mean, global_var = self._mean_var_from_sums(global_sum_vec, global_sq_sum_vec, global_total)
        if global_mean is None or global_var is None:
            return {}
        global_cls_mean, global_cls_var, _ = self._class_mean_var_from_sums(
            global_counts, global_feat_sum, global_feat_sq_sum
        )

        contexts = {}
        all_cluster_ids = set(self.cluster_map.values())
        cluster_mean_vec = {}
        for cluster_id in all_cluster_ids:
            c_counts = cluster_counts.get(cluster_id)
            c_feat_sum = cluster_feat_sum.get(cluster_id)
            if c_counts is None or c_feat_sum is None:
                cluster_mean_vec[cluster_id] = global_mean.clone()
                continue
            c_total = float(c_counts.sum().item())
            if c_total <= 0:
                cluster_mean_vec[cluster_id] = global_mean.clone()
            else:
                cluster_mean_vec[cluster_id] = (c_feat_sum.sum(dim=0) / c_total).float()

        def _compose_negative_stats(cur_cluster_id, cur_counts, cur_feat_sum, cur_feat_sq_sum):
            # Baseline negative context: all other clusters.
            out_counts = (global_counts - cur_counts).clamp_min(0.0)
            out_feat_sum = (global_feat_sum - cur_feat_sum)
            out_feat_sq_sum = (global_feat_sq_sum - cur_feat_sq_sum)

            # Hard-negative context: farthest clusters by feature-mean distance.
            if self.router_server_neg_mode != "farthest_k":
                return out_counts, out_feat_sum, out_feat_sq_sum

            candidates = []
            cur_mean = cluster_mean_vec.get(cur_cluster_id, global_mean)
            for other_id in all_cluster_ids:
                if other_id == cur_cluster_id:
                    continue
                other_counts = cluster_counts.get(other_id)
                other_feat_sum = cluster_feat_sum.get(other_id)
                other_feat_sq_sum = cluster_feat_sq_sum.get(other_id)
                if other_counts is None or other_feat_sum is None or other_feat_sq_sum is None:
                    continue
                other_total = float(other_counts.sum().item())
                if other_total <= 0:
                    continue
                other_mean = cluster_mean_vec.get(other_id, global_mean)
                dist = torch.sum((other_mean - cur_mean).pow(2)).item()
                candidates.append((dist, other_id))

            if not candidates:
                return out_counts, out_feat_sum, out_feat_sq_sum

            candidates.sort(key=lambda x: x[0], reverse=True)
            if self.router_server_neg_topk <= 0:
                pick = candidates
            else:
                pick = candidates[: min(self.router_server_neg_topk, len(candidates))]

            neg_counts = torch.zeros_like(global_counts)
            neg_feat_sum = torch.zeros_like(global_feat_sum)
            neg_feat_sq_sum = torch.zeros_like(global_feat_sq_sum)
            for _, oid in pick:
                neg_counts += cluster_counts[oid]
                neg_feat_sum += cluster_feat_sum[oid]
                neg_feat_sq_sum += cluster_feat_sq_sum[oid]

            neg_total = float(neg_counts.sum().item())
            if neg_total <= 0:
                return out_counts, out_feat_sum, out_feat_sq_sum
            return neg_counts, neg_feat_sum, neg_feat_sq_sum

        for cluster_id in all_cluster_ids:
            c_counts = cluster_counts.get(cluster_id)
            c_feat_sum = cluster_feat_sum.get(cluster_id)
            c_feat_sq_sum = cluster_feat_sq_sum.get(cluster_id)
            if c_counts is None:
                c_counts = torch.zeros_like(global_counts)
                c_feat_sum = torch.zeros_like(global_feat_sum)
                c_feat_sq_sum = torch.zeros_like(global_feat_sq_sum)

            in_total = float(c_counts.sum().item())
            in_sum_vec = c_feat_sum.sum(dim=0)
            in_sq_sum_vec = c_feat_sq_sum.sum(dim=0)

            out_counts, out_feat_sum, out_feat_sq_sum = _compose_negative_stats(
                cluster_id, c_counts, c_feat_sum, c_feat_sq_sum
            )
            out_total = float(out_counts.sum().item())
            out_sum_vec = out_feat_sum.sum(dim=0)
            out_sq_sum_vec = out_feat_sq_sum.sum(dim=0)

            in_mean, in_var = self._mean_var_from_sums(in_sum_vec, in_sq_sum_vec, in_total)
            out_mean, out_var = self._mean_var_from_sums(out_sum_vec, out_sq_sum_vec, out_total)
            if in_mean is None or in_var is None:
                in_mean, in_var = global_mean.clone(), global_var.clone()
            if out_mean is None or out_var is None:
                out_mean, out_var = global_mean.clone(), global_var.clone()

            in_cls_mean, in_cls_var, in_cls_valid = self._class_mean_var_from_sums(
                c_counts, c_feat_sum, c_feat_sq_sum, fallback_mean=global_cls_mean, fallback_var=global_cls_var
            )
            out_cls_mean, out_cls_var, out_cls_valid = self._class_mean_var_from_sums(
                out_counts, out_feat_sum, out_feat_sq_sum, fallback_mean=global_cls_mean, fallback_var=global_cls_var
            )

            contexts[cluster_id] = {
                "in_feature_mean": in_mean.cpu(),
                "in_feature_var": in_var.cpu(),
                "out_feature_mean": out_mean.cpu(),
                "out_feature_var": out_var.cpu(),
                "in_class_mean": in_cls_mean.cpu(),
                "in_class_var": in_cls_var.cpu(),
                "in_class_valid": in_cls_valid.cpu(),
                "out_class_mean": out_cls_mean.cpu(),
                "out_class_var": out_cls_var.cpu(),
                "out_class_valid": out_cls_valid.cpu(),
            }

        return contexts

    def send_router_distribution_context(self, cluster_contexts):
        if not cluster_contexts:
            return 0

        total_broadcast_bytes = 0
        sample_context = next(iter(cluster_contexts.values()))
        context_bytes = self._tensor_state_nbytes(sample_context)
        print(
            "[FedCD] Broadcast Router Context Size: "
            f"{context_bytes / (1024**2):.2f} MB per client (cluster-conditioned)"
        )

        for client in self.clients:
            cluster_id = int(self.cluster_map.get(client.id, 0))
            ctx = cluster_contexts.get(cluster_id, None)
            client.set_router_distribution_context(ctx)
            if ctx is not None:
                total_broadcast_bytes += self._tensor_state_nbytes(ctx)

        return total_broadcast_bytes

    def aggregate_global_gms(self, received_gms):
        """
        Aggregate GM states from all selected clients (weighted FedAvg).
        received_gms: [(client_id, gm_state_dict, weight), ...]
        """
        if not received_gms:
            return None

        valid = []
        total_weight = 0.0
        for _, state, weight in received_gms:
            w = float(weight)
            if w <= 0:
                continue
            valid.append((state, w))
            total_weight += w

        if not valid or total_weight <= 0:
            return None

        template = valid[0][0]
        avg_state = {}
        for key in template.keys():
            acc = None
            for state, w in valid:
                if key not in state:
                    continue
                contrib = state[key] * (w / total_weight)
                acc = contrib if acc is None else (acc + contrib)
            if acc is not None:
                avg_state[key] = acc
        return avg_state

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

    def _weighted_average_state_dicts(self, grouped_states, group_weights):
        valid = []
        total_weight = 0.0
        for group_id, state in grouped_states.items():
            if not state:
                continue
            w = float(group_weights.get(group_id, 1.0))
            if w <= 0:
                continue
            valid.append((state, w))
            total_weight += w

        if not valid or total_weight <= 0:
            return None

        avg_state = {}
        template = valid[0][0]
        for key in template.keys():
            acc = None
            for state, w in valid:
                value = state.get(key, None)
                if value is None:
                    continue
                contrib = value.detach().cpu() * (w / total_weight)
                acc = contrib if acc is None else (acc + contrib)
            if acc is not None:
                avg_state[key] = acc
        return avg_state

    @staticmethod
    def _topk_sorted_indices(score, k):
        n = int(score.numel())
        k = int(max(0, min(k, n)))
        if k <= 0:
            return torch.empty((0,), dtype=torch.long, device=score.device)
        if k == n:
            return torch.arange(n, dtype=torch.long, device=score.device)
        idx = torch.topk(score, k=k, largest=True, sorted=False).indices
        idx, _ = torch.sort(idx)
        return idx

    @staticmethod
    def _slice_or_pad_dim(tensor, dim, target_size, select_idx=None):
        cur = int(tensor.size(dim))
        tgt = int(target_size)
        if cur == tgt:
            return tensor

        if cur > tgt:
            if select_idx is None:
                moved = tensor.abs().movedim(dim, 0).reshape(cur, -1)
                score = moved.mean(dim=1)
                select_idx = FedCD._topk_sorted_indices(score, tgt)
            else:
                select_idx = select_idx.to(tensor.device, dtype=torch.long)
                if select_idx.numel() > tgt:
                    select_idx = select_idx[:tgt]
                elif select_idx.numel() < tgt:
                    remain = tgt - int(select_idx.numel())
                    all_idx = torch.arange(cur, device=tensor.device, dtype=torch.long)
                    mask = torch.ones(cur, device=tensor.device, dtype=torch.bool)
                    if select_idx.numel() > 0:
                        mask[select_idx] = False
                    extra = all_idx[mask][:remain]
                    select_idx = torch.cat([select_idx, extra], dim=0)
                select_idx, _ = torch.sort(select_idx)
            return tensor.index_select(dim, select_idx)

        pad_shape = list(tensor.shape)
        pad_shape[dim] = tgt - cur
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=dim)

    def _match_tensor_shape(self, src, target_tensor):
        if src is None or target_tensor is None:
            return None
        src_t = src.detach().cpu()
        if src_t.dim() != target_tensor.dim():
            return None
        out = src_t
        for dim, tgt in enumerate(target_tensor.shape):
            out = self._slice_or_pad_dim(out, dim, int(tgt))
        return out.to(dtype=target_tensor.dtype).clone()

    @staticmethod
    def _linear_layer_names(module):
        return [name for name, m in module.named_modules() if isinstance(m, nn.Linear)]

    def _select_hidden_mask_indices(self, pm_fc1_weight, pm_fc2_weight, gm_hidden, cache_key):
        pm_hidden = int(pm_fc1_weight.size(0))
        gm_hidden = int(gm_hidden)
        if gm_hidden >= pm_hidden:
            return torch.arange(pm_hidden, dtype=torch.long)

        if self.pm_to_gm_mask_unified:
            cached = self._pm_to_gm_hidden_mask_cache.get(cache_key, None)
            if cached is not None and int(cached.numel()) == gm_hidden:
                return cached.clone()

        score_in = pm_fc1_weight.detach().cpu().abs().mean(dim=1)
        score_out = pm_fc2_weight.detach().cpu().abs().mean(dim=0)
        score = score_in + score_out
        idx = self._topk_sorted_indices(score, gm_hidden).cpu()

        if self.pm_to_gm_mask_unified:
            self._pm_to_gm_hidden_mask_cache[cache_key] = idx.clone()
        return idx

    def _project_pm_state_to_gm_state(self, pm_module_state):
        gm_template = self.generalized_module.state_dict()
        projected = {}

        pm_linear = self._linear_layer_names(self.personalized_module)
        gm_linear = self._linear_layer_names(self.generalized_module)

        if len(pm_linear) >= 2 and len(gm_linear) >= 2:
            pm_fc1_w_key = f"{pm_linear[0]}.weight"
            pm_fc1_b_key = f"{pm_linear[0]}.bias"
            pm_fc2_w_key = f"{pm_linear[1]}.weight"
            pm_fc2_b_key = f"{pm_linear[1]}.bias"
            gm_fc1_w_key = f"{gm_linear[0]}.weight"
            gm_fc1_b_key = f"{gm_linear[0]}.bias"
            gm_fc2_w_key = f"{gm_linear[1]}.weight"
            gm_fc2_b_key = f"{gm_linear[1]}.bias"

            req_pm = {pm_fc1_w_key, pm_fc2_w_key}
            req_gm = {gm_fc1_w_key, gm_fc2_w_key}
            if req_pm.issubset(set(pm_module_state.keys())) and req_gm.issubset(set(gm_template.keys())):
                pm_fc1_w = pm_module_state[pm_fc1_w_key].detach().cpu()
                pm_fc2_w = pm_module_state[pm_fc2_w_key].detach().cpu()
                gm_fc1_w_t = gm_template[gm_fc1_w_key]
                gm_fc2_w_t = gm_template[gm_fc2_w_key]

                if (
                    pm_fc1_w.dim() == 2
                    and pm_fc2_w.dim() == 2
                    and gm_fc1_w_t.dim() == 2
                    and gm_fc2_w_t.dim() == 2
                    and int(pm_fc1_w.size(0)) == int(pm_fc2_w.size(1))
                    and int(gm_fc1_w_t.size(0)) == int(gm_fc2_w_t.size(1))
                ):
                    gm_hidden = int(gm_fc1_w_t.size(0))
                    cache_key = (
                        pm_fc1_w_key,
                        pm_fc2_w_key,
                        int(pm_fc1_w.size(0)),
                        gm_hidden,
                    )
                    hidden_idx = self._select_hidden_mask_indices(pm_fc1_w, pm_fc2_w, gm_hidden, cache_key)

                    in_score = pm_fc1_w.abs().mean(dim=0)
                    in_idx = self._topk_sorted_indices(in_score, int(min(pm_fc1_w.size(1), gm_fc1_w_t.size(1))))
                    gm_fc1_w = pm_fc1_w.index_select(0, hidden_idx)
                    gm_fc1_w = gm_fc1_w.index_select(1, in_idx)
                    gm_fc1_w = self._slice_or_pad_dim(gm_fc1_w, 1, int(gm_fc1_w_t.size(1)))
                    projected[gm_fc1_w_key] = gm_fc1_w.to(dtype=gm_fc1_w_t.dtype).clone()

                    if pm_fc1_b_key in pm_module_state and gm_fc1_b_key in gm_template:
                        gm_fc1_b_t = gm_template[gm_fc1_b_key]
                        pm_fc1_b = pm_module_state[pm_fc1_b_key].detach().cpu()
                        gm_fc1_b = pm_fc1_b.index_select(0, hidden_idx)
                        gm_fc1_b = self._slice_or_pad_dim(gm_fc1_b, 0, int(gm_fc1_b_t.size(0)))
                        projected[gm_fc1_b_key] = gm_fc1_b.to(dtype=gm_fc1_b_t.dtype).clone()

                    gm_fc2_w = pm_fc2_w.index_select(1, hidden_idx)
                    gm_fc2_w = self._slice_or_pad_dim(gm_fc2_w, 0, int(gm_fc2_w_t.size(0)))
                    projected[gm_fc2_w_key] = gm_fc2_w.to(dtype=gm_fc2_w_t.dtype).clone()

                    if pm_fc2_b_key in pm_module_state and gm_fc2_b_key in gm_template:
                        gm_fc2_b_t = gm_template[gm_fc2_b_key]
                        gm_fc2_b = self._match_tensor_shape(pm_module_state[pm_fc2_b_key], gm_fc2_b_t)
                        if gm_fc2_b is not None:
                            projected[gm_fc2_b_key] = gm_fc2_b

        for gm_key, gm_tensor in gm_template.items():
            if gm_key in projected:
                continue
            src = pm_module_state.get(gm_key, None)
            if src is None:
                continue
            matched = self._match_tensor_shape(src, gm_tensor)
            if matched is not None:
                projected[gm_key] = matched

        return projected

    def update_gm_from_pm_fedavg(self, cluster_pms, cluster_counts):
        """
        Build GM by FedAveraging uploaded PMs on server.
        Assumes PM/GM module topology is compatible (e.g., both VGG8 classifiers).
        """
        if not cluster_pms:
            return None

        if not self.pm_to_gm_mask_enable:
            gm_module_template = self.generalized_module.state_dict()
            gm_adapter_template = (
                self.generalized_adapter.state_dict() if self.generalized_adapter is not None else {}
            )

            cluster_gm_states = {}
            for cluster_id, state in cluster_pms.items():
                pm_module_state, pm_adapter_state = self._extract_personalized_state(state)
                converted = {}

                for key, value in pm_module_state.items():
                    if key in gm_module_template and gm_module_template[key].shape == value.shape:
                        converted[f"generalized_module.{key}"] = value.detach().cpu()

                if self.generalized_adapter is not None and pm_adapter_state:
                    for key, value in pm_adapter_state.items():
                        if key in gm_adapter_template and gm_adapter_template[key].shape == value.shape:
                            converted[f"generalized_adapter.{key}"] = value.detach().cpu()

                if converted:
                    cluster_gm_states[cluster_id] = converted

            if not cluster_gm_states:
                print("[FedCD] PM->GM FedAvg skipped: no compatible PM states.")
                return None

            global_gm_state = self._weighted_average_generalized_states(cluster_gm_states, cluster_counts)
            if not global_gm_state:
                return None
        else:
            cluster_pm_modules = {}
            cluster_pm_adapters = {}
            for cluster_id, state in cluster_pms.items():
                pm_module_state, pm_adapter_state = self._extract_personalized_state(state)
                if pm_module_state:
                    cluster_pm_modules[cluster_id] = pm_module_state
                if pm_adapter_state:
                    cluster_pm_adapters[cluster_id] = pm_adapter_state

            if not cluster_pm_modules:
                print("[FedCD] PM->GM masked update skipped: empty PM module states.")
                return None

            avg_pm_module = self._weighted_average_state_dicts(cluster_pm_modules, cluster_counts)
            if not avg_pm_module:
                print("[FedCD] PM->GM masked update skipped: PM averaging failed.")
                return None

            projected_module = self._project_pm_state_to_gm_state(avg_pm_module)
            if not projected_module:
                print("[FedCD] PM->GM masked update skipped: projection produced no GM params.")
                return None

            global_gm_state = {
                f"generalized_module.{k}": v
                for k, v in projected_module.items()
            }

            if self.generalized_adapter is not None and cluster_pm_adapters:
                avg_pm_adapter = self._weighted_average_state_dicts(cluster_pm_adapters, cluster_counts)
                gm_adapter_template = self.generalized_adapter.state_dict()
                if avg_pm_adapter:
                    for key, tensor in gm_adapter_template.items():
                        src = avg_pm_adapter.get(key, None)
                        if src is None:
                            continue
                        matched = self._match_tensor_shape(src, tensor)
                        if matched is not None:
                            global_gm_state[f"generalized_adapter.{key}"] = matched

        # Keep strict load stable even when only a subset of GM keys is updated.
        current_gm_state = self._current_generalized_state()
        for key, value in current_gm_state.items():
            if key not in global_gm_state:
                global_gm_state[key] = value.clone()

        for cluster_id in set(self.cluster_map.values()):
            self.cluster_generalized_states[cluster_id] = {
                k: v.clone() for k, v in global_gm_state.items()
            }
        self._apply_generalized_state_to_server(global_gm_state)
        return global_gm_state

    def update_gm_from_pm_subnet(self, received_pms, client_weights=None):
        """
        Build GM from the FedAvg of uploaded PMs, then project the averaged large PM
        into the smaller GM by structured slicing/masking.

        This is the direct server-side feasibility version of:
          PM(all clients) -> single averaged PM -> smaller GM subnet.
        """
        if not received_pms:
            return None

        if client_weights is None:
            client_weights = {}

        pm_module_states = {}
        pm_adapter_states = {}
        group_weights = {}

        for client_id, state in received_pms:
            pm_module_state, pm_adapter_state = self._extract_personalized_state(state)
            if pm_module_state:
                pm_module_states[int(client_id)] = pm_module_state
                group_weights[int(client_id)] = float(max(1.0, client_weights.get(int(client_id), 1.0)))
            if pm_adapter_state:
                pm_adapter_states[int(client_id)] = pm_adapter_state

        if not pm_module_states:
            print("[FedCD] PM->GM subnet update skipped: empty PM module states.")
            return None

        avg_pm_module = self._weighted_average_state_dicts(pm_module_states, group_weights)
        if not avg_pm_module:
            print("[FedCD] PM->GM subnet update skipped: PM averaging failed.")
            return None

        projected_module = self._project_pm_state_to_gm_state(avg_pm_module)
        if not projected_module:
            print("[FedCD] PM->GM subnet update skipped: projection produced no GM params.")
            return None

        global_gm_state = {
            f"generalized_module.{k}": v
            for k, v in projected_module.items()
        }

        if self.generalized_adapter is not None and pm_adapter_states:
            avg_pm_adapter = self._weighted_average_state_dicts(pm_adapter_states, group_weights)
            gm_adapter_template = self.generalized_adapter.state_dict()
            if avg_pm_adapter:
                for key, tensor in gm_adapter_template.items():
                    src = avg_pm_adapter.get(key, None)
                    if src is None:
                        continue
                    matched = self._match_tensor_shape(src, tensor)
                    if matched is not None:
                        global_gm_state[f"generalized_adapter.{key}"] = matched

        current_gm_state = self._current_generalized_state()
        for key, value in current_gm_state.items():
            if key not in global_gm_state:
                global_gm_state[key] = value.clone()

        for cluster_id in set(self.cluster_map.values()):
            self.cluster_generalized_states[cluster_id] = {
                k: v.clone() for k, v in global_gm_state.items()
            }
        self._apply_generalized_state_to_server(global_gm_state)
        return global_gm_state

    def _collect_pm_prototype_components(self, received_protos):
        """
        Build multimodal prototype components from client PM uploads.
        Each (client, class) pair becomes one component to avoid collapsing
        heterogeneous class modes into a single global mean.
        """
        components = []
        skipped = 0
        expected_classes = None
        expected_feat_dim = None

        for _, state in received_protos:
            if not state:
                continue
            required = {"counts", "feat_sum", "feat_sq_sum", "logit_sum"}
            if not required.issubset(set(state.keys())):
                skipped += 1
                continue

            counts = state["counts"].detach().cpu().float()
            feat_sum = state["feat_sum"].detach().cpu().float()
            feat_sq_sum = state["feat_sq_sum"].detach().cpu().float()
            logit_sum = state["logit_sum"].detach().cpu().float()

            if counts.dim() != 1 or feat_sum.dim() != 2 or feat_sq_sum.dim() != 2 or logit_sum.dim() != 2:
                skipped += 1
                continue
            num_classes = counts.numel()
            if feat_sum.size(0) != num_classes or feat_sq_sum.size(0) != num_classes:
                skipped += 1
                continue
            if logit_sum.size(0) != num_classes or logit_sum.size(1) != num_classes:
                skipped += 1
                continue

            feat_dim = feat_sum.size(1)
            if expected_classes is None:
                expected_classes = num_classes
                expected_feat_dim = feat_dim
            elif expected_classes != num_classes or expected_feat_dim != feat_dim:
                skipped += 1
                continue

            for cls_id in range(num_classes):
                cnt = float(counts[cls_id].item())
                if cnt < float(self.proto_teacher_min_count):
                    continue
                denom = max(cnt, 1e-12)
                mu = feat_sum[cls_id] / denom
                var = (feat_sq_sum[cls_id] / denom) - mu.pow(2)
                var = var.clamp_min(1e-6)
                logits = logit_sum[cls_id] / denom
                components.append({
                    "class_id": cls_id,
                    "count": cnt,
                    "feat_mean": mu,
                    "feat_var": var,
                    "teacher_logits": logits,
                })

        if skipped > 0:
            print(f"[FedCD] Skipped {skipped} invalid PM prototype payload(s).")
        return components, expected_classes, expected_feat_dim

    def _aggregate_proto_state_list(self, proto_states):
        if not proto_states:
            return None
        merged = {}
        for state in proto_states:
            if not state:
                continue
            for key, value in state.items():
                tensor = value.detach().cpu()
                if key not in merged:
                    merged[key] = tensor.clone()
                else:
                    if merged[key].shape != tensor.shape:
                        return None
                    merged[key] += tensor
        return merged if merged else None

    def _aggregate_cluster_proto_states(self, received_protos):
        cluster_buckets = {}
        for client_id, state in received_protos:
            cluster_id = int(self.cluster_map.get(int(client_id), 0))
            cluster_buckets.setdefault(cluster_id, []).append(state)
        out = {}
        for cluster_id, states in cluster_buckets.items():
            merged = self._aggregate_proto_state_list(states)
            if merged:
                out[int(cluster_id)] = merged
        return out

    def _update_cluster_distribution_stats(self, cluster_map=None):
        if cluster_map is None:
            cluster_map = self.cluster_map
        if not self.client_distribution_stats:
            self.cluster_distribution_stats = {}
            return

        grouped = {}
        for client_id, stats in self.client_distribution_stats.items():
            if not stats:
                continue
            mean = stats.get("mean", None)
            var = stats.get("var", None)
            if mean is None or var is None:
                continue
            cluster_id = int(cluster_map.get(int(client_id), 0))
            weight = float(stats.get("weight", max(1, getattr(self.clients[int(client_id)], "train_samples", 1))))
            grouped.setdefault(cluster_id, []).append((mean.float(), var.float(), max(weight, 1e-12)))

        cluster_stats = {}
        for cluster_id, items in grouped.items():
            total_w = sum(w for _, _, w in items)
            if total_w <= 0:
                continue
            mean_acc = None
            second_acc = None
            for mean, var, w in items:
                coeff = w / total_w
                second = var + mean.pow(2)
                mean_acc = mean * coeff if mean_acc is None else (mean_acc + mean * coeff)
                second_acc = second * coeff if second_acc is None else (second_acc + second * coeff)
            if mean_acc is None or second_acc is None:
                continue
            var_acc = (second_acc - mean_acc.pow(2)).clamp_min(1e-6)
            cluster_stats[int(cluster_id)] = {
                "mean": mean_acc.detach().cpu(),
                "var": var_acc.detach().cpu(),
                "weight": float(total_w),
            }

        self.cluster_distribution_stats = cluster_stats

    def _prepare_cluster_dist_meta(self, state, device):
        if not state:
            return None
        mean = state.get("mean", None)
        var = state.get("var", None)
        if mean is None or var is None:
            return None
        mean = mean.detach().to(device).float()
        var = var.detach().to(device).float().clamp_min(1e-6)
        if mean.dim() != 1 or var.dim() != 1 or mean.numel() != var.numel():
            return None
        return {"mean": mean, "var": var}

    def _cluster_distance(self, z, cluster_meta):
        if cluster_meta is None:
            return None
        mean = cluster_meta["mean"]
        var = cluster_meta["var"]
        if z.dim() != 2 or z.size(1) != mean.numel():
            return None
        diff = z - mean.unsqueeze(0)
        if self.pm_teacher_cluster_dist_metric == "euclidean":
            dist = diff.pow(2).mean(dim=1)
        else:
            dist = (diff.pow(2) / var.unsqueeze(0)).mean(dim=1)
        return dist.clamp_min(0.0)

    def _prepare_teacher_proto_meta(self, state, device):
        if not state:
            return None
        required = {"counts", "feat_sum", "feat_sq_sum", "logit_sum"}
        if not required.issubset(set(state.keys())):
            return None
        counts = state["counts"].detach().to(device).float()
        feat_sum = state["feat_sum"].detach().to(device).float()
        if counts.dim() != 1 or feat_sum.dim() != 2 or feat_sum.size(0) != counts.numel():
            return None
        denom = counts.clamp_min(1e-12).unsqueeze(1)
        feat_mean = feat_sum / denom

        correct_count = state.get("correct_count", None)
        if correct_count is not None:
            correct_count = correct_count.detach().to(device).float()
            class_reliability = (correct_count / counts.clamp_min(1e-12)).clamp(0.0, 1.0)
        else:
            class_reliability = torch.ones_like(counts)

        max_count = counts.max().clamp_min(1.0)
        class_support = (torch.log1p(counts) / torch.log1p(max_count)).clamp(0.0, 1.0)
        return {
            "counts": counts,
            "feat_mean": feat_mean,
            "class_reliability": class_reliability,
            "class_support": class_support,
        }

    def _teacher_competence_score(self, z, pm_prob, proto_meta):
        if proto_meta is None:
            return None
        pred = torch.argmax(pm_prob, dim=1)
        rel = proto_meta["class_reliability"].index_select(0, pred)
        support = proto_meta["class_support"].index_select(0, pred)
        proto_mean = proto_meta["feat_mean"].index_select(0, pred)
        z_norm = F.normalize(z, dim=1)
        proto_norm = F.normalize(proto_mean, dim=1)
        proto_match = ((z_norm * proto_norm).sum(dim=1) + 1.0) * 0.5
        proto_match = proto_match.clamp(0.0, 1.0)

        weights = torch.tensor(
            [
                self.pm_teacher_competence_rel_weight,
                self.pm_teacher_competence_support_weight,
                self.pm_teacher_competence_proto_weight,
            ],
            device=z.device,
            dtype=z.dtype,
        )
        total = weights.sum().clamp_min(1e-12)
        competence = (
            weights[0] * rel.to(z.dtype)
            + weights[1] * support.to(z.dtype)
            + weights[2] * proto_match.to(z.dtype)
        ) / total
        return competence.clamp(0.0, 1.0)

    def update_gm_from_pm_prototypes(self, received_protos):
        if not received_protos:
            return None

        components, num_classes, feat_dim = self._collect_pm_prototype_components(received_protos)
        if not components:
            print("[FedCD] No valid PM prototype components. Skip prototype GM update.")
            return None

        comp_labels = torch.tensor([c["class_id"] for c in components], dtype=torch.long)
        comp_weights = torch.tensor([c["count"] for c in components], dtype=torch.float32)
        comp_weights = comp_weights / comp_weights.sum().clamp_min(1e-12)
        feat_mean = torch.stack([c["feat_mean"] for c in components], dim=0).float()
        feat_var = torch.stack([c["feat_var"] for c in components], dim=0).float()
        teacher_logits = torch.stack([c["teacher_logits"] for c in components], dim=0).float()

        valid_classes = int(comp_labels.unique().numel())
        total_proto_samples = float(sum(c["count"] for c in components))
        print(
            f"[FedCD] Prototype GM update (multimodal): components={len(components)}, "
            f"valid_classes={valid_classes}/{num_classes}, feat_dim={feat_dim}, "
            f"total_proto_samples={total_proto_samples:.0f}"
        )

        def _update_once(device):
            use_amp = device == "cuda" and bool(getattr(self.args, "amp", False))
            temp = max(1e-6, float(self.proto_teacher_temp))

            gm_module = copy.deepcopy(self.generalized_module).to(device)
            gm_module.train()
            gm_adapter = None
            if self.generalized_adapter is not None:
                gm_adapter = copy.deepcopy(self.generalized_adapter).to(device)
                gm_adapter.train()

            params = list(gm_module.parameters())
            if gm_adapter is not None:
                params += list(gm_adapter.parameters())
            optimizer = torch.optim.SGD(params, lr=self.proto_teacher_lr)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            ce_loss_fn = nn.CrossEntropyLoss()

            comp_weights_dev = comp_weights.to(device)
            comp_labels_dev = comp_labels.to(device)
            feat_mean_dev = feat_mean.to(device)
            feat_var_dev = feat_var.to(device)
            teacher_logits_dev = teacher_logits.to(device)

            steps = max(1, int(self.proto_teacher_steps))
            batch_size = max(1, int(self.proto_teacher_batch_size))
            noise_scale = max(0.0, float(self.proto_teacher_noise_scale))

            for _ in range(steps):
                sampled_comp = torch.multinomial(comp_weights_dev, batch_size, replacement=True)
                labels = comp_labels_dev[sampled_comp].long()
                mu = feat_mean_dev[sampled_comp]

                if noise_scale > 0:
                    std = torch.sqrt(feat_var_dev[sampled_comp])
                    z = mu + noise_scale * std * torch.randn_like(mu)
                else:
                    z = mu

                with torch.cuda.amp.autocast(enabled=use_amp):
                    z_gm = gm_adapter(z) if gm_adapter is not None else z
                    gm_logits = gm_module(z_gm)

                    loss = gm_logits.new_tensor(0.0)
                    if self.proto_teacher_ce_weight > 0:
                        loss = loss + self.proto_teacher_ce_weight * ce_loss_fn(gm_logits, labels)
                    if self.proto_teacher_kl_weight > 0:
                        teacher_prob = torch.softmax(teacher_logits_dev[sampled_comp] / temp, dim=1)
                        student_log_prob = torch.log_softmax(gm_logits / temp, dim=1)
                        if self.proto_teacher_confidence_weight:
                            teacher_conf = self._teacher_confidence(teacher_prob)
                            kd_weight = self.proto_teacher_confidence_min + (
                                1.0 - self.proto_teacher_confidence_min
                            ) * teacher_conf.pow(self.proto_teacher_confidence_power)
                            kl_per_sample = F.kl_div(
                                student_log_prob, teacher_prob, reduction="none"
                            ).sum(dim=1) * (temp * temp)
                            kl_loss = (kl_per_sample * kd_weight).mean()
                        else:
                            kl_loss = F.kl_div(
                                student_log_prob, teacher_prob, reduction="batchmean"
                            ) * (temp * temp)
                        loss = loss + self.proto_teacher_kl_weight * kl_loss

                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            updated_state = {
                f"generalized_module.{k}": v.detach().cpu()
                for k, v in gm_module.state_dict().items()
            }
            if gm_adapter is not None:
                updated_state.update({
                    f"generalized_adapter.{k}": v.detach().cpu()
                    for k, v in gm_adapter.state_dict().items()
                })
                gm_adapter.to("cpu")
            gm_module.to("cpu")
            if device == "cuda" and self.args.avoid_oom:
                torch.cuda.empty_cache()
            return updated_state

        try:
            updated_state = _update_once(self.device)
        except RuntimeError as err:
            if self.device == "cuda" and "out of memory" in str(err).lower():
                print("[Warn] OOM during prototype GM update. Falling back to CPU.")
                torch.cuda.empty_cache()
                updated_state = _update_once("cpu")
            else:
                raise

        if not updated_state:
            return None

        for cluster_id in set(self.cluster_map.values()):
            self.cluster_generalized_states[cluster_id] = {
                k: v.clone() for k, v in updated_state.items()
            }
        self._apply_generalized_state_to_server(updated_state)
        return updated_state

    def update_gm_hybrid_local_proto(self, received_gms, received_protos):
        """
        Hybrid GM update:
        1) FedAvg local GM uploads (label-supervised local learning signal)
        2) Optional prototype-based refinement from PM uploads
        3) Blend FedAvg and refined GM states for stability
        """
        fedavg_state = self.aggregate_global_gms(received_gms) if received_gms else None
        if fedavg_state is None and not received_protos:
            print("[FedCD] Hybrid GM update skipped: no GM uploads and no PM prototypes.")
            return None

        if fedavg_state is not None:
            # Use FedAvg GM as the base state before refinement.
            self._apply_generalized_state_to_server(fedavg_state)
            for cluster_id in set(self.cluster_map.values()):
                self.cluster_generalized_states[cluster_id] = {
                    k: v.clone() for k, v in fedavg_state.items()
                }

        if not received_protos or self.hybrid_proto_blend <= 0.0:
            if self.hybrid_proto_blend <= 0.0:
                print("[FedCD] Hybrid GM update: prototype refinement disabled by blend=0.")
            return fedavg_state

        refined_state = self.update_gm_from_pm_prototypes(received_protos)
        if refined_state is None:
            return fedavg_state

        if fedavg_state is None:
            print("[FedCD] Hybrid GM update: no local GM uploads, using prototype-refined GM only.")
            return refined_state

        beta = self.hybrid_proto_blend
        if beta >= 1.0:
            return refined_state

        # Blend preserves local supervised signal while injecting PM-prototype knowledge.
        mixed_state = {}
        all_keys = set(fedavg_state.keys()) | set(refined_state.keys())
        for key in all_keys:
            base = fedavg_state.get(key)
            refined = refined_state.get(key)
            if base is None:
                mixed_state[key] = refined.clone()
            elif refined is None:
                mixed_state[key] = base.clone()
            else:
                mixed_state[key] = (1.0 - beta) * base + beta * refined

        self._apply_generalized_state_to_server(mixed_state)
        for cluster_id in set(self.cluster_map.values()):
            self.cluster_generalized_states[cluster_id] = {
                k: v.clone() for k, v in mixed_state.items()
            }
        return mixed_state

    def _extract_personalized_state(self, state):
        personalized_module_state = {
            k.replace("personalized_module.", ""): v
            for k, v in state.items()
            if k.startswith("personalized_module.")
        }
        personalized_adapter_state = {
            k.replace("personalized_adapter.", ""): v
            for k, v in state.items()
            if k.startswith("personalized_adapter.")
        }

        if not personalized_module_state:
            head_state = {k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")}
            final_state = {k.replace("final.", ""): v for k, v in state.items() if k.startswith("final.")}
            if head_state or final_state:
                personalized_module_state = dict(head_state)
                if final_state:
                    if isinstance(self.personalized_module, nn.Sequential) and len(self.personalized_module) > 0:
                        last_idx = str(len(self.personalized_module) - 1)
                        for key, value in final_state.items():
                            personalized_module_state[f"{last_idx}.{key}"] = value
                    else:
                        personalized_module_state.update(final_state)
            if not personalized_adapter_state:
                personalized_adapter_state = {
                    k.replace("adapter.", ""): v for k, v in state.items() if k.startswith("adapter.")
                }

        return personalized_module_state, personalized_adapter_state

    def distill_global_gm_from_pm_teachers(self, received_pms, client_weights=None, teacher_proto_states=None):
        if not received_pms:
            return None
        if not self.pm_teacher_datafree_enable and self.pm_teacher_loader is None:
            print("[FedCD] PM-teacher distillation loader is unavailable. Skip GM update.")
            return None
        if client_weights is None:
            client_weights = {}
        if teacher_proto_states is None:
            teacher_proto_states = {}

        def _distill_once(device):
            self.f_ext.to(device)
            self.f_ext.eval()
            use_amp = device == "cuda" and bool(getattr(self.args, "amp", False))
            temp = max(1e-6, self.pm_teacher_temp)
            ce_loss_fn = nn.CrossEntropyLoss()

            teacher_components = []
            for client_id, state in received_pms:
                pm_module_state, pm_adapter_state = self._extract_personalized_state(state)
                if not pm_module_state:
                    continue

                pm_module = copy.deepcopy(self.personalized_module).to(device)
                pm_module.load_state_dict({k: v.to(device) for k, v in pm_module_state.items()}, strict=True)
                pm_module.eval()
                for p in pm_module.parameters():
                    p.requires_grad = False

                pm_adapter = None
                if self.personalized_adapter is not None:
                    pm_adapter = copy.deepcopy(self.personalized_adapter).to(device)
                    if pm_adapter_state:
                        pm_adapter.load_state_dict({k: v.to(device) for k, v in pm_adapter_state.items()}, strict=True)
                    pm_adapter.eval()
                    for p in pm_adapter.parameters():
                        p.requires_grad = False

                teacher_weight = float(client_weights.get(int(client_id), 1.0))
                proto_meta = None
                if self.pm_teacher_competence_filter or self.pm_teacher_datafree_enable:
                    proto_meta = self._prepare_teacher_proto_meta(
                        teacher_proto_states.get(int(client_id), None),
                        device,
                    )
                cluster_dist_meta = None
                if self.pm_teacher_cluster_dist_weighting and self.pm_teacher_source == "cluster":
                    cluster_dist_meta = self._prepare_cluster_dist_meta(
                        self.cluster_distribution_stats.get(int(client_id), None),
                        device,
                    )
                teacher_components.append({
                    "weight": max(teacher_weight, 1e-12),
                    "pm_module": pm_module,
                    "pm_adapter": pm_adapter,
                    "proto_meta": proto_meta,
                    "cluster_dist_meta": cluster_dist_meta,
                })

            if not teacher_components:
                print("[FedCD] No valid PM teachers for server distillation. Skip GM update.")
                self.f_ext.to("cpu")
                return None
            total_teacher_weight = sum(comp["weight"] for comp in teacher_components)
            total_teacher_weight = max(float(total_teacher_weight), 1e-12)
            for comp in teacher_components:
                comp["norm_weight"] = float(comp["weight"]) / total_teacher_weight

            feature_dim = self.pm_teacher_feature_dim
            if feature_dim is None:
                for comp in teacher_components:
                    proto_meta = comp.get("proto_meta")
                    if proto_meta is not None and "feat_mean" in proto_meta:
                        feature_dim = int(proto_meta["feat_mean"].size(1))
                        break
                    cluster_meta = comp.get("cluster_dist_meta")
                    if cluster_meta is not None and "mean" in cluster_meta:
                        feature_dim = int(cluster_meta["mean"].numel())
                        break
            if feature_dim is None:
                feature_dim = self._resolve_module_input_dim(
                    self.generalized_module,
                    self.personalized_module,
                )
            if feature_dim is None:
                print("[FedCD] Failed to infer feature dimension for PM-teacher distillation. Skip GM update.")
                self.f_ext.to("cpu")
                return None

            feature_generator = None
            feature_generator_optimizer = None
            if self.pm_teacher_datafree_enable and self.pm_teacher_datafree_generator_enable:
                feature_generator = self._ensure_pm_teacher_feature_generator(feature_dim).to(device)
                feature_generator.train()
                feature_generator_optimizer = torch.optim.Adam(
                    feature_generator.parameters(),
                    lr=self.pm_teacher_datafree_generator_lr,
                )

            def _build_teacher_stacks(z):
                teacher_prob_list = []
                teacher_score_list = []
                teacher_conf_list = []
                teacher_competence_list = []
                for comp in teacher_components:
                    pm_adapter = comp["pm_adapter"]
                    z_pm = pm_adapter(z) if pm_adapter is not None else z
                    pm_logits = comp["pm_module"](z_pm)
                    pm_prob = torch.softmax(pm_logits / temp, dim=1)

                    if self.pm_teacher_ensemble_confidence:
                        pm_conf = self._teacher_confidence(pm_prob)
                        pm_conf = self.pm_teacher_confidence_min + (
                            1.0 - self.pm_teacher_confidence_min
                        ) * pm_conf.pow(self.pm_teacher_confidence_power)
                    else:
                        pm_conf = torch.ones(
                            (pm_prob.size(0),),
                            device=pm_prob.device,
                            dtype=pm_prob.dtype,
                        )

                    competence = torch.ones_like(pm_conf)
                    if self.pm_teacher_competence_filter or self.pm_teacher_datafree_enable:
                        competence_score = self._teacher_competence_score(
                            z, pm_prob, comp.get("proto_meta")
                        )
                        if competence_score is not None:
                            competence = competence_score
                            if self.pm_teacher_competence_filter:
                                competence_mask = competence >= self.pm_teacher_competence_min
                                competence = competence * competence_mask.to(competence.dtype)

                    score = pm_conf * competence * float(comp["norm_weight"])
                    teacher_prob_list.append(pm_prob)
                    teacher_score_list.append(score)
                    teacher_conf_list.append(pm_conf)
                    teacher_competence_list.append(competence)

                if not teacher_prob_list:
                    return None

                teacher_prob_stack = torch.stack(teacher_prob_list, dim=0)
                teacher_score_stack = torch.stack(teacher_score_list, dim=0)
                teacher_conf_stack = torch.stack(teacher_conf_list, dim=0)
                teacher_competence_stack = torch.stack(teacher_competence_list, dim=0)
                teacher_pred_stack = torch.argmax(teacher_prob_stack, dim=2)

                use_cluster_dist_prior = (
                    self.pm_teacher_cluster_dist_weighting
                    and self.pm_teacher_source == "cluster"
                    and len(teacher_components) > 1
                    and all(comp.get("cluster_dist_meta") is not None for comp in teacher_components)
                )
                if use_cluster_dist_prior:
                    dist_list = []
                    for comp in teacher_components:
                        dist = self._cluster_distance(z, comp["cluster_dist_meta"])
                        if dist is None:
                            dist_list = []
                            break
                        dist_list.append(dist)
                    if dist_list:
                        dist_stack = torch.stack(dist_list, dim=0)
                        cluster_prior = torch.softmax(
                            -dist_stack / self.pm_teacher_cluster_dist_tau,
                            dim=0,
                        )
                        teacher_score_stack = teacher_score_stack * cluster_prior

                return (
                    teacher_prob_stack,
                    teacher_score_stack,
                    teacher_conf_stack,
                    teacher_competence_stack,
                    teacher_pred_stack,
                )

            def _aggregate_teacher_targets(stacks, y=None):
                if stacks is None:
                    return None
                (
                    teacher_prob_stack,
                    teacher_score_stack,
                    teacher_conf_stack,
                    teacher_competence_stack,
                    teacher_pred_stack,
                ) = stacks

                if self.pm_teacher_teacher_abstain_threshold > 0:
                    teacher_active_mask = (
                        teacher_conf_stack >= self.pm_teacher_teacher_abstain_threshold
                    )
                    teacher_score_stack = teacher_score_stack * teacher_active_mask.to(
                        teacher_score_stack.dtype
                    )

                num_teachers = teacher_score_stack.size(0)
                topk = num_teachers if self.pm_teacher_topk <= 0 else min(self.pm_teacher_topk, num_teachers)
                if topk < num_teachers:
                    topk_scores, topk_idx = torch.topk(
                        teacher_score_stack,
                        k=topk,
                        dim=0,
                        largest=True,
                        sorted=False,
                    )
                    gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, teacher_prob_stack.size(-1))
                    selected_prob = teacher_prob_stack.gather(0, gather_idx)
                    selected_score = topk_scores
                    selected_conf = teacher_conf_stack.gather(0, topk_idx)
                    selected_comp = teacher_competence_stack.gather(0, topk_idx)
                    selected_pred = teacher_pred_stack.gather(0, topk_idx)
                else:
                    selected_prob = teacher_prob_stack
                    selected_score = teacher_score_stack
                    selected_conf = teacher_conf_stack
                    selected_comp = teacher_competence_stack
                    selected_pred = teacher_pred_stack

                score_sum = selected_score.sum(dim=0)
                active_teacher_mask = selected_score > 1e-12
                active_teacher_count = active_teacher_mask.sum(dim=0)
                valid_teacher_mask = score_sum > 1e-12
                valid_teacher_mask = valid_teacher_mask & (
                    active_teacher_count >= self.pm_teacher_min_active_teachers
                )
                if self.pm_teacher_abstain_threshold > 0:
                    selected_conf_active = torch.where(
                        active_teacher_mask,
                        selected_conf,
                        torch.full_like(selected_conf, -1.0),
                    )
                    max_teacher_conf = selected_conf_active.max(dim=0).values
                    valid_teacher_mask = valid_teacher_mask & (
                        max_teacher_conf >= self.pm_teacher_abstain_threshold
                    )
                if self.pm_teacher_consensus_min_ratio > 0:
                    vote_score = torch.zeros(
                        (selected_score.size(1), selected_prob.size(2)),
                        device=selected_score.device,
                        dtype=selected_score.dtype,
                    )
                    vote_score.scatter_add_(
                        1,
                        selected_pred.transpose(0, 1),
                        selected_score.transpose(0, 1),
                    )
                    consensus_ratio = vote_score.max(dim=1).values / score_sum.clamp_min(1e-12)
                    valid_teacher_mask = valid_teacher_mask & (
                        consensus_ratio >= self.pm_teacher_consensus_min_ratio
                    )
                if not bool(valid_teacher_mask.any()):
                    return None

                teacher_prob = (
                    selected_prob * selected_score.unsqueeze(-1)
                ).sum(dim=0) / score_sum.clamp_min(1e-12).unsqueeze(-1)
                if self.pm_teacher_correct_only and y is not None:
                    label_mask = (y >= 0) & (y < teacher_prob.size(1))
                    teacher_pred = torch.argmax(teacher_prob, dim=1)
                    valid_teacher_mask = valid_teacher_mask & label_mask & teacher_pred.eq(y)
                    if not bool(valid_teacher_mask.any()):
                        return None

                competence_weight = None
                if self.pm_teacher_competence_filter or self.pm_teacher_datafree_enable:
                    competence_weight = (
                        (selected_comp * active_teacher_mask.to(selected_comp.dtype)).sum(dim=0)
                        / active_teacher_count.clamp_min(1).to(selected_comp.dtype)
                    ).clamp(0.0, 1.0)

                return teacher_prob, valid_teacher_mask, competence_weight

            def _sample_proto_features(batch_size):
                if not self.pm_teacher_datafree_init_from_proto:
                    return None
                proto_bank = []
                for comp in teacher_components:
                    proto_meta = comp.get("proto_meta")
                    if proto_meta is None:
                        continue
                    feat_mean = proto_meta.get("feat_mean", None)
                    counts = proto_meta.get("counts", None)
                    if feat_mean is None or counts is None:
                        continue
                    valid = counts > 0
                    if bool(valid.any()):
                        proto_bank.append(feat_mean[valid])
                if not proto_bank:
                    return None
                proto_bank = torch.cat(proto_bank, dim=0)
                idx = torch.randint(proto_bank.size(0), (batch_size,), device=device)
                z0 = proto_bank.index_select(0, idx).clone()
                if self.pm_teacher_datafree_noise_scale > 0:
                    z0 = z0 + self.pm_teacher_datafree_noise_scale * torch.randn_like(z0)
                return z0

            def _generate_datafree_features():
                init_z = _sample_proto_features(self.pm_teacher_batch_size)
                if feature_generator is not None:
                    if init_z is None:
                        anchor = torch.zeros(
                            self.pm_teacher_batch_size,
                            feature_dim,
                            device=device,
                        )
                    else:
                        anchor = init_z.detach().clone()
                    noise = torch.randn(
                        self.pm_teacher_batch_size,
                        self.pm_teacher_datafree_generator_noise_dim,
                        device=device,
                    )
                    for _ in range(self.pm_teacher_datafree_generator_steps):
                        z_syn = feature_generator(noise, anchor)
                        stacks = _build_teacher_stacks(z_syn)
                        if stacks is None:
                            break
                        teacher_prob_stack, teacher_score_stack, _, _, _ = stacks
                        score_sum = teacher_score_stack.sum(dim=0)
                        if bool((score_sum > 1e-12).any()):
                            teacher_prob = (
                                teacher_prob_stack * teacher_score_stack.unsqueeze(-1)
                            ).sum(dim=0) / score_sum.clamp_min(1e-12).unsqueeze(-1)
                        else:
                            teacher_prob = teacher_prob_stack.mean(dim=0)
                        sample_entropy = -(
                            teacher_prob * torch.log(teacher_prob.clamp_min(1e-12))
                        ).sum(dim=1).mean()
                        mean_prob = teacher_prob.mean(dim=0)
                        diversity = -(
                            mean_prob * torch.log(mean_prob.clamp_min(1e-12))
                        ).sum()
                        anchor_penalty = (z_syn - anchor).pow(2).mean()
                        generator_loss = (
                            self.pm_teacher_datafree_entropy_weight * sample_entropy
                            - self.pm_teacher_datafree_diversity_weight * diversity
                            + self.pm_teacher_datafree_l2_weight * z_syn.pow(2).mean()
                            + self.pm_teacher_datafree_generator_anchor_weight * anchor_penalty
                        )
                        if not torch.isfinite(generator_loss):
                            break
                        feature_generator_optimizer.zero_grad()
                        generator_loss.backward()
                        feature_generator_optimizer.step()
                    return feature_generator(noise, anchor).detach()
                if init_z is None:
                    scale = self.pm_teacher_datafree_noise_scale if self.pm_teacher_datafree_noise_scale > 0 else 1.0
                    init_z = torch.randn(
                        self.pm_teacher_batch_size,
                        feature_dim,
                        device=device,
                    ) * scale
                z_syn = init_z.detach().clone().requires_grad_(True)
                synth_optimizer = torch.optim.Adam([z_syn], lr=self.pm_teacher_datafree_lr)
                for _ in range(self.pm_teacher_datafree_steps):
                    stacks = _build_teacher_stacks(z_syn)
                    if stacks is None:
                        break
                    teacher_prob_stack, teacher_score_stack, _, _, _ = stacks
                    score_sum = teacher_score_stack.sum(dim=0)
                    if bool((score_sum > 1e-12).any()):
                        teacher_prob = (
                            teacher_prob_stack * teacher_score_stack.unsqueeze(-1)
                        ).sum(dim=0) / score_sum.clamp_min(1e-12).unsqueeze(-1)
                    else:
                        teacher_prob = teacher_prob_stack.mean(dim=0)
                    sample_entropy = -(
                        teacher_prob * torch.log(teacher_prob.clamp_min(1e-12))
                    ).sum(dim=1).mean()
                    mean_prob = teacher_prob.mean(dim=0)
                    diversity = -(
                        mean_prob * torch.log(mean_prob.clamp_min(1e-12))
                    ).sum()
                    synth_loss = (
                        self.pm_teacher_datafree_entropy_weight * sample_entropy
                        - self.pm_teacher_datafree_diversity_weight * diversity
                        + self.pm_teacher_datafree_l2_weight * z_syn.pow(2).mean()
                    )
                    if not torch.isfinite(synth_loss):
                        break
                    synth_optimizer.zero_grad()
                    synth_loss.backward()
                    synth_optimizer.step()
                return z_syn.detach()

            gm_module = copy.deepcopy(self.generalized_module).to(device)
            gm_module.train()
            gm_adapter = None
            if self.generalized_adapter is not None:
                gm_adapter = copy.deepcopy(self.generalized_adapter).to(device)
                gm_adapter.train()

            student_params = list(gm_module.parameters())
            if gm_adapter is not None:
                student_params += list(gm_adapter.parameters())
            optimizer = torch.optim.SGD(student_params, lr=self.pm_teacher_lr)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            for epoch_idx in range(self.pm_teacher_epochs):
                desc = (
                    f"Distill GM from PM teachers [{epoch_idx + 1}/{self.pm_teacher_epochs}]"
                    if self.pm_teacher_epochs > 1
                    else "Distill GM from PM teachers"
                )
                if self.pm_teacher_datafree_enable:
                    batch_iter = range(self.pm_teacher_datafree_batches)
                else:
                    batch_iter = self.pm_teacher_loader

                for batch in tqdm(batch_iter, desc=desc, leave=False):
                    y = None
                    if self.pm_teacher_datafree_enable:
                        z = _generate_datafree_features()
                    else:
                        x, y = batch
                        if type(x) == type([]):
                            x = x[0]
                        x = x.to(device, non_blocking=(device == "cuda"))
                        if self.pm_teacher_ce_weight > 0 or self.pm_teacher_correct_only:
                            y = y.to(device, non_blocking=(device == "cuda")).long()
                        with torch.no_grad():
                            z = self.f_ext(x)
                            if z.dim() > 2:
                                z = torch.flatten(z, 1)

                    with torch.no_grad():
                        targets = _aggregate_teacher_targets(_build_teacher_stacks(z), y=y)
                        if targets is None:
                            continue
                        teacher_prob, valid_teacher_mask, competence_weight = targets

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        z_gm = gm_adapter(z) if gm_adapter is not None else z
                        gm_logits = gm_module(z_gm)
                        student_log_prob = torch.log_softmax(gm_logits / temp, dim=1)
                        kd_per_sample = F.kl_div(
                            student_log_prob,
                            teacher_prob,
                            reduction="none",
                        ).sum(dim=1) * (temp * temp)

                        if self.pm_teacher_confidence_weight:
                            teacher_conf = self._teacher_confidence(teacher_prob)
                            # Down-weight uncertain teacher samples under proxy-target domain mismatch.
                            kd_weight = self.pm_teacher_confidence_min + (
                                1.0 - self.pm_teacher_confidence_min
                            ) * teacher_conf.pow(self.pm_teacher_confidence_power)
                            if competence_weight is not None:
                                kd_weight = kd_weight * competence_weight.clamp(0.0, 1.0)
                            kd_weight = kd_weight * valid_teacher_mask.to(kd_weight.dtype)
                            kd_loss = (kd_per_sample * kd_weight).sum() / kd_weight.sum().clamp_min(1e-12)
                        else:
                            kd_valid = kd_per_sample[valid_teacher_mask]
                            if kd_valid.numel() == 0:
                                continue
                            kd_loss = kd_valid.mean()

                        ce_loss = gm_logits.new_tensor(0.0)
                        gm_logits_valid = gm_logits[valid_teacher_mask]
                        teacher_prob_valid = teacher_prob[valid_teacher_mask]
                        if self.pm_teacher_ce_weight > 0 and y is not None:
                            y_valid = y[valid_teacher_mask]
                            if y_valid.numel() > 0:
                                ce_loss = ce_loss_fn(gm_logits_valid, y_valid)
                        rel_loss = gm_logits.new_tensor(0.0)
                        if self.pm_teacher_rel_weight > 0:
                            rel_loss = self._relation_kd_loss(
                                gm_logits_valid,
                                teacher_prob_valid,
                                max_samples=self.pm_teacher_rel_batch,
                            )
                        loss = (
                            self.pm_teacher_kl_weight * kd_loss
                            + self.pm_teacher_ce_weight * ce_loss
                            + self.pm_teacher_rel_weight * rel_loss
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

            for comp in teacher_components:
                comp["pm_module"].to("cpu")
                if comp["pm_adapter"] is not None:
                    comp["pm_adapter"].to("cpu")
            if feature_generator is not None:
                feature_generator.to("cpu")

            self.f_ext.to("cpu")
            if device == "cuda" and self.args.avoid_oom:
                torch.cuda.empty_cache()

            return distilled_state

        try:
            distilled_state = _distill_once(self.device)
        except RuntimeError as err:
            if self.device == "cuda" and "out of memory" in str(err).lower():
                print("[Warn] OOM during PM-teacher GM distillation. Falling back to CPU.")
                torch.cuda.empty_cache()
                distilled_state = _distill_once("cpu")
            else:
                raise

        if not distilled_state:
            return None

        for cluster_id in set(self.cluster_map.values()):
            self.cluster_generalized_states[cluster_id] = {
                k: v.clone() for k, v in distilled_state.items()
            }
        self._apply_generalized_state_to_server(distilled_state)
        return distilled_state

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
                self.threshold_step = min(abs(self.threshold_step), self.threshold_step_max)
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
        effective_step = min(abs(self.threshold_step), self.threshold_step_max)
        self.current_threshold += self.act_direction * effective_step
        self.current_threshold = max(0.01, min(self.threshold_max, self.current_threshold))
        
        print(f"[ACT] Updated Threshold: {old_th:.4f} -> {self.current_threshold:.4f} (Dir: {self.act_direction}, Step: {effective_step:.4f})")

    def cluster_clients_by_distribution(self):
        # Cluster clients by f_ext feature distribution stats
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.preprocessing import normalize

        features = []
        client_ids = []
        client_stats = {}
        for client in self.clients:
            mean, var = client.get_feature_stats(self.cluster_sample_size)
            # Flatten to ensure 1D vectors before concatenation
            feat = torch.cat([mean.flatten(), var.flatten()], dim=0).cpu().numpy()
            features.append(feat)
            client_ids.append(client.id)
            client_stats[int(client.id)] = {
                "mean": mean.detach().cpu(),
                "var": var.detach().cpu(),
                "weight": float(max(1, getattr(client, "train_samples", 1))),
            }

        self.client_distribution_stats = client_stats

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

    def _build_pm_teacher_loader(self):
        distill_data = None
        distill_source = ""

        try:
            distill_data, distill_source = self._build_proxy_distill_data()
        except Exception as err:
            print(f"[FedCD] Failed to build proxy distill data: {err}")
            distill_data = None

        if distill_data is None:
            if not self.pm_teacher_allow_test_fallback:
                print(
                    "[FedCD] PM-teacher distillation set is unavailable and test fallback is disabled. "
                    "Skipping server distillation."
                )
                return None
            distill_data = []
            for client_id in range(self.num_clients):
                distill_data.extend(read_client_data(self.dataset, client_id, is_train=False))
            distill_source = "client_test_union_fallback"

        if len(distill_data) == 0:
            print("[FedCD] PM-teacher distillation set is empty. Skipping server distillation.")
            return None

        if self.pm_teacher_samples > 0 and self.pm_teacher_samples < len(distill_data):
            rng = random.Random(0)
            sample_indices = rng.sample(range(len(distill_data)), self.pm_teacher_samples)
            distill_data = (
                [distill_data[idx] for idx in sample_indices]
                if isinstance(distill_data, list)
                else torch.utils.data.Subset(distill_data, sample_indices)
            )

        num_workers = int(getattr(self.args, "num_workers", 0))
        pin_memory = bool(getattr(self.args, "pin_memory", False)) and self.device == "cuda"
        loader_kwargs = {
            "batch_size": self.pm_teacher_batch_size,
            "shuffle": True,
            "drop_last": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = int(getattr(self.args, "prefetch_factor", 2))

        print(f"[FedCD] PM-teacher Distill Source: {distill_source}")
        print(f"[FedCD] PM-teacher Distill Set Size: {len(distill_data)}")
        return torch.utils.data.DataLoader(distill_data, **loader_kwargs)

    @staticmethod
    def _teacher_confidence(prob):
        eps = 1e-12
        num_classes = prob.size(1)
        entropy = -(prob * torch.log(prob.clamp_min(eps))).sum(dim=1)
        max_entropy = torch.log(torch.tensor(float(num_classes), device=prob.device)).clamp_min(eps)
        confidence = 1.0 - (entropy / max_entropy)
        return confidence.clamp(0.0, 1.0)

    @staticmethod
    def _relation_kd_loss(student_logits, teacher_prob, max_samples=64):
        # Relational KD on sample-similarity graphs:
        # preserve inter-sample structure from teacher predictions.
        if student_logits.dim() != 2 or teacher_prob.dim() != 2:
            return student_logits.new_zeros(())
        n = student_logits.size(0)
        if n <= 1:
            return student_logits.new_zeros(())
        if max_samples is not None and int(max_samples) > 0 and n > int(max_samples):
            idx = torch.randperm(n, device=student_logits.device)[: int(max_samples)]
            student_logits = student_logits.index_select(0, idx)
            teacher_prob = teacher_prob.index_select(0, idx)

        student_repr = F.normalize(student_logits, p=2, dim=1)
        teacher_repr = torch.log(teacher_prob.clamp_min(1e-12))
        teacher_repr = teacher_repr - teacher_repr.mean(dim=1, keepdim=True)
        teacher_repr = F.normalize(teacher_repr, p=2, dim=1)

        student_rel = torch.matmul(student_repr, student_repr.t())
        teacher_rel = torch.matmul(teacher_repr, teacher_repr.t())
        mask = ~torch.eye(student_rel.size(0), dtype=torch.bool, device=student_rel.device)
        if mask.sum() == 0:
            return student_logits.new_zeros(())
        return F.mse_loss(student_rel[mask], teacher_rel[mask], reduction="mean")

    def _core_root(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    def _default_proxy_root(self, proxy_dataset):
        core_root = self._core_root()
        name = proxy_dataset.lower()
        if name in {"mnist"}:
            return os.path.join(core_root, "dataset", "MNIST", "rawdata")
        if name in {"fashionmnist", "fashion-mnist", "fashion_mnist"}:
            return os.path.join(core_root, "dataset", "FashionMNIST", "rawdata")
        if name in {"cifar100", "cifar-100"}:
            return os.path.join(core_root, "dataset", "Cifar100", "rawdata")
        if name in {"cifar10", "cifar-10"}:
            return os.path.join(core_root, "dataset", "Cifar10", "rawdata")
        if name in {"tinyimagenet", "tiny-imagenet", "tiny_imagenet"}:
            return os.path.join(core_root, "dataset", "TinyImagenet", "rawdata", "tiny-imagenet-200", "train")
        return ""

    def _build_proxy_distill_data(self):
        proxy_name = self.pm_teacher_proxy_dataset.strip()
        if not proxy_name or proxy_name.lower() in {"none", "off", "disabled"}:
            return None, ""

        proxy_split = self.pm_teacher_proxy_split.strip().lower()
        if proxy_split not in {"train", "test", "all"}:
            raise ValueError(
                f"Invalid fedcd_pm_teacher_proxy_split={self.pm_teacher_proxy_split}. "
                "Use one of: train, test, all."
            )

        proxy_root = self.pm_teacher_proxy_root.strip() or self._default_proxy_root(proxy_name)
        if not proxy_root:
            raise ValueError(f"Unknown proxy dataset: {proxy_name}")

        name = proxy_name.lower()
        task_is_mnist = "MNIST" in str(getattr(self, "dataset", ""))
        if name in {"mnist", "fashionmnist", "fashion-mnist", "fashion_mnist"}:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ])
        elif task_is_mnist:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        if name in {"mnist"}:
            splits = []
            if proxy_split in {"train", "all"}:
                splits.append(
                    torchvision.datasets.MNIST(
                        root=proxy_root,
                        train=True,
                        download=self.pm_teacher_proxy_download,
                        transform=transform,
                    )
                )
            if proxy_split in {"test", "all"}:
                splits.append(
                    torchvision.datasets.MNIST(
                        root=proxy_root,
                        train=False,
                        download=self.pm_teacher_proxy_download,
                        transform=transform,
                    )
                )
            if len(splits) == 1:
                return splits[0], f"proxy:MNIST:{proxy_split}"
            return torch.utils.data.ConcatDataset(splits), f"proxy:MNIST:{proxy_split}"

        if name in {"fashionmnist", "fashion-mnist", "fashion_mnist"}:
            splits = []
            if proxy_split in {"train", "all"}:
                splits.append(
                    torchvision.datasets.FashionMNIST(
                        root=proxy_root,
                        train=True,
                        download=self.pm_teacher_proxy_download,
                        transform=transform,
                    )
                )
            if proxy_split in {"test", "all"}:
                splits.append(
                    torchvision.datasets.FashionMNIST(
                        root=proxy_root,
                        train=False,
                        download=self.pm_teacher_proxy_download,
                        transform=transform,
                    )
                )
            if len(splits) == 1:
                return splits[0], f"proxy:FashionMNIST:{proxy_split}"
            return torch.utils.data.ConcatDataset(splits), f"proxy:FashionMNIST:{proxy_split}"

        if name in {"cifar100", "cifar-100"}:
            splits = []
            if proxy_split in {"train", "all"}:
                splits.append(
                    torchvision.datasets.CIFAR100(
                        root=proxy_root,
                        train=True,
                        download=self.pm_teacher_proxy_download,
                        transform=transform,
                    )
                )
            if proxy_split in {"test", "all"}:
                splits.append(
                    torchvision.datasets.CIFAR100(
                        root=proxy_root,
                        train=False,
                        download=self.pm_teacher_proxy_download,
                        transform=transform,
                    )
                )
            if len(splits) == 1:
                return splits[0], f"proxy:CIFAR100:{proxy_split}"
            return torch.utils.data.ConcatDataset(splits), f"proxy:CIFAR100:{proxy_split}"

        if name in {"cifar10", "cifar-10"}:
            splits = []
            if proxy_split in {"train", "all"}:
                splits.append(
                    torchvision.datasets.CIFAR10(
                        root=proxy_root,
                        train=True,
                        download=self.pm_teacher_proxy_download,
                        transform=transform,
                    )
                )
            if proxy_split in {"test", "all"}:
                splits.append(
                    torchvision.datasets.CIFAR10(
                        root=proxy_root,
                        train=False,
                        download=self.pm_teacher_proxy_download,
                        transform=transform,
                    )
                )
            if len(splits) == 1:
                return splits[0], f"proxy:CIFAR10:{proxy_split}"
            return torch.utils.data.ConcatDataset(splits), f"proxy:CIFAR10:{proxy_split}"

        if name in {"tinyimagenet", "tiny-imagenet", "tiny_imagenet"}:
            train_root = proxy_root
            if os.path.basename(os.path.normpath(train_root)) == "train":
                tiny_root = os.path.dirname(train_root)
            else:
                tiny_root = train_root
                train_root = os.path.join(tiny_root, "train")
            val_root = os.path.join(tiny_root, "val")
            image_root = train_root
            if proxy_split == "test":
                image_root = val_root
            if proxy_split == "all":
                train_ds = torchvision.datasets.ImageFolder(root=train_root, transform=transform)
                val_ds = torchvision.datasets.ImageFolder(root=val_root, transform=transform)
                return (
                    torch.utils.data.ConcatDataset([train_ds, val_ds]),
                    "proxy:TinyImagenet:all",
                )
            if not os.path.isdir(image_root):
                raise FileNotFoundError(f"TinyImagenet split path not found: {image_root}")
            return torchvision.datasets.ImageFolder(root=image_root, transform=transform), f"proxy:TinyImagenet:{proxy_split}"

        raise ValueError(f"Unsupported proxy dataset: {proxy_name}")

    def evaluate_local_branch_test_accs(self):
        """
        Evaluate GM-only and PM-only accuracy on each client's local test split.
        Returns:
            (gm_local_acc, pm_local_acc) weighted by total local-test samples.
        """
        if not self.clients:
            self.last_local_gate_stats = None
            return None, None

        device = self.device
        use_non_blocking = device == "cuda" and bool(getattr(self.args, "pin_memory", False))
        total_samples = 0
        gm_correct_total = 0
        pm_correct_total = 0
        pm_weight_sum = 0.0
        pm_weight_min = float("inf")
        pm_weight_max = float("-inf")
        pm_weight_count = 0
        agree_with_pm = 0
        agree_with_gm = 0
        agree_count = 0

        for client in self.clients:
            testloader = client.load_test_data()
            client.model.to(device)
            client.model.eval()

            with torch.no_grad():
                for x, y in testloader:
                    if type(x) == type([]):
                        x = x[0]
                    x = x.to(device, non_blocking=use_non_blocking)
                    y = y.to(device, non_blocking=use_non_blocking)

                    fused_logits, gm_logits, pm_logits, pm_weight = client.infer_fused_logits_with_gate(x)
                    pred_fused = torch.argmax(fused_logits, dim=1)
                    pred_pm = torch.argmax(pm_logits, dim=1)
                    pred_gm = torch.argmax(gm_logits, dim=1)

                    gm_correct_total += (pred_gm == y).sum().item()
                    pm_correct_total += (pred_pm == y).sum().item()
                    total_samples += y.size(0)
                    agree_with_pm += (pred_fused == pred_pm).sum().item()
                    agree_with_gm += (pred_fused == pred_gm).sum().item()
                    agree_count += y.size(0)

                    pw = pm_weight.view(-1)
                    pm_weight_sum += pw.sum().item()
                    pm_weight_count += pw.numel()
                    pm_weight_min = min(pm_weight_min, pw.min().item())
                    pm_weight_max = max(pm_weight_max, pw.max().item())

            client.model.to("cpu")

        if device == "cuda":
            torch.cuda.empty_cache()

        if total_samples <= 0:
            self.last_local_gate_stats = None
            return None, None

        if pm_weight_count > 0:
            self.last_local_gate_stats = {
                "pm_weight_mean": pm_weight_sum / pm_weight_count,
                "pm_weight_min": pm_weight_min,
                "pm_weight_max": pm_weight_max,
                "agree_with_pm": agree_with_pm / max(agree_count, 1),
                "agree_with_gm": agree_with_gm / max(agree_count, 1),
            }
        else:
            self.last_local_gate_stats = None

        gm_local_acc = gm_correct_total / total_samples
        pm_local_acc = pm_correct_total / total_samples
        return gm_local_acc, pm_local_acc

    def evaluate_global_test_acc(self):
        if not self.eval_common_global or self.global_test_loader is None:
            self.last_global_gate_stats = None
            return None

        acc_sum = 0.0
        valid_clients = 0
        device = self.device
        use_non_blocking = device == "cuda" and bool(getattr(self.args, "pin_memory", False))
        pm_weight_sum = 0.0
        pm_weight_min = float("inf")
        pm_weight_max = float("-inf")
        pm_weight_count = 0
        agree_with_pm = 0
        agree_with_gm = 0
        agree_count = 0

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
                    output, gm_logits, pm_logits, pm_weight = client.infer_fused_logits_with_gate(x)
                    pred_fused = torch.argmax(output, dim=1)
                    pred_pm = torch.argmax(pm_logits, dim=1)
                    pred_gm = torch.argmax(gm_logits, dim=1)

                    correct += (pred_fused == y).sum().item()
                    total += y.size(0)
                    agree_with_pm += (pred_fused == pred_pm).sum().item()
                    agree_with_gm += (pred_fused == pred_gm).sum().item()
                    agree_count += y.size(0)

                    pw = pm_weight.view(-1)
                    pm_weight_sum += pw.sum().item()
                    pm_weight_count += pw.numel()
                    pm_weight_min = min(pm_weight_min, pw.min().item())
                    pm_weight_max = max(pm_weight_max, pw.max().item())

            client.model.to("cpu")
            if total > 0:
                acc_sum += correct / total
                valid_clients += 1

        if device == "cuda":
            torch.cuda.empty_cache()

        if pm_weight_count > 0:
            self.last_global_gate_stats = {
                "pm_weight_mean": pm_weight_sum / pm_weight_count,
                "pm_weight_min": pm_weight_min,
                "pm_weight_max": pm_weight_max,
                "agree_with_pm": agree_with_pm / max(agree_count, 1),
                "agree_with_gm": agree_with_gm / max(agree_count, 1),
            }
        else:
            self.last_global_gate_stats = None

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

    def evaluate_pm_only_global_test_acc(self):
        if not self.eval_common_global or self.global_test_loader is None:
            return None

        device = self.device
        use_non_blocking = device == "cuda" and bool(getattr(self.args, "pin_memory", False))
        acc_sum = 0.0
        valid_clients = 0

        for client in self.clients:
            client.model.to(device)
            client.model.eval()

            total = 0
            correct = 0
            with torch.no_grad():
                for x, y in self.global_test_loader:
                    if type(x) == type([]):
                        x = x[0]
                    x = x.to(device, non_blocking=use_non_blocking)
                    y = y.to(device, non_blocking=use_non_blocking)

                    _, _, logits, _ = client.infer_fused_logits_with_gate(x)

                    correct += (torch.argmax(logits, dim=1) == y).sum().item()
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
