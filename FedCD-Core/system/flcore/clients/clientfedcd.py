import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from flcore.clients.clientbase import Client
from flcore.trainmodel.models import SmallFExt
from utils.data_utils import read_client_data

class PMWrapper(nn.Module):
    def __init__(
        self,
        f_ext,
        generalized_module,
        personalized_module,
        num_classes,
        feature_dim=0,
        generalized_adapter=None,
        personalized_adapter=None,
        entropy_temp_pm=1.0,
        entropy_temp_gm=1.0,
        entropy_min_pm_weight=0.1,
        entropy_max_pm_weight=0.9,
        entropy_gate_tau=0.2,
        entropy_pm_bias=0.0,
        entropy_gm_bias=0.0,
        entropy_disagree_gm_boost=0.0,
        entropy_use_class_reliability=True,
        entropy_reliability_scale=0.7,
        entropy_hard_switch_margin=0.15,
        entropy_use_ood_gate=True,
        entropy_ood_scale=1.0,
    ):
        super().__init__()
        self.f_ext = f_ext
        self.generalized_module = generalized_module
        self.personalized_module = personalized_module
        self.generalized_adapter = generalized_adapter
        self.personalized_adapter = personalized_adapter
        self.entropy_temp_pm = max(float(entropy_temp_pm), 1e-6)
        self.entropy_temp_gm = max(float(entropy_temp_gm), 1e-6)
        self.entropy_min_pm_weight = float(entropy_min_pm_weight)
        self.entropy_max_pm_weight = float(entropy_max_pm_weight)
        self.entropy_gate_tau = max(float(entropy_gate_tau), 1e-6)
        self.entropy_pm_bias = float(entropy_pm_bias)
        self.entropy_gm_bias = float(entropy_gm_bias)
        self.entropy_disagree_gm_boost = max(float(entropy_disagree_gm_boost), 0.0)
        self.entropy_use_class_reliability = bool(entropy_use_class_reliability)
        self.entropy_reliability_scale = max(float(entropy_reliability_scale), 0.0)
        self.entropy_hard_switch_margin = max(float(entropy_hard_switch_margin), 0.0)
        self.entropy_use_ood_gate = bool(entropy_use_ood_gate)
        self.entropy_ood_scale = max(float(entropy_ood_scale), 1e-6)
        self.register_buffer(
            "pm_class_reliability",
            torch.full((int(num_classes),), 0.5, dtype=torch.float32),
        )
        self.register_buffer(
            "gm_class_reliability",
            torch.full((int(num_classes),), 0.5, dtype=torch.float32),
        )
        if int(feature_dim) > 0:
            self.register_buffer("feature_mean", torch.zeros((int(feature_dim),), dtype=torch.float32))
            self.register_buffer("feature_var", torch.ones((int(feature_dim),), dtype=torch.float32))
        else:
            self.feature_mean = None
            self.feature_var = None

    def set_class_reliability(self, pm_reliability, gm_reliability):
        with torch.no_grad():
            if pm_reliability is not None:
                pm_reliability = pm_reliability.detach().float().clamp(0.0, 1.0)
                self.pm_class_reliability.copy_(pm_reliability.to(self.pm_class_reliability.device))
            if gm_reliability is not None:
                gm_reliability = gm_reliability.detach().float().clamp(0.0, 1.0)
                self.gm_class_reliability.copy_(gm_reliability.to(self.gm_class_reliability.device))

    def set_feature_stats(self, feature_mean, feature_var):
        if self.feature_mean is None or self.feature_var is None:
            return
        with torch.no_grad():
            if feature_mean is not None:
                feature_mean = feature_mean.detach().float()
                self.feature_mean.copy_(feature_mean.to(self.feature_mean.device))
            if feature_var is not None:
                feature_var = feature_var.detach().float().clamp_min(1e-6)
                self.feature_var.copy_(feature_var.to(self.feature_var.device))

    @staticmethod
    def _normalized_entropy(prob):
        eps = 1e-12
        num_classes = prob.size(1)
        entropy = -(prob * torch.log(prob.clamp_min(eps))).sum(dim=1, keepdim=True)
        norm = torch.log(torch.tensor(float(num_classes), device=prob.device))
        return entropy / norm.clamp_min(eps)

    def mix_prob(self, gm_logits, pm_logits, feat=None):
        # Entropy-based confidence gating:
        # Use relative confidence (PM vs GM) so GM can dominate when PM is uncertain.
        pm_prob = torch.softmax(pm_logits / self.entropy_temp_pm, dim=1)
        gm_prob = torch.softmax(gm_logits / self.entropy_temp_gm, dim=1)
        pm_conf = 1.0 - self._normalized_entropy(pm_prob)
        gm_conf = 1.0 - self._normalized_entropy(gm_prob)
        pred_pm = torch.argmax(pm_prob, dim=1)
        pred_gm = torch.argmax(gm_prob, dim=1)

        if self.entropy_use_class_reliability and self.pm_class_reliability.numel() == pm_prob.size(1):
            pm_rel = self.pm_class_reliability.to(pm_prob.device).index_select(0, pred_pm).unsqueeze(1)
            gm_rel = self.gm_class_reliability.to(gm_prob.device).index_select(0, pred_gm).unsqueeze(1)
            if self.entropy_reliability_scale > 0:
                # Reliability-adjusted confidence amplifies trusted branch and suppresses weak branch.
                pm_conf = pm_conf * (1.0 + self.entropy_reliability_scale * (2.0 * pm_rel - 1.0))
                gm_conf = gm_conf * (1.0 + self.entropy_reliability_scale * (2.0 * gm_rel - 1.0))
                pm_conf = pm_conf.clamp(0.0, 2.0)
                gm_conf = gm_conf.clamp(0.0, 2.0)

        rel_pm_conf = torch.sigmoid(
            ((pm_conf + self.entropy_pm_bias) - (gm_conf + self.entropy_gm_bias))
            / self.entropy_gate_tau
        )
        min_w = float(self.entropy_min_pm_weight)
        max_w = float(self.entropy_max_pm_weight)
        if abs(max_w - min_w) < 1e-12:
            # Explicit fixed-ratio mode (e.g., 0.5/0.5) independent of confidence.
            pm_weight = torch.full_like(rel_pm_conf, min(max(min_w, 0.0), 1.0))
        elif max_w > min_w:
            span = max_w - min_w
            pm_weight = min_w + span * rel_pm_conf
        else:
            # Fallback for malformed range: use relative confidence directly.
            pm_weight = rel_pm_conf

        if self.entropy_hard_switch_margin > 0:
            conf_gap = pm_conf - gm_conf
            pm_weight = torch.where(
                conf_gap >= self.entropy_hard_switch_margin,
                torch.full_like(pm_weight, self.entropy_max_pm_weight),
                pm_weight,
            )
            pm_weight = torch.where(
                conf_gap <= -self.entropy_hard_switch_margin,
                torch.full_like(pm_weight, self.entropy_min_pm_weight),
                pm_weight,
            )

        if self.entropy_disagree_gm_boost > 0:
            disagree = (pred_pm != pred_gm).float().unsqueeze(1)
            gm_better = (gm_conf > pm_conf).float()
            pm_weight = pm_weight - self.entropy_disagree_gm_boost * disagree * gm_better

        if (
            self.entropy_use_ood_gate
            and feat is not None
            and self.feature_mean is not None
            and self.feature_var is not None
            and feat.dim() >= 2
            and feat.size(1) == self.feature_mean.numel()
        ):
            # OOD-aware gating: PM is down-weighted when sample is far from local feature distribution.
            mu = self.feature_mean.to(feat.device).unsqueeze(0)
            var = self.feature_var.to(feat.device).clamp_min(1e-6).unsqueeze(0)
            mahalanobis = ((feat - mu).pow(2) / var).mean(dim=1, keepdim=True)
            in_dist_score = torch.exp(-mahalanobis / self.entropy_ood_scale).clamp(0.0, 1.0)
            pm_floor = min(self.entropy_min_pm_weight, self.entropy_max_pm_weight)
            pm_weight = pm_floor + (pm_weight - pm_floor) * in_dist_score

        pm_weight = pm_weight.clamp(0.0, 1.0)
        gm_weight = 1.0 - pm_weight
        mixed_prob = gm_weight * gm_prob + pm_weight * pm_prob
        return mixed_prob, pm_weight

    def fuse_logits(self, gm_logits, pm_logits, feat=None):
        mixed_prob, _ = self.mix_prob(gm_logits, pm_logits, feat=feat)
        return torch.log(mixed_prob.clamp_min(1e-12))

    def forward(self, x):
        z = self.f_ext(x)
        if z.dim() > 2:
            z = torch.flatten(z, 1)
        z_gm = self.generalized_adapter(z) if self.generalized_adapter is not None else z
        gm_logits = self.generalized_module(z_gm)
        z_pm = self.personalized_adapter(z) if self.personalized_adapter is not None else z
        pm_logits = self.personalized_module(z_pm)
        return self.fuse_logits(gm_logits, pm_logits, feat=z)

class clientFedCD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args

        # 1. 모델 분리 (GM: Teacher/Frozen, PM: Student/Trainable)
        # PFLlib은 self.model에 전체 모델을 로드해줍니다.
        self.gm = copy.deepcopy(self.model) # General Module (CPU)
        pm_model = getattr(args, "pm_model", None)
        if pm_model is not None:
            self.pm = copy.deepcopy(pm_model)
        else:
            self.pm = copy.deepcopy(self.model) # Personalized Module (CPU)
        self.f_ext = self._build_f_ext(args)
        
        # Start from frozen GM backbone; only GM head(+adapter) is trainable.
        for param in self.gm.parameters():
            param.requires_grad = False
        self.gm.eval() 

        self.loss_func = nn.CrossEntropyLoss()
        self.nc_weight = float(getattr(args, "fedcd_nc_weight", 0.0))
        self.nc_target_corr = float(getattr(args, "fedcd_nc_target_corr", -0.1))
        self.fusion_weight = float(getattr(args, "fedcd_fusion_weight", 1.0))
        self.pm_logits_weight = float(getattr(args, "fedcd_pm_logits_weight", 0.5))
        self.pm_only_weight = float(getattr(args, "fedcd_pm_only_weight", 1.5))
        self.gm_logits_weight = float(getattr(args, "fedcd_gm_logits_weight", 1.0))
        self.local_pm_only_objective = bool(getattr(args, "fedcd_local_pm_only_objective", False))
        self.gm_lr_scale = float(getattr(args, "fedcd_gm_lr_scale", 0.1))
        self.gm_update_mode = str(getattr(args, "fedcd_gm_update_mode", "local")).strip().lower()
        if self.gm_update_mode not in {
            "local",
            "server_pm_teacher",
            "server_pm_fedavg",
            "server_proto_teacher",
            "hybrid_local_proto",
        }:
            raise ValueError(f"Unknown fedcd_gm_update_mode: {self.gm_update_mode}")
        self.local_gm_trainable = self.gm_update_mode in {"local", "hybrid_local_proto"}
        self.entropy_temp_pm = float(getattr(args, "fedcd_entropy_temp_pm", 1.0))
        self.entropy_temp_gm = float(getattr(args, "fedcd_entropy_temp_gm", 1.0))
        self.entropy_min_pm_weight = float(getattr(args, "fedcd_entropy_min_pm_weight", 0.1))
        self.entropy_max_pm_weight = float(getattr(args, "fedcd_entropy_max_pm_weight", 0.9))
        self.entropy_gate_tau = float(getattr(args, "fedcd_entropy_gate_tau", 0.2))
        self.entropy_pm_bias = float(getattr(args, "fedcd_entropy_pm_bias", 0.0))
        self.entropy_gm_bias = float(getattr(args, "fedcd_entropy_gm_bias", 0.0))
        self.entropy_disagree_gm_boost = float(getattr(args, "fedcd_entropy_disagree_gm_boost", 0.0))
        self.entropy_use_class_reliability = bool(getattr(args, "fedcd_entropy_use_class_reliability", True))
        self.entropy_reliability_scale = float(getattr(args, "fedcd_entropy_reliability_scale", 0.7))
        self.entropy_hard_switch_margin = float(getattr(args, "fedcd_entropy_hard_switch_margin", 0.15))
        self.entropy_use_ood_gate = bool(getattr(args, "fedcd_entropy_use_ood_gate", True))
        self.entropy_ood_scale = float(getattr(args, "fedcd_entropy_ood_scale", 1.0))
        self.gate_reliability_ema = float(getattr(args, "fedcd_gate_reliability_ema", 0.9))
        self.gate_reliability_samples = int(getattr(args, "fedcd_gate_reliability_samples", 512))
        self.gate_feature_ema = float(getattr(args, "fedcd_gate_feature_ema", 0.9))
        self.gate_feature_samples = int(getattr(args, "fedcd_gate_feature_samples", 512))
        self.warmup_epochs = int(getattr(args, "fedcd_warmup_epochs", 0))
        self.generalized_module = self._extract_module(self.gm)
        self.personalized_module = self._extract_module(self.pm)
        self.f_ext_dim = getattr(self.f_ext, "out_dim", None)
        self.generalized_adapter = self._build_adapter(self.generalized_module)
        self.personalized_adapter = self._build_adapter(self.personalized_module)
        self.pm_class_reliability = torch.full((int(self.num_classes),), 0.5, dtype=torch.float32)
        self.gm_class_reliability = torch.full((int(self.num_classes),), 0.5, dtype=torch.float32)
        self.gate_feature_mean = (
            torch.zeros((int(self.f_ext_dim),), dtype=torch.float32)
            if self.f_ext_dim is not None and int(self.f_ext_dim) > 0
            else None
        )
        self.gate_feature_var = (
            torch.ones((int(self.f_ext_dim),), dtype=torch.float32)
            if self.f_ext_dim is not None and int(self.f_ext_dim) > 0
            else None
        )
        self.model = PMWrapper(
            self.f_ext,
            self.generalized_module,
            self.personalized_module,
            self.num_classes,
            feature_dim=(int(self.f_ext_dim) if self.f_ext_dim is not None else 0),
            generalized_adapter=self.generalized_adapter,
            personalized_adapter=self.personalized_adapter,
            entropy_temp_pm=self.entropy_temp_pm,
            entropy_temp_gm=self.entropy_temp_gm,
            entropy_min_pm_weight=self.entropy_min_pm_weight,
            entropy_max_pm_weight=self.entropy_max_pm_weight,
            entropy_gate_tau=self.entropy_gate_tau,
            entropy_pm_bias=self.entropy_pm_bias,
            entropy_gm_bias=self.entropy_gm_bias,
            entropy_disagree_gm_boost=self.entropy_disagree_gm_boost,
            entropy_use_class_reliability=self.entropy_use_class_reliability,
            entropy_reliability_scale=self.entropy_reliability_scale,
            entropy_hard_switch_margin=self.entropy_hard_switch_margin,
            entropy_use_ood_gate=self.entropy_use_ood_gate,
            entropy_ood_scale=self.entropy_ood_scale,
        )
        self.model.set_class_reliability(self.pm_class_reliability, self.gm_class_reliability)
        self.model.set_feature_stats(self.gate_feature_mean, self.gate_feature_var)

        # Local GM training can be toggled by fedcd_gm_update_mode.
        self._set_local_gm_trainable(self.local_gm_trainable)

        # Optimizer: PM uses base lr, GM uses scaled lr.
        self.optimizer = self._build_optimizer()

        # [Info] Print Model Stats for Client 0
        if self.id == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_size_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
            print(f"\n[Client 0] Model Parameter Stats:")
            print(f"  Total Parameters: {total_params:,}")
            print(f"  Trainable Parameters: {trainable_params:,}")
            print(f"  Estimated Model Size: {total_size_bytes / (1024**2):.2f} MB")

    def _to_device(self, tensor, device):
        if self.pin_memory and device == "cuda":
            return tensor.to(device, non_blocking=True)
        return tensor.to(device)

    @staticmethod
    def _is_oom(err):
        return "out of memory" in str(err).lower()

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

    def _forward_module_with_feature(self, z, module):
        if isinstance(module, nn.Sequential) and len(module) > 1:
            feat = module[:-1](z)
            logits = module[-1](feat)
            return feat, logits
        logits = module(z)
        return z, logits

    def _feature_negative_correlation_loss(self, gm_feat, pm_feat):
        # FedEXT-inspired feature-wise negative correlation:
        # compute per-feature Pearson correlation and push it to a mild negative target.
        if gm_feat.dim() > 2:
            gm_feat = torch.flatten(gm_feat, 1)
        if pm_feat.dim() > 2:
            pm_feat = torch.flatten(pm_feat, 1)
        if gm_feat.size(0) <= 1 or pm_feat.size(0) <= 1:
            return gm_feat.new_zeros(())

        feat_dim = min(gm_feat.size(1), pm_feat.size(1))
        if feat_dim <= 0:
            return gm_feat.new_zeros(())
        if gm_feat.size(1) != feat_dim:
            gm_feat = gm_feat[:, :feat_dim]
        if pm_feat.size(1) != feat_dim:
            pm_feat = pm_feat[:, :feat_dim]

        gm_centered = gm_feat - gm_feat.mean(dim=0, keepdim=True)
        pm_centered = pm_feat - pm_feat.mean(dim=0, keepdim=True)
        gm_std = gm_centered.std(dim=0, unbiased=False).clamp_min(1e-6)
        pm_std = pm_centered.std(dim=0, unbiased=False).clamp_min(1e-6)
        corr = (gm_centered * pm_centered).mean(dim=0) / (gm_std * pm_std)
        corr = corr.clamp(-1.0, 1.0)

        target = torch.full_like(corr, float(self.nc_target_corr))
        return F.mse_loss(corr, target, reduction="mean")

    def _set_local_gm_trainable(self, trainable):
        for p in self.generalized_module.parameters():
            p.requires_grad = bool(trainable)
        if self.generalized_adapter is not None:
            for p in self.generalized_adapter.parameters():
                p.requires_grad = bool(trainable)

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

    def get_feature_stats(self, max_samples=512):
        # Estimate mean/variance of f_ext(z) on local data
        self.f_ext.to("cpu")
        self.f_ext.eval()
        trainloader = self.load_train_data(batch_size=min(self.batch_size, 32))

        count = 0
        sum_feat = None
        sum_sq = None

        with torch.no_grad():
            for x, _ in trainloader:
                if type(x) == type([]):
                    x = x[0]
                x = x.to("cpu")
                feat = self.f_ext(x)
                if feat.dim() > 2:
                    feat = torch.flatten(feat, 1)

                if max_samples and count + feat.size(0) > max_samples:
                    feat = feat[: max_samples - count]

                if sum_feat is None:
                    sum_feat = feat.sum(dim=0)
                    sum_sq = (feat ** 2).sum(dim=0)
                else:
                    sum_feat += feat.sum(dim=0)
                    sum_sq += (feat ** 2).sum(dim=0)

                count += feat.size(0)
                if max_samples and count >= max_samples:
                    break

        denom = max_samples if max_samples and count >= max_samples else count
        mean = sum_feat / denom
        var = sum_sq / denom
        var = var - mean ** 2
        var = torch.clamp(var, min=1e-6)
        return mean, var

    def train(self):
        trained = False

        def _train_once(device, batch_size):
            trainloader = self.load_train_data(batch_size=batch_size)
            # Move models to GPU only during training to reduce memory usage
            self.model.to(device)
            self.f_ext.to(device)
            self.gm.to(device)
            self.generalized_module.to(device)
            self.personalized_module.to(device)
            if self.generalized_adapter is not None:
                self.generalized_adapter.to(device)
            if self.personalized_adapter is not None:
                self.personalized_adapter.to(device)
            self.personalized_module.train()
            if self.local_gm_trainable:
                self.generalized_module.train()
                if self.generalized_adapter is not None:
                    self.generalized_adapter.train()
            else:
                self.generalized_module.eval()
                if self.generalized_adapter is not None:
                    self.generalized_adapter.eval()
            use_amp = device == "cuda" and self.use_amp
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            for _ in range(self.local_epochs):
                for _, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x = x[0]
                    x = self._to_device(x, device)
                    y = self._to_device(y, device)

                    self.optimizer.zero_grad()

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        # Shared feature extractor
                        z = self.f_ext(x)
                        if z.dim() > 2:
                            z = torch.flatten(z, 1)

                        if self.local_gm_trainable:
                            # 1. GM prediction (trainable on local mode)
                            z_gm = self.generalized_adapter(z) if self.generalized_adapter is not None else z
                            gm_feat, logits_gm = self._forward_module_with_feature(z_gm, self.generalized_module)
                        else:
                            # 1. GM prediction (frozen in server_pm_teacher mode)
                            with torch.no_grad():
                                z_gm = self.generalized_adapter(z) if self.generalized_adapter is not None else z
                                gm_feat, logits_gm = self._forward_module_with_feature(z_gm, self.generalized_module)

                        # 2. PM의 예측
                        z_pm = self.personalized_adapter(z) if self.personalized_adapter is not None else z
                        pm_feat, logits_pm = self._forward_module_with_feature(z_pm, self.personalized_module)
                        fused_logits = self.model.fuse_logits(logits_gm, logits_pm, feat=z)

                        # 3. Local objective
                        # PM-only experimental mode: maximize PM local accuracy only.
                        if self.local_pm_only_objective:
                            loss = self.loss_func(logits_pm, y)
                        else:
                            loss = self.fusion_weight * F.nll_loss(fused_logits, y)
                            if self.local_gm_trainable and self.gm_logits_weight > 0:
                                loss = loss + self.gm_logits_weight * self.loss_func(logits_gm, y)
                            if self.pm_logits_weight > 0:
                                loss = loss + self.pm_logits_weight * self.loss_func(logits_pm, y)
                            if self.pm_only_weight > 0:
                                loss = loss + self.pm_only_weight * self.loss_func(logits_pm, y)
                            if self.nc_weight > 0:
                                nc_loss = self._feature_negative_correlation_loss(gm_feat, pm_feat)
                                loss = loss + self.nc_weight * nc_loss
                    
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

            # Move back to CPU after training to free GPU memory
            # [수정] avoid_oom이 True일 때만 CPU로 내림. (속도 향상)
            if self.args.avoid_oom:
                self.f_ext.to("cpu")
                self.gm.to("cpu")
                self.pm.to("cpu")
                self.generalized_module.to("cpu")
                self.personalized_module.to("cpu")
                if self.generalized_adapter is not None:
                    self.generalized_adapter.to("cpu")
                if self.personalized_adapter is not None:
                    self.personalized_adapter.to("cpu")
                self.model.to("cpu")
                if device == "cuda":
                    torch.cuda.empty_cache()

        batch_size = self.batch_size
        if self.device == "cuda":
            if self.gpu_batch_mult > 1:
                batch_size = batch_size * self.gpu_batch_mult
            if self.gpu_batch_max > 0:
                batch_size = min(batch_size, self.gpu_batch_max)
        
        # [Fix] 배치 크기가 데이터 샘플 수보다 크면 학습이 스킵되는 문제 방지 (drop_last=True 때문)
        if batch_size > self.train_samples:
            batch_size = self.train_samples

        try:
            _train_once(self.device, batch_size)
            trained = True
        except RuntimeError as err:
            if self.device == "cuda" and self._is_oom(err):
                print("[Warn] OOM during FedCD client training. Reducing batch size / fallback to CPU.")
                torch.cuda.empty_cache()
                # OOM 발생 시 강제로 CPU로 내리고 정리
                self.model.to("cpu")
                self.f_ext.to("cpu")
                self.gm.to("cpu")
                self.pm.to("cpu")
                self.model.to("cpu")
                reduced = max(1, batch_size // 2)
                if reduced < batch_size:
                    try:
                        _train_once(self.device, reduced)
                        trained = True
                    except RuntimeError as err2:
                        if not self._is_oom(err2):
                            raise
                if not trained:
                    print("[Warn] OOM persists. Falling back to CPU for this client.")
                    _train_once("cpu", reduced)
                    trained = True
            else:
                raise

        if trained:
            stat_samples = max(int(self.gate_reliability_samples), int(self.gate_feature_samples))
            self._update_gate_class_reliability(max_samples=stat_samples)

    def _update_gate_class_reliability(self, max_samples=512):
        ema = min(max(float(self.gate_reliability_ema), 0.0), 1.0)
        feature_ema = min(max(float(self.gate_feature_ema), 0.0), 1.0)
        max_samples = int(max_samples) if max_samples is not None else 0
        if max_samples < 0:
            max_samples = 0

        try:
            # Use local train split (not test split) to avoid evaluation leakage.
            calib_batch_size = min(self.batch_size, max(1, self.train_samples))
            trainloader = self.load_train_data(batch_size=min(calib_batch_size, 64))
        except Exception:
            return

        device = self.device
        pm_correct = torch.zeros(self.num_classes, dtype=torch.float64)
        gm_correct = torch.zeros(self.num_classes, dtype=torch.float64)
        cls_total = torch.zeros(self.num_classes, dtype=torch.float64)
        feat_sum = None
        feat_sq_sum = None
        feat_count = 0
        seen = 0

        self.f_ext.to(device)
        self.generalized_module.to(device)
        self.personalized_module.to(device)
        self.f_ext.eval()
        self.generalized_module.eval()
        self.personalized_module.eval()
        if self.generalized_adapter is not None:
            self.generalized_adapter.to(device)
            self.generalized_adapter.eval()
        if self.personalized_adapter is not None:
            self.personalized_adapter.to(device)
            self.personalized_adapter.eval()

        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x = x[0]
                x = self._to_device(x, device)
                y = self._to_device(y, device).long()

                z = self.f_ext(x)
                if z.dim() > 2:
                    z = torch.flatten(z, 1)
                z_gm = self.generalized_adapter(z) if self.generalized_adapter is not None else z
                z_pm = self.personalized_adapter(z) if self.personalized_adapter is not None else z
                gm_pred = torch.argmax(self.generalized_module(z_gm), dim=1)
                pm_pred = torch.argmax(self.personalized_module(z_pm), dim=1)

                for cls in y.unique():
                    cls_id = int(cls.item())
                    mask = (y == cls)
                    cnt = mask.sum().item()
                    if cnt <= 0:
                        continue
                    cls_total[cls_id] += cnt
                    pm_correct[cls_id] += (pm_pred[mask] == cls).sum().item()
                    gm_correct[cls_id] += (gm_pred[mask] == cls).sum().item()

                if (
                    self.gate_feature_mean is not None
                    and self.gate_feature_var is not None
                    and z.size(1) == self.gate_feature_mean.numel()
                ):
                    zf = z.detach().float()
                    if feat_sum is None:
                        feat_sum = zf.sum(dim=0)
                        feat_sq_sum = (zf * zf).sum(dim=0)
                    else:
                        feat_sum += zf.sum(dim=0)
                        feat_sq_sum += (zf * zf).sum(dim=0)
                    feat_count += zf.size(0)

                seen += y.size(0)
                if max_samples > 0 and seen >= max_samples:
                    break

        valid = cls_total > 0
        if valid.any():
            pm_new = self.pm_class_reliability.clone()
            gm_new = self.gm_class_reliability.clone()
            pm_acc = torch.zeros_like(pm_new)
            gm_acc = torch.zeros_like(gm_new)
            pm_acc[valid] = (pm_correct[valid] / cls_total[valid]).to(pm_new.dtype)
            gm_acc[valid] = (gm_correct[valid] / cls_total[valid]).to(gm_new.dtype)
            pm_new[valid] = ema * pm_new[valid] + (1.0 - ema) * pm_acc[valid]
            gm_new[valid] = ema * gm_new[valid] + (1.0 - ema) * gm_acc[valid]
            self.pm_class_reliability = pm_new.clamp(0.0, 1.0)
            self.gm_class_reliability = gm_new.clamp(0.0, 1.0)
            self.model.set_class_reliability(self.pm_class_reliability, self.gm_class_reliability)

        if (
            feat_count > 0
            and feat_sum is not None
            and feat_sq_sum is not None
            and self.gate_feature_mean is not None
            and self.gate_feature_var is not None
        ):
            feat_mean = feat_sum / float(feat_count)
            feat_var = (feat_sq_sum / float(feat_count)) - feat_mean.pow(2)
            feat_var = feat_var.clamp_min(1e-6)

            # Keep client-side cached stats on CPU, but allow temporary device alignment
            # during EMA update to avoid cpu/cuda mixing errors.
            mean_old = self.gate_feature_mean
            var_old = self.gate_feature_var
            target_device = feat_mean.device
            mean_old_dev = mean_old.to(target_device)
            var_old_dev = var_old.to(target_device)
            mean_new = feature_ema * mean_old_dev + (1.0 - feature_ema) * feat_mean.to(mean_old_dev.dtype)
            var_new = (
                feature_ema * var_old_dev + (1.0 - feature_ema) * feat_var.to(var_old_dev.dtype)
            ).clamp_min(1e-6)
            self.gate_feature_mean = mean_new.detach().cpu()
            self.gate_feature_var = var_new.detach().cpu()
            self.model.set_feature_stats(self.gate_feature_mean, self.gate_feature_var)

        if self.args.avoid_oom:
            self.f_ext.to("cpu")
            self.generalized_module.to("cpu")
            self.personalized_module.to("cpu")
            if self.generalized_adapter is not None:
                self.generalized_adapter.to("cpu")
            if self.personalized_adapter is not None:
                self.personalized_adapter.to("cpu")
            if device == "cuda":
                torch.cuda.empty_cache()

    def warmup_personalized_module(self):
        if self.warmup_epochs <= 0:
            return

        def _warmup_once(device, batch_size):
            self.model.to(device)
            self.f_ext.to(device)
            self.generalized_module.to(device)
            self.f_ext.eval()
            self.personalized_module.to(device)
            if self.generalized_adapter is not None:
                self.generalized_adapter.to(device)
            if self.personalized_adapter is not None:
                self.personalized_adapter.to(device)
            self.personalized_module.train()
            optimizer = torch.optim.SGD(
                list(self.personalized_module.parameters()) +
                (list(self.personalized_adapter.parameters()) if self.personalized_adapter is not None else []),
                lr=self.learning_rate
            )
            trainloader = self.load_train_data(batch_size=batch_size)
            use_amp = device == "cuda" and self.use_amp
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            for _ in range(self.warmup_epochs):
                for x, y in trainloader:
                    if type(x) == type([]):
                        x = x[0]
                    x = self._to_device(x, device)
                    y = self._to_device(y, device)
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        z = self.f_ext(x)
                        if z.dim() > 2:
                            z = torch.flatten(z, 1)
                        z_pm = self.personalized_adapter(z) if self.personalized_adapter is not None else z
                        logits_pm = self.personalized_module(z_pm)
                        z_gm = self.generalized_adapter(z) if self.generalized_adapter is not None else z
                        logits_gm = self.generalized_module(z_gm)
                        logits = self.model.fuse_logits(logits_gm, logits_pm, feat=z)
                        loss = F.nll_loss(logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            # Clean up
            # [수정] avoid_oom이 True일 때만 CPU로 내림.
            if self.args.avoid_oom:
                self.f_ext.to("cpu")
                self.generalized_module.to("cpu")
                self.personalized_module.to("cpu")
                if self.generalized_adapter is not None:
                    self.generalized_adapter.to("cpu")
                if self.personalized_adapter is not None:
                    self.personalized_adapter.to("cpu")
                self.model.to("cpu")
                if device == "cuda":
                    torch.cuda.empty_cache()

        batch_size = self.batch_size
        if self.device == "cuda":
            if self.gpu_batch_mult > 1:
                batch_size = batch_size * self.gpu_batch_mult
            if self.gpu_batch_max > 0:
                batch_size = min(batch_size, self.gpu_batch_max)
        try:
            _warmup_once(self.device, batch_size)
        except RuntimeError as err:
            if self.device == "cuda" and self._is_oom(err):
                print("[Warn] OOM during warmup. Falling back to CPU for this client.")
                torch.cuda.empty_cache()
                # OOM 시 강제 정리
                self.f_ext.to("cpu")
                self.generalized_module.to("cpu")
                self.personalized_module.to("cpu")
                self.model.to("cpu")
                _warmup_once("cpu", max(1, batch_size // 2))
                return
            raise

    def warmup_classifier(self):
        # Backward compatibility for existing call sites.
        self.warmup_personalized_module()

    # [핵심] 서버에서 Generalized Module 파트를 받아 적용
    def set_parameters(self, model):
        # 서버에서 받은 Generalized Module 관련 파라미터를 로드
        if isinstance(model, dict) and model.get("init_parts", None) is not None:
            self.set_initial_parameters(model["init_parts"])
            return
        gm_updated = False
        if isinstance(model, dict):
            gm_parts = model.get("gm_parts", None)
            gm_state = model.get("gm_state", {})
            gm_adapter_state = model.get("gm_adapter", None)
        else:
            gm_parts = None
            gm_state = model.state_dict()
            gm_adapter_state = None

        # New lightweight payload path: generalized_module/generalized_adapter
        if gm_parts is not None:
            if gm_parts and next(iter(gm_parts.values())).is_cuda:
                gm_parts = {k: v.detach().cpu() for k, v in gm_parts.items()}
            generalized_module_state = {
                k.replace("generalized_module.", ""): v
                for k, v in gm_parts.items()
                if k.startswith("generalized_module.")
            }
            generalized_adapter_state = {
                k.replace("generalized_adapter.", ""): v
                for k, v in gm_parts.items()
                if k.startswith("generalized_adapter.")
            }

            # Backward compatibility for old "head/final/adapter" payload
            if not generalized_module_state:
                head_state = {k.replace("head.", ""): v for k, v in gm_parts.items() if k.startswith("head.")}
                final_state = {k.replace("final.", ""): v for k, v in gm_parts.items() if k.startswith("final.")}
                generalized_module_state = self._merge_legacy_module_state(
                    self.generalized_module,
                    head_state,
                    final_state,
                )
                if not generalized_adapter_state:
                    generalized_adapter_state = {
                        k.replace("adapter.", ""): v for k, v in gm_parts.items() if k.startswith("adapter.")
                    }

            if generalized_module_state:
                self.generalized_module.load_state_dict(generalized_module_state, strict=True)
                gm_updated = True
            if self.generalized_adapter is not None and generalized_adapter_state:
                self.generalized_adapter.load_state_dict(generalized_adapter_state, strict=True)
                gm_updated = True
        else:
            # Backward compatibility: full gm_state payload
            if gm_state and next(iter(gm_state.values())).is_cuda:
                gm_state = {k: v.detach().cpu() for k, v in gm_state.items()}
            if gm_state:
                if any(k.startswith("generalized_module.") for k in gm_state.keys()):
                    generalized_module_state = {
                        k.replace("generalized_module.", ""): v
                        for k, v in gm_state.items()
                        if k.startswith("generalized_module.")
                    }
                    if generalized_module_state:
                        self.generalized_module.load_state_dict(generalized_module_state, strict=True)
                        gm_updated = True
                else:
                    self.gm.load_state_dict(gm_state, strict=True)
                    gm_updated = True
            if gm_adapter_state is not None and self.generalized_adapter is not None:
                self.generalized_adapter.load_state_dict(gm_adapter_state, strict=True)
                gm_updated = True

        self.model.generalized_module = self.generalized_module
        if self.generalized_adapter is not None:
            self.model.generalized_adapter = self.generalized_adapter
        if gm_updated:
            self._reset_optimizer()
        
        # PM은 GM과 너무 멀어지지 않게 살짝 당겨주거나(Regularization), 
        # 혹은 그대로 둡니다(Pure Personalization). 연구 의도에 따라 선택.
        # 여기서는 일단 PM은 그대로 유지하는 것으로 둡니다.

    def set_initial_parameters(self, init_state):
        if not init_state:
            return
        state = init_state
        if next(iter(state.values())).is_cuda:
            state = {k: v.detach().cpu() for k, v in state.items()}

        f_ext_state = {
            k.replace("f_ext.", ""): v
            for k, v in state.items()
            if k.startswith("f_ext.")
        }
        generalized_module_state = {
            k.replace("generalized_module.", ""): v
            for k, v in state.items()
            if k.startswith("generalized_module.")
        }
        generalized_adapter_state = {
            k.replace("generalized_adapter.", ""): v
            for k, v in state.items()
            if k.startswith("generalized_adapter.")
        }
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

        if f_ext_state:
            self.f_ext.load_state_dict(f_ext_state, strict=True)
        if generalized_module_state:
            self.generalized_module.load_state_dict(generalized_module_state, strict=True)
        if self.generalized_adapter is not None and generalized_adapter_state:
            self.generalized_adapter.load_state_dict(generalized_adapter_state, strict=True)
        if personalized_module_state:
            self.personalized_module.load_state_dict(personalized_module_state, strict=True)
        if self.personalized_adapter is not None and personalized_adapter_state:
            self.personalized_adapter.load_state_dict(personalized_adapter_state, strict=True)

        self.model.f_ext = self.f_ext
        self.model.generalized_module = self.generalized_module
        self.model.personalized_module = self.personalized_module
        if self.generalized_adapter is not None:
            self.model.generalized_adapter = self.generalized_adapter
        if self.personalized_adapter is not None:
            self.model.personalized_adapter = self.personalized_adapter

        self._reset_optimizer()

    def set_personalized_parameters(self, pm_state):
        # 서버에서 받은 클러스터 대표 Personalized Module을 업데이트
        state = pm_state
        if next(iter(state.values())).is_cuda:
            state = {k: v.detach().cpu() for k, v in state.items()}
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

        # Backward compatibility for old "head/final/adapter" payload
        if not personalized_module_state:
            head_state = {k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")}
            final_state = {k.replace("final.", ""): v for k, v in state.items() if k.startswith("final.")}
            personalized_module_state = self._merge_legacy_module_state(
                self.personalized_module,
                head_state,
                final_state,
            )
            if not personalized_adapter_state:
                personalized_adapter_state = {
                    k.replace("adapter.", ""): v for k, v in state.items() if k.startswith("adapter.")
                }

        if personalized_module_state:
            self.personalized_module.load_state_dict(personalized_module_state, strict=True)
        if self.personalized_adapter is not None and personalized_adapter_state:
            self.personalized_adapter.load_state_dict(personalized_adapter_state, strict=True)
            
        # [Fix] Reset optimizer state when model parameters are forcibly changed
        # This prevents momentum from previous (possibly incompatible) weights 
        # from interfering with the new cluster model.
        self._reset_optimizer()

    def _reset_optimizer(self):
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self):
        pm_params = list(self.personalized_module.parameters())
        if self.personalized_adapter is not None:
            pm_params += list(self.personalized_adapter.parameters())

        gm_params = []
        if self.local_gm_trainable:
            gm_params = list(self.generalized_module.parameters())
            if self.generalized_adapter is not None:
                gm_params += list(self.generalized_adapter.parameters())

        gm_lr = max(self.learning_rate * self.gm_lr_scale, 0.0)
        param_groups = []
        if pm_params:
            param_groups.append({"params": pm_params, "lr": self.learning_rate})
        if gm_params and gm_lr > 0:
            param_groups.append({"params": gm_params, "lr": gm_lr})
        return torch.optim.SGD(param_groups, lr=self.learning_rate)

    def infer_fused_logits_with_gate(self, x):
        z = self.f_ext(x)
        if z.dim() > 2:
            z = torch.flatten(z, 1)
        z_gm = self.generalized_adapter(z) if self.generalized_adapter is not None else z
        gm_logits = self.generalized_module(z_gm)
        z_pm = self.personalized_adapter(z) if self.personalized_adapter is not None else z
        pm_logits = self.personalized_module(z_pm)
        mixed_prob, pm_weight = self.model.mix_prob(gm_logits, pm_logits, feat=z)
        fused_logits = torch.log(mixed_prob.clamp_min(1e-12))
        return fused_logits, gm_logits, pm_logits, pm_weight

    def upload_parameters(self):
        # 업링크 비용 절감: Personalized Module만 전송
        state = {}
        state.update({
            f"personalized_module.{k}": v.detach().cpu()
            for k, v in self.personalized_module.state_dict().items()
        })
        if self.personalized_adapter is not None:
            state.update({
                f"personalized_adapter.{k}": v.detach().cpu()
                for k, v in self.personalized_adapter.state_dict().items()
            })
        return state

    def upload_generalized_parameters(self):
        if not self.local_gm_trainable:
            return {}
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

    def upload_pm_prototypes(self, max_samples=0):
        """
        Upload PM knowledge as class-wise prototype statistics on shared f_ext space.
        Returns:
            {
              "counts": [C],
              "feat_sum": [C, D],
              "feat_sq_sum": [C, D],
              "logit_sum": [C, C],
            }
        """
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        loader_kwargs = {
            "batch_size": min(self.batch_size, max(1, self.train_samples)),
            "drop_last": False,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory and self.device == "cuda",
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        trainloader = DataLoader(train_data, **loader_kwargs)

        device = "cpu"
        self.f_ext.to(device)
        self.f_ext.eval()
        self.personalized_module.to(device)
        self.personalized_module.eval()
        if self.personalized_adapter is not None:
            self.personalized_adapter.to(device)
            self.personalized_adapter.eval()

        C = int(self.num_classes)
        counts = torch.zeros(C, dtype=torch.float32)
        feat_sum = None
        feat_sq_sum = None
        logit_sum = torch.zeros(C, C, dtype=torch.float32)
        seen = 0

        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x = x[0]
                x = x.to(device)
                y = y.to(device).long()

                z = self.f_ext(x)
                if z.dim() > 2:
                    z = torch.flatten(z, 1)

                if feat_sum is None:
                    D = z.size(1)
                    feat_sum = torch.zeros(C, D, dtype=torch.float32)
                    feat_sq_sum = torch.zeros(C, D, dtype=torch.float32)

                z_pm = self.personalized_adapter(z) if self.personalized_adapter is not None else z
                pm_logits = self.personalized_module(z_pm)

                if max_samples > 0 and seen + y.size(0) > max_samples:
                    keep = max_samples - seen
                    if keep <= 0:
                        break
                    y = y[:keep]
                    z = z[:keep]
                    pm_logits = pm_logits[:keep]

                for cls in y.unique():
                    cls_id = int(cls.item())
                    m = (y == cls)
                    cnt = int(m.sum().item())
                    if cnt <= 0:
                        continue
                    z_cls = z[m]
                    counts[cls_id] += float(cnt)
                    feat_sum[cls_id] += z_cls.sum(dim=0).float()
                    feat_sq_sum[cls_id] += (z_cls * z_cls).sum(dim=0).float()
                    logit_sum[cls_id] += pm_logits[m].sum(dim=0).float()

                seen += y.size(0)
                if max_samples > 0 and seen >= max_samples:
                    break

        if feat_sum is None:
            # Edge case: no data
            D = int(getattr(self.f_ext, "out_dim", 0))
            feat_sum = torch.zeros(C, D, dtype=torch.float32)
            feat_sq_sum = torch.zeros(C, D, dtype=torch.float32)

        return {
            "counts": counts.cpu(),
            "feat_sum": feat_sum.cpu(),
            "feat_sq_sum": feat_sq_sum.cpu(),
            "logit_sum": logit_sum.cpu(),
        }
