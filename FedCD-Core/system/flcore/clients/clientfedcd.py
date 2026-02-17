import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from flcore.clients.clientbase import Client
from flcore.trainmodel.models import SmallFExt

class PMWrapper(nn.Module):
    def __init__(
        self,
        f_ext,
        generalized_module,
        personalized_module,
        generalized_adapter=None,
        personalized_adapter=None,
        entropy_temp_pm=1.0,
        entropy_temp_gm=1.0,
        entropy_min_pm_weight=0.1,
        entropy_max_pm_weight=0.9,
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

    @staticmethod
    def _normalized_entropy(prob):
        eps = 1e-12
        num_classes = prob.size(1)
        entropy = -(prob * torch.log(prob.clamp_min(eps))).sum(dim=1, keepdim=True)
        norm = torch.log(torch.tensor(float(num_classes), device=prob.device))
        return entropy / norm.clamp_min(eps)

    def mix_prob(self, gm_logits, pm_logits):
        # Entropy-based confidence gating:
        # lower PM entropy -> higher PM weight, higher entropy -> rely more on GM.
        pm_prob = torch.softmax(pm_logits / self.entropy_temp_pm, dim=1)
        gm_prob = torch.softmax(gm_logits / self.entropy_temp_gm, dim=1)
        pm_conf = 1.0 - self._normalized_entropy(pm_prob)
        if self.entropy_max_pm_weight > self.entropy_min_pm_weight:
            span = self.entropy_max_pm_weight - self.entropy_min_pm_weight
            pm_weight = self.entropy_min_pm_weight + span * pm_conf
        else:
            pm_weight = pm_conf
        pm_weight = pm_weight.clamp(0.0, 1.0)
        gm_weight = 1.0 - pm_weight
        mixed_prob = gm_weight * gm_prob + pm_weight * pm_prob
        return mixed_prob, pm_weight

    def fuse_logits(self, gm_logits, pm_logits):
        mixed_prob, _ = self.mix_prob(gm_logits, pm_logits)
        return torch.log(mixed_prob.clamp_min(1e-12))

    def forward(self, x):
        z = self.f_ext(x)
        if z.dim() > 2:
            z = torch.flatten(z, 1)
        z_gm = self.generalized_adapter(z) if self.generalized_adapter is not None else z
        gm_logits = self.generalized_module(z_gm)
        z_pm = self.personalized_adapter(z) if self.personalized_adapter is not None else z
        pm_logits = self.personalized_module(z_pm)
        return self.fuse_logits(gm_logits, pm_logits)

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
        self.fusion_weight = float(getattr(args, "fedcd_fusion_weight", 1.0))
        self.pm_logits_weight = float(getattr(args, "fedcd_pm_logits_weight", 0.5))
        self.pm_only_weight = float(getattr(args, "fedcd_pm_only_weight", 1.5))
        self.gm_logits_weight = float(getattr(args, "fedcd_gm_logits_weight", 1.0))
        self.gm_lr_scale = float(getattr(args, "fedcd_gm_lr_scale", 0.1))
        self.entropy_temp_pm = float(getattr(args, "fedcd_entropy_temp_pm", 1.0))
        self.entropy_temp_gm = float(getattr(args, "fedcd_entropy_temp_gm", 1.0))
        self.entropy_min_pm_weight = float(getattr(args, "fedcd_entropy_min_pm_weight", 0.1))
        self.entropy_max_pm_weight = float(getattr(args, "fedcd_entropy_max_pm_weight", 0.9))
        self.warmup_epochs = int(getattr(args, "fedcd_warmup_epochs", 0))
        self.generalized_module = self._extract_module(self.gm)
        self.personalized_module = self._extract_module(self.pm)
        self.f_ext_dim = getattr(self.f_ext, "out_dim", None)
        self.generalized_adapter = self._build_adapter(self.generalized_module)
        self.personalized_adapter = self._build_adapter(self.personalized_module)
        self.model = PMWrapper(
            self.f_ext,
            self.generalized_module,
            self.personalized_module,
            self.generalized_adapter,
            self.personalized_adapter,
            entropy_temp_pm=self.entropy_temp_pm,
            entropy_temp_gm=self.entropy_temp_gm,
            entropy_min_pm_weight=self.entropy_min_pm_weight,
            entropy_max_pm_weight=self.entropy_max_pm_weight,
        )

        # Enable training for GM head(+adapter) only (feature extractor stays frozen).
        for p in self.generalized_module.parameters():
            p.requires_grad = True
        if self.generalized_adapter is not None:
            for p in self.generalized_adapter.parameters():
                p.requires_grad = True

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
            self.generalized_module.train()
            if self.generalized_adapter is not None:
                self.generalized_adapter.train()
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

                        # 1. GM prediction (trainable)
                        z_gm = self.generalized_adapter(z) if self.generalized_adapter is not None else z
                        gm_feat, logits_gm = self._forward_module_with_feature(z_gm, self.generalized_module)

                        # 2. PM의 예측
                        z_pm = self.personalized_adapter(z) if self.personalized_adapter is not None else z
                        pm_feat, logits_pm = self._forward_module_with_feature(z_pm, self.personalized_module)
                        fused_logits = self.model.fuse_logits(logits_gm, logits_pm)

                        # 3. Loss 계산 (Task Loss + Feature-wise Negative Correlation)
                        loss = self.fusion_weight * F.nll_loss(fused_logits, y)
                        if self.gm_logits_weight > 0:
                            loss = loss + self.gm_logits_weight * self.loss_func(logits_gm, y)
                        if self.pm_logits_weight > 0:
                            loss = loss + self.pm_logits_weight * self.loss_func(logits_pm, y)
                        if self.pm_only_weight > 0:
                            loss = loss + self.pm_only_weight * self.loss_func(logits_pm, y)
                        if self.nc_weight > 0:
                            fused = (gm_feat + pm_feat) / 2.0
                            nc_term = (gm_feat - fused) * (pm_feat - fused)
                            nc_loss = nc_term.mean()
                            loss = loss + self.nc_weight * nc_loss
                        # 필요시 여기에 GM과의 Distillation Loss 추가 가능
                    
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
                        return
                    except RuntimeError as err2:
                        if not self._is_oom(err2):
                            raise
                print("[Warn] OOM persists. Falling back to CPU for this client.")
                _train_once("cpu", reduced)
                return
            raise

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
                        logits = self.model.fuse_logits(logits_gm, logits_pm)
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

        gm_params = list(self.generalized_module.parameters())
        if self.generalized_adapter is not None:
            gm_params += list(self.generalized_adapter.parameters())

        gm_lr = max(self.learning_rate * self.gm_lr_scale, 0.0)
        param_groups = []
        if pm_params:
            param_groups.append({"params": pm_params, "lr": self.learning_rate})
        if gm_params:
            param_groups.append({"params": gm_params, "lr": gm_lr})
        return torch.optim.SGD(param_groups, lr=self.learning_rate)

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
