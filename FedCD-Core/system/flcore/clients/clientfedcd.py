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
        combiner,
        generalized_adapter=None,
        personalized_adapter=None,
    ):
        super().__init__()
        self.f_ext = f_ext
        self.generalized_module = generalized_module
        self.personalized_module = personalized_module
        self.combiner = combiner
        self.generalized_adapter = generalized_adapter
        self.personalized_adapter = personalized_adapter

    def forward(self, x):
        z = self.f_ext(x)
        if z.dim() > 2:
            z = torch.flatten(z, 1)
        z_gm = self.generalized_adapter(z) if self.generalized_adapter is not None else z
        gm_logits = self.generalized_module(z_gm)
        z_pm = self.personalized_adapter(z) if self.personalized_adapter is not None else z
        pm_logits = self.personalized_module(z_pm)
        fused = torch.cat([gm_logits, pm_logits], dim=1)
        return self.combiner(fused)

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
        
        # GM은 영원히 Freeze (BN 통계량 포함)
        for param in self.gm.parameters():
            param.requires_grad = False
        self.gm.eval() 

        self.loss_func = nn.CrossEntropyLoss()
        self.nc_weight = float(getattr(args, "fedcd_nc_weight", 0.0))
        self.fusion_weight = float(getattr(args, "fedcd_fusion_weight", 1.0))
        self.pm_logits_weight = float(getattr(args, "fedcd_pm_logits_weight", 0.5))
        # New name: pm_only_weight (alias of previous pm_combiner_weight)
        self.pm_only_weight = float(
            getattr(
                args,
                "fedcd_pm_only_weight",
                getattr(args, "fedcd_pm_combiner_weight", 1.5),
            )
        )
        self.prototype_samples = int(getattr(args, "fedcd_prototype_samples", 512))
        self.combiner_calib_epochs = int(getattr(args, "fedcd_combiner_calib_epochs", 1))
        self.combiner_calib_lr_mult = float(getattr(args, "fedcd_combiner_calib_lr_mult", 1.0))
        self.warmup_epochs = int(getattr(args, "fedcd_warmup_epochs", 0))
        self.generalized_module = self._extract_module(self.gm)
        self.personalized_module = self._extract_module(self.pm)
        self.f_ext_dim = getattr(self.f_ext, "out_dim", None)
        self.generalized_adapter = self._build_adapter(self.generalized_module)
        self.personalized_adapter = self._build_adapter(self.personalized_module)
        self.combiner = nn.Linear(self.num_classes * 2, self.num_classes)
        self.model = PMWrapper(
            self.f_ext,
            self.generalized_module,
            self.personalized_module,
            self.combiner,
            self.generalized_adapter,
            self.personalized_adapter,
        )

        # Optimizer는 Personalized Module(+adapter)만 관리
        pm_params = list(self.personalized_module.parameters()) + list(self.combiner.parameters())
        if self.personalized_adapter is not None:
            pm_params += list(self.personalized_adapter.parameters())
        self.optimizer = torch.optim.SGD(pm_params, lr=self.learning_rate)

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

    def _build_feature_extractor(self, model):
        # Build shared feature extractor (f_ext) from model backbone
        if hasattr(model, "features"):
            f_ext = nn.Sequential(model.features, model.avgpool, nn.Flatten())
        elif hasattr(model, "fc"):
            children = list(model.children())[:-1]
            f_ext = nn.Sequential(*children, nn.Flatten())
        else:
            f_ext = nn.Sequential(model, nn.Flatten())
        for param in f_ext.parameters():
            param.requires_grad = False
        f_ext.eval()
        return f_ext

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

    def _sync_feature_extractor(self, target_model):
        # Copy shared extractor weights (GM backbone) into target model
        gm_state = self.gm.state_dict()
        target_state = target_model.state_dict()
        for key in target_state.keys():
            if key in gm_state and not key.startswith(("fc", "classifier")):
                target_state[key] = gm_state[key].clone()
        target_model.load_state_dict(target_state, strict=False)

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
            self.f_ext.to(device)
            self.gm.to(device)
            self.generalized_module.to(device)
            self.personalized_module.to(device)
            self.combiner.to(device)
            if self.generalized_adapter is not None:
                self.generalized_adapter.to(device)
            if self.personalized_adapter is not None:
                self.personalized_adapter.to(device)
            self.personalized_module.train()
            self.combiner.train()
            self.generalized_module.eval()
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

                        # 1. GM의 지식 (Reference)
                        with torch.no_grad():
                            z_gm = self.generalized_adapter(z) if self.generalized_adapter is not None else z
                            gm_feat, logits_gm = self._forward_module_with_feature(z_gm, self.generalized_module)

                        # 2. PM의 예측
                        z_pm = self.personalized_adapter(z) if self.personalized_adapter is not None else z
                        pm_feat, logits_pm = self._forward_module_with_feature(z_pm, self.personalized_module)
                        fused_logits = self.combiner(torch.cat([logits_gm, logits_pm], dim=1))

                        # 3. Loss 계산 (Task Loss + Feature-wise Negative Correlation)
                        loss = self.fusion_weight * self.loss_func(fused_logits, y)
                        if self.pm_logits_weight > 0:
                            loss = loss + self.pm_logits_weight * self.loss_func(logits_pm, y)
                        if self.pm_only_weight > 0:
                            gm_zeros = torch.zeros_like(logits_gm)
                            pm_only_fused_logits = self.combiner(torch.cat([gm_zeros, logits_pm], dim=1))
                            loss = loss + self.pm_only_weight * self.loss_func(pm_only_fused_logits, y)
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
                self.combiner.to("cpu")
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
            self.f_ext.to(device)
            self.generalized_module.to(device)
            self.f_ext.eval()
            self.personalized_module.to(device)
            self.combiner.to(device)
            if self.generalized_adapter is not None:
                self.generalized_adapter.to(device)
            if self.personalized_adapter is not None:
                self.personalized_adapter.to(device)
            self.personalized_module.train()
            self.combiner.train()
            optimizer = torch.optim.SGD(
                list(self.personalized_module.parameters()) +
                list(self.combiner.parameters()) +
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
                        logits = self.combiner(torch.cat([logits_gm, logits_pm], dim=1))
                        loss = self.loss_func(logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            # Clean up
            # [수정] avoid_oom이 True일 때만 CPU로 내림.
            if self.args.avoid_oom:
                self.f_ext.to("cpu")
                self.generalized_module.to("cpu")
                self.personalized_module.to("cpu")
                self.combiner.to("cpu")
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

    def calibrate_combiner(self):
        """
        Post-GM local calibration:
        - Freeze f_ext / GM / PM branches
        - Update combiner only for better alignment with freshly broadcast GM
        """
        if self.combiner_calib_epochs <= 0:
            return

        def _calibrate_once(device, batch_size):
            self.f_ext.to(device)
            self.generalized_module.to(device)
            self.personalized_module.to(device)
            self.combiner.to(device)
            if self.generalized_adapter is not None:
                self.generalized_adapter.to(device)
            if self.personalized_adapter is not None:
                self.personalized_adapter.to(device)

            self.f_ext.eval()
            self.generalized_module.eval()
            self.personalized_module.eval()
            self.combiner.train()

            lr = max(1e-8, self.learning_rate * self.combiner_calib_lr_mult)
            optimizer = torch.optim.SGD(self.combiner.parameters(), lr=lr)
            trainloader = self.load_train_data(batch_size=batch_size)
            use_amp = device == "cuda" and self.use_amp
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            for _ in range(self.combiner_calib_epochs):
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

                        with torch.no_grad():
                            z_gm = self.generalized_adapter(z) if self.generalized_adapter is not None else z
                            logits_gm = self.generalized_module(z_gm)
                            z_pm = self.personalized_adapter(z) if self.personalized_adapter is not None else z
                            logits_pm = self.personalized_module(z_pm)

                        logits = self.combiner(torch.cat([logits_gm, logits_pm], dim=1))
                        loss = self.loss_func(logits, y)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            if self.args.avoid_oom:
                self.f_ext.to("cpu")
                self.generalized_module.to("cpu")
                self.personalized_module.to("cpu")
                self.combiner.to("cpu")
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
        if batch_size > self.train_samples:
            batch_size = self.train_samples

        try:
            _calibrate_once(self.device, batch_size)
        except RuntimeError as err:
            if self.device == "cuda" and self._is_oom(err):
                print("[Warn] OOM during combiner calibration. Falling back to CPU for this client.")
                torch.cuda.empty_cache()
                self.f_ext.to("cpu")
                self.generalized_module.to("cpu")
                self.personalized_module.to("cpu")
                self.combiner.to("cpu")
                if self.generalized_adapter is not None:
                    self.generalized_adapter.to("cpu")
                if self.personalized_adapter is not None:
                    self.personalized_adapter.to("cpu")
                self.model.to("cpu")
                _calibrate_once("cpu", max(1, batch_size // 2))
                return
            raise

    # [핵심] 서버에서 Generalized Module 파트를 받아 적용
    def set_parameters(self, model):
        # 서버에서 받은 Generalized Module 관련 파라미터를 로드
        combiner_updated = False
        if isinstance(model, dict):
            gm_parts = model.get("gm_parts", None)
            gm_state = model.get("gm_state", {})
            gm_adapter_state = model.get("gm_adapter", None)
            global_combiner_state = model.get("global_combiner", None)
        else:
            gm_parts = None
            gm_state = model.state_dict()
            gm_adapter_state = None
            global_combiner_state = None

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
            global_combiner_state = {
                k.replace("global_combiner.", ""): v
                for k, v in gm_parts.items()
                if k.startswith("global_combiner.")
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
            if self.generalized_adapter is not None and generalized_adapter_state:
                self.generalized_adapter.load_state_dict(generalized_adapter_state, strict=True)
            if global_combiner_state:
                self.combiner.load_state_dict(global_combiner_state, strict=True)
                combiner_updated = True
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
                else:
                    self.gm.load_state_dict(gm_state, strict=True)
                if any(k.startswith("global_combiner.") for k in gm_state.keys()):
                    global_combiner_state = {
                        k.replace("global_combiner.", ""): v
                        for k, v in gm_state.items()
                        if k.startswith("global_combiner.")
                    }
            if gm_adapter_state is not None and self.generalized_adapter is not None:
                self.generalized_adapter.load_state_dict(gm_adapter_state, strict=True)
            if global_combiner_state is not None:
                self.combiner.load_state_dict(global_combiner_state, strict=True)
                combiner_updated = True

        self.model.generalized_module = self.generalized_module
        if self.generalized_adapter is not None:
            self.model.generalized_adapter = self.generalized_adapter
        self.model.combiner = self.combiner
        if combiner_updated:
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
        combiner_state = {k.replace("combiner.", ""): v for k, v in state.items() if k.startswith("combiner.")}
        if combiner_state:
            self.combiner.load_state_dict(combiner_state, strict=True)
            
        # [Fix] Reset optimizer state when model parameters are forcibly changed
        # This prevents momentum from previous (possibly incompatible) weights 
        # from interfering with the new cluster model.
        self._reset_optimizer()

    def _reset_optimizer(self):
        pm_params = list(self.personalized_module.parameters()) + list(self.combiner.parameters())
        if self.personalized_adapter is not None:
            pm_params += list(self.personalized_adapter.parameters())
        self.optimizer = torch.optim.SGD(pm_params, lr=self.learning_rate)

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
        state.update({f"combiner.{k}": v.detach().cpu() for k, v in self.combiner.state_dict().items()})
        return state

    def upload_class_prototypes(self, max_samples=None):
        """
        Upload PM-only class prototypes with confidence-weighted sums.
        - class_logit_conf_sum[c]: sum(logits * confidence) for class c
        - class_conf_sum[c]: sum(confidence) for class c
        - class_counts[c]: number of samples for class c
        """
        if max_samples is None:
            max_samples = self.prototype_samples
        max_samples = int(max_samples) if max_samples is not None else 0

        device = "cpu"
        self.f_ext.to(device)
        self.f_ext.eval()
        self.personalized_module.to(device)
        self.personalized_module.eval()
        self.combiner.to(device)
        self.combiner.eval()
        if self.personalized_adapter is not None:
            self.personalized_adapter.to(device)
            self.personalized_adapter.eval()

        class_logit_conf_sum = torch.zeros(self.num_classes, self.num_classes, device=device)
        class_conf_sum = torch.zeros(self.num_classes, device=device)
        class_counts = torch.zeros(self.num_classes, device=device)

        count_seen = 0
        batch_size = min(self.batch_size, 64)
        trainloader = self.load_train_data(batch_size=batch_size)

        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x = x[0]
                x = x.to(device)
                y = y.to(device).long()

                z = self.f_ext(x)
                if z.dim() > 2:
                    z = torch.flatten(z, 1)
                z_pm = self.personalized_adapter(z) if self.personalized_adapter is not None else z
                pm_logits = self.personalized_module(z_pm)
                gm_zeros = torch.zeros_like(pm_logits)
                pm_only_logits = self.combiner(torch.cat([gm_zeros, pm_logits], dim=1))
                pm_only_prob = torch.softmax(pm_only_logits, dim=1)
                confidence = torch.max(pm_only_prob, dim=1).values

                if max_samples > 0 and count_seen + y.size(0) > max_samples:
                    keep = max_samples - count_seen
                    if keep <= 0:
                        break
                    y = y[:keep]
                    pm_only_logits = pm_only_logits[:keep]
                    confidence = confidence[:keep]

                for c in y.unique():
                    cls = int(c.item())
                    mask = (y == c)
                    if not torch.any(mask):
                        continue
                    cls_logits = pm_only_logits[mask]
                    cls_conf = confidence[mask]
                    class_logit_conf_sum[cls] += (cls_logits * cls_conf.unsqueeze(1)).sum(dim=0)
                    class_conf_sum[cls] += cls_conf.sum()
                    class_counts[cls] += mask.sum()

                count_seen += y.size(0)
                if max_samples > 0 and count_seen >= max_samples:
                    break

        return {
            "class_logit_conf_sum": class_logit_conf_sum.detach().cpu(),
            "class_conf_sum": class_conf_sum.detach().cpu(),
            "class_counts": class_counts.detach().cpu(),
        }
