import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from flcore.clients.clientbase import Client
from flcore.trainmodel.models import SmallFExt

class PMWrapper(nn.Module):
    def __init__(self, f_ext, gm_head, gm_final, pm_head, pm_final, combiner, gm_adapter=None, pm_adapter=None):
        super().__init__()
        self.f_ext = f_ext
        self.gm_head = gm_head
        self.gm_final = gm_final
        self.pm_head = pm_head
        self.pm_final = pm_final
        self.combiner = combiner
        self.gm_adapter = gm_adapter
        self.pm_adapter = pm_adapter

    def forward(self, x):
        z = self.f_ext(x)
        if z.dim() > 2:
            z = torch.flatten(z, 1)
        z_gm = self.gm_adapter(z) if self.gm_adapter is not None else z
        gm_logits = self.gm_final(self.gm_head(z_gm))
        if self.pm_adapter is not None:
            z = self.pm_adapter(z)
        pm_logits = self.pm_final(self.pm_head(z))
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
        self.warmup_epochs = int(getattr(args, "fedcd_warmup_epochs", 0))
        self.gm_head, self.gm_final = self._split_classifier(self.gm)
        self.pm_head, self.pm_final = self._split_classifier(self.pm)
        self.f_ext_dim = getattr(self.f_ext, "out_dim", None)
        self.gm_adapter = self._build_adapter(self.gm_head, self.gm_final)
        self.pm_adapter = self._build_adapter(self.pm_head, self.pm_final)
        self.combiner = nn.Linear(self.num_classes * 2, self.num_classes)
        self.model = PMWrapper(
            self.f_ext,
            self.gm_head,
            self.gm_final,
            self.pm_head,
            self.pm_final,
            self.combiner,
            self.gm_adapter,
            self.pm_adapter,
        )

        # Optimizer는 PM head(+adapter)만 관리
        pm_params = list(self.pm_head.parameters()) + list(self.pm_final.parameters()) + list(self.combiner.parameters())
        if self.pm_adapter is not None:
            pm_params += list(self.pm_adapter.parameters())
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

    def _forward_shared(self, z, head, final):
        feat = head(z)
        logits = final(feat)
        return feat, logits

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
            self.pm_head.to(device)
            self.pm_final.to(device)
            self.combiner.to(device)
            if self.gm_adapter is not None:
                self.gm_adapter.to(device)
            if self.pm_adapter is not None:
                self.pm_adapter.to(device)
            self.pm_head.train()
            self.pm_final.train()
            self.combiner.train()
            self.gm.eval() # [중요] BN 통계량 고정
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
                            z_gm = self.gm_adapter(z) if self.gm_adapter is not None else z
                            gm_feat, logits_gm = self._forward_shared(z_gm, self.gm_head, self.gm_final)

                        # 2. PM의 예측
                        z_pm = self.pm_adapter(z) if self.pm_adapter is not None else z
                        pm_feat, logits_pm = self._forward_shared(z_pm, self.pm_head, self.pm_final)
                        fused_logits = self.combiner(torch.cat([logits_gm, logits_pm], dim=1))

                        # 3. Loss 계산 (Task Loss + Feature-wise Negative Correlation)
                        loss = self.loss_func(fused_logits, y)
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
                self.pm_head.to("cpu")
                self.pm_final.to("cpu")
                self.combiner.to("cpu")
                if self.gm_adapter is not None:
                    self.gm_adapter.to("cpu")
                if self.pm_adapter is not None:
                    self.pm_adapter.to("cpu")
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

    def warmup_classifier(self):
        if self.warmup_epochs <= 0:
            return

        def _warmup_once(device, batch_size):
            self.f_ext.to(device)
            self.gm_head.to(device)
            self.gm_final.to(device)
            self.f_ext.eval()
            self.pm_head.to(device)
            self.pm_final.to(device)
            self.combiner.to(device)
            if self.gm_adapter is not None:
                self.gm_adapter.to(device)
            if self.pm_adapter is not None:
                self.pm_adapter.to(device)
            self.pm_head.train()
            self.pm_final.train()
            self.combiner.train()
            optimizer = torch.optim.SGD(
                list(self.pm_head.parameters()) +
                list(self.pm_final.parameters()) +
                list(self.combiner.parameters()) +
                (list(self.pm_adapter.parameters()) if self.pm_adapter is not None else []),
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
                        z_pm = self.pm_adapter(z) if self.pm_adapter is not None else z
                        logits_pm = self.pm_final(self.pm_head(z_pm))
                        z_gm = self.gm_adapter(z) if self.gm_adapter is not None else z
                        logits_gm = self.gm_final(self.gm_head(z_gm))
                        logits = self.combiner(torch.cat([logits_gm, logits_pm], dim=1))
                        loss = self.loss_func(logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            # Clean up
            # [수정] avoid_oom이 True일 때만 CPU로 내림.
            if self.args.avoid_oom:
                self.f_ext.to("cpu")
                self.gm_head.to("cpu")
                self.gm_final.to("cpu")
                self.pm_head.to("cpu")
                self.pm_final.to("cpu")
                self.combiner.to("cpu")
                if self.gm_adapter is not None:
                    self.gm_adapter.to("cpu")
                if self.pm_adapter is not None:
                    self.pm_adapter.to("cpu")
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
                self.gm_head.to("cpu")
                self.gm_final.to("cpu")
                self.pm_head.to("cpu")
                self.model.to("cpu")
                _warmup_once("cpu", max(1, batch_size // 2))
                return
            raise

    # [핵심] 서버로 보낼 때: GM은 안 보내고 PM만 보냄 (Zero-Uplink for GM)
    def set_parameters(self, model):
        # 서버에서 받은 글로벌 GM(+gm_adapter)을 내 GM에 업데이트
        if isinstance(model, dict):
            gm_state = model.get("gm_state", {})
            gm_adapter_state = model.get("gm_adapter", None)
        else:
            gm_state = model.state_dict()
            gm_adapter_state = None
        if gm_state and next(iter(gm_state.values())).is_cuda:
            gm_state = {k: v.detach().cpu() for k, v in gm_state.items()}
        if gm_state:
            self.gm.load_state_dict(gm_state, strict=True)
        if gm_adapter_state is not None and self.gm_adapter is not None:
            self.gm_adapter.load_state_dict(gm_adapter_state, strict=True)
        self.model.gm_head = self.gm_head
        self.model.gm_final = self.gm_final
        if self.gm_adapter is not None:
            self.model.gm_adapter = self.gm_adapter
        
        # PM은 GM과 너무 멀어지지 않게 살짝 당겨주거나(Regularization), 
        # 혹은 그대로 둡니다(Pure Personalization). 연구 의도에 따라 선택.
        # 여기서는 일단 PM은 그대로 유지하는 것으로 둡니다.

    def set_personalized_parameters(self, pm_state):
        # 서버에서 받은 클러스터 대표 PM을 내 PM에 업데이트
        state = pm_state
        if next(iter(state.values())).is_cuda:
            state = {k: v.detach().cpu() for k, v in state.items()}
        head_state = {k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")}
        final_state = {k.replace("final.", ""): v for k, v in state.items() if k.startswith("final.")}
        adapter_state = {k.replace("adapter.", ""): v for k, v in state.items() if k.startswith("adapter.")}
        if head_state:
            self.pm_head.load_state_dict(head_state, strict=True)
        if final_state:
            self.pm_final.load_state_dict(final_state, strict=True)
        if self.pm_adapter is not None and adapter_state:
            self.pm_adapter.load_state_dict(adapter_state, strict=True)
        combiner_state = {k.replace("combiner.", ""): v for k, v in state.items() if k.startswith("combiner.")}
        if combiner_state:
            self.combiner.load_state_dict(combiner_state, strict=True)
            
        # [Fix] Reset optimizer state when model parameters are forcibly changed
        # This prevents momentum from previous (possibly incompatible) weights 
        # from interfering with the new cluster model.
        self._reset_optimizer()

    def _reset_optimizer(self):
        pm_params = list(self.pm_head.parameters()) + list(self.pm_final.parameters()) + list(self.combiner.parameters())
        if self.pm_adapter is not None:
            pm_params += list(self.pm_adapter.parameters())
        self.optimizer = torch.optim.SGD(pm_params, lr=self.learning_rate)

    def upload_parameters(self):
        # 업링크 비용 절감: PM만 전송
        state = {}
        state.update({f"head.{k}": v.detach().cpu() for k, v in self.pm_head.state_dict().items()})
        state.update({f"final.{k}": v.detach().cpu() for k, v in self.pm_final.state_dict().items()})
        if self.pm_adapter is not None:
            state.update({f"adapter.{k}": v.detach().cpu() for k, v in self.pm_adapter.state_dict().items()})
        state.update({f"combiner.{k}": v.detach().cpu() for k, v in self.combiner.state_dict().items()})
        return state