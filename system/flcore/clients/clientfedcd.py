import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientbase import Client

class clientFedCD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # 1. 모델 분리 (GM: Teacher/Frozen, PM: Student/Trainable)
        # PFLlib은 self.model에 전체 모델을 로드해줍니다.
        self.gm = copy.deepcopy(self.model) # General Module (CPU)
        self.pm = copy.deepcopy(self.model) # Personalized Module (CPU)
        self.f_ext = self._build_feature_extractor(self.gm)
        
        # GM은 영원히 Freeze (BN 통계량 포함)
        for param in self.gm.parameters():
            param.requires_grad = False
        self.gm.eval() 

        # Optimizer는 PM만 관리
        self.optimizer = torch.optim.SGD(self.pm.parameters(), lr=self.learning_rate)
        self.loss_func = nn.CrossEntropyLoss()
        self.nc_weight = float(getattr(args, "fedcd_nc_weight", 0.0))
        self.warmup_epochs = int(getattr(args, "fedcd_warmup_epochs", 0))
        self.gm_head, self.gm_final = self._split_classifier(self.gm)
        self.pm_head, self.pm_final = self._split_classifier(self.pm)

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
        trainloader = self.load_train_data()
        # Move models to GPU only during training to reduce memory usage
        self.f_ext.to(self.device)
        self.gm.to(self.device)
        self.pm.to(self.device)
        self.pm.train()
        self.gm.eval() # [중요] BN 통계량 고정

        for epoch in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                # Shared feature extractor
                z = self.f_ext(x)

                # 1. GM의 지식 (Reference)
                with torch.no_grad():
                    gm_feat, logits_gm = self._forward_shared(z, self.gm_head, self.gm_final)

                # 2. PM의 예측
                pm_feat, logits_pm = self._forward_shared(z, self.pm_head, self.pm_final)

                # 3. Loss 계산 (Task Loss + Feature-wise Negative Correlation)
                loss = self.loss_func(logits_pm, y)
                if self.nc_weight > 0:
                    fused = (gm_feat + pm_feat) / 2.0
                    nc_term = (gm_feat - fused) * (pm_feat - fused)
                    nc_loss = nc_term.mean()
                    loss = loss + self.nc_weight * nc_loss
                # 필요시 여기에 GM과의 Distillation Loss 추가 가능
                
                loss.backward()
                self.optimizer.step()

        # Sync base model for evaluation
        self._sync_feature_extractor(self.pm)
        self.model.load_state_dict(self.pm.state_dict(), strict=True)

        # Move back to CPU after training to free GPU memory
        self.f_ext.to("cpu")
        self.gm.to("cpu")
        self.pm.to("cpu")
        self.model.to("cpu")
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def warmup_classifier(self):
        if self.warmup_epochs <= 0:
            return

        classifier = self._select_classifier(self.pm)
        if classifier is None:
            return

        # Freeze all but classifier
        for param in self.pm.parameters():
            param.requires_grad = False
        for param in classifier.parameters():
            param.requires_grad = True

        self.pm.to(self.device)
        self.pm.train()
        optimizer = torch.optim.SGD(classifier.parameters(), lr=self.learning_rate)
        trainloader = self.load_train_data()

        for _ in range(self.warmup_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                logits = self.pm(x)
                loss = self.loss_func(logits, y)
                loss.backward()
                optimizer.step()

        # Sync and clean up
        self.model.load_state_dict(self.pm.state_dict(), strict=True)
        self.pm.to("cpu")
        self.model.to("cpu")
        if self.device == "cuda":
            torch.cuda.empty_cache()

    # [핵심] 서버로 보낼 때: GM은 안 보내고 PM만 보냄 (Zero-Uplink for GM)
    def set_parameters(self, model):
        # 서버에서 받은 글로벌 GM을 내 GM에 업데이트
        state = model.state_dict()
        if next(iter(state.values())).is_cuda:
            state = {k: v.detach().cpu() for k, v in state.items()}
        self.gm.load_state_dict(state, strict=True)
        self.f_ext = self._build_feature_extractor(self.gm)
        
        # PM은 GM과 너무 멀어지지 않게 살짝 당겨주거나(Regularization), 
        # 혹은 그대로 둡니다(Pure Personalization). 연구 의도에 따라 선택.
        # 여기서는 일단 PM은 그대로 유지하는 것으로 둡니다.

    def set_personalized_parameters(self, pm_state):
        # 서버에서 받은 클러스터 대표 PM을 내 PM에 업데이트
        state = pm_state
        if next(iter(state.values())).is_cuda:
            state = {k: v.detach().cpu() for k, v in state.items()}
        self.pm.load_state_dict(state, strict=True)
        self.model.load_state_dict(state, strict=True)

    def upload_parameters(self):
        # 업링크 비용 절감: PM만 전송
        return {k: v.detach().cpu() for k, v in self.pm.state_dict().items()}