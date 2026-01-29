import copy
import torch
import torch.nn as nn
from flcore.clients.clientbase import Client

class clientFedCD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # 1. 모델 분리 (GM: Teacher/Frozen, PM: Student/Trainable)
        # PFLlib은 self.model에 전체 모델을 로드해줍니다.
        self.gm = copy.deepcopy(self.model) # General Module
        self.pm = copy.deepcopy(self.model) # Personalized Module (실제 학습 대상)
        
        # GM은 영원히 Freeze (BN 통계량 포함)
        for param in self.gm.parameters():
            param.requires_grad = False
        self.gm.eval() 

        # Optimizer는 PM만 관리
        self.optimizer = torch.optim.SGD(self.pm.parameters(), lr=self.learning_rate)
        self.loss_func = nn.CrossEntropyLoss()

    def train(self):
        trainloader = self.load_train_data()
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

                # 1. GM의 지식 (Reference)
                with torch.no_grad():
                    logits_gm = self.gm(x)

                # 2. PM의 예측
                logits_pm = self.pm(x)

                # 3. Loss 계산 (Task Loss + Optional Distillation)
                loss = self.loss_func(logits_pm, y)
                # 필요시 여기에 GM과의 Distillation Loss 추가 가능
                
                loss.backward()
                self.optimizer.step()

    # [핵심] 서버로 보낼 때: GM은 안 보내고 PM만 보냄 (Zero-Uplink for GM)
    def set_parameters(self, model):
        # 서버에서 받은 글로벌 GM을 내 GM에 업데이트
        for new_param, old_param in zip(model.parameters(), self.gm.parameters()):
            old_param.data = new_param.data.clone()
        
        # PM은 GM과 너무 멀어지지 않게 살짝 당겨주거나(Regularization), 
        # 혹은 그대로 둡니다(Pure Personalization). 연구 의도에 따라 선택.
        # 여기서는 일단 PM은 그대로 유지하는 것으로 둡니다.

    def upload_parameters(self):
        # 업링크 비용 절감: PM만 전송
        return copy.deepcopy(self.pm.state_dict())