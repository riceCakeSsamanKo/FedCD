import copy
import torch
import torch.nn as nn
from flcore.clients.clientfedcd import clientFedCD
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data 

class FedCD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # 클라이언트 클래스 연결
        self.set_clients(clientFedCD)
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        # [중요] 서버용 Proxy Data 로드 (Knowledge Distillation용)
        # 일단 테스트 데이터셋의 일부를 Proxy로 쓴다고 가정 (실제 연구에선 별도 데이터 권장)
        self.proxy_data_loader = self.load_proxy_data() 

    def train(self):
        for i in range(self.global_rounds + 1):
            self.selected_clients = self.select_clients()
            
            # 1. 로컬 학습 수행 (Local Training)
            for client in self.selected_clients:
                client.train()

            # 2. PM 수집 (Receive PMs)
            received_pms = []
            for client in self.selected_clients:
                received_pms.append(client.upload_parameters())
            
            # 3. 서버 측 앙상블 증류 (Server-side Ensemble Distillation)
            self.aggregate_and_distill(received_pms)
            
            # 4. 업데이트된 GM을 모든 클라이언트에게 배포 (Downlink)
            self.send_models()
            
            if i % self.eval_gap == 0:
                print(f"\n------------- Round number: {i} -------------")
                self.evaluate()

    def aggregate_and_distill(self, received_pm_states):
        print("Server: Distilling Knowledge from PM Ensemble to GM...")
        
        # PM 앙상블 준비
        teachers = []
        for state in received_pm_states:
            temp_model = copy.deepcopy(self.global_model)
            temp_model.load_state_dict(state)
            temp_model.eval()
            teachers.append(temp_model)
        
        # GM (Student) 학습 준비
        self.global_model.train()
        optimizer = torch.optim.SGD(self.global_model.parameters(), lr=0.01)
        kl_loss = nn.KLDivLoss(reduction='batchmean')

        # Proxy Data로 증류 (예: 1 Epoch)
        for x, _ in self.proxy_data_loader:
            x = x.to(self.device)
            
            # Teacher Ensemble Logits
            with torch.no_grad():
                teacher_logits = [t(x) for t in teachers]
                ensemble_logits = torch.stack(teacher_logits).mean(dim=0)
                target_prob = torch.softmax(ensemble_logits, dim=1)
            
            # Student (GM) Logits
            student_logits = self.global_model(x)
            student_log_prob = torch.log_softmax(student_logits, dim=1)
            
            loss = kl_loss(student_log_prob, target_prob)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    def load_proxy_data(self):
        # 원래는 별도의 공용 데이터셋을 로드해야 하지만, 
        # 실험을 위해 일단 Test 데이터셋 로더를 가져와서 씁니다.
        # (PFLlib은 args에 데이터 정보가 들어있음)
        from utils.data_utils import read_client_data
        
        # 임시: 첫 번째 클라이언트의 테스트 데이터를 Proxy로 사용
        # (실제 연구에선 이렇게 하면 안 되지만 코드 검증용으로는 OK)
        train_data, test_data = read_client_data(self.args.dataset, 0, is_train=False)
        return torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)