import copy
import torch
import torch.nn as nn
import numpy as np
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
        self.global_period = max(1, int(getattr(args, "global_period", 1)))
        self.cluster_sample_size = int(getattr(args, "cluster_sample_size", 512))
        self.cluster_map = {c.id: (c.id % self.num_clusters) for c in self.clients}

    def train(self):
        for i in range(self.global_rounds + 1):
            self.selected_clients = self.select_clients()
            print(f"\n[FedCD] Round {i}: training {len(self.selected_clients)} clients")
            
            # 1. 로컬 학습 수행 (Local Training)
            for client in self.selected_clients:
                client.train()
            print(f"[FedCD] Round {i}: local training done")

            # 2. PM 수집 (Receive PMs)
            received_pms = []
            for client in self.selected_clients:
                received_pms.append((client.id, client.upload_parameters()))
            print(f"[FedCD] Round {i}: received {len(received_pms)} PMs")

            # 2.5 클러스터링 갱신
            if i % self.cluster_period == 0:
                self.cluster_map = self.cluster_clients_by_distribution()
                cluster_counts = {}
                for cid in self.cluster_map.values():
                    cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
                print(f"[FedCD] Round {i}: cluster sizes = {cluster_counts}")

            # 3. 클러스터 내 PM 집계 및 배포
            cluster_pms = self.aggregate_cluster_pms(received_pms)
            if i % self.cluster_period == 0 and cluster_pms:
                self.send_cluster_pms(cluster_pms)
                print(f"[FedCD] Round {i}: sent cluster PMs")

            # 4. 서버 측 앙상블 증류 (Server-side Ensemble Distillation)
            if i % self.global_period == 0 and cluster_pms:
                self.aggregate_and_distill(list(cluster_pms.values()))
                # 업데이트된 GM을 모든 클라이언트에게 배포 (Downlink)
                self.send_models()
                print(f"[FedCD] Round {i}: distillation + GM broadcast done")
                # Warm-up classifier after GM update
                if getattr(self.args, "fedcd_warmup_epochs", 0) > 0:
                    for client in self.selected_clients:
                        client.warmup_classifier()
            
            if i % self.eval_gap == 0:
                print(f"\n------------- Round number: {i} -------------")
                self.evaluate()

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
        for client in self.clients:
            cluster_id = self.cluster_map.get(client.id, 0)
            if cluster_id in cluster_pms:
                client.set_personalized_parameters(cluster_pms[cluster_id])

    def cluster_clients_by_distribution(self):
        # Cluster clients by f_ext feature distribution stats
        from sklearn.cluster import KMeans

        features = []
        client_ids = []
        for client in self.clients:
            mean, var = client.get_feature_stats(self.cluster_sample_size)
            feat = torch.cat([mean, var], dim=0).cpu().numpy()
            features.append(feat)
            client_ids.append(client.id)

        n_clusters = min(self.num_clusters, len(client_ids))
        if n_clusters <= 1:
            return {cid: 0 for cid in client_ids}

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = kmeans.fit_predict(np.stack(features, axis=0))
        return {cid: int(label) for cid, label in zip(client_ids, labels)}

    @staticmethod
    def average_state_dicts(states):
        avg_state = {}
        for key in states[0].keys():
            avg_state[key] = sum(state[key] for state in states) / len(states)
        return avg_state

    def aggregate_and_distill(self, received_pm_states):
        print("Server: Distilling Knowledge from PM Ensemble to GM...")
        
        # GM (Student) 학습 준비
        self.global_model.to(self.device)
        self.global_model.train()
        optimizer = torch.optim.SGD(self.global_model.parameters(), lr=0.01)
        kl_loss = nn.KLDivLoss(reduction='batchmean')

        # Proxy Data로 증류 (예: 1 Epoch)
        # Use a single teacher model on GPU to avoid OOM
        teacher = copy.deepcopy(self.global_model).to(self.device)
        teacher.eval()
        for x, _ in self.proxy_data_loader:
            x = x.to(self.device)
            
            # Teacher Ensemble Logits (sequential to save memory)
            with torch.no_grad():
                ensemble_logits = None
                for state in received_pm_states:
                    teacher.load_state_dict(state, strict=True)
                    logits = teacher(x)
                    if ensemble_logits is None:
                        ensemble_logits = logits
                    else:
                        ensemble_logits = ensemble_logits + logits
                ensemble_logits = ensemble_logits / len(received_pm_states)
                target_prob = torch.softmax(ensemble_logits, dim=1)
            
            # Student (GM) Logits
            student_logits = self.global_model(x)
            student_log_prob = torch.log_softmax(student_logits, dim=1)
            
            loss = kl_loss(student_log_prob, target_prob)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Move back to CPU to reduce GPU memory pressure
        teacher.to("cpu")
        self.global_model.to("cpu")
        torch.cuda.empty_cache()
            
    def load_proxy_data(self):
        # 원래는 별도의 공용 데이터셋을 로드해야 하지만, 
        # 실험을 위해 일단 Test 데이터셋 로더를 가져와서 씁니다.
        # (PFLlib은 args에 데이터 정보가 들어있음)
        from utils.data_utils import read_client_data
        
        # 임시: 첫 번째 클라이언트의 테스트 데이터를 Proxy로 사용
        # (실제 연구에선 이렇게 하면 안 되지만 코드 검증용으로는 OK)
        test_data = read_client_data(self.args.dataset, 0, is_train=False)
        return torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)