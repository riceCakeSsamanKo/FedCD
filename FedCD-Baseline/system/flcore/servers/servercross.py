import torch
import numpy as np
import copy
import time
from flcore.servers.serverbase import Server
from flcore.trainmodel.models import *
from flcore.clients.clientcross import clientCross
import torch.nn.functional as F

class FedCross(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientCross)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

        self.first_stage_bound = args.first_stage_bound
        self.cross_alpha = args.fedcross_alpha
        self.collaberative_model_select_strategy = args.collaberative_model_select_strategy
        
        # Store local models for cross aggregation
        self.w_locals = []
        self.w_locals_num = self.num_join_clients
        for i in range(self.w_locals_num):
            self.w_locals.append(copy.deepcopy(self.global_model))

        self.w_locals_by_client = {}
        if self.dynamic_client_enabled:
            self.w_locals_by_client = {
                int(client.id): copy.deepcopy(self.global_model)
                for client in self.clients
            }

    def _on_dynamic_clients_activated(self, new_clients):
        for client in new_clients:
            client.set_parameters(self.global_model)
            self.w_locals_by_client[int(client.id)] = copy.deepcopy(self.global_model)

    def train(self):
        for round_idx in range(self.global_rounds+1):
            s_t = time.time()

            self.selected_clients = self.select_clients()

            if round_idx%self.eval_gap == 0:
                print(f"\n-------------Round number: {round_idx}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            selected_local_models = []
            for local_idx, client in enumerate(self.selected_clients):
                start_time = time.time()
                if self.dynamic_client_enabled:
                    local_model = self.w_locals_by_client[int(client.id)]
                else:
                    local_model = self.w_locals[local_idx]
                client.set_parameters(local_model)
                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
                client.train()
                client.clone_model(client.model, local_model)
                selected_local_models.append(local_model)

            self.downlink_MB += len(self.selected_clients) * self.model_size_MB

            # Receive models from clients
            self.receive_models()
            if self.dlg_eval and round_idx%self.dlg_gap == 0:
                self.call_dlg(round_idx)
            ### args.client_drop_rate should be 0
            ### equal to aggregate with w_locals
            self.aggregate_parameters()

            # Calculate similarity between models
            models_for_cross = (
                selected_local_models
                if self.dynamic_client_enabled
                else self.w_locals
            )
            sim_tab, sim_value = self.calculate_similarity(models_for_cross)

            # Update global model
            if round_idx >= self.first_stage_bound and len(models_for_cross) > 1:
                # Cross aggregation
                updated_models = self.cross_aggregation(
                    round_idx,
                    sim_tab,
                    models=models_for_cross,
                )
            else:
                updated_models = models_for_cross
                for local_model in updated_models:
                    for param, global_param in zip(local_model.parameters(), self.global_model.parameters()):
                        param.data = global_param.data.clone()

            if self.dynamic_client_enabled:
                for client, local_model in zip(self.selected_clients, updated_models):
                    self.w_locals_by_client[int(client.id)] = local_model
            else:
                self.w_locals = updated_models

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientCross)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def calculate_similarity(self, models=None):
        models = self.w_locals if models is None else models
        model_num = len(models)
        if model_num <= 1:
            return [[1.0]] if model_num == 1 else [], 0.0
        sim_tab = [[0 for _ in range(model_num)] for _ in range(model_num)]
        sum_sim = 0.0

        w_locals_dict = [model.state_dict() for model in models]
        
        for k in range(model_num):
            for j in range(k):
                s = 0.0
                dict_a = torch.Tensor(0)
                dict_b = torch.Tensor(0)
                cnt = 0
                
                for p in w_locals_dict[k].keys():
                    a = w_locals_dict[k][p]
                    b = w_locals_dict[j][p]
                    a = a.view(-1)
                    b = b.view(-1)

                    if cnt == 0:
                        dict_a = a
                        dict_b = b
                    else:
                        dict_a = torch.cat((dict_a, a), dim=0)
                        dict_b = torch.cat((dict_b, b), dim=0)

                    if cnt % 2 == 0:
                        sub_a = a
                        sub_b = b
                    else:
                        sub_a = torch.cat((sub_a, a), dim=0)
                        sub_b = torch.cat((sub_b, b), dim=0)

                    if cnt % 2 == 1:
                        s += F.cosine_similarity(sub_a, sub_b, dim=0)
                    cnt += 1
                
                s += F.cosine_similarity(sub_a, sub_b, dim=0)
                sim_tab[k][j] = s
                sim_tab[j][k] = s
                sum_sim += copy.deepcopy(s)

        l = int(len(w_locals_dict[0].keys()) / 5) + 1.0
        sum_sim /= (l * model_num * (model_num - 1) / 2.0)
        
        return sim_tab, sum_sim

    def cross_aggregation(self, iter, sim_tab, models=None):
        source_models = self.w_locals if models is None else models
        model_num = len(source_models)
        if model_num <= 1:
            return copy.deepcopy(source_models)
        w_locals_new = copy.deepcopy(source_models)
        crosslist = []

        for j in range(model_num):
            maxtag = 0
            submax = 1
            mintag = (j + 1) % model_num
            
            for p in range(model_num):
                if sim_tab[j][p] > sim_tab[j][maxtag]:
                    submax = maxtag
                    maxtag = p
                elif sim_tab[j][p] > sim_tab[j][submax]:
                    submax = p
                if sim_tab[j][p] < sim_tab[j][mintag] and p != j:
                    mintag = p

            rlist = []
            offset = iter % (model_num - 1) + 1
            sub_list = []
            
            for k in range(model_num):
                if k == j:
                    rlist.append(self.cross_alpha)
                    sub_list.append(copy.deepcopy(source_models[j]))

                if self.collaberative_model_select_strategy == 0:
                    if (j + offset) % model_num == k:
                        rlist.append(1.0 - self.cross_alpha)
                        sub_list.append(copy.deepcopy(source_models[k]))
                elif self.collaberative_model_select_strategy == 1:
                    if mintag == k:
                        rlist.append(1.0 - self.cross_alpha)
                        sub_list.append(copy.deepcopy(source_models[mintag]))
                elif self.collaberative_model_select_strategy == 2:
                    if maxtag == k:
                        rlist.append(1.0 - self.cross_alpha)
                        sub_list.append(copy.deepcopy(source_models[maxtag]))

            # Aggregate selected models
            w_cc = self.aggregate_parameters_cross(sub_list, rlist)
            crosslist.append(w_cc)

        for k in range(model_num):
            w_locals_new[k] = crosslist[k]

        return w_locals_new

    def aggregate_parameters_cross(self, models=None, weights=None):
        if models is None:
            models = self.uploaded_models
            weights = self.uploaded_weights
            
        aggregated_model = copy.deepcopy(models[0])
        for param in aggregated_model.parameters():
            param.data.zero_()
            
        total_count = sum(weights)
        for w, client_model in zip(weights, models):
            for aggregated_model_param, client_param in zip(aggregated_model.parameters(), client_model.parameters()):
                aggregated_model_param.data += client_param.data.clone() * w / total_count
            
        return aggregated_model
