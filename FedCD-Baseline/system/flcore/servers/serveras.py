import time
import copy
import numpy as np
# from flcore.clients.clientavg import clientAVG
from flcore.clients.clientas import clientAS
from flcore.servers.serverbase import Server
from utils.model_state import average_module_states


class FedAS(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAS)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def all_clients(self):
        return self._dynamic_client_active_clients()

    def _on_dynamic_clients_activated(self, new_clients):
        progress = max(0, self.dynamic_client_round) / max(1, self.global_rounds)
        for client in new_clients:
            client.set_parameters(copy.deepcopy(self.global_model), progress)

    def send_selected_models(self, selected_ids, epoch):
        assert (len(self.clients) > 0)

        # for client in self.clients:
        for client in [client for client in self.clients if (client.id in selected_ids)]:
            start_time = time.time()

            progress = epoch / self.global_rounds
            
            client.set_parameters(copy.deepcopy(self.global_model), progress)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)    

        # FedAS uses custom dispatch path; track downlink explicitly.
        self.downlink_MB += len(selected_ids) * self.model_size_MB
    
    def aggregate_wrt_fisher(self):
        assert (len(self.uploaded_models) > 0)

        # calculate the aggregrate weight with respect to the FIM value of model
        FIM_weight_list = []
        for id in self.uploaded_ids:
            FIM_weight_list.append(self.clients[id].fim_trace_history[-1])
        # normalization to obtain weight
        FIM_weight_list = [FIM_value/sum(FIM_weight_list) for FIM_value in FIM_weight_list]

        self.global_model = average_module_states(
            self.uploaded_models,
            FIM_weight_list,
        )

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.alled_clients = self.all_clients()

            selected_ids = [client.id for client in self.selected_clients]


            # self.send_models()

            # evaluate personalized models, ie FedAvg-C
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            # self.send_models()
            self.send_selected_models(selected_ids, i)

            # print(f'send selected models done')

            # for client in self.selected_clients:
            #     client.train()
        

            for client in self.alled_clients:
                # print("===============")
                client.train(client.id in selected_ids)
            # assert 1==0


            self.print_fim_histories()



            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)


            self.aggregate_wrt_fisher()

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

        print(f'+++++++++++++++++++++++++++++++++++++++++')
        gen_acc = self.avg_generalization_metrics()
        print(f'Generalization Acc: {gen_acc}')
        print(f'+++++++++++++++++++++++++++++++++++++++++')

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAS)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def print_fim_histories(self):
        avg_fim_histories = []

        # Print FIM trace history for each client
        # for client in self.selected_clients:
        for client in self.alled_clients:
            formatted_history = [f"{value:.1f}" for value in client.fim_trace_history]
            print(f"Client{client.id} : {formatted_history}")
            avg_fim_histories.append(client.fim_trace_history)

        if self.dynamic_client_enabled:
            latest_values = [history[-1] for history in avg_fim_histories if history]
            latest_avg = float(np.mean(latest_values)) if latest_values else 0.0
            print(f"Avg Current Sum_T_FIM : {latest_avg:.1f}")
            return

        # Calculate and print average FIM trace history across clients
        avg_fim_histories = np.mean(avg_fim_histories, axis=0)
        formatted_avg = [f"{value:.1f}" for value in avg_fim_histories]
        print(f"Avg Sum_T_FIM : {formatted_avg}")
