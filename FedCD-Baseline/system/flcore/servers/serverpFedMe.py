import os
import time
import copy
import h5py
import numpy as np
import torch
from flcore.clients.clientpFedMe import clientpFedMe
from flcore.servers.serverbase import Server
from threading import Thread


class pFedMe(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientpFedMe)

        self.beta = args.beta
        self.rs_train_acc_per = []
        self.rs_train_loss_per = []
        self.rs_test_acc_per = []
        self.rs_global_test_acc_per = []

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []

    def _on_dynamic_clients_activated(self, new_clients):
        for client in new_clients:
            client.set_parameters(self.global_model)
            client.personalized_params = client._clone_param_list(
                self.global_model.parameters()
            )

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                global_metrics = self.evaluate_global_model()
                print("\nEvaluate personalized model")
                personalized_metrics = self.evaluate_personalized()
                self.log_usage_combined(global_metrics, personalized_metrics)

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.previous_global_model = copy.deepcopy(list(self.global_model.parameters()))
            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            self.beta_aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc_per], top_cnt=self.top_cnt):
                break

        # print("\nBest global accuracy.")
        # # self.print_(max(self.rs_test_acc), max(
        # #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(self.rs_test_acc))

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc_per), max(
        #     self.rs_train_acc_per), min(self.rs_train_loss_per))
        print(max(self.rs_test_acc_per))
        print("\nAverage time cost per round.")
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:])/len(self.Budget[1:]))
        elif len(self.Budget) == 1:
            print(self.Budget[0])
        else:
            print(0.0)


        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientpFedMe)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def evaluate_personalized_global_test_acc(self):
        if not self.eval_common_global or self.global_test_loader is None:
            return None

        acc_sum = 0.0
        valid_clients = 0

        eval_clients = self._evaluation_clients()
        for client in eval_clients:
            client.update_parameters(client.model, client.personalized_params)
            client.model.to(self.device)
            client.model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in self.global_test_loader:
                    if type(x) == type([]):
                        x = x[0]
                    x = x.to(self.device)
                    y = y.to(self.device)
                    output = client.model(x)
                    correct += (torch.argmax(output, dim=1) == y).sum().item()
                    total += y.size(0)

            client.model.cpu()
            if total > 0:
                acc_sum += correct / total
                valid_clients += 1

        if self.device == "cuda":
            torch.cuda.empty_cache()

        if valid_clients == 0:
            return None
        return acc_sum / valid_clients

    def evaluate_global_model(self):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        total_test_samples = sum(stats[1])
        total_train_samples = sum(stats_train[1])

        if total_test_samples > 0:
            local_test_acc = sum(stats[2]) * 1.0 / total_test_samples
        else:
            local_test_acc = 0.0

        if total_train_samples > 0:
            train_loss = sum(stats_train[2]) * 1.0 / total_train_samples
        else:
            train_loss = 0.0

        global_test_acc = self.evaluate_global_test_acc()
        split_metrics = self.evaluate_label_split_metrics()
        accs = [a / n for a, n in zip(stats[2], stats[1]) if n > 0]

        self.rs_test_acc.append(local_test_acc)
        self.rs_train_loss.append(train_loss)
        if global_test_acc is not None:
            self.rs_global_test_acc.append(global_test_acc)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Local Test Accuracy: {:.4f}".format(local_test_acc))
        if global_test_acc is not None:
            print("Averaged Global Test Accuracy: {:.4f}".format(global_test_acc))
        print("Std Test Accuracy: {:.4f}".format(float(np.std(accs)) if len(accs) > 0 else 0.0))
        if split_metrics is not None:
            id_text = self._format_optional_metric(split_metrics.get("id_test_acc")) or "N/A"
            ood_text = self._format_optional_metric(split_metrics.get("ood_test_acc")) or "N/A"
            print(f"Averaged Global ID/OOD Test Accuracy: {id_text}/{ood_text}")

        return {
            "local_test_acc": local_test_acc,
            "global_test_acc": global_test_acc,
            "train_loss": train_loss,
            "split_metrics": split_metrics,
        }


    def beta_aggregate_parameters(self):
        # aggregate avergage model with previous model using parameter beta
        for pre_param, param in zip(self.previous_global_model, self.global_model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data

    def test_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        eval_clients = self._evaluation_clients()
        for c in eval_clients:
            ct, ns = c.test_metrics_personalized()
            if not bool(getattr(self, "eval_common_global", True)) and ns > 0:
                tot_correct.append(ct * 1.0 / ns)
                num_samples.append(1)
            else:
                tot_correct.append(ct*1.0)
                num_samples.append(ns)
        ids = [c.id for c in eval_clients]

        return ids, num_samples, tot_correct

    def train_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        tot_correct = []
        losses = []
        eval_clients = self._evaluation_clients()
        for c in eval_clients:
            ct, cl, ns = c.train_metrics_personalized()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in eval_clients]

        return ids, num_samples, tot_correct, losses

    def evaluate_personalized(self):
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_acc = sum(stats_train[2])*1.0 / sum(stats_train[1])
        train_loss = sum(stats_train[3])*1.0 / sum(stats_train[1])
        global_test_acc = self.evaluate_personalized_global_test_acc()
        split_metrics = self.evaluate_label_split_metrics("test_label_split_metrics_personalized")
        accs = [a / n for a, n in zip(stats[2], stats[1]) if n > 0]
        
        self.rs_test_acc_per.append(test_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        if global_test_acc is not None:
            self.rs_global_test_acc_per.append(global_test_acc)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Local Test Accuracy: {:.4f}".format(test_acc))
        if global_test_acc is not None:
            print("Averaged Global Test Accuracy: {:.4f}".format(global_test_acc))
        print("Averaged Personalized Train Accuracy: {:.4f}".format(train_acc))
        print("Std Test Accuracy: {:.4f}".format(float(np.std(accs)) if len(accs) > 0 else 0.0))
        if split_metrics is not None:
            id_text = self._format_optional_metric(split_metrics.get("id_test_acc")) or "N/A"
            ood_text = self._format_optional_metric(split_metrics.get("ood_test_acc")) or "N/A"
            print(f"Averaged Personalized ID/OOD Test Accuracy: {id_text}/{ood_text}")

        return {
            "local_test_acc": test_acc,
            "global_test_acc": global_test_acc,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "split_metrics": split_metrics,
        }

    def _evaluate_personalized_local_metrics_on_dataset(self, dataset):
        original_dataset = self.dataset
        original_client_datasets = [client.dataset for client in self.clients]
        try:
            self.dataset = dataset
            for client in self.clients:
                client.dataset = dataset
            stats = self.test_metrics_personalized()
            total_test_samples = sum(stats[1])
            if total_test_samples <= 0:
                local_acc = 0.0
            else:
                local_acc = sum(stats[2]) * 1.0 / total_test_samples
            split_metrics = self.evaluate_label_split_metrics("test_label_split_metrics_personalized")
            return local_acc, split_metrics
        finally:
            self.dataset = original_dataset
            for client, client_dataset in zip(self.clients, original_client_datasets):
                client.dataset = client_dataset

    def _write_combined_usage_row(
        self,
        file_path,
        round_num,
        global_metrics,
        personalized_metrics,
        round_uplink,
        round_downlink,
    ):
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(
                    "round,global_local_test_acc,global_global_test_acc,global_train_loss,"
                    "global_id_test_acc,global_ood_test_acc,global_id_test_count,global_ood_test_count,"
                    "personalized_local_test_acc,personalized_global_test_acc,personalized_train_loss,"
                    "personalized_id_test_acc,personalized_ood_test_acc,"
                    "personalized_id_test_count,personalized_ood_test_count,"
                    "uplink_mb,downlink_mb,total_mb\n"
                )

        total_mb = round_uplink + round_downlink
        global_global_acc = (
            f"{global_metrics['global_test_acc']:.4f}"
            if global_metrics["global_test_acc"] is not None
            else ""
        )
        personal_global_acc = (
            f"{personalized_metrics['global_test_acc']:.4f}"
            if personalized_metrics["global_test_acc"] is not None
            else ""
        )
        global_split = global_metrics.get("split_metrics")
        personal_split = personalized_metrics.get("split_metrics")
        with open(file_path, "a") as f:
            f.write(
                f"{round_num},"
                f"{global_metrics['local_test_acc']:.4f},{global_global_acc},{global_metrics['train_loss']:.4f},"
                f"{self._format_optional_metric(global_split.get('id_test_acc') if global_split else None)},"
                f"{self._format_optional_metric(global_split.get('ood_test_acc') if global_split else None)},"
                f"{self._format_optional_count(global_split.get('id_test_count') if global_split else None)},"
                f"{self._format_optional_count(global_split.get('ood_test_count') if global_split else None)},"
                f"{personalized_metrics['local_test_acc']:.4f},{personal_global_acc},{personalized_metrics['train_loss']:.4f},"
                f"{self._format_optional_metric(personal_split.get('id_test_acc') if personal_split else None)},"
                f"{self._format_optional_metric(personal_split.get('ood_test_acc') if personal_split else None)},"
                f"{self._format_optional_count(personal_split.get('id_test_count') if personal_split else None)},"
                f"{self._format_optional_count(personal_split.get('ood_test_count') if personal_split else None)},"
                f"{round_uplink:.2f},{round_downlink:.2f},{total_mb:.2f}\n"
            )

    def log_multi_rho_eval_combined(self, global_metrics, personalized_metrics):
        if not self.multi_rho_eval:
            return
        round_num = len(self.rs_test_acc_per)
        round_uplink, round_downlink = self._last_eval_round_comm
        for item in self.eval_rho_items:
            for client in self.clients:
                client.update_parameters(client.model, client.local_params)
            global_acc, global_split = self._evaluate_local_metrics_on_dataset(item["dataset"])
            personal_acc, personal_split = self._evaluate_personalized_local_metrics_on_dataset(item["dataset"])
            print(
                f"[Multi-Rho Eval] rho={item['rho']:.1f} "
                f"Global/Personalized Local Test Accuracy: {global_acc:.4f}/{personal_acc:.4f}"
            )
            file_path = self.eval_rho_log_paths.get(item["label"])
            if file_path:
                self._write_combined_usage_row(
                    file_path,
                    round_num,
                    {
                        "local_test_acc": global_acc,
                        "global_test_acc": None,
                        "train_loss": global_metrics["train_loss"],
                        "split_metrics": global_split,
                    },
                    {
                        "local_test_acc": personal_acc,
                        "global_test_acc": None,
                        "train_loss": personalized_metrics["train_loss"],
                        "split_metrics": personal_split,
                    },
                    round_uplink,
                    round_downlink,
                )

    def log_usage_combined(self, global_metrics, personalized_metrics):
        file_path = getattr(self.args, "log_path", "usage.csv")
        round_num = len(self.rs_test_acc_per)
        round_uplink = max(0.0, self.uplink_MB - self._last_logged_uplink_MB)
        round_downlink = max(0.0, self.downlink_MB - self._last_logged_downlink_MB)
        self._last_logged_uplink_MB = self.uplink_MB
        self._last_logged_downlink_MB = self.downlink_MB
        self._last_eval_round_comm = (round_uplink, round_downlink)
        self._write_combined_usage_row(
            file_path,
            round_num,
            global_metrics,
            personalized_metrics,
            round_uplink,
            round_downlink,
        )
        self.log_multi_rho_eval_combined(global_metrics, personalized_metrics)
        self._maybe_log_dynamic_client_metrics()

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # if (len(self.rs_test_acc) & len(self.rs_train_acc) & len(self.rs_train_loss)):
        #     algo1 = algo + "_" + self.goal + "_" + str(self.times)
        #     with h5py.File(result_path + "{}.h5".format(algo1), 'w') as hf:
        #         hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
        #         hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
        #         hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

        if (len(self.rs_test_acc_per)):
            algo2 = algo + "_" + self.goal + "_" + str(self.times)
            with h5py.File(result_path + "{}.h5".format(algo2), 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
                if len(self.rs_global_test_acc_per) > 0:
                    hf.create_dataset('rs_global_test_acc', data=self.rs_global_test_acc_per)
                    hf.create_dataset('rs_common_test_acc', data=self.rs_global_test_acc_per)
