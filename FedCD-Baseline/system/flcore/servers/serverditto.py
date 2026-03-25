import numpy as np
import os
import time
import torch
from flcore.clients.clientditto import clientDitto
from flcore.servers.serverbase import Server
from threading import Thread


class Ditto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientDitto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.rs_test_acc_per = []
        self.rs_train_loss_per = []
        self.rs_global_test_acc_per = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global models")
                global_metrics = self.evaluate_global_models()

            if i%self.eval_gap == 0:
                print("\nEvaluate personalized models")
                personalized_metrics = self.evaluate_personalized()
                self.log_usage_combined(global_metrics, personalized_metrics)

            for client in self.selected_clients:
                client.ptrain()
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest global-model accuracy.")
        print(max(self.rs_test_acc))
        if len(self.rs_test_acc_per) > 0:
            print("\nBest personalized accuracy.")
            print(max(self.rs_test_acc_per))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientDitto)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def evaluate_global_models(self):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        total_test_samples = sum(stats[1])
        total_train_samples = sum(stats_train[1])

        if total_test_samples > 0:
            local_test_acc = sum(stats[2]) * 1.0 / total_test_samples
            test_auc = sum(stats[3]) * 1.0 / total_test_samples
        else:
            local_test_acc = 0.0
            test_auc = 0.0

        if total_train_samples > 0:
            train_loss = sum(stats_train[2]) * 1.0 / total_train_samples
        else:
            train_loss = 0.0
        global_test_acc = self.evaluate_global_test_acc()

        accs = [a / n for a, n in zip(stats[2], stats[1]) if n > 0]
        aucs = [a / n for a, n in zip(stats[3], stats[1]) if n > 0]
        std_acc = float(np.std(accs)) if len(accs) > 0 else 0.0
        std_auc = float(np.std(aucs)) if len(aucs) > 0 else 0.0

        self.rs_test_acc.append(local_test_acc)
        self.rs_train_loss.append(train_loss)
        if global_test_acc is not None:
            self.rs_global_test_acc.append(global_test_acc)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Local Test Accuracy: {:.4f}".format(local_test_acc))
        if global_test_acc is not None:
            print("Averaged Global Test Accuracy: {:.4f}".format(global_test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accuracy: {:.4f}".format(std_acc))
        print("Std Test AUC: {:.4f}".format(std_auc))

        return {
            "local_test_acc": local_test_acc,
            "global_test_acc": global_test_acc,
            "train_loss": train_loss,
            "test_auc": test_auc,
        }

    def evaluate_personalized_global_test_acc(self):
        if not self.eval_common_global or self.global_test_loader is None:
            return None

        acc_sum = 0.0
        valid_clients = 0

        for client in self.clients:
            client.model_per.to(self.device)
            client.model_per.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in self.global_test_loader:
                    if type(x) == type([]):
                        x = x[0]
                    x = x.to(self.device)
                    y = y.to(self.device)
                    output = client.model_per(x)
                    correct += (torch.argmax(output, dim=1) == y).sum().item()
                    total += y.size(0)

            client.model_per.cpu()
            if total > 0:
                acc_sum += correct / total
                valid_clients += 1

        if self.device == "cuda":
            torch.cuda.empty_cache()

        if valid_clients == 0:
            return None
        return acc_sum / valid_clients


    def test_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics_personalized()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics_personalized()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate_personalized(self, acc=None, loss=None):
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        global_test_acc = self.evaluate_personalized_global_test_acc()
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc_per.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss_per.append(train_loss)
        else:
            loss.append(train_loss)
        if global_test_acc is not None:
            self.rs_global_test_acc_per.append(global_test_acc)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Local Test Accuracy: {:.4f}".format(test_acc))
        if global_test_acc is not None:
            print("Averaged Global Test Accuracy: {:.4f}".format(global_test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        return {
            "local_test_acc": test_acc,
            "global_test_acc": global_test_acc,
            "train_loss": train_loss,
            "test_auc": test_auc,
        }

    def log_usage_combined(self, global_metrics, personalized_metrics):
        file_path = getattr(self.args, "log_path", "usage.csv")
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(
                    "round,global_local_test_acc,global_global_test_acc,global_train_loss,"
                    "personalized_local_test_acc,personalized_global_test_acc,personalized_train_loss,"
                    "uplink_mb,downlink_mb,total_mb\n"
                )

        round_num = len(self.rs_test_acc)
        round_uplink = max(0.0, self.uplink_MB - self._last_logged_uplink_MB)
        round_downlink = max(0.0, self.downlink_MB - self._last_logged_downlink_MB)
        total_mb = round_uplink + round_downlink
        self._last_logged_uplink_MB = self.uplink_MB
        self._last_logged_downlink_MB = self.downlink_MB

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

        with open(file_path, "a") as f:
            f.write(
                f"{round_num},"
                f"{global_metrics['local_test_acc']:.4f},{global_global_acc},{global_metrics['train_loss']:.4f},"
                f"{personalized_metrics['local_test_acc']:.4f},{personal_global_acc},{personalized_metrics['train_loss']:.4f},"
                f"{round_uplink:.2f},{round_downlink:.2f},{total_mb:.2f}\n"
            )
