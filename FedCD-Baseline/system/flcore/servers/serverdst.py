import copy
import math
import random
import time

import torch
from tqdm import tqdm

from flcore.clients.clientdst import clientDST
from flcore.servers.serverbase import Server
from flcore.servers.feddst_utils import (
    apply_masks,
    bits_to_mb,
    clone_masks,
    dense_parameter_bits,
    make_initial_masks,
    masks_equal,
    sparse_payload_bits,
)


class FedDST(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.target_sparsity = float(getattr(args, "feddst_sparsity", 0.3))
        final_sparsity_arg = getattr(args, "feddst_final_sparsity", None)
        self.final_sparsity = self.target_sparsity if final_sparsity_arg is None else float(final_sparsity_arg)
        self.readjustment_ratio = float(getattr(args, "feddst_readjustment_ratio", 0.5))
        self.rounds_between_readjustments = int(getattr(args, "feddst_rounds_between_readjustments", 10))
        self.rate_decay_method = str(getattr(args, "feddst_rate_decay_method", "cosine")).lower()
        self.rate_decay_end = int(getattr(args, "feddst_rate_decay_end", max(1, self.global_rounds // 2)))
        self.sparsity_distribution = str(getattr(args, "feddst_sparsity_distribution", "erk"))
        self.min_votes = int(getattr(args, "feddst_min_votes", 0))
        self.fp16 = bool(getattr(args, "feddst_fp16", False))
        self.remember_old = bool(getattr(args, "feddst_remember_old", False))

        self.masks = make_initial_masks(self.global_model, self.target_sparsity, self.sparsity_distribution)
        apply_masks(self.global_model, self.masks)
        self.model_size_MB = bits_to_mb(dense_parameter_bits(self.global_model, fp16=False))

        self.set_slow_clients()
        self.set_clients(clientDST)

        self.uploaded_masks = []
        self.uploaded_states = []
        self.Budget = []
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(
            "FedDST config: "
            f"sparsity={self.target_sparsity}, readjustment_ratio={self.readjustment_ratio}, "
            f"readjustment_interval={self.rounds_between_readjustments}, distribution={self.sparsity_distribution}"
        )
        print("Finished creating FedDST server and clients.")

    def _readjustment_ratio_for_round(self, server_round):
        ratio = self.readjustment_ratio
        if self.rate_decay_method == "cosine":
            end = max(1, self.rate_decay_end)
            if server_round >= end:
                return 0.0
            ratio *= 0.5 * (1.0 + math.cos(math.pi * server_round / end))
        return max(0.0, ratio)

    def _on_dynamic_clients_activated(self, new_clients):
        for client in new_clients:
            client.set_sparse_parameters(self.global_model, self.masks)

    def _sparsity_for_round(self, server_round):
        end = max(1, self.rate_decay_end)
        if server_round <= end:
            alpha = server_round / end
            return (1.0 - alpha) * self.target_sparsity + alpha * self.final_sparsity
        return self.final_sparsity

    def _send_sparse_models(self):
        clients = (
            self.selected_clients
            if len(self.selected_clients) > 0
            else self._dynamic_client_active_clients()
        )
        for client in tqdm(clients, desc="Distributing sparse models", leave=False):
            start_time = time.time()
            mask_changed = not masks_equal(getattr(client, "_last_received_masks", None), self.masks)
            client.set_sparse_parameters(self.global_model, self.masks)
            payload_bits = sparse_payload_bits(
                self.global_model,
                self.masks,
                include_mask_bits=mask_changed,
                fp16=False,
            )
            self.downlink_MB += bits_to_mb(payload_bits)
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

    def _receive_sparse_models(self, include_mask_bits=False):
        assert len(self.selected_clients) > 0
        keep = int((1 - self.client_drop_rate) * self.current_num_join_clients)
        active_clients = random.sample(self.selected_clients, keep)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_states = []
        self.uploaded_masks = []
        total_samples = 0
        for client in tqdm(active_clients, desc="Collecting sparse models", leave=False):
            try:
                client_time_cost = (
                    client.train_time_cost["total_cost"] / client.train_time_cost["num_rounds"]
                    + client.send_time_cost["total_cost"] / client.send_time_cost["num_rounds"]
                )
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost > self.time_threthold:
                continue
            total_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_states.append({
                name: value.detach().cpu().clone()
                for name, value in client.model.state_dict().items()
            })
            self.uploaded_masks.append(clone_masks(client.masks))
            self.uplink_MB += bits_to_mb(
                sparse_payload_bits(client.model, client.masks, include_mask_bits=include_mask_bits, fp16=self.fp16)
            )

        if total_samples == 0:
            return
        self.uploaded_weights = [weight / total_samples for weight in self.uploaded_weights]

    def _aggregate_sparse_parameters(self):
        if not self.uploaded_states:
            return

        old_state = self.global_model.state_dict()
        new_state = copy.deepcopy(old_state)
        new_masks = clone_masks(self.masks)
        train_sample_weights = {
            client_id: self.clients[client_id].train_samples
            for client_id in self.uploaded_ids
            if 0 <= client_id < len(self.clients)
        }
        total_samples = sum(train_sample_weights.values()) or 1

        for name, param in self.global_model.named_parameters():
            if name in self.masks:
                numerator = torch.zeros_like(param.detach().cpu(), dtype=torch.float32)
                denom = torch.zeros_like(param.detach().cpu(), dtype=torch.float32)
                vote_count = torch.zeros_like(param.detach().cpu(), dtype=torch.float32)
                for client_id, state, masks in zip(self.uploaded_ids, self.uploaded_states, self.uploaded_masks):
                    weight = float(train_sample_weights.get(client_id, 0))
                    if weight <= 0:
                        continue
                    mask = masks[name].float().cpu()
                    value = state[name].float().cpu()
                    numerator += weight * value * mask
                    denom += weight * mask
                    vote_count += mask
                keep = vote_count > float(self.min_votes)
                averaged = torch.zeros_like(numerator)
                valid = denom > 0
                averaged[valid] = numerator[valid] / denom[valid]
                if self.remember_old:
                    averaged[~valid] = old_state[name].detach().cpu().float()[~valid]
                averaged[~keep] = 0.0
                new_state[name] = averaged.to(dtype=old_state[name].dtype)
                new_masks[name] = keep.bool().cpu()
            else:
                avg = torch.zeros_like(param.detach().cpu(), dtype=torch.float32)
                for client_id, state in zip(self.uploaded_ids, self.uploaded_states):
                    weight = float(train_sample_weights.get(client_id, 0)) / total_samples
                    avg += weight * state[name].float().cpu()
                new_state[name] = avg.to(dtype=old_state[name].dtype)

        parameter_names = set(dict(self.global_model.named_parameters()))
        for name, reference in old_state.items():
            if name in parameter_names:
                continue
            if torch.is_floating_point(reference) or torch.is_complex(reference):
                averaged = torch.zeros_like(reference.detach().cpu())
            else:
                averaged = torch.zeros_like(reference.detach().cpu(), dtype=torch.float64)
            for client_id, state in zip(self.uploaded_ids, self.uploaded_states):
                weight = float(train_sample_weights.get(client_id, 0)) / total_samples
                value = state[name].detach().cpu().to(dtype=averaged.dtype)
                averaged.add_(value, alpha=weight)
            if not (torch.is_floating_point(reference) or torch.is_complex(reference)):
                averaged = averaged.round().to(dtype=reference.dtype)
            new_state[name] = averaged

        self.global_model.load_state_dict(new_state, strict=True)
        self.masks = new_masks
        apply_masks(self.global_model, self.masks)

    def train(self):
        for i in tqdm(range(self.global_rounds + 1), desc="Global Rounds"):
            start_time = time.time()
            self.selected_clients = self.select_clients()
            self._send_sparse_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate sparse global model")
                self.evaluate()

            round_sparsity = self._sparsity_for_round(i)
            ratio = self._readjustment_ratio_for_round(i)
            readjust = (
                i > 0
                and self.rounds_between_readjustments > 0
                and i % self.rounds_between_readjustments == 0
                and ratio > 0
            )
            if readjust:
                print(f"[FedDST] Round {i}: readjust sparse masks with ratio={ratio:.6f}")

            for client in tqdm(self.selected_clients, desc="Training FedDST clients", leave=False):
                client.train(readjust=readjust, readjustment_ratio=ratio, target_sparsity=round_sparsity)

            self._receive_sparse_models(include_mask_bits=readjust)
            self._aggregate_sparse_parameters()

            self.Budget.append(time.time() - start_time)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc) if self.rs_test_acc else 0.0)
        print("\nAverage time cost per round.")
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        else:
            print(0.0)
        self.save_results()
        self.save_global_model()


