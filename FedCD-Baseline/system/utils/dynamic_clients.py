import csv
import json
import os

import numpy as np

from utils.data_utils import read_client_data


class DynamicClientExperimentMixin:
    _DYNAMIC_METRIC_KEYS = (
        "existing_id",
        "existing_ood",
        "newcomer_id",
        "newcomer_ood",
    )

    def _init_dynamic_client_experiment(self, args):
        self.dynamic_client_enabled = self._dynamic_bool(
            getattr(args, "dynamic_client_enabled", False)
        )
        self.dynamic_client_join_round = max(
            0, int(getattr(args, "dynamic_client_join_round", 51))
        )
        self.dynamic_client_old_classes = self._parse_dynamic_classes(
            getattr(args, "dynamic_client_old_classes", "0,1,2,3,4,5")
        )
        self.dynamic_client_new_classes = self._parse_dynamic_classes(
            getattr(args, "dynamic_client_new_classes", "6,7,8,9")
        )
        self.dynamic_client_expected_existing = max(
            0, int(getattr(args, "dynamic_client_expected_existing_clients", 30))
        )
        self.dynamic_client_expected_newcomers = max(
            0, int(getattr(args, "dynamic_client_expected_newcomer_clients", 20))
        )
        self.dynamic_client_require_contiguous_ids = self._dynamic_bool(
            getattr(args, "dynamic_client_require_contiguous_ids", True)
        )
        self.dynamic_client_round = -1
        self.dynamic_client_phase_changed = False
        self._dynamic_round_prepared = False
        self._dynamic_client_groups_initialized = False
        self.dynamic_phase1_client_ids = []
        self.dynamic_new_client_ids = []
        self.dynamic_client_train_class_map = {}
        self._eval_clients_override = None
        self._dynamic_logged_metric_keys = set()
        self._dynamic_comm_by_round = {}

    @staticmethod
    def _dynamic_bool(value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _parse_dynamic_classes(value):
        if isinstance(value, (list, tuple, set)):
            tokens = value
        else:
            tokens = str(value or "").replace(";", ",").split(",")
        result = []
        for token in tokens:
            token = str(token).strip()
            if token:
                class_idx = int(token)
                if class_idx not in result:
                    result.append(class_idx)
        return result

    @staticmethod
    def _dynamic_label_to_int(label):
        if hasattr(label, "detach"):
            return int(label.detach().cpu().item())
        if hasattr(label, "item"):
            return int(label.item())
        return int(label)

    def _dynamic_client_train_classes(self, client):
        if hasattr(client, "get_label_hist"):
            hist = np.asarray(client.get_label_hist()).reshape(-1)
            return {int(idx) for idx in np.flatnonzero(hist > 0)}
        train_data = read_client_data(
            self.dataset, int(client.id), is_train=True, few_shot=self.few_shot
        )
        return {self._dynamic_label_to_int(sample[1]) for sample in train_data}

    def _ensure_dynamic_client_groups(self):
        if not self.dynamic_client_enabled or self._dynamic_client_groups_initialized:
            return
        if not getattr(self, "clients", None):
            return
        old_classes = set(self.dynamic_client_old_classes)
        new_classes = set(self.dynamic_client_new_classes)
        if not old_classes or not new_classes or old_classes.intersection(new_classes):
            raise ValueError("Dynamic client old/new class sets must be non-empty and disjoint.")

        phase1_ids, new_ids, invalid = [], [], []
        train_class_map = {}
        for client in self.clients:
            client_id = int(client.id)
            train_classes = self._dynamic_client_train_classes(client)
            train_class_map[client_id] = sorted(train_classes)
            if train_classes and train_classes.issubset(old_classes):
                phase1_ids.append(client_id)
            elif train_classes and train_classes.issubset(new_classes):
                new_ids.append(client_id)
            else:
                invalid.append((client_id, sorted(train_classes)))
        if invalid:
            detail = ", ".join(f"client {cid}: {classes}" for cid, classes in invalid)
            raise ValueError("Each client must belong to exactly one dynamic phase; " + detail)
        if not phase1_ids or not new_ids:
            raise ValueError("Dynamic experiment requires both phase-1 and new-client groups.")

        phase1_ids = sorted(phase1_ids)
        new_ids = sorted(new_ids)
        if self.dynamic_client_expected_existing and len(phase1_ids) != self.dynamic_client_expected_existing:
            raise ValueError(
                "Dynamic experiment expected "
                f"{self.dynamic_client_expected_existing} existing clients, got {len(phase1_ids)}."
            )
        if self.dynamic_client_expected_newcomers and len(new_ids) != self.dynamic_client_expected_newcomers:
            raise ValueError(
                "Dynamic experiment expected "
                f"{self.dynamic_client_expected_newcomers} newcomers, got {len(new_ids)}."
            )
        if self.dynamic_client_require_contiguous_ids:
            expected_existing_ids = list(range(len(phase1_ids)))
            expected_new_ids = list(range(len(phase1_ids), len(phase1_ids) + len(new_ids)))
            if phase1_ids != expected_existing_ids or new_ids != expected_new_ids:
                raise ValueError(
                    "Dynamic experiment requires zero-based contiguous IDs: "
                    f"existing={expected_existing_ids}, newcomers={expected_new_ids}; "
                    f"found existing={phase1_ids}, newcomers={new_ids}."
                )

        self.dynamic_phase1_client_ids = phase1_ids
        self.dynamic_new_client_ids = new_ids
        self.dynamic_client_train_class_map = train_class_map
        self._dynamic_client_groups_initialized = True
        print(
            "[Dynamic Clients] "
            f"existing={len(phase1_ids)} classes={sorted(old_classes)}, "
            f"newcomers={len(new_ids)} classes={sorted(new_classes)}, "
            f"join_round={self.dynamic_client_join_round}"
        )
        self._write_dynamic_client_config()

    def _dynamic_client_config_dir(self):
        exp_dir = str(getattr(self.args, "exp_dir", "") or "").strip()
        if exp_dir:
            return exp_dir
        for name in ("log_usage_path", "log_path"):
            path = str(getattr(self.args, name, "") or "").strip()
            if path:
                return os.path.dirname(path)
        return ""

    def _write_dynamic_client_config(self):
        output_dir = self._dynamic_client_config_dir()
        if not output_dir:
            return
        os.makedirs(output_dir, exist_ok=True)
        payload = {
            "enabled": True,
            "dataset": str(self.dataset),
            "join_round": int(self.dynamic_client_join_round),
            "old_classes": list(self.dynamic_client_old_classes),
            "new_classes": list(self.dynamic_client_new_classes),
            "existing_client_ids": list(self.dynamic_phase1_client_ids),
            "newcomer_client_ids": list(self.dynamic_new_client_ids),
            "client_id_convention": "zero_based; paper clients 1-30 map to IDs 0-29",
            "client_train_classes": {
                str(client_id): list(classes)
                for client_id, classes in sorted(self.dynamic_client_train_class_map.items())
            },
            "accuracy_aggregation": "macro mean of per-client accuracies",
            "existing_id_definition": "each existing client's local training classes",
            "existing_ood_definition": (
                "all task classes except each existing client's local training classes"
            ),
            "newcomer_id_definition": "each newcomer's local training classes",
            "newcomer_ood_definition": (
                "all task classes except each newcomer's local training classes"
            ),
            "communication_unit": "MiB (column suffix retained as _mb for compatibility)",
        }
        path = os.path.join(output_dir, "dynamic_client_config.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)

    def set_dynamic_client_round(self, round_idx):
        if not self.dynamic_client_enabled:
            return
        self._ensure_dynamic_client_groups()
        previous_phase = self._dynamic_client_phase()
        self.dynamic_client_round = int(round_idx)
        self.dynamic_client_phase_changed = previous_phase != self._dynamic_client_phase()
        self._dynamic_round_prepared = True
        if self.dynamic_client_phase_changed:
            new_clients = self._dynamic_new_clients()
            self._on_dynamic_clients_activated(new_clients)
            print(
                f"[Dynamic Clients] Round {self.dynamic_client_round}: "
                f"activated {len(new_clients)} newcomers."
            )

    def _dynamic_new_clients(self):
        new_ids = set(self.dynamic_new_client_ids)
        return [client for client in self.clients if int(client.id) in new_ids]

    def _on_dynamic_clients_activated(self, new_clients):
        """Give newly activated clients the current shared model before cold-start evaluation."""
        global_model = getattr(self, "global_model", None)
        if global_model is None:
            return
        for client in new_clients:
            if hasattr(client, "set_parameters"):
                try:
                    client.set_parameters(global_model)
                except TypeError:
                    # Algorithms with specialized dispatch override this hook.
                    continue

    def _advance_dynamic_client_round(self):
        if not self.dynamic_client_enabled:
            return
        if self._dynamic_round_prepared:
            self._dynamic_round_prepared = False
            return
        self.set_dynamic_client_round(self.dynamic_client_round + 1)
        self._dynamic_round_prepared = False

    def _dynamic_client_phase(self, round_idx=None):
        round_idx = self.dynamic_client_round if round_idx is None else int(round_idx)
        return "phase2" if round_idx >= self.dynamic_client_join_round else "phase1"

    def _dynamic_client_ids_for_round(self, round_idx=None):
        self._ensure_dynamic_client_groups()
        if self._dynamic_client_phase(round_idx) == "phase1":
            return list(self.dynamic_phase1_client_ids)
        return list(self.dynamic_phase1_client_ids) + list(self.dynamic_new_client_ids)

    def _dynamic_client_active_clients(self, round_idx=None):
        if not self.dynamic_client_enabled:
            return list(self.clients)
        active_ids = set(self._dynamic_client_ids_for_round(round_idx))
        return [client for client in self.clients if int(client.id) in active_ids]

    def _evaluation_clients(self):
        if self._eval_clients_override is not None:
            return list(self._eval_clients_override)
        return self._dynamic_client_active_clients()

    def _dynamic_client_metric_method(self, client):
        if self.algorithm in {"pFedMe", "Ditto"} and hasattr(
            client, "test_metrics_personalized"
        ):
            return client.test_metrics_personalized
        return client.test_metrics

    @staticmethod
    def _empty_dynamic_metric():
        return {
            "accuracy": None,
            "std": None,
            "clients": 0,
            "correct": 0.0,
            "samples": 0,
        }

    def _dynamic_client_macro_subset_metric(self, client_ids, class_selector, dataset):
        client_ids = {int(client_id) for client_id in client_ids}
        accuracies = []
        correct = 0.0
        total = 0
        for client in self.clients:
            client_id = int(client.id)
            if client_id not in client_ids:
                continue
            classes = class_selector(client_id) if callable(class_selector) else class_selector
            classes = sorted({int(value) for value in classes})
            if not classes:
                continue
            old_dataset = client.dataset
            old_filter = getattr(client, "eval_class_filter", None)
            client.dataset = dataset
            client.set_eval_class_filter(classes)
            try:
                loader = client.load_test_data()
                if len(loader.dataset) <= 0:
                    continue
                result = self._dynamic_client_metric_method(client)()
                client_correct = float(result[0])
                client_total = int(result[1])
                if client_total <= 0:
                    continue
                accuracies.append(client_correct / client_total)
                correct += client_correct
                total += client_total
            finally:
                client.dataset = old_dataset
                client.set_eval_class_filter(old_filter)
        if not accuracies:
            return self._empty_dynamic_metric()
        return {
            "accuracy": float(np.mean(accuracies)),
            "std": float(np.std(accuracies)),
            "clients": len(accuracies),
            "correct": correct,
            "samples": total,
        }

    @staticmethod
    def _format_dynamic_float(value):
        return "" if value is None or not np.isfinite(value) else f"{float(value):.8f}"

    def _dynamic_metric_dataset_items(self):
        if bool(getattr(self, "multi_rho_eval", False)):
            items = []
            for item in getattr(self, "eval_rho_items", []):
                acc_path = getattr(self, "eval_rho_log_paths", {}).get(item["label"])
                output_dir = (
                    os.path.dirname(acc_path)
                    if acc_path
                    else self._dynamic_client_config_dir()
                )
                items.append((item["label"], item["dataset"], output_dir))
            return items
        return [("dynamic_test", self.dataset, self._dynamic_client_config_dir())]

    def _dynamic_client_comm_snapshot(self, round_idx):
        round_idx = int(round_idx)
        if round_idx not in self._dynamic_comm_by_round:
            if hasattr(self, "_fedccmv22_v2_comm_values"):
                values = list(self._fedccmv22_v2_comm_values(round_idx))
                uplink_mb = float(values[-3])
                downlink_mb = float(values[-2])
            else:
                uplink_mb, downlink_mb = getattr(self, "_last_eval_round_comm", (0.0, 0.0))
                uplink_mb = float(uplink_mb)
                downlink_mb = float(downlink_mb)
            self._dynamic_comm_by_round[round_idx] = (uplink_mb, downlink_mb)

        uplink_mb, downlink_mb = self._dynamic_comm_by_round[round_idx]
        ordered = [self._dynamic_comm_by_round[key] for key in sorted(self._dynamic_comm_by_round)]
        cumulative_uplink = float(sum(item[0] for item in ordered))
        cumulative_downlink = float(sum(item[1] for item in ordered))
        logged_rounds = len(ordered)
        return {
            "round_uplink_mb": uplink_mb,
            "round_downlink_mb": downlink_mb,
            "round_total_mb": uplink_mb + downlink_mb,
            "cumulative_uplink_mb": cumulative_uplink,
            "cumulative_downlink_mb": cumulative_downlink,
            "cumulative_total_mb": cumulative_uplink + cumulative_downlink,
            "mean_total_mb_per_logged_round": (
                (cumulative_uplink + cumulative_downlink) / logged_rounds
                if logged_rounds > 0 else 0.0
            ),
        }

    def _dynamic_metric_summaries(self, dataset):
        existing_ids = list(self.dynamic_phase1_client_ids)
        newcomer_ids = (
            list(self.dynamic_new_client_ids)
            if self._dynamic_client_phase() == "phase2" else []
        )
        id_selector = lambda client_id: self.dynamic_client_train_class_map.get(client_id, [])
        all_classes = set(self.dynamic_client_old_classes).union(
            self.dynamic_client_new_classes
        )
        ood_selector = lambda client_id: all_classes.difference(
            self.dynamic_client_train_class_map.get(client_id, [])
        )
        summaries = {
            "existing_id": self._dynamic_client_macro_subset_metric(
                existing_ids, id_selector, dataset
            ),
            "existing_ood": self._dynamic_client_macro_subset_metric(
                existing_ids, ood_selector, dataset
            ),
            "newcomer_id": self._empty_dynamic_metric(),
            "newcomer_ood": self._empty_dynamic_metric(),
        }
        if newcomer_ids:
            summaries["newcomer_id"] = self._dynamic_client_macro_subset_metric(
                newcomer_ids, id_selector, dataset
            )
            summaries["newcomer_ood"] = self._dynamic_client_macro_subset_metric(
                newcomer_ids, ood_selector, dataset
            )
        return summaries

    def _build_dynamic_metric_row(self, recorded_round, dataset_label, dataset, summaries):
        newcomer_count = (
            len(self.dynamic_new_client_ids)
            if self._dynamic_client_phase(recorded_round) == "phase2" else 0
        )
        row = [
            int(recorded_round),
            self._dynamic_client_phase(recorded_round),
            dataset_label,
            str(dataset),
            len(self.dynamic_phase1_client_ids),
            newcomer_count,
            len(self.dynamic_phase1_client_ids) + newcomer_count,
        ]
        row.extend(
            self._format_dynamic_float(summaries[key]["accuracy"])
            for key in self._DYNAMIC_METRIC_KEYS
        )
        row.extend(
            self._format_dynamic_float(summaries[key]["std"])
            for key in self._DYNAMIC_METRIC_KEYS
        )
        row.extend(int(summaries[key]["clients"]) for key in self._DYNAMIC_METRIC_KEYS)
        for key in self._DYNAMIC_METRIC_KEYS:
            row.extend([
                f"{float(summaries[key]['correct']):.0f}",
                int(summaries[key]["samples"]),
            ])
        comm = self._dynamic_client_comm_snapshot(recorded_round)
        row.extend(f"{comm[key]:.8f}" for key in (
            "round_uplink_mb",
            "round_downlink_mb",
            "round_total_mb",
            "cumulative_uplink_mb",
            "cumulative_downlink_mb",
            "cumulative_total_mb",
            "mean_total_mb_per_logged_round",
        ))
        return row

    @staticmethod
    def _print_dynamic_metric_row(row):
        display = lambda value: value if value != "" else "N/A"
        print(
            f"[Dynamic Clients][{row[2]}][round={row[0]}][{row[1]}] "
            f"existing ID/OOD={display(row[7])}/{display(row[8])} | "
            f"newcomer ID/OOD={display(row[9])}/{display(row[10])} | "
            f"comm={float(row[-5]):.4f} MiB/round, cumulative={float(row[-2]):.4f} MiB"
        )

    @classmethod
    def _dynamic_metric_header(cls):
        header = [
            "round",
            "phase",
            "dataset_label",
            "dataset",
            "existing_client_count",
            "newcomer_client_count",
            "active_client_count",
        ]
        header.extend(f"{key}_acc" for key in cls._DYNAMIC_METRIC_KEYS)
        header.extend(f"{key}_client_std" for key in cls._DYNAMIC_METRIC_KEYS)
        header.extend(f"{key}_evaluated_clients" for key in cls._DYNAMIC_METRIC_KEYS)
        for key in cls._DYNAMIC_METRIC_KEYS:
            header.extend([f"{key}_correct", f"{key}_samples"])
        header.extend([
            "round_uplink_mb",
            "round_downlink_mb",
            "round_total_mb",
            "cumulative_uplink_mb",
            "cumulative_downlink_mb",
            "cumulative_total_mb",
            "mean_total_mb_per_logged_round",
        ])
        return header

    def _maybe_log_dynamic_client_metrics(self):
        if not self.dynamic_client_enabled or self.dynamic_client_round < 0:
            return
        self._ensure_dynamic_client_groups()
        for dataset_label, dataset, output_dir in self._dynamic_metric_dataset_items():
            if not output_dir:
                continue
            key = (int(self.dynamic_client_round), str(dataset_label))
            if key in self._dynamic_logged_metric_keys:
                continue
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "dynamic_client_metrics.csv")
            file_exists = os.path.exists(path)
            summaries = self._dynamic_metric_summaries(dataset)
            row = self._build_dynamic_metric_row(
                self.dynamic_client_round,
                dataset_label,
                dataset,
                summaries,
            )
            with open(path, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(self._dynamic_metric_header())
                writer.writerow(row)
            self._print_dynamic_metric_row(row)
            self._dynamic_logged_metric_keys.add(key)
