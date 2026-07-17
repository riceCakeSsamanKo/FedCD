import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class TinyFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128, num_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projector = nn.Linear(128, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def features(self, x):
        z = self.encoder(x).flatten(1)
        z = self.projector(z)
        return F.normalize(z, dim=1)

    def forward(self, x):
        return self.classifier(self.features(x))


class LARClassifier(nn.Module):
    def __init__(self, feature_dim=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    if "data" in data.files:
        payload = data["data"].item()
        return payload["x"].astype(np.float32), payload["y"].astype(np.int64)
    return data["x"].astype(np.float32), data["y"].astype(np.int64)


def load_split(dataset_dir, split):
    xs, ys = [], []
    for path in sorted((dataset_dir / split).glob("*.npz"), key=lambda p: int(p.stem)):
        x, y = load_npz(path)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def stratified_indices(y, fraction, seed, num_classes=10):
    rng = np.random.default_rng(seed)
    selected = []
    for cls in range(num_classes):
        idx = np.flatnonzero(y == cls)
        if len(idx) == 0:
            continue
        take = max(1, int(round(len(idx) * fraction)))
        selected.extend(rng.choice(idx, size=min(take, len(idx)), replace=False).tolist())
    selected = np.array(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * y.numel()
        total_correct += int((logits.argmax(1) == y).sum().item())
        total += y.numel()
    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def eval_classifier(model, loader, device):
    model.eval()
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        total_correct += int((logits.argmax(1) == y).sum().item())
        total += y.numel()
    return total_correct / max(total, 1)


@torch.no_grad()
def extract_features(model, x, batch_size, device):
    model.eval()
    feats = []
    tensor_x = torch.from_numpy(x)
    for start in range(0, len(tensor_x), batch_size):
        batch = tensor_x[start : start + batch_size].to(device)
        feats.append(model.features(batch).cpu())
    return torch.cat(feats, dim=0).numpy().astype(np.float32)


def split_lar_train_val(x, y, val_ratio, seed):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in range(10):
        idx = np.flatnonzero(y == cls)
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_ratio))
        if len(idx) >= 5:
            n_val = max(1, n_val)
        n_val = min(n_val, max(len(idx) - 1, 0))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def train_lar(features, labels, args, device):
    train_idx, val_idx = split_lar_train_val(features, labels, args.lar_val_ratio, args.seed)
    lar = LARClassifier(feature_dim=features.shape[1], num_classes=10).to(device)
    optimizer = torch.optim.AdamW(lar.parameters(), lr=args.lar_lr, weight_decay=args.lar_weight_decay)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(features[train_idx]), torch.from_numpy(labels[train_idx])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = None
    if len(val_idx) > 0:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(features[val_idx]), torch.from_numpy(labels[val_idx])),
            batch_size=args.batch_size,
            shuffle=False,
        )

    history = []
    for epoch in range(1, args.lar_epochs + 1):
        loss, acc = train_epoch(lar, train_loader, optimizer, device)
        val_acc = eval_classifier(lar, val_loader, device) if val_loader is not None else 0.0
        history.append({"epoch": epoch, "loss": loss, "train_acc": acc, "val_acc": val_acc})
    return lar, train_idx, val_idx, history


@torch.no_grad()
def lar_predict(lar, features, batch_size, device):
    lar.eval()
    logits = []
    tensor = torch.from_numpy(features)
    for start in range(0, len(tensor), batch_size):
        batch = tensor[start : start + batch_size].to(device)
        logits.append(lar(batch).cpu())
    logits = torch.cat(logits, dim=0)
    probs = torch.softmax(logits, dim=1).numpy()
    pred = probs.argmax(axis=1)
    max_conf = probs.max(axis=1)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    return probs, pred, max_conf, entropy


def prototype_scores(train_features, train_labels, probe_features):
    prototypes = []
    proto_classes = []
    scales = []
    for cls in range(10):
        cls_feats = train_features[train_labels == cls]
        if len(cls_feats) == 0:
            continue
        proto = cls_feats.mean(axis=0)
        distances = np.linalg.norm(cls_feats - proto[None, :], axis=1)
        scale = float(np.percentile(distances, 90)) if len(distances) > 1 else float(distances.mean() + 1e-3)
        prototypes.append(proto)
        proto_classes.append(cls)
        scales.append(max(scale, 1e-3))
    if not prototypes:
        return np.zeros(len(probe_features), dtype=np.float32), np.full(len(probe_features), -1, dtype=np.int64)
    prototypes = np.stack(prototypes, axis=0)
    scales = np.array(scales, dtype=np.float32)
    dist = np.linalg.norm(probe_features[:, None, :] - prototypes[None, :, :], axis=2)
    score = np.exp(-dist / scales[None, :])
    best = score.argmax(axis=1)
    return score.max(axis=1), np.array(proto_classes, dtype=np.int64)[best]


def rankdata(values):
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def bin_name(count, high_seen_threshold, low_seen_threshold):
    if count >= high_seen_threshold:
        return "high_seen"
    if count >= low_seen_threshold:
        return "low_seen"
    if count > 0:
        return "rare_seen"
    return "zero_seen"


def summarize_by_class(labels, pred, max_conf, true_prob, entropy, proto_score, proto_pred, counts, ood_threshold, args):
    rows = []
    for cls in range(10):
        mask = labels == cls
        if not np.any(mask):
            continue
        count = int(counts[cls])
        rows.append(
            {
                "class": cls,
                "local_train_count": count,
                "exposure_bin": bin_name(count, args.high_seen_threshold, args.low_seen_threshold),
                "probe_samples": int(mask.sum()),
                "lar_acc": float((pred[mask] == labels[mask]).mean()),
                "lar_max_conf_mean": float(max_conf[mask].mean()),
                "lar_max_conf_std": float(max_conf[mask].std()),
                "lar_true_prob_mean": float(true_prob[mask].mean()),
                "lar_entropy_mean": float(entropy[mask].mean()),
                "lar_ood_rate": float((max_conf[mask] < ood_threshold).mean()),
                "prototype_score_mean": float(proto_score[mask].mean()),
                "prototype_acc": float((proto_pred[mask] == labels[mask]).mean()),
            }
        )
    return rows


def summarize_by_bin(class_rows):
    out = []
    for name in ["high_seen", "low_seen", "rare_seen", "zero_seen"]:
        rows = [row for row in class_rows if row["exposure_bin"] == name]
        if not rows:
            continue
        total = sum(row["probe_samples"] for row in rows)
        weighted = {}
        for key in [
            "lar_acc",
            "lar_max_conf_mean",
            "lar_true_prob_mean",
            "lar_entropy_mean",
            "lar_ood_rate",
            "prototype_score_mean",
            "prototype_acc",
        ]:
            weighted[key] = sum(row[key] * row["probe_samples"] for row in rows) / max(total, 1)
        out.append({"exposure_bin": name, "classes": [row["class"] for row in rows], "probe_samples": total, **weighted})
    return out


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Toy check for LAR overconfidence in CIFAR-10 Dirichlet setting.")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Users\mulso\Documents\GitHub\fl_data"))
    parser.add_argument("--dataset", type=str, default="Cifar10_dir0.1_nc20")
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--fext-fraction", type=float, default=0.10)
    parser.add_argument("--fext-epochs", type=int, default=3)
    parser.add_argument("--lar-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fext-lr", type=float, default=1e-3)
    parser.add_argument("--lar-lr", type=float, default=1e-3)
    parser.add_argument("--lar-weight-decay", type=float, default=1e-4)
    parser.add_argument("--lar-val-ratio", type=float, default=0.2)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/toy_lar_dir_overconfidence"))
    parser.add_argument("--high-seen-threshold", type=int, default=200)
    parser.add_argument("--low-seen-threshold", type=int, default=30)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    device = torch.device(device)

    dataset_dir = args.data_root / args.dataset
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Missing train/test split under {dataset_dir}")

    x_all_train, y_all_train = load_split(dataset_dir, "train")
    x_all_test, y_all_test = load_split(dataset_dir, "test")
    x_client, y_client = load_npz(train_dir / f"{args.client_id}.npz")
    client_counts = np.bincount(y_client, minlength=10)

    fext_idx = stratified_indices(y_all_train, args.fext_fraction, args.seed, num_classes=10)
    fext_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_all_train[fext_idx]), torch.from_numpy(y_all_train[fext_idx])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_eval_idx = stratified_indices(y_all_test, min(1.0, 2000 / max(len(y_all_test), 1)), args.seed + 1, num_classes=10)
    fext_eval_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_all_test[test_eval_idx]), torch.from_numpy(y_all_test[test_eval_idx])),
        batch_size=args.batch_size,
        shuffle=False,
    )

    fext = TinyFeatureExtractor(feature_dim=args.feature_dim, num_classes=10).to(device)
    optimizer = torch.optim.AdamW(fext.parameters(), lr=args.fext_lr, weight_decay=1e-4)
    fext_history = []
    for epoch in range(1, args.fext_epochs + 1):
        loss, train_acc = train_epoch(fext, fext_loader, optimizer, device)
        eval_acc = eval_classifier(fext, fext_eval_loader, device)
        fext_history.append({"epoch": epoch, "loss": loss, "train_acc": train_acc, "probe_acc": eval_acc})
        print(f"[fext] epoch={epoch} loss={loss:.4f} train_acc={train_acc:.4f} probe_acc={eval_acc:.4f}")

    for param in fext.parameters():
        param.requires_grad = False

    client_features = extract_features(fext, x_client, args.batch_size, device)
    test_features = extract_features(fext, x_all_test, args.batch_size, device)

    lar, lar_train_idx, lar_val_idx, lar_history = train_lar(client_features, y_client, args, device)
    print(
        f"[lar] final train_acc={lar_history[-1]['train_acc']:.4f} "
        f"val_acc={lar_history[-1]['val_acc']:.4f} train_samples={len(lar_train_idx)} val_samples={len(lar_val_idx)}"
    )

    train_probs, _, train_conf, _ = lar_predict(lar, client_features[lar_train_idx], args.batch_size, device)
    if len(lar_val_idx) > 0:
        _, _, val_conf, _ = lar_predict(lar, client_features[lar_val_idx], args.batch_size, device)
        threshold_source = val_conf
    else:
        threshold_source = train_conf
    ood_threshold = float(np.percentile(threshold_source, 5))

    probs, pred, max_conf, entropy = lar_predict(lar, test_features, args.batch_size, device)
    true_prob = probs[np.arange(len(y_all_test)), y_all_test]
    proto_score, proto_pred = prototype_scores(client_features[lar_train_idx], y_client[lar_train_idx], test_features)

    class_rows = summarize_by_class(
        y_all_test,
        pred,
        max_conf,
        true_prob,
        entropy,
        proto_score,
        proto_pred,
        client_counts,
        ood_threshold,
        args,
    )
    bin_rows = summarize_by_bin(class_rows)
    corr = spearman(
        [row["local_train_count"] for row in class_rows],
        [row["lar_max_conf_mean"] for row in class_rows],
    )
    true_corr = spearman(
        [row["local_train_count"] for row in class_rows],
        [row["lar_true_prob_mean"] for row in class_rows],
    )

    run_name = f"{args.dataset}_client{args.client_id}_seed{args.seed}"
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "class_summary.csv", class_rows)
    write_csv(out_dir / "bin_summary.csv", bin_rows)

    summary = {
        "dataset": args.dataset,
        "client_id": args.client_id,
        "device": str(device),
        "seed": args.seed,
        "fext_fraction": args.fext_fraction,
        "fext_train_samples": int(len(fext_idx)),
        "client_train_samples": int(len(y_client)),
        "client_class_counts": client_counts.astype(int).tolist(),
        "ood_threshold_max_softmax_p05": ood_threshold,
        "spearman_count_vs_lar_max_conf": corr,
        "spearman_count_vs_lar_true_prob": true_corr,
        "fext_history": fext_history,
        "lar_history_tail": lar_history[-5:],
        "class_summary_csv": str(out_dir / "class_summary.csv"),
        "bin_summary_csv": str(out_dir / "bin_summary.csv"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nClient class counts:", client_counts.astype(int).tolist())
    print(f"LAR OOD threshold (5th percentile ID confidence): {ood_threshold:.4f}")
    print(f"Spearman count vs max confidence: {corr:.4f}")
    print(f"Spearman count vs true-class probability: {true_corr:.4f}")
    print("\nBin summary:")
    for row in bin_rows:
        print(
            f"  {row['exposure_bin']:>9} classes={row['classes']} "
            f"max_conf={row['lar_max_conf_mean']:.4f} true_prob={row['lar_true_prob_mean']:.4f} "
            f"ood_rate={row['lar_ood_rate']:.4f} acc={row['lar_acc']:.4f} "
            f"proto_score={row['prototype_score_mean']:.4f}"
        )
    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
