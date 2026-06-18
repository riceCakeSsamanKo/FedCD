#!/usr/bin/env python
import argparse
import csv
from pathlib import Path

import numpy as np


def _npz_files(path_text):
    path = Path(path_text)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Missing dump path: {path}")
    files = sorted(path.glob("client_*_router_tsne.npz"))
    if not files:
        files = sorted(path.rglob("*_router_tsne.npz"))
    if not files:
        raise FileNotFoundError(f"No router t-SNE npz files under: {path}")
    return files


def _read_array(data, key, n, fill=np.nan, dtype=np.float32):
    if key in data:
        arr = np.asarray(data[key])
        if arr.ndim == 0:
            arr = np.full(n, arr.item())
        return arr.reshape(-1)[:n]
    return np.full(n, fill, dtype=dtype)


def load_condition(path_text, condition, feature_key):
    features = []
    rows = []
    for path in _npz_files(path_text):
        with np.load(path, allow_pickle=True) as data:
            if feature_key not in data:
                raise KeyError(f"{path} does not contain feature key '{feature_key}'")
            feat = np.asarray(data[feature_key], dtype=np.float32)
            if feat.ndim != 2:
                raise ValueError(f"{path}:{feature_key} must be 2-D, got {feat.shape}")
            n = feat.shape[0]
            labels = _read_array(data, "label", n, fill=-1, dtype=np.int64).astype(np.int64)
            is_seen = _read_array(data, "is_seen", n, fill=-1, dtype=np.int64).astype(np.int64)
            client_id = _read_array(data, "client_id", n, fill=-1, dtype=np.int64).astype(np.int64)
            router_prob = _read_array(data, "router_prob", n)
            server_pm_prob = _read_array(data, "server_pm_prob", n)
            pm_weight = _read_array(data, "pm_weight", n)
            pm_pred = _read_array(data, "pm_pred", n, fill=-1, dtype=np.int64).astype(np.int64)
            gm_pred = _read_array(data, "gm_pred", n, fill=-1, dtype=np.int64).astype(np.int64)

            start = len(rows)
            features.append(feat)
            for i in range(n):
                rows.append(
                    {
                        "condition": condition,
                        "source_file": str(path),
                        "row": start + i,
                        "client_id": int(client_id[i]),
                        "label": int(labels[i]),
                        "is_seen": int(is_seen[i]),
                        "router_prob": float(router_prob[i]),
                        "server_pm_prob": float(server_pm_prob[i]),
                        "pm_weight": float(pm_weight[i]),
                        "pm_pred": int(pm_pred[i]),
                        "gm_pred": int(gm_pred[i]),
                    }
                )
    return np.concatenate(features, axis=0), rows


def _sample_condition(features, rows, max_points, seed):
    if max_points <= 0 or features.shape[0] <= max_points:
        return features, rows
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(features.shape[0], size=max_points, replace=False))
    return features[idx], [rows[int(i)] for i in idx]


def _standardize(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (x - mean) / std


def _run_tsne(x, perplexity, iterations, seed):
    try:
        from sklearn.manifold import TSNE
    except ImportError as err:
        raise SystemExit("scikit-learn is required: pip install scikit-learn") from err

    n = x.shape[0]
    if n < 4:
        raise ValueError("Need at least 4 samples for t-SNE comparison.")
    perplexity = min(float(perplexity), max(2.0, float((n - 1) // 3)))
    kwargs = dict(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    )
    try:
        return TSNE(max_iter=int(iterations), **kwargs).fit_transform(x)
    except TypeError:
        return TSNE(n_iter=int(iterations), **kwargs).fit_transform(x)


def _finite_color(values):
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if finite.any():
        fill = float(np.nanmedian(values[finite]))
    else:
        fill = 0.0
    return np.where(finite, values, fill)


def write_csv(path, rows, xy):
    fields = [
        "condition",
        "source_file",
        "row",
        "client_id",
        "label",
        "is_seen",
        "router_prob",
        "server_pm_prob",
        "pm_weight",
        "pm_pred",
        "gm_pred",
        "tsne_x",
        "tsne_y",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row, coord in zip(rows, xy):
            out = dict(row)
            out["tsne_x"] = float(coord[0])
            out["tsne_y"] = float(coord[1])
            writer.writerow(out)


def plot_compare(path, rows, xy, no_ood_label, with_ood_label):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise SystemExit("matplotlib is required: pip install matplotlib") from err

    conditions = np.asarray([row["condition"] for row in rows])
    is_seen = np.asarray([row["is_seen"] for row in rows], dtype=np.int64)
    router_prob = np.asarray([row["router_prob"] for row in rows], dtype=np.float32)
    labels = [no_ood_label, with_ood_label]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for col, label in enumerate(labels):
        mask = conditions == label
        sub_xy = xy[mask]
        sub_seen = is_seen[mask]
        ax = axes[0, col]
        seen_mask = sub_seen == 1
        ood_mask = sub_seen == 0
        ax.scatter(sub_xy[seen_mask, 0], sub_xy[seen_mask, 1], s=10, alpha=0.65, c="#1f77b4", label="seen")
        ax.scatter(sub_xy[ood_mask, 0], sub_xy[ood_mask, 1], s=12, alpha=0.75, c="#d62728", label="OOD")
        ax.set_title(f"{label}: seen/OOD (n={int(mask.sum())}, OOD={int(ood_mask.sum())})")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="best", frameon=False)

        ax = axes[1, col]
        colors = _finite_color(router_prob[mask])
        sc = ax.scatter(sub_xy[:, 0], sub_xy[:, 1], s=10, alpha=0.75, c=colors, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"{label}: router PM probability")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare FedCD locality-aware router dumps with t-SNE.")
    parser.add_argument("--no-ood", required=True, help="Dump directory or .npz from router trained without OOD supervision.")
    parser.add_argument("--with-ood", required=True, help="Dump directory or .npz from router trained with OOD supervision.")
    parser.add_argument("--output-dir", required=True, help="Directory for PNG and CSV outputs.")
    parser.add_argument("--feature-key", default="router_feature", choices=["router_feature", "raw_feature"])
    parser.add_argument("--no-ood-label", default="no_ood_train")
    parser.add_argument("--with-ood-label", default="with_ood_train")
    parser.add_argument("--max-points-per-condition", type=int, default=3000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_no, rows_no = load_condition(args.no_ood, args.no_ood_label, args.feature_key)
    x_yes, rows_yes = load_condition(args.with_ood, args.with_ood_label, args.feature_key)
    x_no, rows_no = _sample_condition(x_no, rows_no, args.max_points_per_condition, args.seed)
    x_yes, rows_yes = _sample_condition(x_yes, rows_yes, args.max_points_per_condition, args.seed + 1)

    x = np.concatenate([x_no, x_yes], axis=0)
    rows = rows_no + rows_yes
    xy = _run_tsne(_standardize(x), args.perplexity, args.iterations, args.seed)

    csv_path = out_dir / f"router_tsne_compare_{args.feature_key}.csv"
    png_path = out_dir / f"router_tsne_compare_{args.feature_key}.png"
    write_csv(csv_path, rows, xy)
    plot_compare(png_path, rows, xy, args.no_ood_label, args.with_ood_label)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
