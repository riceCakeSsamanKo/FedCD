import os
import ujson
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from matplotlib.patches import Ellipse

"""
Description:
이 코드는 FedCD 클러스터링의 핵심인 '데이터 분포의 형상(Shape)'을 분석합니다.
1. Cosine Similarity Heatmap: 각 클라이언트의 [평균, 분산] 통계 벡터 간의 유사도를 측정합니다.
2. t-SNE Regions: 임베딩 공간에서 각 클라이언트의 데이터가 차지하는 영역을 타원(Ellipse)으로 시각화합니다.
이를 통해 왜 Dirichlet 방식에서 클러스터링 거리가 가깝게 측정되는지(타원이 겹치는지) 설명할 수 있습니다.
"""

def get_vgg16_extractor():
    base_model = torchvision.models.vgg16(pretrained=True)
    f_ext = nn.Sequential(base_model.features, base_model.avgpool, nn.Flatten())
    return f_ext

def draw_confidence_ellipse(x, y, ax, n_std=1.5, facecolor='none', **kwargs):
    if len(x) < 3: return
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x, ell_radius_y = np.sqrt(1 + pearson), np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
    scale_x, scale_y = np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std
    import matplotlib.transforms as transforms
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(np.mean(x), np.mean(y))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def visualize_distribution_shape(dataset_name):
    dir_path = f"../dataset/{dataset_name}/"
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")

    with open(config_path, 'r') as f:
        config = ujson.load(f)
    num_clients = config['num_clients']
    partition = config.get('partition', 'unknown')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_vgg16_extractor().to(device)
    model.eval()

    client_stats, all_embs, client_ids = [], [], []

    with torch.no_grad():
        for i in range(num_clients):
            file_path = os.path.join(train_path, f"{i}.npz")
            if not os.path.exists(file_path): continue
            data = np.load(file_path, allow_pickle=True)['data'].item()
            idx = np.random.choice(len(data['x']), min(len(data['x']), 100), replace=False)
            emb = model(torch.tensor(data['x'][idx]).float().to(device)).cpu().numpy()
            client_stats.append(np.concatenate([np.mean(emb, axis=0), np.var(emb, axis=0)]))
            all_embs.append(emb)
            client_ids.append(np.full(len(emb), i))

    res_dir = f"../data_visualization_result/{partition}"
    os.makedirs(res_dir, exist_ok=True)

    # 1. Similarity Heatmap
    norm_stats = normalize(np.stack(client_stats), axis=1)
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.dot(norm_stats, norm_stats.T), cmap="YlOrRd", vmin=0.8)
    plt.title(f"Client Shape Similarity (Cosine) - {partition}")
    plt.savefig(os.path.join(res_dir, "shape_similarity.png"))

    # 2. t-SNE with Ellipses
    tsne_results = TSNE(n_components=2, random_state=42).fit_transform(np.concatenate(all_embs))
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, num_clients))
    all_ids = np.concatenate(client_ids)
    for i in range(num_clients):
        pts = tsne_results[all_ids == i]
        plt.scatter(pts[:, 0], pts[:, 1], color=colors[i], alpha=0.1, s=5)
        draw_confidence_ellipse(pts[:, 0], pts[:, 1], plt.gca(), edgecolor=colors[i], linewidth=2)
    plt.title(f"Embedding Shapes (t-SNE Regions) - {partition}")
    plt.savefig(os.path.join(res_dir, "shape_territory.png"))
    print(f"Saved results to {res_dir}")

if __name__ == "__main__":
    visualize_distribution_shape("Cifar10")
