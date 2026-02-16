import os
import ujson
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import norm

"""
Description:
이 코드는 각 클라이언트의 고차원 임베딩을 PCA를 통해 가장 지배적인 1차원 축으로 투영한 뒤,
그 축 위에서의 데이터 분포를 정규분포 곡선(종 모양)으로 시각화합니다.
각 곡선의 '폭(분산)'과 '겹침 정도'를 통해, 왜 Dirichlet 방식에서 클라이언트들이 서로 구분되기 힘든지(분산이 너무 커서 겹치는지) 분석합니다.
"""

def get_vgg16_extractor():
    base_model = torchvision.models.vgg16(pretrained=True)
    f_ext = nn.Sequential(base_model.features, base_model.avgpool, nn.Flatten())
    return f_ext

def visualize_client_curves(dataset_name):
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

    client_data = []
    print("Extracting embeddings...")
    with torch.no_grad():
        for i in range(num_clients):
            file_path = os.path.join(train_path, f"{i}.npz")
            if not os.path.exists(file_path): continue
            data = np.load(file_path, allow_pickle=True)['data'].item()
            idx = np.random.choice(len(data['x']), min(len(data['x']), 200), replace=False)
            emb = model(torch.tensor(data['x'][idx]).float().to(device)).cpu().numpy()
            client_data.append(emb)

    # PCA to 1D
    combined = np.concatenate(client_data, axis=0)
    pca = PCA(n_components=1)
    pca.fit(combined)

    plt.figure(figsize=(12, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, num_clients))
    x_axis = np.linspace(-150, 150, 1000)

    for i in range(num_clients):
        projected = pca.transform(client_data[i]).flatten()
        mu, std = norm.fit(projected)
        p = norm.pdf(x_axis, mu, std)
        plt.plot(x_axis, p, color=colors[i], linewidth=2, label=f'C{i} (std={std:.1f})')
        plt.fill_between(x_axis, p, alpha=0.1, color=colors[i])

    plt.title(f"Client Embedding Curves (PCA 1D) - {partition}")
    plt.xlabel("Principal Feature Axis")
    plt.ylabel("Probability Density")
    
    res_dir = f"../data_visualization_result/{partition}"
    os.makedirs(res_dir, exist_ok=True)
    plt.savefig(os.path.join(res_dir, "distribution_curves.png"))
    print(f"Saved results to {res_dir}")

if __name__ == "__main__":
    visualize_client_curves("Cifar10")
