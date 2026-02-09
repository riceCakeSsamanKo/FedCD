import os
import ujson
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

"""
Description:
이 코드는 VGG16 Feature Extractor를 통해 추출된 데이터의 고차원 임베딩을 t-SNE를 이용해 2차원으로 시각화합니다.
1. 클라이언트별 시각화: 각 클라이언트의 데이터가 임베딩 공간에서 어떻게 섞여 있는지 보여줍니다. (partition 폴더에 저장)
2. 클래스별 시각화: 데이터의 실제 클래스(레이블)가 어떻게 군집화되는지 보여줍니다. (class 폴더에 저장)
이를 통해 데이터 분배 방식에 따른 클라이언트 간의 경계 모호성을 분석할 수 있습니다.
"""

class SmallFExt(nn.Module):
    def __init__(self, in_channels=3, out_dim=512):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.flatten = nn.Flatten()
        self.out_dim = out_dim 
    def forward(self, x):
        return self.flatten(self.conv3(self.conv2(self.conv1(x))))

def get_vgg16_extractor():
    base_model = torchvision.models.vgg16(pretrained=True)
    f_ext = nn.Sequential(base_model.features, base_model.avgpool, nn.Flatten())
    return f_ext

def visualize_embedding(dataset_name, model_type="VGG16", samples_per_client=100):
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

    all_embeddings, all_client_ids, all_labels = [], [], []

    with torch.no_grad():
        for i in range(num_clients):
            file_path = os.path.join(train_path, f"{i}.npz")
            if not os.path.exists(file_path): continue
            data = np.load(file_path, allow_pickle=True)['data'].item()
            X, y = data['x'], data['y']
            idx = np.random.choice(len(X), min(len(X), samples_per_client), replace=False)
            X_tensor = torch.tensor(X[idx]).float().to(device)
            emb = model(X_tensor).cpu().numpy()
            all_embeddings.append(emb)
            all_client_ids.extend([i] * len(emb))
            all_labels.extend(y[idx])

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(all_embeddings)

    # 1. 시각화 - 클라이언트별 (partition 폴더)
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_client_ids, cmap='tab20', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Client ID')
    plt.title(f"t-SNE by Client ({partition})")
    res_dir = f"../data_visualization_result/{partition}"
    os.makedirs(res_dir, exist_ok=True)
    plt.savefig(os.path.join(res_dir, f"embedding_by_client.png"))

    # 2. 시각화 - 클래스별 (class 폴더)
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Class Label')
    plt.title(f"t-SNE by Class ({partition})")
    class_dir = "../data_visualization_result/class"
    os.makedirs(class_dir, exist_ok=True)
    plt.savefig(os.path.join(class_dir, f"embedding_by_class_{partition}.png"))
    print(f"Saved results to {res_dir} and {class_dir}")

if __name__ == "__main__":
    visualize_embedding("Cifar10")
