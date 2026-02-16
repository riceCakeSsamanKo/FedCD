import os
import ujson
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
Description:
이 코드는 각 클라이언트가 보유한 클래스(레이블)별 데이터 개수를 히트맵 형태로 시각화합니다.
X축은 클래스 ID, Y축은 클라이언트 ID를 나타내며, 색상이 진할수록 해당 클래스의 데이터가 많음을 의미합니다.
이를 통해 데이터셋의 Non-IID 정도(Pathological vs Dirichlet)를 직관적으로 확인할 수 있습니다.
"""

def visualize_distribution(dataset_name):
    # 경로 설정 (소스가 data_visualization_code 폴더에 있으므로 상위로 이동)
    dir_path = f"../dataset/{dataset_name}/"
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")

    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = ujson.load(f)

    num_clients = config['num_clients']
    num_classes = config['num_classes']
    partition = config.get('partition', 'unknown')
    
    dist_matrix = np.zeros((num_clients, num_classes))

    for i in range(num_clients):
        file_path = os.path.join(train_path, f"{i}.npz")
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True)['data'].item()
            y = data['y']
            unique, counts = np.unique(y, return_counts=True)
            for label, count in zip(unique, counts):
                dist_matrix[i, int(label)] = count

    plt.figure(figsize=(12, 8))
    sns.heatmap(dist_matrix, annot=False, cmap="YlGnBu", fmt='g')
    plt.title(f"Label Distribution per Client ({dataset_name} - {partition})")
    plt.xlabel("Class ID")
    plt.ylabel("Client ID")
    
    # 결과 저장 경로 설정
    result_dir = f"../data_visualization_result/{partition}"
    os.makedirs(result_dir, exist_ok=True)
    output_file = os.path.join(result_dir, f"label_dist_{dataset_name}.png")
    
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    visualize_distribution("Cifar10")
