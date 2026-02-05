import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def visualize_single_experiment(exp_path):
    """
    Visualizes results for a single experiment folder.
    Generates:
    1. accuracy_curve.png (from acc.csv)
    2. cluster_accuracy_curve.png (from cluster_acc.csv, if exists)
    """
    try:
        # Load config to get metadata for title
        config_file = os.path.join(exp_path, "config.json")
        title_suffix = ""
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
                algo = config.get("algorithm", "Unknown")
                n_clients = config.get("num_clients", "?")
                n_clusters = config.get("num_clusters", "?")
                
                # Determine partition type from copied dataset config name if possible
                partition = "?"
                ds_configs = glob.glob(os.path.join(exp_path, "dataset_config_*.json"))
                if ds_configs:
                    fname = os.path.basename(ds_configs[0])
                    if "_pat_" in fname: partition = "Pat"
                    elif "_dir_" in fname: partition = "Dir"
                
                title_suffix = f"({algo}, {partition}, Clients={n_clients}, Clusters={n_clusters})"

        # 1. Plot Main Accuracy Curve
        acc_file = os.path.join(exp_path, "acc.csv")
        if os.path.exists(acc_file):
            df = pd.read_csv(acc_file)
            if not df.empty:
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=df, x="round", y="test_acc", label="Test Accuracy", marker="o")
                # Also plot train loss on secondary axis if needed, or just accuracy
                
                plt.title(f"Test Accuracy Curve {title_suffix}")
                plt.xlabel("Round")
                plt.ylabel("Accuracy")
                plt.grid(True)
                plt.ylim(0, 1.05)
                
                save_path = os.path.join(exp_path, "accuracy_curve.png")
                plt.savefig(save_path)
                plt.close()
                print(f"Generated: {save_path}")

        # 2. Plot Cluster Accuracy Curve (if available)
        cluster_file = os.path.join(exp_path, "cluster_acc.csv")
        if os.path.exists(cluster_file):
            df_c = pd.read_csv(cluster_file)
            if not df_c.empty and "cluster_id" in df_c.columns:
                plt.figure(figsize=(12, 8))
                
                # Pivot or filter to plot lines for each cluster
                # unique_clusters = df_c["cluster_id"].unique()
                sns.lineplot(data=df_c, x="round", y="accuracy", hue="cluster_id", 
                             palette="tab10", marker="o")
                
                plt.title(f"Cluster-wise Accuracy {title_suffix}")
                plt.xlabel("Round")
                plt.ylabel("Accuracy")
                plt.grid(True)
                plt.ylim(0, 1.05)
                plt.legend(title="Cluster ID")
                
                save_path = os.path.join(exp_path, "cluster_accuracy_curve.png")
                plt.savefig(save_path)
                plt.close()
                print(f"Generated: {save_path}")

    except Exception as e:
        print(f"Error processing {exp_path}: {e}")

def main():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        print(f"Directory '{log_dir}' not found.")
        return

    # Find all experiment directories (subdirectories of logs/)
    exp_dirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    
    print(f"Found {len(exp_dirs)} experiment directories.")
    
    for exp_path in exp_dirs:
        # Check if it looks like an experiment folder (has config or csvs)
        if glob.glob(os.path.join(exp_path, "*.csv")) or glob.glob(os.path.join(exp_path, "*.json")):
            visualize_single_experiment(exp_path)

if __name__ == "__main__":
    main()