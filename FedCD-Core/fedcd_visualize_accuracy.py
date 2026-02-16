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
                
                # Extract partition and alpha from path if possible
                path_parts = exp_path.replace("\\", "/").split("/")
                # Example: logs/FedCD/GM_.../dir/0.1/NC_20/...
                partition = "Unknown"
                if "dir" in path_parts: partition = "Dir"
                elif "pat" in path_parts: partition = "Pat"
                
                title_suffix = f"({algo}, {partition}, NC={n_clients})"

        # 1. Plot Main Accuracy Curve
        acc_file = os.path.join(exp_path, "acc.csv")
        if os.path.exists(acc_file):
            df = pd.read_csv(acc_file)
            if not df.empty and "test_acc" in df.columns:
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=df, x="round", y="test_acc", label="Test Accuracy", marker="o")
                
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
                sns.lineplot(data=df_c, x="round", y="accuracy", hue="cluster_id", 
                             palette="tab10", marker="o")
                
                plt.title(f"Cluster-wise Accuracy {title_suffix}")
                plt.xlabel("Round")
                plt.ylabel("Accuracy")
                plt.grid(True)
                plt.ylim(0, 1.05)
                plt.legend(title="Cluster ID", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                
                save_path = os.path.join(exp_path, "cluster_accuracy_curve.png")
                plt.savefig(save_path)
                plt.close()
                print(f"Generated: {save_path}")

    except Exception as e:
        print(f"Error processing {exp_path}: {e}")

def main():
    log_dir = "logs/FedCD/"
    if not os.path.exists(log_dir):
        print(f"Directory '{log_dir}' not found.")
        return

    print(f"Searching for experiments in {log_dir} ...")
    
    found_count = 0
    # Walk through all subdirectories
    for root, dirs, files in os.walk(log_dir):
        # We consider a directory as an experiment if it contains acc.csv
        if "acc.csv" in files:
            print(f"\nProcessing Experiment: {root}")
            visualize_single_experiment(root)
            found_count += 1
            
    if found_count == 0:
        print("No experiments with 'acc.csv' found.")
    else:
        print(f"\nFinished processing {found_count} experiments.")

if __name__ == "__main__":
    main()
