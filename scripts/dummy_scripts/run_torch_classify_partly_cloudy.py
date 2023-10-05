import subprocess
import pandas as pd 
from tqdm import trange 

def run_classify_torch():
    base_command = [
        "python", "classify_torch.py",
        "--dataset", "partly_cloudy",
        "--task_type", "EMOTION3",
        "--hidden_dim", "16",
        "--learning_rate", ".01"
    ]

    models = ["GAT", "GCN", "GIN"]
    results_list = []
    dataset_size = 155 
    for i in trange(dataset_size):
        for model in models:
            sub_dataset_value = f"{i:04d}"
            command = base_command + [
                "--model", model,
                "--sub_dataset", sub_dataset_value
            ]
            result = subprocess.run(command, capture_output = True, text=True).stdout.strip().split(",")
            acc = float(result[0])
            stdev = float(result[1])

            new_row = {
                "sub_dataset": sub_dataset_value,
                "model": model,
                "acc": acc,
                "stdev": stdev,
                "hidden_dim": 16, 
                "learning_rate":.01
            }

            results_list.append(new_row)
    
    df_results = pd.DataFrame(results_list)
    df_results.to_csv(f'partly_cloudy_GNN.csv', index = False)

if __name__ == "__main__":
    run_classify_torch()