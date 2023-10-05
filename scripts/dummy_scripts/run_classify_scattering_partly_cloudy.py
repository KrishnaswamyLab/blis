import subprocess
import pandas as pd
from tqdm import trange

def run_classify_scattering():
    pca_var = .999
    base_command = [
        "python", "classify_scattering.py",
        "--largest_scale", "4",
        "--moment_list", "1",
        "--layer_list", "3",
        "--dataset", "partly_cloudy",
        "--task_type", "EMOTION3",
        "--PCA_variance", f"{pca_var}"
    ]
    
    scattering_types = ["blis", "modulus"]
    models = ['MLP', 'XGB']

    # DataFrame to store results
    results_list = []
    dataset_size = 155 #155 for full dataset
    for scattering_type in scattering_types:
        for i in trange(dataset_size):  # 155 because it will generate numbers from 0 to 154 inclusive
            for model in models:
                # Formatting the number to be 4 digits
                sub_dataset_value = f"{i:04d}"
                # Construct the command
                command = base_command + [
                    "--scattering_type", scattering_type,
                    "--sub_dataset", sub_dataset_value,
                    "--model", model
                ]
                # Assuming your script returns output like: "0.85,0.05"
                result = subprocess.run(command, capture_output=True, text=True).stdout.strip().split(',')
                score = float(result[0])
                stdev = float(result[1])
                ncomp = float(result[2])

                # Storing results in the DataFrame
                new_row = {
                    'scattering_type': scattering_type,
                    'sub_dataset': sub_dataset_value,
                    'model': model,
                    'score': score,
                    'stdev': stdev,
                    'ncomp': ncomp,
                    'pca_var':pca_var}
                results_list.append(new_row)
                
    df_results = pd.DataFrame(results_list)
    # Save results to a CSV file
    df_results.to_csv(f'results_{dataset_size}.csv', index=False)

if __name__ == "__main__":
    run_classify_scattering()
