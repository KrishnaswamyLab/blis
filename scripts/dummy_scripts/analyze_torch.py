import pandas as pd
import numpy as np


# Load the DataFrame
df = pd.read_csv('partly_cloudy_GNN_16_lr_0_01.csv')
models = ['GAT', 'GIN', 'GCN']
for model in models:
    # Filter the DataFrame for only rows where the model is 'MLP'
    model_df = df[df['model'] == model]

    avg_score = model_df['acc'].mean()
    avg_CV_std = model_df['stdev'].mean()
    avg_std = model_df['acc'].std()
    print(f'for model {model}, avg_score is {avg_score} pm {avg_std}, CV stdev is {avg_CV_std}')