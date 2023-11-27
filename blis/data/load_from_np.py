import torch 
import numpy as np
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import os
import tqdm

from blis import DATA_DIR

# Convert adjacency matrix to edge indices
def adjacency_to_edge_indices(A):
    edge_indices = torch.nonzero(A, as_tuple=False).t()
    return edge_indices

# Create a dataset and data loader
def create_dataset(X, y, A, transform = None):
    edge_index = adjacency_to_edge_indices(torch.Tensor(A))
    edge_weight = torch.Tensor(A[edge_index[0], edge_index[1]])
    data_list = []

    #print("Creating dataset....") 
    for i in tqdm.tqdm(range(X.shape[0])):
        num_nodes = X[i].shape[0]
        
        data = Data(x=torch.Tensor(X[i]), edge_index=edge_index, y=torch.Tensor([y[i]]).long(), num_nodes = num_nodes, edge_weight = edge_weight)
        
        if transform is not None:
            data = transform(data)
        data_list.append(data)
    #print("Done!")
    return data_list

# usage:
# kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
# # Iterate over the folds
# for train_index, test_index in kf.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

#     # Create the dataset and data loader
#     train_data = create_dataset(X_train, y_train, A)
#     train_loader = DataLoader(train_data, batch_size=32, shuffle=True)