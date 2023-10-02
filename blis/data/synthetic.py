import torch 
import numpy as np
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import os

from blis import DATA_DIR
from blis.data.load_from_np import create_dataset


def synthetic_data_loader(seed, subdata_type, task_type, batch_size):
    
    label_path = os.path.join(DATA_DIR,"synthetic",subdata_type,task_type,"label.npy")
    graph_path = os.path.join(DATA_DIR,"synthetic",subdata_type,"adjacency_matrix.npy")
    signal_path = os.path.join(DATA_DIR,"synthetic",subdata_type,"graph_signals.npy")

    # Load data
    X = np.load(signal_path)
    y = np.load(label_path)
    A = np.load(graph_path)

    data = create_dataset(X, y, A)

    train_idx, val_idx = train_test_split(np.arange(len(data)), test_size=0.3, random_state=seed)
    val_idx, test_idx = train_test_split(val_idx, test_size=0.5, random_state=seed)

    train_ds = Subset(data,train_idx)
    val_ds = Subset(data,val_idx)
    test_ds = Subset(data,test_idx)


    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_dl, val_dl, test_dl


def synthetic_scattering_data_loader(seed, subdata_type, task_type, batch_size, scattering_dict = None):
    """
    Extract the scattering features according to the following options

    Scattering_dict =
    { scattering_type : ["blis" or "modulus"],
    scale_type : ["largest_scale_4]}
    layers :[[1],[1,2]],
    moments : [[1,2,3],...]}

    Returns :
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    """
    label_path = os.path.join(DATA_DIR,"synthetic",subdata_type,task_type,"label.npy")
    layer_paths = [os.path.join(DATA_DIR,"synthetic",subdata_type,"processed",
                                scattering_dict["scattering_type"],
                                scattering_dict["scale_type"],
                                f"layer_{layer}") for layer in scattering_dict["layers"]]
   
    
    moments = []
    for layer_path in layer_paths:
        for moment in scattering_dict["moments"]:
            moments.append(np.load(os.path.join(layer_path, "moment_{}.npy".format(moment))))

    X = np.concatenate(moments,1)
    y = np.load(label_path)

    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.3, random_state=seed)
    val_idx, test_idx = train_test_split(val_idx, test_size=0.5, random_state=seed)
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])



if __name__ == "__main__":
    #tr, vl, ts = traffic_data_loader(42, "PEMS04", "DAY", 32)

    scattering_dict = { "scattering_type" : "blis",
    "scale_type" : "largest_scale_4",
    "layers" :[1],
    "moments" : [1,2]}
    
    tr, vl, ts = synthetic_scattering_data_loader(42, "PEMS04", "DAY", 32, scattering_dict = scattering_dict)
    for i,b in enumerate(tr):
        breakpoint()
