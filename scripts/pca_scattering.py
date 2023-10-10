from blis.data import traffic, cloudy, synthetic
import argparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neural_network import MLPClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.decomposition import PCA
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

def run_pca_scattering_cloudy(args,scattering_dict):
    full_test_scores = []
    full_train_scores = []
    full_n_pca = []
    pca_dict = {}
    scattering_type_list = ['modulus', 'blis']
    # assume that the dataset is cloudy
    for scattering_type in scattering_type_list:
        pca_dict[scattering_type] = list()
        scattering_dict['scattering_type'] = scattering_type
        for sub_dataset_num in range(155):
            sub_dataset_str = f"{sub_dataset_num:04d}"
            for seed in [42]:
                if args.dataset == "traffic":
                    (X_train, y_train),  (X_test, y_test) = traffic.traffic_scattering_data_loader(seed=seed,
                                                                                                            subdata_type=args.sub_dataset,
                                                                                                            task_type=args.task_type,
                                                                                                            scattering_dict=scattering_dict)
                elif args.dataset == "partly_cloudy":
                    (X_train, y_train),  (X_test, y_test) = cloudy.cloudy_scattering_data_loader(seed=seed,
                                                                                                            subdata_type=sub_dataset_str,
                                                                                                            task_type=args.task_type,
                                                                                                            scattering_dict=scattering_dict)
                elif args.dataset == "synthetic":
                    (X_train, y_train),  (X_test, y_test) = synthetic.synthetic_scattering_data_loader(seed=seed,
                                                                                                            subdata_type=args.sub_dataset,
                                                                                                            task_type=args.task_type,
                                                                                                            scattering_dict=scattering_dict)
                else:
                    raise ValueError("Invalid dataset")



                X_train = X_train.reshape(X_train.shape[0],-1) 
                X_test = X_test.reshape(X_test.shape[0],-1)

                pca = PCA()
                X_pca = pca.fit_transform(X_train)
                explained_variance = pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance)

                pca_dict[scattering_type].append(cumulative_variance)
        
    import pdb; pdb.set_trace()
    for scattering_type in scattering_type_list:
        full_n_pca_arr = np.array(pca_dict[scattering_type])
        mean = [full_n_pca_arr[:,ind].mean() for ind in range(full_n_pca_arr.shape[1])]
        plt.plot(np.array(mean)[0:60], label = scattering_type)
    plt.legend()
    plt.xlabel("Number of PC")
    plt.ylabel("Explained variance")
    plt.title("Partly cloudy scattering")
    plt.savefig("PCA_partly_cloudy.png")
        

    return np.average(np.array(full_n_pca))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse arguments for the program.")

    parser.add_argument("--scattering_type", choices=['blis', 'modulus'], help="Type of scattering: 'blis' or 'modulus'.")
    parser.add_argument("--largest_scale", type=int, help="Largest (dyadic) scale as a positive integer.")
    parser.add_argument("--moment_list", nargs='+', type=int, default=[1], help="List of moments as positive integers. E.g., --moment_list 1 2 3.") 
    parser.add_argument("--dataset", choices=['traffic', 'partly_cloudy', 'synthetic'], help="Dataset: 'traffic' or 'partly_cloudy' or 'synthetic'.")
    parser.add_argument("--sub_dataset", help="Sub-dataset value depending on the dataset chosen.")
    parser.add_argument("--layer_list", nargs='+', type=int, default=[2], help="List of layers as positive integers. E.g., --layer_list 1 2.")
    parser.add_argument("--model", choices=['RF', 'SVC', 'KNN', 'MLP', 'LR','XGB'], type=str, default="LR", help="Classification model to use. Options: 'RF', 'SVC', 'KNN', 'MLP', 'LR','XGB'")
    parser.add_argument("--task_type", type=str,  help="The task type to use for the classification")
    parser.add_argument("--PCA_variance", type=float, default=1, help="PCA variance to retain (int between 0 and 1, default: 1)")

    args = parser.parse_args()
    if not 0 < args.PCA_variance <= 1:
        raise ValueError("PCA variance should be between 0 and 1")

    scattering_dict = {"scattering_type": args.scattering_type,
                       "scale_type": f"largest_scale_{args.largest_scale}",
                       "layers": args.layer_list,
                       "moments" : args.moment_list}
    
    final_score, final_stdev, n_comp = run_pca_scattering_cloudy(args,scattering_dict)
    print(f"{final_score},{final_stdev}, {n_comp}")

    #Example : python classify_scattering.py --dataset=traffic --largest_scale=4 --sub_dataset=PEMS04 --scattering_type=blis --task_type=DAY
    #Example : python pca_scattering.py --dataset=partly_cloudy --sub_dataset=0001 --largest_scale=4 --scattering_type=blis --task_type=EMOTION3 --moment_list 1 --layer_list 3 --model SVC

