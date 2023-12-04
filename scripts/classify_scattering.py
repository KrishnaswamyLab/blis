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
import pandas as pd
import warnings
import os
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

def run_classifier_scattering(args,scattering_dict):
    full_test_scores = []
    full_train_scores = []
    full_n_pca = []
    for seed in [42,43,44,45,56]:
        if args.dataset == "traffic":
            (X_train, y_train),  (X_test, y_test) = traffic.traffic_scattering_data_loader(seed=seed,
                                                                                                    subdata_type=args.sub_dataset,
                                                                                                    task_type=args.task_type,
                                                                                                    scattering_dict=scattering_dict,
                                                                                                    ignore_graph=args.ignore_graph)
        elif args.dataset == "partly_cloudy":
            (X_train, y_train),  (X_test, y_test) = cloudy.cloudy_scattering_data_loader(seed=seed,
                                                                                                    subdata_type=args.sub_dataset,
                                                                                                    task_type=args.task_type,
                                                                                                    scattering_dict=scattering_dict,
                                                                                                    ignore_graph=args.ignore_graph)
        elif args.dataset == "synthetic":
            (X_train, y_train),  (X_test, y_test) = synthetic.synthetic_scattering_data_loader(seed=seed,
                                                                                                    subdata_type=args.sub_dataset,
                                                                                                    task_type=args.task_type,
                                                                                                    scattering_dict=scattering_dict,
                                                                                                    ignore_graph=args.ignore_graph)
        else:
            raise ValueError("Invalid dataset")

        X_train = X_train.reshape(X_train.shape[0],-1) 
        X_test = X_test.reshape(X_test.shape[0],-1)
        n_comp = -1
        if (args.PCA_variance != 1):
            if args.PCA_variance > 1:
                args.PCA_variance = int(args.PCA_variance)
            pca = PCA(n_components=args.PCA_variance)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            n_comp = pca.n_components_
            #print(f"num principal components = {pca.n_components_}")


        if args.model == "LR":
            base_model = LogisticRegression() 
        if args.model == "SVC":
            base_model = SVC() 
        if args.model == "KNN":
            base_model = KNeighborsClassifier()
        if args.model == "MLP":
            base_model = MLPClassifier() 
        if args.model == "RF":
            base_model = RandomForestClassifier()
        if args.model == "XGB":
            base_model = xgb.XGBClassifier()

        in_shape = X_train.shape[1]

        # Create a pipeline that first applies the standard scaler, then the model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', base_model)
        ])

        # Define hyperparameters grid for each model (with 'model__' prefix for parameters)
        if isinstance(base_model, RandomForestClassifier):
            param_grid = {
                'model__n_estimators': [50, 100, 150],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5]
            }
        elif isinstance(base_model, SVC):
            param_grid = {
                'model__C': [0.1, 1, 10],
                'model__kernel': ['linear', 'rbf'],
                'model__gamma': ['scale','auto', .1, 1, 10]
            }
        elif isinstance(base_model, KNeighborsClassifier):
            param_grid = {
                'model__n_neighbors': [3, 5, 7],
                'model__weights': ['uniform', 'distance']
            }
        elif isinstance(base_model, MLPClassifier):
            param_grid = {
                'model__hidden_layer_sizes': [(in_shape//2, in_shape//4), (in_shape//2, in_shape//4, in_shape//8), (150, 50)],
                'model__activation': ['relu'],
                'model__alpha': [.01]
            }
        elif isinstance(base_model, LogisticRegression):
            param_grid = {
                'model__C': [0.1, 1, 10],
                'model__solver': ['lbfgs', 'liblinear']
            }
        elif isinstance(base_model, xgb.XGBClassifier):
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__learning_rate': [0.05, 0.1]
            }

        clf = GridSearchCV(pipeline, param_grid, cv = 3)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Best parameters found: ",clf.best_params_)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        #print("Train score : ", train_score)
        #print("Test score : ", test_score)
        full_test_scores.append(test_score)
        full_train_scores.append(train_score)
        full_n_pca.append(n_comp)
    
    final_score = np.average(np.array(full_test_scores))
    final_stdev = np.std(np.array(full_test_scores))
    #print("Final score: ", final_score )
    #print("Final stdev: ", final_stdev)
    return final_score, final_stdev, np.average(np.array(full_n_pca))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse arguments for the program.")

    parser.add_argument("--scattering_type", choices=['blis', 'modulus', 'all'], help="Type of scattering: 'blis' or 'modulus'.")
    parser.add_argument("--largest_scale", type=int, help="Largest (dyadic) scale as a positive integer.")
    parser.add_argument("--moment_list", nargs='+', type=int, default=[1], help="List of moments as positive integers. E.g., --moment_list 1 2 3.") 
    parser.add_argument("--dataset", choices=['traffic', 'partly_cloudy', 'synthetic'], help="Dataset: 'traffic' or 'partly_cloudy' or 'synthetic'.")
    parser.add_argument("--sub_dataset", help="Sub-dataset value depending on the dataset chosen. Use 'full' for entire dataset")
    parser.add_argument("--layer_list", nargs='+', type=int, default=[2], help="List of layers as positive integers. E.g., --layer_list 1 2.")
    parser.add_argument("--model", choices=['RF', 'SVC', 'KNN', 'MLP', 'LR','XGB', 'all'], type=str, default="LR", help="Classification model to use. Options: 'RF', 'SVC', 'KNN', 'MLP', 'LR','XGB', 'all'")
    parser.add_argument("--task_type", type=str,  help="The task type to use for the classification")
    parser.add_argument("--PCA_variance", type=float, default=1, help="PCA variance to retain (int between 0 and 1, default: 1)")
    parser.add_argument("--wavelet_type", choices=['W1','W2', 'all'], default = 'W2', help='Type of wavelet, either W1 or W2')
    parser.add_argument("--ignore_graph", type=bool, default=False, help="Ignore the graph structure of the data")

    args = parser.parse_args()

    # generate the list of sub datasets 
    if args.dataset == 'traffic' and args.sub_dataset == 'full':
        sub_datasets = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']
    elif args.dataset == 'partly_cloudy' and args.sub_dataset == 'full':
        sub_datasets = [f'{i:04d}' for i in range(155)]
    elif args.dataset == 'synthetic' and args.sub_dataset == 'full':
        dataset_types = ['camel_pm', 'gaussian_pm']
        sub_datasets = []
        for dataset_type in dataset_types:
            for i in range(5):
                sub_datasets.append(f'{dataset_type}_{i}')
    else:
        sub_datasets = [args.sub_dataset]

    if args.model == 'all':
        models = ['RF', 'SVC', 'KNN', 'MLP', 'LR', 'XGB']
    else:
        models = [args.model]
    
    if args.scattering_type == 'all':
        scattering_types = ['blis', 'modulus']
    else:
        scattering_types = [args.scattering_type]

    if args.wavelet_type == 'all':
        wavelet_types = ['W1', 'W2']
    else:
        wavelet_types = [args.wavelet_type]

    results_list = []

    for wavelet_type in wavelet_types:
        for scattering_type in scattering_types:
            for model in models:
                for sub_dataset in sub_datasets:
                    args.model = model 
                    args.scattering_type = scattering_type 
                    args.sub_dataset = sub_dataset 

                    scattering_dict = {"scattering_type": scattering_type,
                        "scale_type": f"largest_scale_{args.largest_scale}",
                        "layers": args.layer_list,
                        "moments" : args.moment_list,
                        "wavelet_type": wavelet_type}
                                        
                    final_score, final_stdev, n_comp = run_classifier_scattering(args, scattering_dict)
                    
                    # store the results

                    new_row = {
                        'scattering_type': scattering_type, 
                        'sub_dataset': sub_dataset,
                        'model': model,
                        'score': final_score,
                        'stdev': final_stdev, 
                        'ncomp': n_comp,
                        'task': args.task_type,
                        'pca_var': args.PCA_variance,
                        'moment_list': '1',
                        'layer_list': ','.join(map(str, args.layer_list)),
                        'wavelet_type': wavelet_type,
                        'dataset': args.dataset,
                        'largest_scale': args.largest_scale,
                        'ignore_graph': args.ignore_graph
                    }
                    results_list.append(new_row)
    
    df_results = pd.DataFrame(results_list)
    if len(sub_datasets) > 1:
        sub_dataset = 'full'
    if len(models) > 1:
        model = 'full'
    if len(wavelet_types) > 1:
        wavelet_type = 'W12'
    if len(scattering_types) > 1:
        scattering_type = 'blis_mod'
    layer_list = ','.join(map(str, args.layer_list))

    if args.ignore_graph:
        save_name = f'{args.dataset}_{sub_dataset}_{wavelet_type}_{scattering_type}_{args.task_type}_{layer_list}_{args.largest_scale}_ignore_graph.csv'
    else:
        save_name = f'{args.dataset}_{sub_dataset}_{wavelet_type}_{scattering_type}_{args.task_type}_{layer_list}_{args.largest_scale}.csv'
    df_results.to_csv(os.path.join('run_results', save_name), index = False)

    #Example : python classify_scattering.py --dataset=traffic --largest_scale=4 --sub_dataset=PEMS04 --scattering_type=blis --task_type=DAY
    #Example : python classify_scattering.py --dataset=partly_cloudy --sub_dataset=0001 --largest_scale=4 --scattering_type=blis --task_type=EMOTION3 --moment_list 1 --layer_list 1 2 3 --model SVC
    #Example: python classify_scattering.py --dataset synthetic --sub_dataset full --largest_scale 4 --scattering_type modulus --task_type PLUSMINUS --moment_list 1 --layer_list 1 2 --model LR
