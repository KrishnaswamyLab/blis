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

import warnings
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
                                                                                                    scattering_dict=scattering_dict)
        elif args.dataset == "partly_cloudy":
            (X_train, y_train),  (X_test, y_test) = cloudy.cloudy_scattering_data_loader(seed=seed,
                                                                                                    subdata_type=args.sub_dataset,
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
        n_comp = -1
        if (args.PCA_variance != 1):
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
                'model__gamma': ['scale','auto', .01, .1, 1]
            }
        elif isinstance(base_model, KNeighborsClassifier):
            param_grid = {
                'model__n_neighbors': [3, 5, 7],
                'model__weights': ['uniform', 'distance']
            }
        elif isinstance(base_model, MLPClassifier):
            param_grid = {
                'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
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
                'model__n_estimators': [50, 100, 150],
                'model__learning_rate': [0.01, 0.05, 0.1]
            }

        clf = GridSearchCV(pipeline, param_grid, cv = 3)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        #print("Best parameters found: ",clf.best_params_)
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
    
    final_score, final_stdev, n_comp = run_classifier_scattering(args,scattering_dict)
    print(f"{final_score},{final_stdev}, {n_comp}")

    #Example : python classify_scattering.py --dataset=traffic --largest_scale=4 --sub_dataset=PEMS04 --scattering_type=blis --task_type=DAY
    #Example : python classify_scattering.py --dataset=partly_cloudy --sub_dataset=0001 --largest_scale=4 --scattering_type=blis --task_type=EMOTION3 --moment_list 1 --layer_list 1 2 3 --model SVC

