from blis.data import traffic
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neural_network import MLPClassifier 
from sklearn.linear_model import LogisticRegression 
import xgboost as xgb

def main(args,scattering_dict):

    for seed in [42,43,44,45,56]:

        if args.dataset == "traffic":
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = traffic.traffic_scattering_data_loader(seed=42,
                                                                                                    subdata_type=args.sub_dataset,
                                                                                                    task_type='DAY',
                                                                                                    batch_size=32,
                                                                                                    scattering_dict=scattering_dict)
        import pdb; pdb.set_trace() 
        X_train = X_train.reshape(X_train.shape[0],-1)
        X_val = X_val.reshape(X_val.shape[0],-1)
        X_test = X_test.reshape(X_test.shape[0],-1)

        if args.model = "LR":
            base_model = LogisticRegression() 
        if args.model = "SVC":
            base_model = SVC() 
        if args.model = "KNN":
            base_model = KNeighborsClassifier()
        if args.model = "MLP":
            base_model = MLPClassifier() 
        if args.model = "RF":
            base_model = RandomForest()
        if args.model = "XGB":
            base_model = xgb.XGBClassifier()

        # Create a pipeline that first applies the standard scaler, then the model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', base_model)
        ])

        # Define hyperparameters grid for each model (with 'model__' prefix for parameters)
        if isinstance(base_model, RandomForestClassifier):
            param_grid = {
                'model__n_estimators': [10, 50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10]
            }
        elif isinstance(base_model, SVC):
            param_grid = {
                'model__C': [0.01, 0.1, 1, 10],
                'model__kernel': ['linear', 'rbf'],
                'model__gamma': ['scale','auto', .001, .01, .1, 1, 10]
            }
        elif isinstance(base_model, KNeighborsClassifier):
            param_grid = {
                'model__n_neighbors': [3, 5, 7, 11],
                'model__weights': ['uniform', 'distance']
            }
        elif isinstance(base_model, MLPClassifier):
            param_grid = {
                'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'model__activation': ['relu']
            }
        elif isinstance(base_model, LogisticRegression):
            param_grid = {
                'model__C': [0.1, 1, 10, 100],
                'model__solver': ['newton-cg', 'lbfgs', 'liblinear']
            }
        elif isinstance(base_model, xgb.XGBClassifier):
            param_grid = {
                'model__n_estimators': [50, 100, 150],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [3, 5, 7]
            }

        clf = GridSearchCV(pipeline, param_grid, cv = 3)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Best parameters found: "clf.best_params_)
        print("Train score : ", clf.score(X_train, y_train))
        print("Test score : ", clf.score(X_test, y_test))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse arguments for the program.")

    parser.add_argument("--scattering_type", choices=['blis', 'modulus'], help="Type of scattering: 'blis' or 'modulus'.")
    parser.add_argument("--largest_scale", type=int, help="Largest (dyadic) scale as a positive integer.")
    parser.add_argument("--highest_moment", type=int, default=1, help="Highest moment as a positive integer. Defaults to 1.")
    parser.add_argument("--dataset", choices=['traffic', 'partly_cloudy', 'synthetic'], help="Dataset: 'traffic' or 'partly_cloudy' or 'synthetic'.")
    parser.add_argument("--sub_dataset", help="Sub-dataset value depending on the dataset chosen.")
    parser.add_argument("--num_layers", type=int, default=2, help="Largest scattering layer")
    parser.add_argument("--model", choices=['RF, SVC, KNN, MLP, LR', "XGB"], type=str, default="LR", help="Classification model to use. Options: 'RF, SVC, KNN, MLP, LR, XGB'")

    args = parser.parse_args()

    scattering_dict = {"scattering_type": args.scattering_type,
                       "scale_type": f"largest_scale_{args.largest_scale}",
                       "layers": [i+1 for i in range(args.num_layers)],
                       "moments" : [i+1 for i in range(args.highest_moment)]}
    
    main(args,scattering_dict)

    #Example : python classify_scattering.py --dataset=traffic --largest_scale=4 --sub_dataset=PEMS04 --scattering_type=blis


