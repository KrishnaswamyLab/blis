import numpy as np 
import os 
import argparse 
import configparser 
import torch 

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neural_network import MLPClassifier 
from sklearn.linear_model import LogisticRegression 
import xgboost as xgb

# Assuming the dataloaders are already defined
X_train, y_train = train_dataloader.tensors()
X_val, y_val = val_dataloader.tensors()
X_test, y_test = test_dataloader.tensors()

# Choose the model you'd like to use. For this example, I'll use RandomForest
base_model = RandomForestClassifier()

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

# Use GridSearchCV to find the best hyperparameters
clf = GridSearchCV(pipeline, param_grid, cv=3)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Report classification accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy on Test Set: {accuracy * 100:.2f}%")
