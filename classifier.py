import os
import argparse
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from new_class import GraphScattering


def get_model_and_grid(model_name, input_dim):
    if model_name == "LR":
        model = LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            )
        grid = {
            "clf__C": [0.1, 1, 10],
            }
        return model, grid

    if model_name == "RF":
        model = RandomForestClassifier(n_jobs=-1)
        grid = {
            "clf__n_estimators": [50, 100, 150],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5],
        }
        return model, grid

    if model_name == "SVC":
        model = SVC()
        grid = {
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["linear", "rbf"],
            "clf__gamma": ["scale", "auto", 0.1, 1, 10],
        }
        return model, grid

    if model_name == "KNN":
        model = KNeighborsClassifier(n_jobs=-1)
        grid = {
            "clf__n_neighbors": [3, 5, 7],
            "clf__weights": ["uniform", "distance"],
        }
        return model, grid

    if model_name == "MLP":
        model = MLPClassifier(max_iter=500)
        grid = {
            "clf__hidden_layer_sizes": [
                (150, 50),
                (256, 128),
            ],
            "clf__activation": ["relu"],
            "clf__alpha": [0.01],
        }
        return model, grid

    if model_name == "XGB":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")

        model = XGBClassifier(
            eval_metric="mlogloss",
            n_jobs=-1,
            tree_method="hist",
        )

        grid = {
            "clf__n_estimators": [50, 100],
            "clf__learning_rate": [0.05, 0.1],
        }
        return model, grid

    raise ValueError(f"Unknown model: {model_name}")


def compute_graphscattering_one_layer(
    gs,
    A_real,
    X_raw,
    largest_scale,
    highest_moment,
    layer_num,
    moment_list,
):
    if X_raw.ndim == 2:
        X_raw = X_raw[:, :, None]

    T, N, F = X_raw.shape
    outputs = []

    for f in range(F):
        print(f"Computing layer_{layer_num}, channel {f + 1}/{F}")

        X_f = X_raw[:, :, f].T

        B_f = gs.calculate_scattering(
            A_real,
            X_f,
            J=largest_scale,
            Q=highest_moment,
            layer_num=layer_num,
        )

        moment_indices = [m - 1 for m in moment_list]
        B_f = B_f[:, :, :, moment_indices]

        outputs.append(B_f)

    B = np.concatenate(outputs, axis=2)

    return B


def build_feature_matrix(
    gs,
    A_real,
    X_real,
    largest_scale,
    layer_list,
    moment_list,
):
    highest_moment = max(moment_list)
    feature_blocks = []

    for layer_num in layer_list:
        B = compute_graphscattering_one_layer(
            gs=gs,
            A_real=A_real,
            X_raw=X_real,
            largest_scale=largest_scale,
            highest_moment=highest_moment,
            layer_num=layer_num,
            moment_list=moment_list,
        )

        print(f"layer_{layer_num} B shape: {B.shape}")

        X_layer = B.reshape(B.shape[0], -1).astype(np.float32)

        print(f"layer_{layer_num} flattened shape: {X_layer.shape}")

        feature_blocks.append(X_layer)

    X_features = np.concatenate(feature_blocks, axis=1).astype(np.float32)

    return X_features


def train_one_seed(
    X_features,
    y,
    model_name,
    seed,
    pca_variance,
    test_size,
    cv,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    model, param_grid = get_model_and_grid(model_name, X_features.shape[1])

    steps = []
    steps.append(("scaler", StandardScaler()))

    if pca_variance < 1.0:
        steps.append(("pca", PCA(n_components=pca_variance, random_state=seed)))

    steps.append(("clf", model))

    pipe = Pipeline(steps)

    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    return {
        "seed": seed,
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "best_params": search.best_params_,
    }


def run_classifiers(
    X_features,
    y,
    models,
    seeds,
    pca_variance,
    test_size,
    cv,
):
    all_results = []

    for model_name in models:
        print("\n" + "=" * 90)
        print(f"MODEL: {model_name}")
        print("=" * 90)

        model_results = []

        for seed in seeds:
            print("\n" + "-" * 90)
            print(f"Running {model_name}, seed={seed}")
            print("-" * 90)

            result = train_one_seed(
                X_features=X_features,
                y=y,
                model_name=model_name,
                seed=seed,
                pca_variance=pca_variance,
                test_size=test_size,
                cv=cv,
            )

            model_results.append(result)
            all_results.append(result)

            print(f"seed={seed}")
            print(f"accuracy={result['accuracy']:.6f}")
            print(f"macro_f1={result['macro_f1']:.6f}")
            print(f"weighted_f1={result['weighted_f1']:.6f}")
            print(f"best_params={result['best_params']}")

        accuracies = np.array([r["accuracy"] for r in model_results])
        macro_f1s = np.array([r["macro_f1"] for r in model_results])
        weighted_f1s = np.array([r["weighted_f1"] for r in model_results])

        print("\n" + "*" * 90)
        print(f"SUMMARY FOR {model_name}")
        print("*" * 90)
        print(f"accuracy mean: {accuracies.mean():.6f}")
        print(f"accuracy std:  {accuracies.std():.6f}")
        print(f"macro_f1 mean: {macro_f1s.mean():.6f}")
        print(f"macro_f1 std:  {macro_f1s.std():.6f}")
        print(f"weighted_f1 mean: {weighted_f1s.mean():.6f}")
        print(f"weighted_f1 std:  {weighted_f1s.std():.6f}")

    return all_results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="traffic")
    parser.add_argument("--sub_dataset", type=str, default="PEMS08")
    parser.add_argument("--task_type", type=str, default="DAY")

    parser.add_argument("--largest_scale", type=int, default=4)
    parser.add_argument("--layer_list", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--moment_list", nargs="+", type=int, default=[1, 2, 3])

    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["LR", "RF", "SVC", "KNN", "MLP", "XGB"],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 56],
    )

    parser.add_argument("--pca_variance", type=float, default=0.99)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=3)

    args = parser.parse_args()

    if len(args.models) == 1 and args.models[0] == "all":
        args.models = ["LR", "RF", "SVC", "KNN", "MLP", "XGB"]

    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(
        base_dir,
        "data",
        args.dataset,
        args.sub_dataset,
    )

    label_path = os.path.join(data_path, args.task_type, "label.npy")
    adjacency_path = os.path.join(data_path, "adjacency_matrix.npy")
    signal_path = os.path.join(data_path, "graph_signals.npy")

    print("\n" + "=" * 90)
    print("GRAPHSCATTERING IN-MEMORY CLASSIFIER TEST")
    print("=" * 90)
    print(f"data_path: {data_path}")
    print(f"label_path: {label_path}")
    print(f"adjacency_path: {adjacency_path}")
    print(f"signal_path: {signal_path}")
    print(f"layers: {args.layer_list}")
    print(f"moments: {args.moment_list}")
    print(f"models: {args.models}")
    print(f"seeds: {args.seeds}")

    gs = GraphScattering(label_path, adjacency_path, signal_path)

    A_real, X_real, labels_real = gs.load_data()

    X_features = build_feature_matrix(
        gs=gs,
        A_real=A_real,
        X_real=X_real,
        largest_scale=args.largest_scale,
        layer_list=args.layer_list,
        moment_list=args.moment_list,
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels_real)

    print("\n" + "=" * 90)
    print("FEATURE INFO")
    print("=" * 90)
    print(f"X_features shape: {X_features.shape}")
    print(f"y shape: {y.shape}")
    print(f"number of classes: {len(np.unique(y))}")

    all_results = run_classifiers(
        X_features=X_features,
        y=y,
        models=args.models,
        seeds=args.seeds,
        pca_variance=args.pca_variance,
        test_size=args.test_size,
        cv=args.cv,
    )

    print("\n" + "=" * 90)
    print("FINAL RESULTS")
    print("=" * 90)

    for model_name in args.models:
        model_results = [r for r in all_results if r["model"] == model_name]

        if len(model_results) == 0:
            continue

        accuracies = np.array([r["accuracy"] for r in model_results])
        macro_f1s = np.array([r["macro_f1"] for r in model_results])
        weighted_f1s = np.array([r["weighted_f1"] for r in model_results])

        print(
            f"{model_name:<6} | "
            f"accuracy={accuracies.mean():.6f} ± {accuracies.std():.6f} | "
            f"macro_f1={macro_f1s.mean():.6f} ± {macro_f1s.std():.6f} | "
            f"weighted_f1={weighted_f1s.mean():.6f} ± {weighted_f1s.std():.6f}"
        )


if __name__ == "__main__":
    main()