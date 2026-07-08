import os
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
import torch

from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from new_class import GraphScattering


def make_graph_scattering(device):
    try:
        gs = GraphScattering(device=device)
    except TypeError:
        gs = GraphScattering()

    return gs


def load_moment_features(
    data_dir,
    dataset,
    sub_dataset,
    wavelet_type,
    largest_scale,
    layer_list,
    moment_list,
):
    base_dir = os.path.join(
        data_dir,
        dataset,
        sub_dataset,
        "processed",
        "blis",
        wavelet_type,
        f"largest_scale_{largest_scale}",
    )

    features = []

    for layer in layer_list:
        for moment in moment_list:
            path = os.path.join(
                base_dir,
                f"layer_{layer}",
                f"moment_{moment}.npy",
            )

            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing moment file: {path}")

            print(f"Loading: {path}")

            X = np.load(path)
            X = X.reshape(X.shape[0], -1)
            features.append(X)

    X_features = np.concatenate(features, axis=1)
    X_features = X_features.astype(np.float32)

    return X_features


def load_labels(data_dir, dataset, sub_dataset, task_type):
    label_path = os.path.join(
        data_dir,
        dataset,
        sub_dataset,
        task_type,
        "label.npy",
    )

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing label file: {label_path}")

    y = np.load(label_path)

    classes = np.unique(y)
    class_to_id = {c: i for i, c in enumerate(classes)}

    y = np.array([class_to_id[c] for c in y], dtype=np.int64)

    return y


def split_like_blis(X_features, y, seed):
    all_idx = np.arange(len(X_features))

    train_idx, val_test_idx = train_test_split(
        all_idx,
        test_size=0.30,
        random_state=seed,
    )

    val_idx, test_idx = train_test_split(
        val_test_idx,
        test_size=0.50,
        random_state=seed,
    )

    train_val_idx = np.concatenate([train_idx, val_idx], axis=0)

    X_train = X_features[train_val_idx]
    y_train = y[train_val_idx]

    X_test = X_features[test_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def set_seed(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def torch_fit_pca(X_train, pca_variance):
    """
    Match the BLIS-Net behavior:

    pca_variance == 1      -> no PCA
    0 < pca_variance < 1   -> keep enough components for that variance
    pca_variance > 1       -> use int(pca_variance) fixed components
    """
    if pca_variance == 1:
        return X_train, None, -1

    if pca_variance <= 0:
        raise ValueError("pca_variance must be positive.")

    mean = X_train.mean(dim=0, keepdim=True)
    X_train_centered = X_train - mean

    U, S, Vh = torch.linalg.svd(
        X_train_centered,
        full_matrices=False,
    )

    max_components = Vh.shape[0]

    if pca_variance > 1:
        num_components = int(pca_variance)
        num_components = min(num_components, max_components)

    else:
        eigvals = S ** 2
        explained = eigvals / eigvals.sum()
        cumulative = torch.cumsum(explained, dim=0)

        num_components = int(torch.searchsorted(cumulative, pca_variance).item()) + 1
        num_components = min(num_components, max_components)

    V = Vh[:num_components].T

    X_train_pca = X_train_centered @ V

    pca_state = {
        "mean": mean,
        "V": V,
    }

    print(f"PCA components: {num_components}")

    return X_train_pca, pca_state, num_components


def torch_apply_pca(X, pca_state):
    if pca_state is None:
        return X

    return (X - pca_state["mean"]) @ pca_state["V"]


def torch_fit_standardizer(X_train, eps=1e-12):
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True)

    std = torch.where(
        std > eps,
        std,
        torch.ones_like(std),
    )

    scaler_state = {
        "mean": mean,
        "std": std,
    }

    return scaler_state


def torch_apply_standardizer(X, scaler_state):
    return (X - scaler_state["mean"]) / scaler_state["std"]


def torch_standardize_fit_transform(X_train, X_test):
    scaler_state = torch_fit_standardizer(X_train)

    X_train = torch_apply_standardizer(X_train, scaler_state)
    X_test = torch_apply_standardizer(X_test, scaler_state)

    return X_train, X_test, scaler_state


class TorchLR(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.linear = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class TorchMLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layer_sizes):
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_layer_sizes:
            layers.append(torch.nn.Linear(previous_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            previous_dim = hidden_dim

        layers.append(torch.nn.Linear(previous_dim, num_classes))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_torch_model(model_name, input_dim, num_classes, params):
    if model_name == "LR":
        return TorchLR(input_dim, num_classes)

    if model_name == "MLP":
        return TorchMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layer_sizes=params["hidden_layer_sizes"],
        )

    raise ValueError(
        f"Unknown torch model: {model_name}. "
        "SVC is intentionally left out for now."
    )


def compute_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    return metrics


def train_torch_model(
    model_name,
    X_train,
    y_train,
    params,
    epochs,
    batch_size,
    device,
    verbose=False,
):
    input_dim = X_train.shape[1]
    num_classes = int(torch.max(y_train).item()) + 1

    model = build_torch_model(
        model_name=model_name,
        input_dim=input_dim,
        num_classes=num_classes,
        params=params,
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    n_train = X_train.shape[0]

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n_train, device=device)

        model.train()
        total_loss = 0.0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)

            idx = perm[start:end]

            xb = X_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()

            scores = model(xb)
            loss = loss_fn(scores, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.shape[0]

        if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == epochs):
            print(f"Epoch {epoch}, loss = {total_loss / n_train:.6f}")

    return model


def predict_torch_model(model, X):
    model.eval()

    with torch.no_grad():
        scores = model(X)
        y_pred = torch.argmax(scores, dim=1)

    return y_pred


def get_torch_param_grid(model_name, input_dim, args):
    """
    Manual Torch grid search.

    LR approximates sklearn LogisticRegression C by using weight_decay = 1 / C.
    MLP uses the same hidden layer choices as the BLIS-Net sklearn grid.
    """
    lr_values = args.lr_values if args.lr_values is not None else [args.lr]

    if model_name == "LR":
        grid = []

        for C, lr in itertools.product([0.1, 1.0, 10.0], lr_values):
            grid.append(
                {
                    "C": C,
                    "lr": lr,
                    "weight_decay": 1.0 / C,
                }
            )

        return grid

    if model_name == "MLP":
        hidden_options = [
            (input_dim // 2, input_dim // 4),
            (input_dim // 2, input_dim // 4, input_dim // 8),
            (150, 50),
        ]

        grid = []

        for hidden_layer_sizes, lr in itertools.product(hidden_options, lr_values):
            grid.append(
                {
                    "hidden_layer_sizes": hidden_layer_sizes,
                    "activation": "relu",
                    "alpha": 0.01,
                    "lr": lr,
                    "weight_decay": 0.01,
                }
            )

        return grid

    raise ValueError(
        f"Unknown torch model for grid search: {model_name}. "
        "SVC is intentionally left out for now."
    )


def torch_cross_val_score(
    model_name,
    X_train,
    y_train,
    params,
    epochs,
    batch_size,
    cv_folds,
    device,
    seed,
):
    y_train_np = y_train.detach().cpu().numpy()

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=seed,
    )

    fold_scores = []

    for fold_id, (inner_train_idx, inner_val_idx) in enumerate(
        cv.split(np.zeros(len(y_train_np)), y_train_np),
        start=1,
    ):
        inner_train_idx = torch.as_tensor(
            inner_train_idx,
            dtype=torch.long,
            device=device,
        )

        inner_val_idx = torch.as_tensor(
            inner_val_idx,
            dtype=torch.long,
            device=device,
        )

        X_inner_train = X_train[inner_train_idx]
        y_inner_train = y_train[inner_train_idx]

        X_inner_val = X_train[inner_val_idx]
        y_inner_val = y_train[inner_val_idx]

        # StandardScaler is inside the BLIS-Net sklearn Pipeline,
        # so we fit it inside each CV fold.
        X_inner_train, X_inner_val, _ = torch_standardize_fit_transform(
            X_inner_train,
            X_inner_val,
        )

        set_seed(seed + fold_id, device)

        model = train_torch_model(
            model_name=model_name,
            X_train=X_inner_train,
            y_train=y_inner_train,
            params=params,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            verbose=False,
        )

        y_pred = predict_torch_model(model, X_inner_val)

        score = accuracy_score(
            y_inner_val.detach().cpu().numpy(),
            y_pred.detach().cpu().numpy(),
        )

        fold_scores.append(score)

    return float(np.mean(fold_scores))


def train_torch_with_grid_search(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    args,
    device,
    seed,
):
    input_dim = X_train.shape[1]

    param_grid = get_torch_param_grid(
        model_name=model_name,
        input_dim=input_dim,
        args=args,
    )

    best_score = -np.inf
    best_params = None

    print(f"Manual Torch grid search for {model_name}")
    print(f"Number of settings: {len(param_grid)}")

    for params in param_grid:
        cv_score = torch_cross_val_score(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            params=params,
            epochs=args.epochs,
            batch_size=args.batch_size,
            cv_folds=args.cv_folds,
            device=device,
            seed=seed,
        )

        print(f"params={params}, cv_accuracy={cv_score:.6f}")

        if cv_score > best_score:
            best_score = cv_score
            best_params = params

    print(f"Best params: {best_params}")
    print(f"Best CV accuracy: {best_score:.6f}")

    # Refit StandardScaler on the full training split before final model,
    # matching GridSearchCV refit behavior.
    X_train_scaled, X_test_scaled, _ = torch_standardize_fit_transform(
        X_train,
        X_test,
    )

    set_seed(seed, device)

    model = train_torch_model(
        model_name=model_name,
        X_train=X_train_scaled,
        y_train=y_train,
        params=best_params,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        verbose=True,
    )

    y_pred = predict_torch_model(model, X_test_scaled)

    y_true_np = y_test.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    metrics = compute_metrics(y_true_np, y_pred_np)
    metrics["best_cv_accuracy"] = best_score
    metrics["best_params"] = str(best_params)

    return metrics


def train_torch_without_grid_search(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    args,
    device,
    seed,
):
    if model_name == "LR":
        params = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }

    elif model_name == "MLP":
        params = {
            "hidden_layer_sizes": (256, 128),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }

    else:
        raise ValueError(
            f"Unknown torch model: {model_name}. "
            "SVC is intentionally left out for now."
        )

    X_train_scaled, X_test_scaled, _ = torch_standardize_fit_transform(
        X_train,
        X_test,
    )

    set_seed(seed, device)

    model = train_torch_model(
        model_name=model_name,
        X_train=X_train_scaled,
        y_train=y_train,
        params=params,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        verbose=True,
    )

    y_pred = predict_torch_model(model, X_test_scaled)

    y_true_np = y_test.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    metrics = compute_metrics(y_true_np, y_pred_np)
    metrics["best_cv_accuracy"] = np.nan
    metrics["best_params"] = str(params)

    return metrics


def get_xgb_param_grid():
    grid = []

    for n_estimators, learning_rate in itertools.product([50, 100], [0.05, 0.1]):
        grid.append(
            {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": 6,
            }
        )

    return grid


def build_xgb_model(params, device):
    from xgboost import XGBClassifier

    xgb_device = "cuda" if device.type == "cuda" else "cpu"

    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device=xgb_device,
    )

    return model


def xgb_cross_val_score(
    X_train,
    y_train,
    params,
    cv_folds,
    device,
    seed,
):
    X_train_np = X_train.detach().cpu().numpy()
    y_train_np = y_train.detach().cpu().numpy()

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=seed,
    )

    fold_scores = []

    for fold_id, (inner_train_idx, inner_val_idx) in enumerate(
        cv.split(X_train_np, y_train_np),
        start=1,
    ):
        X_inner_train = torch.as_tensor(
            X_train_np[inner_train_idx],
            dtype=torch.float32,
            device=device,
        )

        X_inner_val = torch.as_tensor(
            X_train_np[inner_val_idx],
            dtype=torch.float32,
            device=device,
        )

        # StandardScaler is inside the BLIS-Net sklearn Pipeline,
        # so we fit it inside each CV fold.
        X_inner_train, X_inner_val, _ = torch_standardize_fit_transform(
            X_inner_train,
            X_inner_val,
        )

        X_inner_train_np = X_inner_train.detach().cpu().numpy()
        X_inner_val_np = X_inner_val.detach().cpu().numpy()

        y_inner_train_np = y_train_np[inner_train_idx]
        y_inner_val_np = y_train_np[inner_val_idx]

        model = build_xgb_model(
            params=params,
            device=device,
        )

        model.fit(X_inner_train_np, y_inner_train_np)

        y_pred = model.predict(X_inner_val_np)

        score = accuracy_score(y_inner_val_np, y_pred)
        fold_scores.append(score)

    return float(np.mean(fold_scores))


def train_xgb_with_grid_search(
    X_train,
    y_train,
    X_test,
    y_test,
    args,
    device,
    seed,
):
    param_grid = get_xgb_param_grid()

    best_score = -np.inf
    best_params = None

    print("Manual XGB grid search")
    print(f"Number of settings: {len(param_grid)}")

    for params in param_grid:
        cv_score = xgb_cross_val_score(
            X_train=X_train,
            y_train=y_train,
            params=params,
            cv_folds=args.cv_folds,
            device=device,
            seed=seed,
        )

        print(f"params={params}, cv_accuracy={cv_score:.6f}")

        if cv_score > best_score:
            best_score = cv_score
            best_params = params

    print(f"Best params: {best_params}")
    print(f"Best CV accuracy: {best_score:.6f}")

    X_train_scaled, X_test_scaled, _ = torch_standardize_fit_transform(
        X_train,
        X_test,
    )

    X_train_np = X_train_scaled.detach().cpu().numpy()
    X_test_np = X_test_scaled.detach().cpu().numpy()

    y_train_np = y_train.detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy()

    model = build_xgb_model(
        params=best_params,
        device=device,
    )

    model.fit(X_train_np, y_train_np)

    y_pred = model.predict(X_test_np)

    metrics = compute_metrics(y_test_np, y_pred)
    metrics["best_cv_accuracy"] = best_score
    metrics["best_params"] = str(best_params)

    return metrics


def train_xgb_without_grid_search(
    X_train,
    y_train,
    X_test,
    y_test,
    args,
    device,
):
    params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
    }

    X_train_scaled, X_test_scaled, _ = torch_standardize_fit_transform(
        X_train,
        X_test,
    )

    X_train_np = X_train_scaled.detach().cpu().numpy()
    X_test_np = X_test_scaled.detach().cpu().numpy()

    y_train_np = y_train.detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy()

    model = build_xgb_model(
        params=params,
        device=device,
    )

    model.fit(X_train_np, y_train_np)

    y_pred = model.predict(X_test_np)

    metrics = compute_metrics(y_test_np, y_pred)
    metrics["best_cv_accuracy"] = np.nan
    metrics["best_params"] = str(params)

    return metrics


def run_one_seed(
    X_features,
    y,
    model_name,
    seed,
    args,
    device,
):
    set_seed(seed, device)

    X_train_np, X_test_np, y_train_np, y_test_np = split_like_blis(
        X_features,
        y,
        seed,
    )

    X_train = torch.as_tensor(
        X_train_np,
        dtype=torch.float32,
        device=device,
    )

    X_test = torch.as_tensor(
        X_test_np,
        dtype=torch.float32,
        device=device,
    )

    y_train = torch.as_tensor(
        y_train_np,
        dtype=torch.long,
        device=device,
    )

    y_test = torch.as_tensor(
        y_test_np,
        dtype=torch.long,
        device=device,
    )

    # Match BLIS-Net order:
    # flatten -> PCA if requested -> StandardScaler inside CV/final training.
    X_train, pca_state, n_pca = torch_fit_pca(
        X_train,
        args.pca_variance,
    )

    X_test = torch_apply_pca(
        X_test,
        pca_state,
    )

    if model_name == "SVC":
        raise ValueError("SVC is intentionally left out for now.")

    if model_name == "XGB":
        if args.no_grid_search:
            metrics = train_xgb_without_grid_search(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                args=args,
                device=device,
            )

        else:
            metrics = train_xgb_with_grid_search(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                args=args,
                device=device,
                seed=seed,
            )

    else:
        if args.no_grid_search:
            metrics = train_torch_without_grid_search(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                args=args,
                device=device,
                seed=seed,
            )

        else:
            metrics = train_torch_with_grid_search(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                args=args,
                device=device,
                seed=seed,
            )

    metrics["n_pca"] = n_pca

    return metrics


def summarize_results(rows, metric_names):
    df = pd.DataFrame(rows)

    summary_rows = []

    group_cols = [
        "dataset",
        "task",
        "wavelet_type",
        "largest_scale",
        "layers",
        "moments",
        "model",
    ]

    for keys, group in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))

        for metric in metric_names:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=0)

        row["n_pca_mean"] = group["n_pca"].mean()

        if "best_cv_accuracy" in group.columns:
            row["best_cv_accuracy_mean"] = group["best_cv_accuracy"].mean()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    return summary_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="traffic")

    parser.add_argument(
        "--sub_datasets",
        nargs="+",
        type=str,
        default=["PEMS03", "PEMS04", "PEMS07", "PEMS08"],
    )

    parser.add_argument(
        "--task_types",
        nargs="+",
        type=str,
        default=["HOUR", "DAY", "WEEK"],
    )

    parser.add_argument("--wavelet_type", type=str, default="W2")
    parser.add_argument("--largest_scale", type=int, default=4)
    parser.add_argument("--highest_moment", type=int, default=3)

    parser.add_argument("--layer_list", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--moment_list", nargs="+", type=int, default=[1])

    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["LR", "MLP", "XGB"],
        help="Torch/XGB models to run. SVC is intentionally left out for now.",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 56],
    )

    parser.add_argument(
        "--pca_variance",
        type=float,
        default=1.0,
        help="PCA behavior: 1 means no PCA, 0.99 keeps 99 percent variance, >1 uses fixed components.",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument(
        "--lr_values",
        nargs="+",
        type=float,
        default=None,
        help="Optional learning-rate values for manual Torch grid search.",
    )

    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--cv_folds", type=int, default=3)
    parser.add_argument("--no_grid_search", action="store_true")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--skip_moments", action="store_true")

    parser.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()

    if args.sub_datasets == ["full"]:
        args.sub_datasets = ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 90)
    print("BLIS MOMENTS FROM new_class.py + TORCH CLASSIFICATION")
    print("SVC intentionally left out for now")
    print("=" * 90)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    gs = make_graph_scattering(device=device)

    all_rows = []

    for sub_dataset in args.sub_datasets:
        print("=" * 90)
        print(f"DATASET: {sub_dataset}")
        print("=" * 90)

        data_path = os.path.join(
            args.data_dir,
            args.dataset,
            sub_dataset,
        )

        adjacency_path = os.path.join(data_path, "adjacency_matrix.npy")
        signal_path = os.path.join(data_path, "graph_signals.npy")

        if not os.path.exists(adjacency_path):
            raise FileNotFoundError(f"Missing adjacency file: {adjacency_path}")

        if not os.path.exists(signal_path):
            raise FileNotFoundError(f"Missing graph signal file: {signal_path}")

        A = np.load(adjacency_path).astype(np.float32)
        X = np.load(signal_path).astype(np.float32)

        print(f"A shape: {A.shape}")
        print(f"X shape: {X.shape}")

        if not args.skip_moments:
            gs.save_scattering_moments(
                A=A,
                X=X,
                data_dir=args.data_dir,
                dataset=args.dataset,
                sub_dataset=sub_dataset,
                scattering_type="blis",
                wavelet_type=args.wavelet_type,
                largest_scale=args.largest_scale,
                highest_moment=args.highest_moment,
                layer_list=args.layer_list,
                force_recompute=args.force_recompute,
            )

        X_features = load_moment_features(
            data_dir=args.data_dir,
            dataset=args.dataset,
            sub_dataset=sub_dataset,
            wavelet_type=args.wavelet_type,
            largest_scale=args.largest_scale,
            layer_list=args.layer_list,
            moment_list=args.moment_list,
        )

        print(f"X_features shape: {X_features.shape}")

        for task_type in args.task_types:
            print("-" * 90)
            print(f"TASK: {task_type}")
            print("-" * 90)

            y = load_labels(
                data_dir=args.data_dir,
                dataset=args.dataset,
                sub_dataset=sub_dataset,
                task_type=task_type,
            )

            if len(y) != X_features.shape[0]:
                raise ValueError(
                    f"Label/features mismatch for {sub_dataset} {task_type}: "
                    f"len(y)={len(y)}, X_features.shape[0]={X_features.shape[0]}"
                )

            print(f"y shape: {y.shape}")
            print(f"classes: {np.unique(y)}")

            for model_name in args.models:
                if model_name == "SVC":
                    raise ValueError("SVC is intentionally left out for now.")

                print("-" * 90)
                print(f"MODEL: {model_name}")
                print("-" * 90)

                for seed in args.seeds:
                    print(f"Seed: {seed}")

                    metrics = run_one_seed(
                        X_features=X_features,
                        y=y,
                        model_name=model_name,
                        seed=seed,
                        args=args,
                        device=device,
                    )

                    row = {
                        "dataset": sub_dataset,
                        "task": task_type,
                        "wavelet_type": args.wavelet_type,
                        "largest_scale": args.largest_scale,
                        "layers": " ".join(map(str, args.layer_list)),
                        "moments": " ".join(map(str, args.moment_list)),
                        "model": model_name,
                        "seed": seed,
                        "pca_variance": args.pca_variance,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "lr_values": str(args.lr_values),
                        "weight_decay": args.weight_decay,
                        "cv_folds": args.cv_folds,
                        "no_grid_search": args.no_grid_search,
                        "device": str(device),
                    }

                    row.update(metrics)

                    all_rows.append(row)

                    print(f"accuracy = {metrics['accuracy']:.6f}")
                    print(f"balanced_accuracy = {metrics['balanced_accuracy']:.6f}")
                    print(f"macro_f1 = {metrics['macro_f1']:.6f}")
                    print(f"weighted_f1 = {metrics['weighted_f1']:.6f}")
                    print(f"macro_precision = {metrics['macro_precision']:.6f}")
                    print(f"macro_recall = {metrics['macro_recall']:.6f}")
                    print(f"n_pca = {metrics['n_pca']}")
                    print(f"best_cv_accuracy = {metrics['best_cv_accuracy']}")
                    print(f"best_params = {metrics['best_params']}")

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
    ]

    results_df = pd.DataFrame(all_rows)
    summary_df = summarize_results(all_rows, metric_names)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sub_dataset_name = "_".join(args.sub_datasets)
    task_name = "_".join(args.task_types)
    model_name = "_".join(args.models)
    layer_name = "_".join(map(str, args.layer_list))
    moment_name = "_".join(map(str, args.moment_list))

    base_name = (
        f"torch_blis_{sub_dataset_name}_{task_name}_{model_name}_"
        f"{args.wavelet_type}_L{layer_name}_M{moment_name}_{timestamp}"
    )

    per_seed_path = os.path.join(args.results_dir, f"{base_name}_per_seed.csv")
    summary_path = os.path.join(args.results_dir, f"{base_name}_summary.csv")

    results_df.to_csv(per_seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("=" * 90)
    print(f"Saved per-seed results to: {per_seed_path}")
    print(f"Saved summary results to: {summary_path}")
    print("=" * 90)

    print(summary_df)


if __name__ == "__main__":
    main()