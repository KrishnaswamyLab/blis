import os
import numpy as np
import torch

from new_class import GraphScattering


def find_traffic_datasets(data_root):
    datasets = []

    for name in sorted(os.listdir(data_root)):
        folder = os.path.join(data_root, name)

        if not os.path.isdir(folder):
            continue

        adjacency_path = os.path.join(folder, "adjacency_matrix.npy")
        signal_path = os.path.join(folder, "graph_signals.npy")

        if os.path.exists(adjacency_path) and os.path.exists(signal_path):
            datasets.append(name)

    return datasets


def find_old_moment_dir(old_root, sub_dataset, layer_num):
    matches = []

    for root, dirs, files in os.walk(old_root):
        if os.path.basename(root) == f"layer_{layer_num}":
            moment_1 = os.path.join(root, "moment_1.npy")
            moment_2 = os.path.join(root, "moment_2.npy")
            moment_3 = os.path.join(root, "moment_3.npy")

            if (
                os.path.exists(moment_1)
                and os.path.exists(moment_2)
                and os.path.exists(moment_3)
            ):
                parent = os.path.dirname(root)

                if sub_dataset in parent:
                    matches.append(parent)

    matches = sorted(matches)

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No old moment directory found for {sub_dataset}, layer_{layer_num} inside {old_root}"
        )

    if len(matches) > 1:
        print(f"\nWARNING: multiple old output folders found for {sub_dataset}, layer_{layer_num}")
        for i, m in enumerate(matches):
            print(f"{i}: {m}")
        print(f"Using first match: {matches[0]}")

    return matches[0]


def load_old_moments(old_save_dir, layer_num, Q):
    arrays = []

    for q in range(1, Q + 1):
        path = os.path.join(old_save_dir, f"layer_{layer_num}", f"moment_{q}.npy")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing old moment file: {path}")

        arr = np.load(path)
        arrays.append(arr)

    old_B = np.stack(arrays, axis=-1)

    return old_B


def compare_outputs(old_B, new_B, dataset, layer_num, atol=1e-8, rtol=1e-5):
    shape_match = old_B.shape == new_B.shape

    result = {
        "dataset": dataset,
        "layer": f"layer_{layer_num}",
        "old_shape": str(old_B.shape),
        "new_shape": str(new_B.shape),
        "shape_match": shape_match,
        "np_allclose": False,
        "torch_allclose": False,
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "relative_error": None,
    }

    if not shape_match:
        return result

    old_B = old_B.astype(np.float64)
    new_B = new_B.astype(np.float64)

    diff = new_B - old_B

    max_abs_diff = np.max(np.abs(diff))
    mean_abs_diff = np.mean(np.abs(diff))

    old_norm = np.linalg.norm(old_B.reshape(-1))
    diff_norm = np.linalg.norm(diff.reshape(-1))

    if old_norm == 0:
        relative_error = diff_norm
    else:
        relative_error = diff_norm / old_norm

    np_close = np.allclose(new_B, old_B, atol=atol, rtol=rtol)

    old_torch = torch.from_numpy(old_B).double()
    new_torch = torch.from_numpy(new_B).double()

    torch_close = torch.allclose(
        new_torch,
        old_torch,
        atol=atol,
        rtol=rtol,
    )

    result["np_allclose"] = bool(np_close)
    result["torch_allclose"] = bool(torch_close)
    result["max_abs_diff"] = float(max_abs_diff)
    result["mean_abs_diff"] = float(mean_abs_diff)
    result["relative_error"] = float(relative_error)

    if not np_close:
        bad = np.where(np.abs(diff) > (atol + rtol * np.abs(old_B)))

        if bad[0].size > 0:
            idx = tuple(dim[0] for dim in bad)
            result["first_mismatch_index"] = str(idx)
            result["old_value"] = float(old_B[idx])
            result["new_value"] = float(new_B[idx])
            result["difference"] = float(diff[idx])

    return result


def compute_new_all_features(gs, A_real, X_raw, largest_scale, highest_moment, layer_num):
    if X_raw.ndim == 2:
        X_raw = X_raw[:, :, None]

    T, N, F = X_raw.shape

    outputs = []

    for f in range(F):
        X_f = X_raw[:, :, f].T

        B_f = gs.calculate_scattering(
            A_real,
            X_f,
            J=largest_scale,
            Q=highest_moment,
            layer_num=layer_num,
        )

        outputs.append(B_f)

    new_B = np.concatenate(outputs, axis=2)

    return new_B


def compare_one_dataset_one_layer(
    data_root,
    old_root,
    sub_dataset,
    layer_num,
    largest_scale=4,
    highest_moment=3,
    atol=1e-8,
    rtol=1e-5,
):
    data_path = os.path.join(data_root, sub_dataset)

    label_path = os.path.join(data_path, "DAY", "label.npy")
    adjacency_path = os.path.join(data_path, "adjacency_matrix.npy")
    signal_path = os.path.join(data_path, "graph_signals.npy")

    gs = GraphScattering(label_path, adjacency_path, signal_path)

    A_real, X_real, labels_real = gs.load_data()

    new_B = compute_new_all_features(
        gs,
        A_real,
        X_real,
        largest_scale=largest_scale,
        highest_moment=highest_moment,
        layer_num=layer_num,
    )

    old_save_dir = find_old_moment_dir(
        old_root=old_root,
        sub_dataset=sub_dataset,
        layer_num=layer_num,
    )

    old_B = load_old_moments(
        old_save_dir=old_save_dir,
        layer_num=layer_num,
        Q=highest_moment,
    )

    result = compare_outputs(
        old_B=old_B,
        new_B=new_B,
        dataset=sub_dataset,
        layer_num=layer_num,
        atol=atol,
        rtol=rtol,
    )

    return result


def print_summary(all_results):
    print("\n" + "=" * 120)
    print("FINAL SUMMARY")
    print("=" * 120)

    print(
        f"{'dataset':<10} "
        f"{'layer':<8} "
        f"{'shape':<8} "
        f"{'np_close':<10} "
        f"{'torch_close':<12} "
        f"{'max_abs_diff':<16} "
        f"{'mean_abs_diff':<16} "
        f"{'relative_error':<16}"
    )

    print("-" * 120)

    for r in all_results:
        if "error" in r:
            print(f"{r['dataset']:<10} {r['layer']:<8} ERROR: {r['error']}")
            continue

        max_abs_diff = r["max_abs_diff"]
        mean_abs_diff = r["mean_abs_diff"]
        relative_error = r["relative_error"]

        if max_abs_diff is None:
            max_abs_diff_str = "None"
        else:
            max_abs_diff_str = f"{max_abs_diff:.6e}"

        if mean_abs_diff is None:
            mean_abs_diff_str = "None"
        else:
            mean_abs_diff_str = f"{mean_abs_diff:.6e}"

        if relative_error is None:
            relative_error_str = "None"
        else:
            relative_error_str = f"{relative_error:.6e}"

        print(
            f"{r['dataset']:<10} "
            f"{r['layer']:<8} "
            f"{str(r['shape_match']):<8} "
            f"{str(r['np_allclose']):<10} "
            f"{str(r['torch_allclose']):<12} "
            f"{max_abs_diff_str:<16} "
            f"{mean_abs_diff_str:<16} "
            f"{relative_error_str:<16}"
        )


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_root = os.path.join(base_dir, "data", "traffic")
    old_root = os.path.join(base_dir, "blis", "data", "traffic")

    largest_scale = 4
    highest_moment = 3

    layers_to_compare = [1, 2, 3]

    datasets = find_traffic_datasets(data_root)

    print("\nFound traffic datasets:")
    for d in datasets:
        print(f"  {d}")

    all_results = []

    for sub_dataset in datasets:
        for layer_num in layers_to_compare:
            print("\n" + "#" * 90)
            print(f"RUNNING {sub_dataset}, layer_{layer_num}")
            print("#" * 90)

            try:
                result = compare_one_dataset_one_layer(
                    data_root=data_root,
                    old_root=old_root,
                    sub_dataset=sub_dataset,
                    layer_num=layer_num,
                    largest_scale=largest_scale,
                    highest_moment=highest_moment,
                    atol=1e-8,
                    rtol=1e-5,
                )

                all_results.append(result)

            except Exception as e:
                all_results.append(
                    {
                        "dataset": sub_dataset,
                        "layer": f"layer_{layer_num}",
                        "error": str(e),
                    }
                )

    print_summary(all_results)


if __name__ == "__main__":
    main()