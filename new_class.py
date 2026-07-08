import os
import numpy as np


class GraphScattering:
    def __init__(self, label_path=None, adjacency_path=None, signal_path=None):
        self.label_path = label_path
        self.adjacency_path = adjacency_path
        self.signal_path = signal_path

    def relu(self, x):
        return np.maximum(0, x)

    def reverse_relu(self, x):
        return np.maximum(0, -x)

    def load_data(self):
        self.labels = np.load(self.label_path) if self.label_path is not None else None
        self.A = np.load(self.adjacency_path) if self.adjacency_path is not None else None
        self.X = np.load(self.signal_path) if self.signal_path is not None else None

        if self.A is not None:
            print(f"Loaded adjacency matrix A with shape: {self.A.shape}")

        if self.X is not None:
            print(f"Loaded graph signals X with shape: {self.X.shape}")

        if self.labels is not None:
            print(f"Loaded labels with shape: {self.labels.shape}")

        return self.A, self.X, self.labels

    def get_P(self, A):
        A = np.asarray(A, dtype=float)

        d_arr = np.sum(A, axis=0)

        d_arr_inv = np.divide(
            1.0,
            d_arr,
            out=np.zeros_like(d_arr, dtype=float),
            where=d_arr != 0,
        )

        D_inv = np.diag(d_arr_inv)

        P = 0.5 * (np.eye(A.shape[0]) + A @ D_inv)

        return P

    def compute_W_2_transform(self, A, X, largest_scale, low_pass_as_wavelet=True):
        A = np.asarray(A, dtype=float)
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X[:, None]

        n_nodes = A.shape[0]

        if X.shape[0] != n_nodes:
            raise ValueError(
                f"Node mismatch: X has {X.shape[0]} rows, but A is {n_nodes}x{n_nodes}."
            )

        P = self.get_P(A)
        I = np.eye(n_nodes)

        coeffs = []

        C0 = (I - P) @ X
        coeffs.append(C0)

        a = X.copy()

        for j in range(1, largest_scale):
            pow_val = 2 ** (j - 1)

            prev = a.copy()

            for _ in range(pow_val):
                prev = P @ prev

            curr = prev.copy()

            for _ in range(pow_val):
                curr = P @ curr

            a_j = prev - curr

            coeffs.append(a_j)

        if low_pass_as_wavelet:
            if largest_scale == 0:
                low_pass_steps = 0
            else:
                low_pass_steps = 2 ** (largest_scale - 1)

            low_pass = X.copy()

            for _ in range(low_pass_steps):
                low_pass = P @ low_pass

            coeffs.append(low_pass)

        out = np.concatenate(coeffs, axis=1)

        return out

    def calculate_scattering(self, A, X, J, Q, layer_num=1):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X[:, None]

        n_nodes, T = X.shape

        num_wavelets = J + 2
        num_activations = 2
        num_paths = (num_wavelets * num_activations) ** layer_num

        B = np.zeros((T, num_paths, 1, Q), dtype=float)

        path_index = 0

        def build_layer(current_layer, current_signal):
            nonlocal path_index

            if current_layer == layer_num:
                for q in range(1, Q + 1):
                    moment = np.sum(np.abs(current_signal) ** q, axis=0)
                    B[:, path_index, 0, q - 1] = moment

                path_index += 1

                return

            x1 = self.compute_W_2_transform(
                A,
                current_signal,
                largest_scale=J + 1,
                low_pass_as_wavelet=True,
            )

            signal_width = current_signal.shape[1]
            num_wavelets_here = x1.shape[1] // signal_width

            for j in range(num_wavelets_here):
                start = j * signal_width
                end = (j + 1) * signal_width

                wavelet_signal = x1[:, start:end]

                build_layer(current_layer + 1, self.relu(wavelet_signal))
                build_layer(current_layer + 1, self.reverse_relu(wavelet_signal))

        build_layer(0, X)

        return B

    def save_scattering_moments(
        self,
        A,
        X,
        data_dir,
        dataset="traffic",
        sub_dataset="PEMS08",
        scattering_type="blis",
        wavelet_type="W2",
        largest_scale=4,
        highest_moment=3,
        layer_list=(1, 2, 3),
        force_recompute=False,
    ):
        X = np.asarray(X, dtype=float)

        if X.ndim == 2:
            X = X[:, :, None]

        T, n_nodes, n_features = X.shape

        save_base = os.path.join(
            data_dir,
            dataset,
            sub_dataset,
            "processed",
            scattering_type,
            wavelet_type,
            f"largest_scale_{largest_scale}",
        )

        os.makedirs(save_base, exist_ok=True)

        print(f"Saving scattering moments to: {save_base}")
        print(f"A shape: {A.shape}")
        print(f"X shape: {X.shape}")
        print(f"T: {T}")
        print(f"n_nodes: {n_nodes}")
        print(f"n_features: {n_features}")

        for layer_num in layer_list:
            print("=" * 80)
            print(f"Computing layer_{layer_num}")
            print("=" * 80)

            layer_dir = os.path.join(save_base, f"layer_{layer_num}")
            os.makedirs(layer_dir, exist_ok=True)

            expected_files = [
                os.path.join(layer_dir, f"moment_{q}.npy")
                for q in range(1, highest_moment + 1)
            ]

            if not force_recompute and all(os.path.exists(path) for path in expected_files):
                print(f"Skipping layer_{layer_num}; moment files already exist.")
                continue

            all_moments = [[] for _ in range(highest_moment)]

            for f in range(n_features):
                print(f"Computing channel {f + 1}/{n_features}")

                X_channel = X[:, :, f].T

                print(f"X_channel shape: {X_channel.shape}")

                B = self.calculate_scattering(
                    A,
                    X_channel,
                    J=largest_scale,
                    Q=highest_moment,
                    layer_num=layer_num,
                )

                print(f"B shape: {B.shape}")

                for q in range(highest_moment):
                    all_moments[q].append(B[:, :, :, q])

            for q in range(highest_moment):
                moment_q = np.concatenate(all_moments[q], axis=2)
                moment_q = moment_q.astype(np.float32)

                save_path = os.path.join(layer_dir, f"moment_{q + 1}.npy")

                np.save(save_path, moment_q)

                print(f"Saved: {save_path}")
                print(f"moment_{q + 1} shape: {moment_q.shape}")

        print("Finished saving scattering moments.")


if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "traffic",
        "PEMS08",
    )

    label_path = os.path.join(data_path, "DAY", "label.npy")
    adjacency_path = os.path.join(data_path, "adjacency_matrix.npy")
    signal_path = os.path.join(data_path, "graph_signals.npy")

    gs = GraphScattering(
        label_path=label_path,
        adjacency_path=adjacency_path,
        signal_path=signal_path,
    )

    A_real, X_real, labels_real = gs.load_data()

    data_dir = os.path.join(os.path.dirname(__file__), "data")

    gs.save_scattering_moments(
        A=A_real,
        X=X_real,
        data_dir=data_dir,
        dataset="traffic",
        sub_dataset="PEMS08",
        scattering_type="blis",
        wavelet_type="W2",
        largest_scale=4,
        highest_moment=3,
        layer_list=(1, 2, 3),
        force_recompute=True,
    )