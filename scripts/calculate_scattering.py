import numpy as np
import matplotlib.pyplot as plt 
import os
import blis.models.scattering_transform as st 
import blis.models.wavelets as wav 
from blis import DATA_DIR
import argparse
import time

# example usage: python calculate_scattering.py --scattering_type blis --wavelet_type W2 --largest_scale 4 --highest_moment 3 --dataset traffic --sub_dataset PEMS08

def validate_args(args):
    # Check if dataset is 'traffic' and sub_dataset is valid
    if args.dataset == 'traffic' and args.sub_dataset not in ['PEMS08', 'PEMS07', 'PEMS04', 'PEMS03']:
        raise ValueError("Invalid sub_dataset for dataset 'traffic'.")
    # Check if dataset is 'partly_cloudy' and sub_dataset is valid
    if args.dataset == 'partly_cloudy':
        try:
            sub_dataset_val = int(args.sub_dataset)
            if not (0 <= sub_dataset_val <= 154):
                raise ValueError("Invalid sub_dataset for dataset 'partly_cloudy'. Should be between 0 and 154.")
            # Check for 4-digit formatting and format if necessary
            args.sub_dataset = f"{sub_dataset_val:04}"
        except ValueError:
            raise ValueError("Sub_dataset for dataset 'partly_cloudy' should be an integer between 0 and 154.")
    if args.dataset == 'synthetic':
        raise ValueError("Synthetic still needs to be incorporated")

def main():
    parser = argparse.ArgumentParser(description="Parse arguments for the program.")

    parser.add_argument("--scattering_type", choices=['blis', 'modulus'], help="Type of scattering: 'blis' or 'modulus'.")
    parser.add_argument("--wavelet_type", choices=['W1', 'W2'], default = 'W2', help="Type of wavelet: 'W1' or 'W2'.")
    parser.add_argument("--largest_scale", type=int, help="Largest (dyadic) scale as a positive integer.")
    parser.add_argument("--highest_moment", type=int, default=1, help="Highest moment as a positive integer. Defaults to 1.")
    parser.add_argument("--dataset", choices=['traffic', 'partly_cloudy', 'synthetic'], help="Dataset: 'traffic' or 'partly_cloudy' or 'synthetic'.")
    parser.add_argument("--sub_dataset", help="Sub-dataset value depending on the dataset chosen.")
    parser.add_argument("--num_layers", type=int, default=3, help="Largest scattering layer")

    args = parser.parse_args()

    # Additional validation for the arguments
    validate_args(args)

    # dataset directory specifies up to the sub dataset level 
    dataset_dir = os.path.join(DATA_DIR, args.dataset, args.sub_dataset)

    # the processed directory records scattering type and the largest wavelet scale
    processed_dir = os.path.join(dataset_dir, 'processed', args.scattering_type, f'largest_scale_{args.largest_scale}')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # load adjacency matrix and signal
    A = np.load(os.path.join(dataset_dir, 'adjacency_matrix.npy'))
    x = np.load(os.path.join(dataset_dir, 'graph_signals.npy'))

    if args.wavelet_type == 'W2':
        wavelets = wav.get_W_2(A, args.largest_scale, low_pass_as_wavelet=(args.scattering_type == 'blis'))
    else:
        wavelets = wav.get_W_1(A, args.largest_scale, low_pass_as_wavelet=(args.scattering_type == 'blis'))
    
    st.scattering_transform(x, args.scattering_type, wavelets, args.num_layers, args.highest_moment, processed_dir)
    
if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    main()
    end_time = time.time()  # Record the end time
    
    elapsed_time = end_time - start_time  # Compute the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

