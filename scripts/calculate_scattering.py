import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt 

import blis.models.scattering_transform as st 
import blis.models.wavelets as wav 
from blis import DATA_DIR
import argparse
import time

# example usage: python calculate_scattering.py --scattering_type blis --wavelet_type W2 --largest_scale 4 --highest_moment 3 --dataset traffic --sub_dataset PEMS08

print('Started')
def validate_args(args):
    # Check if dataset is 'traffic' and sub_dataset is valid
    if args.dataset == 'traffic' and args.sub_dataset not in ['PEMS08', 'PEMS07', 'PEMS04', 'PEMS03']:
        raise ValueError("Invalid sub_dataset for dataset 'traffic'.")
    # Check if dataset is 'partly_cloudy' and sub_dataset is valid
    if args.dataset == 'partly_cloudy' and args.sub_dataset != 'full_data':
        try:
            sub_dataset_val = int(args.sub_dataset)
            if not (0 <= sub_dataset_val <= 154):
                raise ValueError("Invalid sub_dataset for dataset 'partly_cloudy'. Should be between 0 and 154.")
            # Check for 4-digit formatting and format if necessary
            args.sub_dataset = f"{sub_dataset_val:04}"
        except ValueError:
            raise ValueError("Sub_dataset for dataset 'partly_cloudy' should be an integer between 0 and 154.")
    # if args.dataset == 'synthetic' and args.sub_dataset not in ['bimodal_normal', 'bimodal_camel']:
    #     raise ValueError("Invalid sub dataset for dataset synthetic")

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
    processed_dir = os.path.join(dataset_dir,  'processed', args.scattering_type, args.wavelet_type, f'largest_scale_{args.largest_scale}')
    print(processed_dir)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # load adjacency matrix and signal
    A = np.load(os.path.join(dataset_dir, 'adjacency_matrix.npy'))
    x = np.load(os.path.join(dataset_dir, 'graph_signals.npy'))
    print(f'this is the shape of x initially: {x.shape}')
    print('Adjacency matrix and signals loaded)')
    #import pdb; pdb.set_trace()
    if len(x.shape) == 2:
        x = x[:,:,None]
        print(f'this is the shape of x in calculate scattering: {x.shape}')
    # ensure that we're working with symmetric matrices!
    assert((A == A.T).all())
    if args.wavelet_type == 'W2':
        wavelets= wav.get_W_2(A, args.largest_scale, low_pass_as_wavelet=(args.scattering_type == 'blis'))
        #wavelets = wav.compute_W_2_transform(A,x,args.largest_scale,low_pass_as_wavelet=(args.scattering_type == 'blis'))
        print(f'this is the shape of wavelets in calculate: {wavelets.shape}')
    else:
        wavelets = wav.get_W_1(A, args.largest_scale, low_pass_as_wavelet=(args.scattering_type == 'blis'))
    print('Started calculating')
    print(f"this is the shape of x in pre-calculating scattering: {x.shape}")
    st.scattering_transform(x, args.scattering_type, wavelets, args.num_layers, args.highest_moment, processed_dir,args.wavelet_type)
    
if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    main()
    end_time = time.time()  # Record the end time
    
    elapsed_time = end_time - start_time  # Compute the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

