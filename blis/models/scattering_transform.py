import numpy as np
from itertools import product 
import os

def relu(x):
    return x * (x > 0)

def reverse_relu(x):
    return relu(-x)

def scattering_transform(x, scattering_type, wavelets, num_layers, highest_moment, save_dir):
    '''
    Computes the graph scattering transform

    Inputs
    scattering_type - a string of either "blis" or "modulus"
    wavelets - a np array of wavelets (possibly containing the lowpass as a wavelet)
    num_layers - The number of wavelet matricies in each transform paths
    x - a torch tensor of shape N x num_features
    save_dir: a directory to the data. it should contain information about scattering type, and wavelets
    '''
    if scattering_type not in ["blis", "modulus"]:
        raise ValueError("Invalid scattering type. Accepted values are 'blis' or 'modulus'.")
    
    N ,num_features = x.shape 
    J = len(wavelets)
    
    if scattering_type == 'blis':
        combinations = list(product(range(J), [relu, reverse_relu], repeat = num_layers))
    else:
        combinations = list(product(range(J), [np.abs], repeat = num_layers)) 

    # store the outputs in a dictionary
    coeff_dict = {}
    for layer in range(num_layers):
        for moment in range(1, highest_moment + 1):
            coeff_dict[(layer,moment)] = list()
            

    # save each layers output and each statistical moment separately
    for ind, comb in enumerate(combinations):
        layer_out = x 
        for layer in range(num_layers):
            wavelet_index = comb[layer * 2]
            activation = comb[layer * 2 + 1]
            wavelet = wavelets[wavelet_index] 
            layer_out = activation(wavelet @ layer_out) 
            # note: for layer = n, it is interpreted as the output of layer n+1

            # compute moments on the layer output 
            for moment in range(1, highest_moment + 1):
                feat = np.sum(np.power(layer_out, moment))
                coeff_dict[(layer, moment)].append(feat)
    
    # write all of coeff_dict to memory
    for layer in range(num_layers):
        for moment in range(1, highest_moment + 1):
            np_array = np.array(data_list)
            full_path = os.path.join(save_path, f"layer_{layer}_moment_{moment}.npy")
            np.save(full_path, np_array)
    

     