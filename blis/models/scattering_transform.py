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
    x - a torch tensor of shape num_signals x N x num_features
    save_dir: a directory to the data. it should contain information about scattering type, and wavelets (i.e. highest scale)
    '''
    if scattering_type not in ["blis", "modulus"]:
        raise ValueError("Invalid scattering type. Accepted values are 'blis' or 'modulus'.")
    
    num_signals, N ,num_features = x.shape 
    J = len(wavelets)
    
    # if scattering_type == 'blis':
    #     combinations = list(product(range(J), [relu, reverse_relu], repeat = num_layers))
    # else:
    #     combinations = list(product(range(J), [np.abs], repeat = num_layers)) 

    # # store the outputs in a dictionary
    # coeff_dict = {}
    # for layer in range(num_layers):
    #     for moment in range(1, highest_moment + 1):
    #         coeff_dict[(layer,moment)] = list()
    
    # num_layers is the LARGEST layer size
    # layer_num is the largest layer size within the loop
    # layer is the layer number looping up to layer_num 
    for layer_num in range(1, num_layers+1):
        # note that this code has redundant calculations for each layer!
        if scattering_type == 'blis':
            combinations = list(product(range(J), [relu, reverse_relu], repeat = layer_num))
            num_activation = 2
        else:
            combinations = list(product(range(J), [np.abs], repeat = layer_num))
            num_activation = 1

        # store the output
        coeffs = np.zeros((num_signals, (J*num_activation)**layer_num, num_features, highest_moment))

        for ind, comb in enumerate(combinations):
            layer_out = x 
            for layer in range(layer_num):
                wavelet_index = comb[layer * 2]
                activation = comb[layer * 2 + 1]
                wavelet = wavelets[wavelet_index]
                wavelet_transform = np.einsum('ik, nkf->nif', wavelet, layer_out)
                layer_out = activation(wavelet_transform)
            
            # the scattering transform along one path has now been calculated for all signals
            # layer_out has shape [num_signals, num_vertices, num_features]
            for moment in range(1, highest_moment + 1):
                coeffs[:, ind, :, moment-1] = np.sum(np.power(layer_out, moment), axis = 1)
        
        # all of the coeffs have been calculated for a given layer and number of moments
        # write them to memory

        #create a directory for each layer 
        layer_dir = os.path.join(save_dir, f'layer_{layer_num}')
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)
        for moment_ind in range(highest_moment):
            full_path = os.path.join(layer_dir, f"moment_{moment_ind + 1}.npy")
            np.save(full_path, coeffs[:,:,:, moment_ind])
        
        


            

    # # save each layers output and each statistical moment separately
    # for ind, comb in enumerate(combinations):
    #     layer_out = x 
    #     for layer in range(num_layers):
    #         wavelet_index = comb[layer * 2]
    #         activation = comb[layer * 2 + 1]
    #         wavelet = wavelets[wavelet_index] 
    #         layer_out = activation(wavelet @ layer_out) 
    #         # note: for layer = n, it is interpreted as the output of layer n+1

    #         # compute moments on the layer output 
    #         for moment in range(1, highest_moment + 1):
    #             feat = np.sum(np.power(layer_out, moment))
    #             coeff_dict[(layer, moment)].append(feat)
    
    # # write all of coeff_dict to memory
    # for layer in range(num_layers):
    #     for moment in range(1, highest_moment + 1):
    #         np_array = np.array(coeff_dict[(layer, moment)])
    #         full_path = os.path.join(save_path, f"layer_{layer}_moment_{moment}.npy")
    #         np.save(full_path, np_array)
    #         # note: this is not correct! saving at each layer results in duplicates
    

     
