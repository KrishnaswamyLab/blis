import numpy as np
from itertools import product 
import os

print('Started running the scattering transform file')
def relu(x):
    return x * (x > 0)

def reverse_relu(x):
    return relu(-x)

def scattering_transform(x, scattering_type, input_wavelets, num_layers, highest_moment, save_dir,wavelet_type):
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

    if len(x.shape) == 3:
        num_signals, N ,num_features = x.shape
    if len(x.shape) == 2:
        num_signals, N = x.shape
        num_features= 1

    #print(f'this is x at the beginning : {x[0]}')

    print('The number of signals is', num_signals)
    print('The number of features is', num_features)
    print('The number of wavelets is', N)
    print(f'this is the shape of x initially in scattering transform: {x.shape}')

    print(f'this is the shape of wavelets in scattering transform initially: {input_wavelets.shape}')
    J = len(input_wavelets)
    print(f'this is J = {J}')

    # save the zero order scattering coefficients:
    zero_save_dir = os.path.join(save_dir, f'layer_0')
    print(f'This is the zero save dir: {zero_save_dir}')
    if not os.path.exists(zero_save_dir):
        os.makedirs(zero_save_dir)
    for moment_ind in range(highest_moment):
        full_path = os.path.join(zero_save_dir, f"moment_{moment_ind + 1}.npy")
        coeffs_zero = np.zeros((num_signals, 1, num_features, highest_moment))

        for moment in range(1, highest_moment + 1):
            coeffs_zero[:, 0, :, moment-1] = np.sum(np.power(x, moment), axis = 1)

            np.save(full_path, coeffs_zero[:,:,:,moment_ind])

    # num_layers is the LARGEST layer size
    # layer_num is the largest layer size within the loop
    # layer is the layer number looping up to layer_num

    for layer_num in range(1, num_layers+1):

        if save_dir is not None:
            layer_dir = os.path.join(save_dir, f'layer_{layer_num}')
            print(f'this is the layer dir: {layer_dir}')

            if os.path.exists(layer_dir):
                # pass over this iteration of the for loop
                continue

        # note that this code has redundant calculations for each layer!
        if scattering_type == 'blis':
            combinations = list(product(range(J), [relu, reverse_relu], repeat = layer_num))
            num_activation = 2
            print(f'this is the number of combinations: {len(combinations)}')
        else:
            combinations = list(product(range(J), [np.abs], repeat = layer_num))
            num_activation = 1

        # store the output
        coeffs = np.zeros((num_signals, (J*num_activation)**layer_num, num_features, highest_moment))
        print(f'This is the shape of coeffs: {coeffs.shape}')


        for ind, comb in enumerate(combinations):
            print(f'this is the number of iterations: {ind+1} out of {len(combinations)}')
            layer_out = x
            #print(f'this is the shape of layer out: {layer_out.shape}')
            for layer in range(layer_num):
                wavelet_index = comb[layer * 2]
                print(f'This is the wavelet index: {wavelet_index}')
                #print(f'this is the shape of wavelets in scattering transform initially: {wavelets.shape}')
                activation = comb[layer * 2 + 1]
                print(f'This is the activation function: {activation}')
                if wavelet_type == 'W2':
                    wavelet=input_wavelets[wavelet_index]
                    #wavelet_transform=input_wavelets
                    #print(f'this is the shape of wavelet in scattering transform: {wavelet.shape}')
                    #print(f'this is the shape of layer out right before einstein summation: {layer_out.shape}')
                    wavelet_transform=np.einsum('ik, nkf->nif', wavelet, layer_out)
                    print(f"this is the dimension of the wavelet transform:{wavelet_transform.shape}")
                else:
                    wavelet = input_wavelets[wavelet_index]
                    #print(f'this is the shape of wavelet in scattering transform: {wavelet.shape}')
                    wavelet_transform = np.einsum('ik, nkf->nif', wavelet, layer_out)
                    #print(f'this is the shape of wavelet transform: {wavelet_transform.shape}')

                layer_out = activation(wavelet_transform)
                print(f'this is the shape of layer out after transform: {layer_out.shape}')
 
    # the scattering transform along one path has now been calculated for all signals
    # layer_out has shape [num_signals, num_vertices, num_features]
            for moment in range(1, highest_moment + 1):
                
                print(moment)
                print(f'this is the layer out dimension in moment: {layer_out.shape}')
                coeffs[:, ind, :, moment-1] = np.sum(np.power(layer_out, moment), axis=1)
                print(f'this is the shape of coeffs after moment: {coeffs.shape}')


        # all of the coeffs have been calculated for a given layer and number of moments
        # write them to memory

        #create a directory for each layer
        if save_dir is not None:
            if not os.path.exists(layer_dir):
                os.makedirs(layer_dir)
            for moment_ind in range(highest_moment):
                full_path = os.path.join(layer_dir, f"moment_{moment_ind + 1}.npy")
                print(f'This is the full path {full_path}')
                np.save(full_path, coeffs[:,:,:, moment_ind])

    #return coeffs
