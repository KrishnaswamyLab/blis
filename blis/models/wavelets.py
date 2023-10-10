import networkx as nx
import numpy as np
import numpy.linalg as LA

def get_P(A: np.ndarray) -> np.ndarray:
    d_arr = np.sum(A, axis=1)
    d_arr_inv = 1/d_arr
    d_arr_inv[np.isinf(d_arr_inv)] = 0
    D_inv = np.diag(d_arr_inv)
    P = 0.5 * (np.eye(D_inv.shape[0]) + A @ D_inv)
    return P

def get_T(A: np.ndarray) -> np.ndarray:
    d_arr = np.sum(A, axis = 1) 
    d_arr_inv = 1/d_arr 
    d_arr_inv[np.isinf(d_arr_inv)] = 0 
    D_inv_sqrt = np.diag(np.sqrt(d_arr_inv))
    T = 0.5 * (np.eye(A.shape[0]) + D_inv_sqrt @ A @ D_inv_sqrt)
    return T

def get_M(A: np.ndarray) -> np.ndarray:
    M = np.diag(1 / np.sqrt(np.sum(A, axis=1)))
    return M

def get_W_2(A, largest_scale, low_pass_as_wavelet=False):
    P = get_P(A)
    N = P.shape[0]
    powered_P = P
    if low_pass_as_wavelet:
        wavelets = np.zeros((largest_scale + 2, *P.shape))
    else:
        wavelets = np.zeros((largest_scale + 1, *P.shape))
    wavelets[0,:,:] = np.eye(N) - powered_P
    for scale in range(1, largest_scale + 1):
        Phi = powered_P @ (np.eye(N) - powered_P)
        wavelets[scale,:,:] = Phi
        powered_P = powered_P @ powered_P
    low_pass = powered_P
    if low_pass_as_wavelet:
        wavelets[-1,:,:] = low_pass
    return wavelets

def get_W_1(A: np.ndarray, largest_scale: int, low_pass_as_wavelet=False) -> list:
    #import pdb; pdb.set_trace()
    T_matrix = get_T(A)
    w, U = LA.eigh(T_matrix)
    w = np.maximum(w, 0)  # ReLU operation

    d_arr = np.sum(A, axis = 1)
    d_arr_inv = 1/d_arr 
    d_arr_inv[np.isinf(d_arr_inv)] = 0 
    M = np.diag(np.sqrt(d_arr_inv)) 
    #M = np.diag(1/ np.sqrt(np.sum(A, axis=1)))
    M_inv = np.diag(np.sqrt(d_arr))
    if low_pass_as_wavelet:
        wavelets = np.zeros((largest_scale + 2, *T_matrix.shape))
    else:
        wavelets = np.zeros((largest_scale + 1, *T_matrix.shape))
    eig_filter = np.sqrt(np.maximum(np.ones(len(w)) - w, 0))
    Psi = M_inv @ U @ np.diag(eig_filter) @ U.T @ M 
    wavelets[0,:,:] = Psi
    for scale in range(1, largest_scale + 1):
        eig_filter = np.sqrt(np.maximum(w ** (2 **(scale-1) ) - w ** (2 ** scale), 0))
        Psi = M_inv @ U @ np.diag(eig_filter) @ U.T @ M
        wavelets[scale,:,:] = Psi
    low_pass = M_inv @ U @ np.diag(np.sqrt(w ** (2 ** largest_scale))) @ U.T @ M
    if low_pass_as_wavelet:
        wavelets[-1,:,:] = low_pass
    return wavelets
