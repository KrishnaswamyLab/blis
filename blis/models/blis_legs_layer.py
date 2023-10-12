import numpy as np

import torch
import torch.nn as nn
from torch.nn import Linear
from torch_scatter import scatter_mean
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
import torch_geometric
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn import GCNConv

device = torch.device("cpu")

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, add_self_loops=False, dtype=None):

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 1, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    
    return edge_index, deg_inv_sqrt[col] * edge_weight


class LazyLayer(torch.nn.Module):
    
    """ Currently a single elementwise multiplication with one laziness parameter per
    channel. this is run through a softmax so that this is a real laziness parameter
    """

    def __init__(self, n):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.Tensor(2, n))

    def forward(self, x, propogated):
        inp = torch.stack((x, propogated), dim=1)
        s_weights = torch.nn.functional.softmax(self.weights, dim=0)
        return torch.sum(inp * s_weights, dim=-2)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weights)


class Diffuse(MessagePassing):

    """ Implements low pass walk with optional weights
    """

    def __init__(self, in_channels, out_channels, trainable_laziness=False, fixed_weights=True):

        super().__init__(aggr="add",  flow = "target_to_source", node_dim=-3)  # "Add" aggregation.
        assert in_channels == out_channels
        self.trainable_laziness = trainable_laziness
        self.fixed_weights = fixed_weights
        if trainable_laziness:
            self.lazy_layer = LazyLayer(in_channels)
        if not self.fixed_weights:
            self.lin = torch.nn.Linear(in_channels, out_channels)


    def forward(self, x, edge_index, edge_weight=None):

        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 2: Linearly transform node feature matrix.
        # turn off this step for simplicity
        if not self.fixed_weights:
            x = self.lin(x)

        # Step 3: Compute normalization
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(self.node_dim), dtype=x.dtype)

        # Step 4-6: Start propagating messages.
        propogated = self.propagate(
            edge_index, edge_weight=edge_weight, size=None, x=x,
        )
        if not self.trainable_laziness:
            return 0.5 * (x + propogated), edge_index, edge_weight

        return self.lazy_layer(x, propogated), edge_index, edge_weight


    def message(self, x_j, edge_weight):
        
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return edge_weight.view(-1, 1, 1) * x_j


    #def message_and_aggregate(self, adj_t, x):
    #
    #    return matmul(adj_t, x, reduce=self.aggr)


    def update(self, aggr_out):

        # aggr_out has shape [N, out_channels]
        # Step 6: Return new node embeddings.
        return aggr_out
    
class Blis(torch.nn.Module):

    def __init__(self, in_channels, trainable_laziness=False, trainable_scales = False, activation = "blis"):

        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_channels = in_channels
        self.trainable_laziness = trainable_laziness
        self.diffusion_layer1 = Diffuse(in_channels, in_channels, trainable_laziness)
        # self.diffusion_layer2 = Diffuse(
        #     4 * in_channels, 4 * in_channels, trainable_laziness
        # )
        self.wavelet_constructor = torch.nn.Parameter(torch.tensor([
            [1, -1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ], requires_grad=trainable_scales))

        if activation == "blis":
            self.activations = [lambda x: torch.relu(x), lambda x: torch.relu(-x)]
        elif activation == None:
            self.activations = [lambda x : x]

    def forward(self, data):

        """ This performs  Px with P = 1/2(I + AD^-1) (column stochastic matrix) at the different scales"""

        x, edge_index = data.x, data.edge_index
        s0 = x[:,:,None]
        avgs = [s0]
        for i in range(16):
            avgs.append(self.diffusion_layer1(avgs[-1], edge_index)[0])
        for j in range(len(avgs)):
            avgs[j] = avgs[j][None, :, :, :]  # add an extra dimension to each tensor to avoid data loss while concatenating TODO: is there a faster way to do this?
        
        # Combine the diffusion levels into a single tensor.
        diffusion_levels = torch.cat(avgs)
        
        # Reshape the 3d tensor into a 2d tensor and multiply with the wavelet_constructor matrix
        # This simulates the below subtraction:
        # filter0 = avgs[0] - avgs[1]
        # filter1 = avgs[1] - avgs[2] 
        # filter2 = avgs[2] - avgs[4]
        # filter3 = avgs[4] - avgs[8]
        # filter4 = avgs[8] - avgs[16] 
        # filter5 = avgs[16]
        wavelet_coeffs = torch.einsum("ij,jklm->iklm", self.wavelet_constructor, diffusion_levels) # J x num_nodes x num_features x 1
        #subtracted = subtracted.view(6, x.shape[0], x.shape[1]) # reshape into given input shape
        activated = [self.activations[i](wavelet_coeffs) for i in range(len(self.activations))]
        
        s = torch.cat(activated, axis=-1).transpose(1,0)
        
        return s
    
    def out_features(self):
        return 12 * self.in_channels

class BlisNet(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, edge_in_channels = None, trainable_laziness=False, layout = ['blis','gcn','gcn'],  **kwargs):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_in_channels = edge_in_channels
        self.trainable_laziness = trainable_laziness

        self.layout = layout
        #self.layers = nn.ModuleList()
        self.layers = []
        self.out_dimensions = [in_channels]

        for layout_ in layout:
            if layout_ == 'blis':
                self.layers.append(Blis(self.out_dimensions[-1], trainable_laziness=trainable_laziness))
                self.out_dimensions.append(self.layers[-1].out_features())
            elif layout_ == 'gcn':
                self.layers.append(GCNConv(self.out_dimensions[-1], hidden_channels))
                self.out_dimensions.append(hidden_channels)
            elif layout_ == 'dim_reduction':
                self.layers.append(Linear(self.out_dimensions[-1], hidden_channels))
                self.out_dimensions.append(hidden_channels)
        self.layers = nn.ModuleList(self.layers)
        self.batch_norm = BatchNorm(self.out_dimensions[-1])
        self.lin1 = Linear(self.out_dimensions[-1], self.out_dimensions[-1]//2 )
        self.mean = global_mean_pool
        self.lin2 = Linear(self.out_dimensions[-1]//2, out_channels)
        self.lin3 = Linear(out_channels, out_channels)

        self.act = torch.nn.ReLU()


    def forward(self, data):

        for il, layer in enumerate(self.layers):
            #import pdb; pdb.set_trace()
            if self.layout[il] == "blis":
                x = layer(data).reshape(data.x.shape[0],-1)
            elif self.layout[il] == "dim_reduction":
                x = layer(data.x)
            else:
                x = layer(data.x, data.edge_index)
            data.x =x
        
        x = self.batch_norm(data.x)
        x = self.lin1(x)
        x = self.act(x)
        x = self.mean(x,data.batch)
        x = self.lin2(x)
        x = self.act(x)
        x = self.lin3(x)

        return x


if __name__ == "__main__":
    print("Testing the BLIS Legs module")
    from blis import DATA_DIR
    import os
    import blis.models.scattering_transform as st 
    import blis.models.wavelets as wav 

    dataset = 'traffic'
    sub_dataset = "PEMS04"
    label= "DAY"
    largest_scale = 4
    scattering_type = 'blis'
    num_layers = 1
    highest_moment = 4
    wavelet_type = 'W2'

    dataset_dir = os.path.join(DATA_DIR, dataset, sub_dataset)
    processed_dir =  os.path.join(dataset_dir, 'processed', scattering_type, wavelet_type, f'largest_scale_{largest_scale}')


    dataset_dir = os.path.join(DATA_DIR, dataset, sub_dataset)


    # load adjacency matrix and signal
    A = np.load(os.path.join(dataset_dir, 'adjacency_matrix.npy'), allow_pickle = True)
    x = np.load(os.path.join(dataset_dir, 'graph_signals.npy'), allow_pickle = True)
    y = np.load(os.path.join(dataset_dir, label, 'label.npy'), allow_pickle = True)
    if len(x.shape) == 2:
        x = x[:,:,None]

    x = x[:10]

    if wavelet_type == 'W2':
        wavelets = wav.get_W_2(A, largest_scale, low_pass_as_wavelet=(scattering_type == 'blis'))
    else:
        wavelets = wav.get_W_1(A, largest_scale, low_pass_as_wavelet=(scattering_type == 'blis'))
    #coeffs = st.scattering_transform(x, scattering_type, wavelets, num_layers, highest_moment, processed_dir)

    coeffs = np.stack([np.einsum('ik, nkf->nif', wavelets[j], x) for j in range(len(wavelets))],1).transpose(0,2,1,3) # Ngraphs x Nnodes x Nscales x Ndim

    from blis.data.load_from_np import create_dataset

    data_list = create_dataset(x, y , A)

    blis_mod = Blis(in_channels = 3, trainable_laziness=False, activation = None)
    out_coeffs  = blis_mod(data_list[0])

    diff = np.abs(coeffs[0] - out_coeffs.detach().numpy()[...,0]).max()

    print(f"Max difference between BLIS and BLIS Legs is {diff}")
