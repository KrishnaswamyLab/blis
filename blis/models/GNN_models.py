import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import GINConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU


class GCN(nn.Module):
    def __init__(self, in_features,hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
        self.final_nonlin = nn.Softmax(dim = 1)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)

        #x = F.dropout(x, p = 0.2, training=self.training)
        x = self.lin(x)
        # Apparently when using cross entropy loss, the softmax is automatically applied.
        #x = self.final_nonlin(x)

        return x

class GIN(nn.Module):
    def __init__(self, in_features, hidden_channels, num_classes):
        super(GIN, self).__init__()

        # Define the neural network for GINConv (this can be adjusted)
        nn1 = Sequential(Linear(in_features, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1)
        
        nn2 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv2 = GINConv(nn2)
        
        self.lin = Linear(hidden_channels, num_classes)
        self.final_nonlin = nn.Softmax(dim=1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x

class GAT(nn.Module):
    def __init__(self, in_features, hidden_channels, num_classes, heads=1):
        super(GAT, self).__init__()

        # The number of heads can be adjusted for multi-head attention.
        self.conv1 = GATConv(in_features, hidden_channels, heads=heads, concat=True)
        # If we use multi-head attention in the first layer, we need to adjust the input dimension of the next layer.
        self.conv2 = GATConv(heads * hidden_channels, hidden_channels)

        self.lin = Linear(hidden_channels, num_classes)
        self.final_nonlin = nn.Softmax(dim=1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x
