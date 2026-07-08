import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)

from torch_geometric.nn import GINEConv, GPSConv,GCN, global_add_pool,global_mean_pool
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn import GCNConv
from typing import Any, Dict, Optional



class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1

class GPS(torch.nn.Module):
    def __init__(self, in_features: int, channels: int, pe_dim: int, num_layers: int, num_classes:int,
                 attn_dropout: float):
        super().__init__()

        self.node_emb = Linear(in_features, channels - pe_dim)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(4, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels = channels, conv = GINEConv(nn), heads=4)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, num_classes),
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=None) #1000 if attn_type == 'performer' else None)

    def forward(self, data):
        x = data.x
        pe = data.pe
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        x_pe = self.pe_norm(pe)
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        
        x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)

        if edge_attr is not None:
            edge_attr = self.edge_emb(edge_attr)
        else:
            edge_attr = self.edge_emb(torch.zeros(len(edge_index[0]),device=x.device).long())

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)

        x = global_add_pool(x, batch)
        return self.mlp(x)

class GPS_(nn.Module):
    def __init__(self, in_features,hidden_channels, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(in_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)
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

