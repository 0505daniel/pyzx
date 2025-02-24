import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv
from utils_train import get_normalization, get_readout

import random

class ZXGNN(nn.Module):
    def __init__(self, encoding_dim, heads=4, beta=True, dropout=0.2, normalization='batch_norm', num_layers=2, activation='elu', readout='attention'):
        super(ZXGNN, self).__init__()

        self.activation = getattr(F, activation)
        self.transformer_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.readout = readout

        self.encoding_layer = nn.Linear(2, encoding_dim) # vtype, phase -> in_channels
        
        for _ in range(num_layers): # Message Passing
            self.transformer_layers.append(TransformerConv(encoding_dim, encoding_dim // heads, heads=heads, concat=True, beta=beta, dropout=dropout, edge_dim=1))
            self.norm_layers.append(get_normalization(normalization, encoding_dim))

        self.decoding_layer = nn.Linear(encoding_dim, 1)  # Output layer for scalar projection
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # If batch is None, create a default batch where all nodes belong to the same graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.encoding_layer(x)

        for i, transformer_layer in enumerate(self.transformer_layers):
            x = transformer_layer(x, edge_index, edge_attr=edge_attr)
            # print(f'After layer {i}, x shape: {x.shape}')
            x = self.norm_layers[i](x)
            x = self.activation(x)

        # Graph readout
        h_G = get_readout(self.readout, x, batch)

        # Apply the linear transformation
        output = self.decoding_layer(h_G) + self.bias

        return output
    
    