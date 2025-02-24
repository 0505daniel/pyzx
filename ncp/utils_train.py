import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GraphNorm, LayerNorm, InstanceNorm
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, Set2Set
from tqdm import tqdm

import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pyzx as zx

def get_normalization(normalization, num_features):
    if normalization == 'batch_norm':
        return nn.BatchNorm1d(num_features)
    elif normalization == 'layer_norm':
        return LayerNorm(num_features)
    elif normalization == 'instance_norm':
        return InstanceNorm(num_features)
    elif normalization == 'graph_norm':
        return GraphNorm(num_features)
    else:
        return nn.Identity()
    
def get_readout(readout_type, x, batch):
    if readout_type == 'mean':
        h_G = global_mean_pool(x, batch)
    elif readout_type == 'max':
        h_G = global_max_pool(x, batch)
    elif readout_type == 'sum':
        h_G = global_add_pool(x, batch)
    elif readout_type == 'attention':
        # Attention-based pooling
        node_weights = F.softmax(x, dim=1)
        h_G = scatter(node_weights * x, batch, dim=0, reduce='sum')
    #BUG: Not working
    elif readout_type == 'set2set': 
        set2set = Set2Set(x.size(1), processing_steps=2).to(x.device)
        h_G = set2set(x, batch)
    else:
        raise ValueError(f'Unknown readout type: {readout_type}')
    
    return h_G

def get_loss_function(loss_function):
    if loss_function == 'mse':
        return nn.MSELoss()
    elif loss_function == 'mae':
        return nn.L1Loss()
    elif loss_function == 'huber':
        return nn.HuberLoss()
    elif loss_function == 'smooth_l1':
        return nn.SmoothL1Loss()
    
def get_optimizer(optimizer, model, lr):
    if optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=lr)
    
def zx_to_pyg_data(g):
    """
    Convert a PyZX graph 'g' to a torch_geometric Data object.
    - Node features: [vertex_type, phase]
    - Edge features: [edge_type]
    """
    vertices = list(g.graph.keys())
    idx_map = {v: i for i, v in enumerate(vertices)}

    # Node features
    # vertex_type -> integer, phase -> float
    x_list = []
    for v in vertices:
        vtype = g.ty[v].value + 1  # BOUNDARY=0, Z=1, X=2, ...
        phase = float(g._phase[v])  # fraction -> float
        x_list.append([vtype, phase])

    # Build edges
    edge_indices = []
    edge_attrs = []
    visited = set()
    for v in vertices:
        for w, etype in g.graph[v].items():
            # To avoid duplicating edges, only take (v, w) if v < w in index
            iv, iw = idx_map[v], idx_map[w]
            if (iw, iv) not in visited and (iv, iw) not in visited:
                visited.add((iv, iw))
                visited.add((iw, iv))
                edge_indices.append([iv, iw])
                # Edge type = etype.value (SIMPLE=1, HADAMARD=2, etc.)
                edge_attrs.append([etype.value])

    x_tensor = torch.tensor(x_list, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)
    return data

class ZXGraphDataset(Dataset):
    """
    Creates or loads a list of (Data, label) pairs where:
      Data = PyTorch Geometric Data from a ZX-diagram
      label = Weighted Gate Count
    """
    def __init__(self, root, name="zx_data", num_samples=100, transform=None):
        self.root = root
        self.name = name
        self.num_samples = num_samples
        self.transform = transform
        self.data_list = None

        # If file not already present, generate and save
        if not os.path.exists(self.processed_file_path()):
            self.process()
        # Load from file
        self.load_data()

    def processed_file_path(self):
        return os.path.join(self.root, f"{self.name}_data_{self.num_samples}.pt")

    def process(self):
        data_list = []

        for _ in tqdm(range(self.num_samples), desc="Generating ZX graphs"):
            qubit_amount = random.randint(5, 15) 
            gate_count = random.randint(50, 500)    

            g = zx.generate.cliffordT(qubit_amount, gate_count)
            zx.simplify.full_reduce(g)

            c = zx.extract.extract_circuit(g).to_basic_gates()
            c = zx.optimize.basic_optimization(c)

            # total = len(c.gates)
            # two_qubit_count = c.twoqubitcount()
            # single_qubit_count = total - two_qubit_count

            # label
            y_val = zx.local_search.scores.wgc(c, two_qb_weight=10)

            # to pyg
            data = zx_to_pyg_data(g)
            data.y = torch.tensor([y_val], dtype=torch.float)
            data_list.append(data)
        torch.save(data_list, self.processed_file_path())

    def load_data(self):
        self.data_list = torch.load(self.processed_file_path())

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        d = self.data_list[idx]
        if self.transform is not None:
            d = self.transform(d)
        return d