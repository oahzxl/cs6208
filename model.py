import einops
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv 
from torch_geometric.nn import global_mean_pool

from encoding import CentralityEncoding, EdgeEncoding, SpatialEncoding


class Graphormer(nn.Module):
    def __init__(self, dim, head_num, layer_num, num_class=10):
        super().__init__()

        # here are the three key encoding in the paper
        self.centrality_encoding = CentralityEncoding(dim)
        self.spatial_encoding = SpatialEncoding(dim, head_num)
        self.edge_encoding = EdgeEncoding(head_num)  # here I implement one improvement

        # graphormer layers
        self.layers = nn.ModuleList(
            [GraphormerLayer(dim, head_num) for _ in range(layer_num)]
        )
        self.fc_out1 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.fc_out2 = nn.Linear(dim, num_class)

    def forward(self, x, edge_index, edge_attr, batch):
        x, bias = self.pre_process(x, edge_index, edge_attr)
        for layer in self.layers:
            x = layer(x, bias, edge_index)
        x = self.post_process(x, batch)
        return x

    def pre_process(self, x, edge_idx, edge_attr):
        x = self.centrality_encoding(x, edge_idx)
        bias = self.spatial_encoding(x, edge_idx)
        bias = bias + self.edge_encoding(x, edge_idx, edge_attr)
        return x, bias

    def post_process(self, x, batch):
        x = global_mean_pool(x[:-1], batch)
        x = self.fc_out1(x)
        x = torch.nn.functional.relu(x)
        x = self.ln(x)
        x = self.fc_out2(x)
        return x


class GraphormerLayer(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.attn = Attention(head, dim)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        self.gcn = GCN(dim)

    def forward(self, x, bias, edge):
        residual = x
        x = self.ln1(x)
        x = self.attn(x, bias)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.ln2(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.ln3(x)
        x = self.gcn(x, edge)
        x = residual + x

        return x


class Attention(nn.Module):
    def __init__(self, head, dim):
        super().__init__()
        self.num_heads = head
        self.head_dim = dim // head
        self.scaling = self.head_dim**-0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, x, bias):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = einops.rearrange(q, "s (h hd) -> h s hd", h=self.num_heads)
        k = einops.rearrange(k, "s (h hd) -> h s hd", h=self.num_heads)
        v = einops.rearrange(v, "s (h hd) -> h s hd", h=self.num_heads)
        x = torch.matmul(q * self.scaling, k.transpose(-1, -2)) + bias
        x = torch.softmax(x, dim=-1)
        x = torch.matmul(x, v)
        x = einops.rearrange(x, "h s hd  -> s (h hd)")
        x = self.o(x)
        return x


class GCN(torch.nn.Module): 
    def __init__( 
        self, 
        hidden_dim: int = 16, 
        dropout_rate: float = 0.5, 
    ) -> None: 
        super().__init__() 
        self.dropout1 = torch.nn.Dropout(dropout_rate) 
        self.conv1 = GCNConv(hidden_dim, hidden_dim) 
        self.relu = torch.nn.ReLU() 
        self.dropout2 = torch.nn.Dropout(dropout_rate) 

    def forward(self, x, edge_index) -> torch.Tensor: 
        x = self.dropout1(x) 
        x = self.conv1(x, edge_index) 
        x = self.relu(x) 
        x = self.dropout2(x) 
        return x 
