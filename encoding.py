import einops
import torch
from torch import nn
from torch_geometric.utils import degree, to_networkx
from torch_geometric.data import Data
import networkx as nx


class CentralityEncoding(nn.Module):
    """
    Compute node degree.
    """

    def __init__(self, hidden_dim):
        super(CentralityEncoding, self).__init__()
        self.feature_encoder = nn.Linear(7, hidden_dim)
        self.degree_encoder = nn.Embedding(10, hidden_dim)

    def forward(self, x, edge_idx):
        # encode nodes
        x = self.feature_encoder(x)

        # get in and out degree
        x_degree = degree(edge_idx.view(-1), num_nodes=x.shape[0]).int() // 2
        x_degree = self.degree_encoder(x_degree)

        x = x + x_degree
        return x


class EdgeEncoding(nn.Module):
    """
    Compute edge bias.
    """

    def __init__(self, dim, head_num):
        super(EdgeEncoding, self).__init__()
        self.edge_encoder = nn.Linear(4, head_num)
        self.head_num = head_num

    def forward(self, x, edge_idx, edge_attr):
        edge_data = Data(edge_index=edge_idx, num_nodes=x.shape[0])
        edge_data = to_networkx(edge_data)
        shortest_path = nx.shortest_path(edge_data)
        out = torch.zeros((x.shape[0], x.shape[0], self.head_num), device=x.device)
        for i, d in shortest_path.items():
            for j, path_route in d.items():
                for e in path_route:
                    e_attr = edge_attr[e]
                    e_tensor = self.edge_encoder(e_attr)
                    out[i, j] += e_tensor
                if len(path_route) > 0:
                    out[i, j] /= len(path_route)
        out = einops.rearrange(out, "h w n -> n h w")
        return out


class SpatialEncoding(nn.Module):
    """
    Compute spatial bias.
    """

    def __init__(self, dim, head_num):
        super(SpatialEncoding, self).__init__()
        self.spatial_encoder = nn.Embedding(50, head_num)
        self.head_num = head_num

    def forward(self, x, edge_idx):
        edge_data = Data(edge_index=edge_idx, num_nodes=x.shape[0])
        edge_data = to_networkx(edge_data)
        shortest_path = list(nx.shortest_path_length(edge_data))
        edge_data = torch.zeros(
            (x.shape[0], x.shape[0]), device=x.device, dtype=torch.int
        )
        for i in range(len(shortest_path)):
            j, d = shortest_path[i]
            for k, v in d.items():
                edge_data[j, k] = v
        out = self.spatial_encoder(edge_data)
        out = einops.rearrange(out, "h w n -> n h w")
        return out
