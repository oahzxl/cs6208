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
        self.degree_encoder = nn.Embedding(15, hidden_dim)
        self.special_node = nn.Embedding(1, hidden_dim)

    def forward(self, x, edge_idx):
        # encode nodes
        x = self.feature_encoder(x)
        x = torch.cat([x, self.special_node.weight.data], dim=0)

        # get in and out degree
        x_degree = degree(edge_idx.view(-1), num_nodes=x.shape[0]).int() // 2
        x_degree = self.degree_encoder(x_degree)

        x = x + x_degree
        return x


class EdgeEncoding(nn.Module):
    """
    Compute edge bias.
    """

    def __init__(self, head_num):
        super(EdgeEncoding, self).__init__()
        self.edge_encoder = nn.Linear(4, head_num)
        self.head_num = head_num

    def forward(self, x, edge_idx, edge_attr):
        edge_data = Data(edge_index=edge_idx, num_nodes=x.shape[0])
        edge_data = to_networkx(edge_data)
        out = torch.zeros((x.shape[0], x.shape[0], self.head_num), device=x.device)
        # NOTE: here I implement the multi path improvement
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                shortest_paths = nx.all_shortest_paths(edge_data, source=i, target=j)
                try:
                    shortest_paths = list(shortest_paths)
                except:
                    continue  # no path
                if len(shortest_paths) > 3:
                    shortest_paths = shortest_paths[:3]
                for path in shortest_paths:
                    val = torch.zeros((self.head_num), device=x.device)
                    for e in path:
                        e_attr = edge_attr[e]
                        e_tensor = self.edge_encoder(e_attr)
                        val += e_tensor
                    if len(path) > 0:
                        val /= len(path)
                    out[i, j] += val
                out[i, j] /= len(shortest_paths)
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
