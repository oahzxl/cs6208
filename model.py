import einops
import torch
import torch.nn as nn

from encoding import CentralityEncoding, EdgeEncoding, SpatialEncoding, SpecialEncoding


class Graphormer(nn.Module):
    def __init__(
        self, dim, head_num, layer_num, num_class=10, num_patch=8, patch_size=4
    ):
        super().__init__()
        self.centrality_encoding = CentralityEncoding(
            head_num, num_patch, patch_size, 3, dim
        )
        self.spatial_encoding = SpatialEncoding(head_num, num_patch)
        self.special_encoding = SpecialEncoding(dim)
        self.edge_encoding = EdgeEncoding(head_num, num_patch)
        self.layers = nn.ModuleList(
            [GraphormerLayer(dim, head_num) for _ in range(layer_num)]
        )
        self.fc_out1 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.fc_out2 = nn.Linear(dim, num_class)

    def forward(self, x):
        x, bias = self.pre_process(x)
        for layer in self.layers:
            x = layer(x, bias)
        x = self.post_process(x)
        return x

    def pre_process(self, x):
        x = self.centrality_encoding(x)
        x = self.special_encoding(x)
        bias = self.spatial_encoding()
        bias = bias + self.edge_encoding()
        return x, bias

    def post_process(self, x):
        x = torch.mean(x, dim=1)
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

    def forward(self, x, bias):
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
        q = einops.rearrange(q, "b s (h hd) -> b h s hd", h=self.num_heads)
        k = einops.rearrange(k, "b s (h hd) -> b h s hd", h=self.num_heads)
        v = einops.rearrange(v, "b s (h hd) -> b h s hd", h=self.num_heads)
        x = torch.matmul(q * self.scaling, k.transpose(-1, -2)) + bias
        x = torch.softmax(x, dim=-1)
        x = torch.matmul(x, v)
        x = einops.rearrange(x, "b h s hd  -> b s (h hd)")
        x = self.o(x)
        return x
