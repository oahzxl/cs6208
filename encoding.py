import einops
import torch
from torch import nn


class CentralityEncoding(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(self, num_heads, num_patches, patch_size, num_degree, hidden_dim):
        super(CentralityEncoding, self).__init__()
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_encoder = nn.Linear(3 * patch_size * patch_size, hidden_dim)
        self.degree_encoder = nn.Embedding(num_degree, hidden_dim)

    def forward(self, x):
        x = einops.rearrange(
            x,
            "b c (h ph) (w pw) -> b (h w) (ph pw c)",
            h=self.num_patches,
            w=self.num_patches,
        )

        x = self.patch_encoder(x)

        degree = (
            torch.zeros([self.num_patches, self.num_patches], dtype=torch.int).cuda()
            + 2
        )
        degree[0, :] -= 1
        degree[-1, :] -= 1
        degree[:, 0] -= 1
        degree[:, -1] -= 1
        degree = self.degree_encoder(degree)
        degree = einops.rearrange(degree, "h w d -> (h w) d").unsqueeze(0)
        x = x + degree
        return x


class SpatialEncoding(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(self, num_heads, num_patches):
        super(SpatialEncoding, self).__init__()
        self.spatial_encoder = nn.Embedding(2 * num_patches, num_heads)
        self.num_heads = num_heads
        self.num_patches = num_patches
        self.spatial_bias = torch.zeros(
            [self.num_patches**2 + 1, self.num_patches**2 + 1], dtype=torch.int
        ).cuda()
        for i in range(self.spatial_bias.shape[0] - 1):
            for j in range(self.spatial_bias.shape[1] - 1):
                h = abs(i - j) // self.num_patches
                v = abs(i - j) % self.num_patches
                self.spatial_bias[i][j] = h + v

    def forward(self):
        spatial_bias = self.spatial_encoder(self.spatial_bias)
        spatial_bias = einops.rearrange(spatial_bias, "h w hd -> hd h w").unsqueeze(0)
        return spatial_bias


class SpecialEncoding(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(self, dim):
        super(SpecialEncoding, self).__init__()
        self.special_encoder = nn.Embedding(1, dim)

    def forward(self, x):
        special_encoder = self.special_encoder.weight.data
        special_encoder = special_encoder.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = torch.cat([x, special_encoder], dim=1)
        return x


class EdgeEncoding(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(self, num_heads, num_patches):
        super(EdgeEncoding, self).__init__()
        self.edge_encoder = nn.Embedding(12, num_heads)
        self.num_heads = num_heads
        self.num_patches = num_patches
        self.edge_bias = torch.zeros(
            [self.num_patches**2 + 1, self.num_patches**2 + 1], dtype=torch.int
        ).cuda()
        for i in range(self.edge_bias.shape[0] - 1):
            for j in range(self.edge_bias.shape[1] - 1):
                h = abs(i - j) // self.num_patches
                v = abs(i - j) % self.num_patches
                edge = torch.angle(
                    torch.complex(torch.tensor(float(h)), torch.tensor(float(v)))
                )
                edge = int((edge / 3.14159 + 1) * 6)
                self.edge_bias[i][j] = edge

    def forward(self):
        edge_bias = self.edge_encoder(self.edge_bias)
        edge_bias = einops.rearrange(edge_bias, "h w hd -> hd h w").unsqueeze(0)
        return edge_bias
