import torch.nn as nn
import torch
import einops


class Graphormer(nn.Module):
    def __init__(self, dim, head_num, layer_num, num_class=10):
        super().__init__()
        self.fc_in = nn.Linear(3, dim)
        self.layers = nn.ModuleList(
            [GraphormerLayer(dim, head_num) for _ in range(layer_num)]
        )
        self.fc_out1 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.fc_out2 = nn.Linear(dim, num_class)

    def forward(self, x):
        x = self.pre_process(x)
        for layer in self.layers:
            x = layer(x)
        x = self.post_process(x)
        return x

    def pre_process(self, x):
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        x = self.fc_in(x)
        return x

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

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x = self.attn(x)
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

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = einops.rearrange(q, "b s (h hd) -> b h s hd", h=self.head_dim)
        k = einops.rearrange(k, "b s (h hd) -> b h s hd", h=self.head_dim)
        v = einops.rearrange(v, "b s (h hd) -> b h s hd", h=self.head_dim)
        x = torch.matmul(q * self.scaling, k.transpose(-1, -2))
        x = torch.softmax(x, dim=-1)
        x = torch.matmul(x, v)
        x = einops.rearrange(x, "b h s hd  -> b s (h hd)")
        x = self.o(x)
        return x
