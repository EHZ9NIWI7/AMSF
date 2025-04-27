import torch.nn.functional as F
from torch import nn


class SA(nn.Module):
    def __init__(self, channel, in_dim, out_dim, ks=3, **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm([channel, in_dim])

        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim, bias=False),
        )

        self.q = nn.Linear(channel * in_dim, out_dim)
        self.k = nn.Linear(channel * in_dim, out_dim)
        self.v = nn.Linear(channel * in_dim, out_dim)

    def forward(self, t):
        x = self.ln(t).reshape(t.size(0), -1)
        q, k, v = self.q(x), self.k(x), self.v(x)
        r = F.scaled_dot_product_attention(q, k, v)
        r = self.mlp(r)
        r = r.reshape(t.size(0), -1)

        return r


class MSCA(nn.Module):
    def __init__(self, channel, in_dim, ks=1, r=3, **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm([channel, in_dim])

        inter_channel = int(channel // r)
        
        self.local_att = nn.Sequential(
            nn.Conv1d(channel, inter_channel, ks, stride=1, padding=int((ks - 1) / 2)),
            nn.LayerNorm([inter_channel, in_dim]),
            nn.GELU(),
            nn.Conv1d(inter_channel, channel, ks, stride=1, padding=int((ks - 1) / 2)),
            nn.LayerNorm([channel, in_dim])
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, inter_channel, ks, stride=1, padding=int((ks - 1) / 2)),
            nn.LayerNorm([inter_channel, 1]),
            nn.GELU(),
            nn.Conv1d(inter_channel, channel, ks, stride=1, padding=int((ks - 1) / 2)),
            nn.LayerNorm([channel, 1])
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, t):
        x = self.ln(t)
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        w = self.sigmoid(xlg)
        r = w * x
        r = r.reshape(r.size(0), -1)

        return r


class MHA_block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, heads, bias=False)
    
    def forward(self, x):
        x = self.ln(x)
        x = x + self.mha(x, x, x, need_weights=False)[0]

        return x


class MHA(nn.Module):
    def __init__(self, channel, in_dim, out_dim, num_layers=1, num_heads=8, **kwargs):
        super().__init__()
        self.mha = nn.Sequential(*[MHA_block(in_dim, num_heads) for _ in range(num_layers)])

        self.mlp = nn.Sequential(
            nn.LayerNorm(channel * in_dim),
            nn.Linear(channel * in_dim, in_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim, bias=False),
        )

    def forward(self, t):
        x = t.permute(1, 0, 2)
        x = self.mha(x)
        x = x.permute(1, 0, 2)
        x = x.reshape(x.size(0), -1)
        r = self.mlp(x)

        return r
