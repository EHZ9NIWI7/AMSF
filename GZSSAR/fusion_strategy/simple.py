import pdb

import torch
from torch import nn


class Prompt(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, t):
        return t


class Concat(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(self, t):
        r = t.reshape(t.size(0), -1)
        
        return r


class Add(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(self, t):
        r = torch.sum(t, dim=1)

        return r


class Weighted_Add(nn.Module):
    def __init__(self, channel, in_dim, **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.w = torch.nn.Parameter(torch.ones(channel, 1), requires_grad=True)

    def forward(self, t):
        r = (self.ln(t) * self.w.expand_as(t)).sum(1)

        return r


class Linear(nn.Module):
    def __init__(self, channel, in_dim, out_dim, **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm(channel * in_dim)
        self.linear = nn.Linear(in_dim * channel, out_dim, bias=False)

    def forward(self, t):
        x = t.reshape(t.size(0), -1)
        r = self.linear(self.ln(x))

        return r


class MLP(nn.Module):
    def __init__(self, channel, in_dim, out_dim, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(channel * in_dim),
            nn.Linear(channel * in_dim, in_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim, bias=False),
        )

    def forward(self, t):
        x = t.reshape(t.size(0), -1)
        r = self.mlp(x)

        return r


class Conv(nn.Module):
    def __init__(self, channel, ks=9, **kwargs): # ks = 1, 3, 5, 7, 9
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channel, 1, ks, stride=1, padding=(ks-1)//2)
        )
    def forward(self, t):
        r = self.conv(t).squeeze(1)

        return r

