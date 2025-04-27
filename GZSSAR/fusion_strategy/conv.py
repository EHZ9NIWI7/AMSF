import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, channel, dim, ks=3): # ks = 3, 5, 7, 9
        super().__init__()
        self.conv = nn.Conv1d(channel, 1, ks, padding=(ks - 1) / 2)
        self.linear = nn.Linear(dim, channel * dim)

    def forward(self, t):
        x = self.conv(t).squeeze(1)
        r = self.linear(x)

        return r


class Conv_1(nn.Module):
    def __init__(self, channel, dim):
        super().__init__()
        self.conv = nn.Conv1d(channel, 1, 3, padding=1) # in, out, kernel_size, stride, padding
        self.linear = nn.Linear(dim, channel * dim)
    
    def forward(self, t):
        r = self.conv(t).squeeze(1)
        r = self.linear(r)

        return r


class Conv_2(nn.Module):
    def __init__(self, channel, dim) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channel, 1, 5, padding=2)
    
    def forward(self, t):
        r = self.conv(t).squeeze(1)

        return r
    

class Conv_3(nn.Module):
    def __init__(self, channel, dim):
        super().__init__()
        self.conv = nn.Conv1d(channel, 1, 7, padding=3)
    
    def forward(self, t):
        r = self.conv(t).squeeze(1)

        return r


class Conv_4(nn.Module):
    def __init__(self, channel, dim) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channel, 1, 9, padding=4)
    
    def forward(self, t):
        r = self.conv(t).squeeze(1)

        return r
