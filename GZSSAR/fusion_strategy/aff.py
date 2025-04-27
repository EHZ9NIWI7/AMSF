import torch
import torch.nn as nn


class MS_CAM(nn.Module):
    def __init__(self, dim=512, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(dim // r)

        self.local_att = nn.Sequential(
            nn.Conv1d(dim, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(dim),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(dim),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        w = self.sigmoid(xlg)
        return w
    
    
class AFF(nn.Module):
    def __init__(self, dim=512, r=4, **kwargs):
        super(AFF, self).__init__()
        self.cam = MS_CAM(dim, r)

    def forward(self, t):
        x = t[:, 0, :].unsqueeze(-1)
        for i in range(1, t.shape[1]):
            y = t[:, i, :].unsqueeze(-1)
            w = self.cam(x + y)
            x = (w * x + (1 - w) * y) * 2

        x = x.squeeze(-1)

        return x


class iAFF(nn.Module):
    def __init__(self, dim=512, r=4, **kwargs):
        super(iAFF, self).__init__()
        self.cam1 = MS_CAM(dim, r)
        self.cam2 = MS_CAM(dim, r)
    
    def forward(self, t):
        x = t[:, 0, :].unsqueeze(-1)
        for i in range(1, t.shape[1]):
            y = t[:, i, :].unsqueeze(-1)
            # x = self.fusion(x, y)
            w1 = self.cam1(x + y)
            mid = x * w1 + y * (1 - w1)
            w2 = self.cam2(mid)
            x = x * w2 + y * (1 - w2)

        x = x.squeeze(-1)

        return x


class AFM(nn.Module):
    def __init__(self, channel, dim=512, r=4, norm='sigmoid') -> None:
        super().__init__()
        inter_channels = dim // r
        self.channel = channel
        self.norm = nn.Sigmoid() if norm == 'sigmoid' else nn.Softmax(-1)
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(dim),
            )
            for _ in range(channel)
        ])
        
    def forward(self, x):
        l = [self.branches[i](x[:, :, i].unsqueeze(-1)) for i in range(self.channel)]
        w = self.norm(torch.concat(l, dim=-1))
        
        return w


class AF_1(nn.Module):
    def __init__(self, channel, dim=512, r=4, **kwargs):
        super().__init__()
        self.afm = AFM(channel, dim, r, 'softmax')
        
    def forward(self, t):
        t = t.transpose(1, 2)
        w = self.afm(t)
        t = (t * w).sum(-1).squeeze(-1) * 3
        
        return t
    
 
class AF_2(nn.Module):
    def __init__(self, channel, dim=512, r=4, **kwargs):
        super().__init__()
        self.nb = channel
        self.ml = nn.ModuleList([AFM(1, dim, r, 'sigmoid') for _ in range(self.nb)])
        
    def forward(self, t):
        t = t.transpose(1, 2)
        s = t.sum(-1).unsqueeze(-1)
        w = [self.ml[i](s).squeeze(-1) for i in range(self.nb)]
        r = sum([w[j] * t[:, :, j] for j in range(self.nb)]) * 3
        
        return r
    

class AF_3(AF_2):
    def __init__(self, channel, dim=512, r=4, **kwargs):
        super().__init__(channel, dim, r)
        self.ml2 = nn.ModuleList([AFM(1, dim, r, 'sigmoid') for _ in range(self.nb)])
        
    def forward(self, t):
        t = t.transpose(1, 2)
        s = t.sum(-1).unsqueeze(-1)
        v = [self.ml[i](s) for i in range(self.nb)]
        m = sum([v[j] * t[:, :, j].unsqueeze(-1) for j in range(self.nb)])
        w = [self.ml2[k](m).squeeze(-1) for k in range(self.nb)]
        r = sum([w[i] * t[:, :, i] for i in range(self.nb)])
        
        return r

    
class AF_4(AF_2):
    def __init__(self, channel, dim=512, r=4, **kwargs):
        super().__init__(channel, dim, r)
        self.scale = nn.Parameter(torch.tensor(1.), requires_grad=True)

    def forward(self, t):
        r = super().forward(t) * self.scale
        
        return r
