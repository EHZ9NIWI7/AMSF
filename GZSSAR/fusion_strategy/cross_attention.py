import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x


class CAM(nn.Module):
    def __init__(self, dim, bias=False, mlp_ratio=4., act_layer=nn.GELU, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.tensor(0.125), requires_grad=True)
        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)
        self.v = nn.Linear(dim, dim, bias=bias)
        self.proj = nn.Linear(dim, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
    
    def qkv(self, x, y):
        q = self.q(self.norm1(x))
        k = self.k(self.norm1(y))
        v = self.v(self.norm1(y))
        
        return q, k, v

    def forward(self):
        pass


class CA_1(CAM):
    def forward(self, t):
        size = t.size(1)
        x = t[:, 0, :]
        for i in range(1, size):
            y = t[:, i, :]
            q, k, v = self.qkv(x, y)
            att = q * k * self.scale
            att = att.softmax(dim=-1)
            x = att * v
                
        return x

    
class CA_2(CAM):
    def forward(self, t):
        size = t.size(1)
        x = t[:, 0, :]
        rl = []
        for i in range(size):
            x = t[:, i, :]
            q = self.q(self.norm1(x))
            for j in range(t.shape[1]):
                y = t[:, j, :]
                k = self.k(self.norm1(y))
                v = self.v(self.norm1(y))
                att = torch.bmm(q.unsqueeze(-1), k.unsqueeze(-2)) * self.scale
                att = att.softmax(dim=-1)
                r = torch.bmm(att, v.unsqueeze(-1)).squeeze(-1)
                rl.append(r)
        result = sum(rl) / len(rl)
        
        return result


class CA_3(CAM):
    def forward(self, t):
        size = t.size(1)
        rl = []
        for i in range(size):
            x = t[:, i, :]
            rl.append(x)
            for j in range(size):
                if j == i: continue
                y = t[:, j, :]
                q, k, v = self.qkv(x, y)
                att = torch.bmm(q.unsqueeze(-1), k.unsqueeze(-2)) * self.scale
                att = att.softmax(dim=-1)
                a = torch.bmm(att, v.unsqueeze(-1)).squeeze(-1)
                rl.append(a)
        result = sum(rl) / len(rl)

        return result


class CA_4(CAM):
    def __init__(self, dim, **kwargs):
        super().__init__(dim)
    
    def forward(self, t):
        size = t.size(1)
        x = t[:, 0, :]
        for i in range(1, size):
            y = t[:, i, :]
            q, k, v = self.qkv(x, y)
            att = torch.bmm(q.unsqueeze(-1), k.unsqueeze(-2)) * self.scale
            att = att.softmax(dim=-1)
            a = torch.bmm(att, v.unsqueeze(-1)).squeeze(-1)
            x = x + a
        
        return x

