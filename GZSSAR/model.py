import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)

        nn.init.xavier_uniform_(m.weight,gain=0.5)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Encoder(nn.Module):
    def __init__(self, layer_sizes):
        super(Encoder, self).__init__()
    
        layers = []
        for i in range(len(layer_sizes)-2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
            
        self.model = nn.Sequential(*layers)

        self.mu_transformation = nn.Sequential(
            nn.Linear(layer_sizes[-2], layer_sizes[-1])
        )
        
        self.logvar_transformation = nn.Sequential(
            nn.Linear(layer_sizes[-2], layer_sizes[-1])
        )
        
        self.apply(weights_init)

    def forward(self, x):

        h = self.model(x)
        mu = self.mu_transformation(h)
        logvar = self.logvar_transformation(h)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, layer_sizes):
        super(Decoder, self).__init__()

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
            
        self.model = nn.Sequential(*layers)
        
        self.apply(weights_init)

    def forward(self, x):

        out = self.model(x)        
        return out
        
class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)
        self.apply(weights_init)
    
    def forward(self, x):
        return self.model(x)
    
class LogisticRegression(nn.Module):
    def __init__(self, num_class, num_unseen):
        super(LogisticRegression, self).__init__()
        self.num_u = num_unseen
        self.num_c = num_class
        
        self.linear = nn.Linear(self.num_u * 2, 1) # + self.num_c
        self.l2 = nn.Linear(64, 1)
        self.bn = nn.BatchNorm1d(64)
        self.apply(weights_init)

    def forward(self, x):
        # k = torch.cat((self.k_u.expand(self.num_u), self.k_s.expand(self.num_c)), dim=0)
        # m = k * x
        
        u = x[:, :self.num_u]
        s = x[:, self.num_u:].sort(1, descending=True)[0][:, :self.num_u]
        
        # eu, es = torch.exp(u), torch.exp(s)
        # nu, ns = eu / sum(u), es / sum(s)
        
        # nu, ns = self.ul(u), self.sl(s)
        
        # nu, ns = self.ubn(self.ul(self.ubn(u))), self.sbn(self.sl(self.sbn(s)))
        
        x = torch.cat((u, s), dim=-1)
        
        r = self.linear(x).squeeze(-1)
        
        # r = self.l2(self.bn(r))
        
        return r