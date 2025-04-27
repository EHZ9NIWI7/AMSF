from torch import nn


class Transformer_Block(nn.Module):
    def __init__(self, width, n_head, **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.mha = nn.MultiheadAttention(width, n_head, bias=False)
        self.mlp = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.LayerNorm(width * 4),
            nn.Linear(width * 4, width),
        )
    
    def forward(self, x):
        x = self.ln(x)
        x = x + self.mha(x, x, x, need_weights=False)[0]
        x = x + self.mlp(x)

        return x
    

class Former(nn.Module):
    def __init__(self, channel, in_dim, out_dim, num_layers=1, num_heads=8, **kwargs):
        super().__init__()
        # self.positional_embedding = nn.Parameter(torch.empty(channel, in_dim), requires_grad=True)
        # nn.init.normal_(self.positional_embedding, std=0.01)
        self.transformer = nn.Sequential(*[Transformer_Block(width=in_dim, n_head=num_heads) for _ in range(num_layers)])
        self.linear = nn.Sequential(
            nn.LayerNorm(channel * in_dim),
            nn.Linear(channel * in_dim, out_dim, bias=False)
        )

    def forward(self, t):
        # x = t + self.positional_embedding.to(t.device)
        x = t.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        r = x.reshape(x.size(0), -1)
        r = self.linear(r)

        return r
