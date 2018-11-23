import torch
from torch import nn
from torch.nn import functional


class BiMLP(nn.Module):
    def __init__(self, emb_dim):
        super(BiMLP, self).__init__()
        self.emb_dim = emb_dim
        self.bil = nn.Bilinear(
            in1_features=2 * self.emb_dim,
            # in1_features=self.emb_dim,
            in2_features=2 * self.emb_dim,
            # in2_features=self.emb_dim,
            out_features=self.emb_dim)
        self.mlp = nn.Linear(
                in_features=emb_dim,
                # in_features= 2 * emb_dim,
                out_features=1)

    def forward(self, geek, job):
        x = self.bil(geek, job)
        # x = torch.cat((geek, job), dim=1)
        x = torch.sigmoid(x)
        # x = functional.relu(x)
        x = self.mlp(x)
        x = torch.sigmoid(x)
        return x



