import torch
from torch import nn


class BiMLP(nn.Module):
    def __init__(self, emb_dim):
        super(BiMLP, self).__init__()
        self.emb_dim = emb_dim
        self.bil = nn.Bilinear(
            in1_features=2 * self.emb_dim,
            in2_features=2 * self.emb_dim,
            out_features=self.emb_dim)
        self.mlp = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(
                in_features=emb_dim,
                out_features=1),
            nn.Sigmoid())

    def forward(self, geek, job):
        mix = self.bil(geek, job)
        match_score = self.mlp(mix)
        return match_score



