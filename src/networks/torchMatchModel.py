import torch
from torch import nn
from .torchCNN import TextCNN
from .torchAttention import Attention
from .torchBiMLP import BiMLP


class MatchModel(nn.Module):
    def __init__(self, n_word, emb_dim, sent_len, doc_len, emb_pretrain=[]):
        super(MatchModel, self).__init__()
        self.emb_dim = emb_dim
        self.sent_len = sent_len
        self.doc_len = doc_len
        self.n_word = n_word
        self.emb_pretrain = emb_pretrain
        self.jd_cnn = TextCNN(n_word, emb_dim, doc_len, sent_len, emb_pretrain)
        self.cv_cnn = TextCNN(n_word, emb_dim, doc_len, sent_len, emb_pretrain)
        self.att = Attention(doc_len)
        self.mlp = BiMLP(emb_dim)

    def forward(self, jd, cv, inf_mask, zero_mask):
        jd = self.jd_cnn.forward(jd)
        cv = self.cv_cnn.forward(cv)
        jd, cv = self.att.forward(jd, cv, inf_mask, zero_mask)
        # jd = torch.max(jd, dim=1)[0]
        # cv = torch.max(cv, dim=1)[0]
        # jd = torch.mean(jd, dim=1)
        # cv = torch.mean(cv, dim=1)
        score = self.mlp.forward(jd, cv)
        return score

    def get_masks(self, jd_data_np, cv_data_np):
        return self.att.get_masks(jd_data_np, cv_data_np)

