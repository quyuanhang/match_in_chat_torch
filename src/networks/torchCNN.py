import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, emb_dim, sent_len):
        super(CNN, self).__init__()
        self.sent_len = sent_len
        self.emb_dim = emb_dim
        self.cnn1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.emb_dim,
                kernel_size=(5, self.emb_dim),
                stride=1,
                padding=(2, 0)),
            nn.ReLU())
        self.cnn2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.emb_dim,
                kernel_size=(3, self.emb_dim),
                stride=1,
                padding=(1, 0)),
            nn.ReLU())
        self.max_pool = nn.MaxPool2d(kernel_size=(self.sent_len, 1))

    def forward(self, x):
        """
        :param x: 4d array: (batch * doc_len) * sent_len * emb_dim * 1
        :return: 4d array: (batch * doc_len) * 1 * 1 * emb_dim
        """
        x = self.cnn1(x)
        x = x.permute(0, 3, 2, 1)
        x = self.cnn2(x)
        x = self.max_pool(x)
        return x


class TextCNN(nn.Module):
    def __init__(self, n_word, emb_dim, doc_len, sent_len, emb_weights=[]):
        super(TextCNN, self).__init__()
        self.doc_len = doc_len
        self.sent_len = sent_len
        self.emb_dim = emb_dim
        self.emb_weights = nn.Embedding(num_embeddings=n_word, embedding_dim=emb_dim, padding_idx=0)
        if len(emb_weights) > 0:
            self.emb_weights.weight.data.copy_(torch.FloatTensor(emb_weights))
        self.cnn = CNN(emb_dim, sent_len)

    def forward(self, x):
        x = self.emb_weights(x)
        x = x.view(-1, 1, self.sent_len, self.emb_dim)
        x = self.cnn(x)
        x = x.view(-1, self.doc_len, self.emb_dim)
        return x
