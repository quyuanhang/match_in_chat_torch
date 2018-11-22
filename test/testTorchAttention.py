import torch
import numpy as np
import sys
sys.path.append('../')
from networks.torchCNN import TextCNN
from networks.torchAttention import Attention


if __name__ == '__main__':

    data1 = np.random.randint(200, size=[10, 25, 50])
    data2 = np.random.randint(200, size=[10, 25, 50])
    for i in range(1, 10):
        data1[i, -i:, :] = 0
        data1[i, :, -i:] = 0

    X1 = torch.LongTensor(data1)
    X2 = torch.LongTensor(data2)

    cnn1 = TextCNN(200, 100, 25, 50)
    Y1 = cnn1.forward(X1)

    cnn2 = TextCNN(200, 100, 25, 50)
    Y2 = cnn2.forward(X2)

    attention = Attention(25)
    inf_mask, zero_mask = attention.get_masks(data1, data2)
    INF_MASK = torch.FloatTensor(inf_mask)
    ZERO_MASK = torch.FloatTensor(zero_mask)
    out1, out2 = attention.forward(Y1, Y2, INF_MASK, ZERO_MASK)

    print(out2.shape)



