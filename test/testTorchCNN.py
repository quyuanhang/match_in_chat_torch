import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
sys.path.append('../src/')
import numpy as np
import torch
from networks.torchCNN import TextCNN


if __name__ == '__main__':
    cnn = TextCNN(200, 100, 25, 50)
    data = np.random.randint(200, size=[10, 25, 50])
    data = torch.LongTensor(data)
    out = cnn.forward(data)
    print(out.shape)