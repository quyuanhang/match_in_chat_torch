import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
sys.path.append('../src/')
import torch
from torch import nn
import numpy as np
from networks.torchMatchModel import MatchModel
import tensorboardX
from tensorboardX import SummaryWriter

if __name__ == '__main__':

    model = MatchModel(200, 100, 50, 25)

    data1 = np.random.randint(200, size=[10, 25, 50])
    data2 = np.random.randint(200, size=[10, 25, 50])
    inf_mask, zero_mask = model.get_masks(data1, data2)
    label = np.random.randint(2, size=[10])

    JD = torch.LongTensor(data1)
    CV = torch.LongTensor(data2)
    INF_MASK = torch.FloatTensor(inf_mask)
    ZERO_MASK = torch.FloatTensor(zero_mask)

    out = model.forward(JD, CV, INF_MASK, ZERO_MASK)

    writer = SummaryWriter()
    writer.add_graph(model=model, input_to_model=(JD, CV, INF_MASK, ZERO_MASK))

    print(out)

