import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
sys.path.append('../src/')
import torch
import numpy as np
from networks.torchBiMLP import BiMLP

if __name__ == '__main__':
    jd = np.random.randint(10, size=[128, 200])
    cv = np.random.randint(10, size=[128, 200])

    JD = torch.Tensor(jd)
    CV = torch.Tensor(cv)
    bimlp = BiMLP(100)
    out = bimlp.forward(JD, CV)
    print(out)
