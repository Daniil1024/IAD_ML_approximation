import FrEIA.framework as Ff
import FrEIA.modules as Fm
from typing import Iterable, Tuple, List
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.utils.data import DataLoader
import config as c
from data_processing.DIS_dataset import DIS_dataset

def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 100), nn.ReLU(),
                         nn.Linear(100, dims_out))

class MySmallINN(Ff.SequenceINN):
    def __init__(self, N_DIM):
        super(MySmallINN, self).__init__(N_DIM)
        for k in range(4):
            self.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    def forward(self, x_or_z: Tensor, c: Iterable[Tensor] = None,
                rev: bool = False, jac: bool = False) -> Tensor:
        return super(MySmallINN, self).forward(x_or_z, c, rev, jac)[0]
    def forward_with_jac(self, x_or_z: Tensor, c: Iterable[Tensor] = None,
                rev: bool = False, jac: bool = False) -> Tensor:
        return super(MySmallINN, self).forward(x_or_z, c, rev, jac)