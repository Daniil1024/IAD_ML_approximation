import FrEIA.framework as Ff
import FrEIA.modules as Fm
from typing import Iterable
from torch import nn, Tensor


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 60), nn.ReLU(),
                         nn.Linear(60, 60), nn.ReLU(),
                         nn.Linear(60, 60), nn.ReLU(),
                         nn.Linear(60, dims_out))

class MyINN(Ff.SequenceINN):
    def __init__(self, N_DIM):
        super(MyINN, self).__init__(N_DIM)
        for k in range(12):
            self.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    def forward(self, x_or_z: Tensor, c: Iterable[Tensor] = None,
                rev: bool = False, jac: bool = True) -> Tensor:
        return super(MyINN, self).forward(x_or_z, c, rev, jac)[0]
    def forward_with_jac(self, x_or_z: Tensor, c: Iterable[Tensor] = None,
                rev: bool = False, jac: bool = True) -> Tensor:
        return super(MyINN, self).forward(x_or_z, c, rev, jac)
