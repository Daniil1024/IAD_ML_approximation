from torch import nn
class FCNN(nn.Sequential):
    def __init__(self, N_DIM):
        super(FCNN, self).__init__(nn.Linear(N_DIM, 60), nn.ReLU(),
                         nn.Linear(60, 120), nn.ReLU(),
                         nn.Linear(120, 120), nn.ReLU(),
                         nn.Linear(120, 60), nn.ReLU(),
                         nn.Linear(60, N_DIM))

