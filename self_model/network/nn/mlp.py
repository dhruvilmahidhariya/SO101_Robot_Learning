import torch.nn as nn

class WorldModel(nn.module):
    " predict delta q, delta q_d from [q,dq,a]"

    def __init__(self) -> None:
        super().__init__()
        self.net=nn.sequencial(
            nn.Linear(18,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,12),
        )
    def forward(self,x):
        return self.net(x)