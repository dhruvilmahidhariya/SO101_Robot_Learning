import torch.nn as nn


class WorldModel(nn.Module):
    """Predict delta q, delta qd from [q, qd, a]."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,12),
        )
    def forward(self,x):
        return self.net(x)