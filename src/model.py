import torch.nn as nn

class FullyConnected(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.net(x)