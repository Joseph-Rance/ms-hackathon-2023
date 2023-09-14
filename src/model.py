import torch.nn as nn

class FullyConnected(nn.Module):

    def __init__(self, classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.net(x)