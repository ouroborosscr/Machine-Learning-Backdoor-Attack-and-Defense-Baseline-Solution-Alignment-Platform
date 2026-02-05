import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_channel=3):
        super(SimpleCNN, self).__init__()
        self.m1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.m2 = nn.Sequential(
            nn.Dropout(0.5),

            nn.Linear(3200, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        n = x.size(0)
        x = self.m1(x)
        x = F.adaptive_avg_pool2d(x, (5, 5))
        x = x.view(n, -1)
        x = self.m2(x)
        return x