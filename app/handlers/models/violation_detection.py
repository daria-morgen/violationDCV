import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.act = nn.LeakyReLU(0.2)
        self.maxPool = nn.MaxPool2d(2,2)

        self.conv0 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.adaptivePool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64,10)
        self.linear2 = nn.Linear(10,3)

    def forward(self, X):
        out = self.conv0(X)
        out = self.act(out)
        out = self.maxPool(out)

        out = self.conv1(out)
        out = self.act(out)
        out = self.maxPool(out)

        out = self.conv2(out)
        out = self.act(out)
        out = self.maxPool(out)

        out = self.conv3(out)
        out = self.act(out)

        out = self.adaptivePool(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)

        return out



