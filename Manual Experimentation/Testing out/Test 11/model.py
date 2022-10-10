from torch.nn import Module
from torch import nn


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(5, 8, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32,15)
        self.relu3 = nn.ReLU()
        self.fc3 = nn.Linear(15, 10)
        self.sf = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc3(y)
        y = self.sf(y)
        return y
