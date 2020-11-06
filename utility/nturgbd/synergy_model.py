import torch
from torch import nn


class Net_one(nn.Module):   
    def __init__(self):
        super(Net_one, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=(3, 5), stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(4, 4, kernel_size=(3, 5), stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 60)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x