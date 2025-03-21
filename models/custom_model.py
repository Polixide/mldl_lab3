import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.flatten = nn.Flatten()

        # Define layers of the neural network
        self.layers = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=3, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.MaxPool2d(2,2),
          nn.Conv2d(64, 128, kernel_size=3, padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),
          nn.Flatten(),
          nn.Linear(128 * 56 * 56, 200)
        )

    def forward(self, x):
        logits = self.layers(x)

        return logits