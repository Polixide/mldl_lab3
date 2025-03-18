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
          nn.GELU(),
          nn.Conv2d(64, 128, kernel_size=3, padding=1),
          nn.BatchNorm2d(128),
          nn.GELU(),
          nn.MaxPool2d(2, 2),

          nn.Conv2d(128, 256, kernel_size=3, padding=1),
          nn.BatchNorm2d(256),
          nn.GELU(),
          nn.Conv2d(256, 512, kernel_size=3, padding=1),
          nn.BatchNorm2d(512),
          nn.GELU(),
          nn.MaxPool2d(2, 2),

          nn.Conv2d(512, 1024, kernel_size=3, padding=1),
          nn.BatchNorm2d(1024),
          nn.GELU(),
          nn.MaxPool2d(2, 2),

          nn.Flatten(),
          nn.Linear(1024 * 28 * 28, 4096),
          nn.GELU(),
          nn.Dropout(0.5),
          nn.Linear(4096, 200),
        )

    def forward(self, x):
        logits = self.layers(x)

        return logits