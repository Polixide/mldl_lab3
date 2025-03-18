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
            nn.GELU(),
            nn.MaxPool2d(2, 2),  # Reduce spatial size from 224x224 -> 112x112
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),  # Reduce spatial size from 112x112 -> 56x56
            nn.Flatten(),
            nn.Linear(128 * 56 * 56, 200),  # Adjusted size based on pooling ,200 is the number of classes in TinyImageNet , last output channel size has to be equal to the number of classes we want to classify

        )

    def forward(self, x):
        logits = self.layers(x)

        return logits