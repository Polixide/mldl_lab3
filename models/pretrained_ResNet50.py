import torch
import torchvision.models as models
import torch.nn as nn

# Definisci la classe del modello ResNet50
class ResNet50Model(nn.Module):
    def __init__(self, num_classes=200):
        super(ResNet50Model, self).__init__()
        # Carica ResNet50 pre-addestrato
        self.model = models.resnet50(pretrained=True)
        
        # Modifica l'ultimo layer per adattarlo al numero di classi desiderato
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)