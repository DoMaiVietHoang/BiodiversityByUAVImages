import torch
from torchvision import models
import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(self, num_class=1000, softmax = False) :
        super().__init__()
        self.softmax= softmax
        self.model = models.resnet50(weights = 'ResNet50_Weights.DEFAULT' )
        self.model.fc = nn.Sequential(
            nn.BatchNorm2d(num_features=2048, eps=1e-4),
            nn.Linear(2048,num_class,bias=True)
        )
    def forward(self,x):
        x = self.model(x)
        if self.softmax == True:
            x = nn.Softmax(dim=1)
        return x
        
model = ResNet50()

