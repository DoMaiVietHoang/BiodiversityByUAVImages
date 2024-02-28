import torch
import torch.nn as nn
from torchvision import models

class EfficientNet(nn.Module):
    def __init__(self, num_classes = 7, softmax = True) -> None:
        super().__init__()
        self.softmax= softmax
        self.model = models.efficientnet_v2_l(weights = models.EfficientNet_V2_L_Weights)
        self.model.classifier[1] = nn.Sequential(
            nn.BatchNorm2d(1280),
            nn.Dropout(p=0.5,inplace = True),
            nn.Linear(1280,num_classes,bias=True)
            
        )
    def forward(self,x):
        x = self.model(x)
        if self.softmax:
            x = nn.Softmax(dim=1)
        return x
model = EfficientNet()
print(model)
