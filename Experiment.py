import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision import transforms, datasets
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
data_dir = './Data'
e_pochs = 100
l_r=0.0001
batch_size = 8
criter = nn.CrossEntropyLoss()

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])
data = datasets.ImageFolder(root = data_dir, transform = preprocess)
train_loader = DataLoader(dataset=data, shuffle=False, batch_size=8)
EfficientNet = models.efficientnet_b7(weights= models.EfficientNet)
EfficientNet.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features=2046, out_features=7),
    nn.Softmax(dim=1)
)
optimizer = optim.SGD(EfficientNet.parameters(), lr=l_r)
for ep in range(e_pochs):
    for img, label in train_loader:
        batch_size = img.shape[0]
        out = EfficientNet(img.view(batch_size,-1))
        loss = loss(out, torch.tensor([label]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: %d, Loss: %f" % (ep, float(loss)))
