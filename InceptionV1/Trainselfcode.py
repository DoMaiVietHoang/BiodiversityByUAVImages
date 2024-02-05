import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from Model import Inceptionv1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = 'E:/ModelDeepfromScratch/Data'
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
  
])
dataset = datasets.ImageFolder(data_dir, transform=transform)
train_size = int(0.7*len(dataset))
test_size  = len(dataset)-train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
