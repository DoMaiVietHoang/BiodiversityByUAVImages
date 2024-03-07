from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
#defineMetrics
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
epochs = 100
criterion = nn.CrossEntropyLoss()
# define model
import torch
from torchvision import models
import os
import torch.nn as nn
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, softmax = False) :
        super().__init__()
        self.softmax= softmax
        self.model = models.resnet50(weights = 'ResNet50_Weights.DEFAULT' )
        self.model.fc = nn.Sequential(
            #nn.BatchNorm2d(num_features=2048, eps=1e-4),
            nn.Linear(2048,num_classes,bias=True)
        )
    def forward(self,x):
        x = self.model(x)
        return x
    
class EfficientNet(nn.Module):
    def __init__(self, num_classes = 7, softmax = True) -> None:
        super().__init__()
        self.softmax= softmax
        self.model = models.efficientnet_v2_l(weights = models.EfficientNet_V2_L_Weights)
        self.model.classifier[1] = nn.Sequential(
           
            nn.Dropout(p=0.5,inplace = True),
            nn.Linear(1280,num_classes,bias=True)
            
        )
    def forward(self,x):
        x = self.model(x)
        return x
class Mobinet(nn.Module):
    def __init__(self, num_classes = 7, softmax = True) -> None:
        super().__init__()
        self.softmax= softmax
        self.model = models.mobilenet_v2(weights = models.MobileNetV2)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3,inplace = True),
            nn.Linear(1280,num_classes,bias=True)
        )
    def forward(self,x):
        x = self.model(x)
        return x

### Data preprocessing
data_transform = transforms.Compose([
    #transforms.RandomRotation(degrees=(-90,90)),
    transforms.Resize((224,224)),
    #transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std =[0.5,0.5,0.5]
    )

])
if __name__ == "__main__":
    Losshistory = []
    Acchistory   = []
    model = Mobinet(num_classes=7, softmax=False)
    # model.cuda()
    print(device)
    train_data = datasets.ImageFolder(root = "./Train", transform=data_transform)
    test_data  = datasets.ImageFolder(root = "./Test",transform=data_transform)
    train_loader = DataLoader(train_data,batch_size=4,num_workers=4,shuffle=True)
    test_loader  = DataLoader(test_data,batch_size=4,num_workers=4,shuffle=True)
    optimizer = optim.SGD(model.parameters(),lr=1e-3)
    loss = 0
    print(len(train_loader.dataset))
    print(len(test_loader.dataset))
    for i in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        print('Epoch {}/{}'.format(i+1, epochs))
        for images, labels in tqdm(train_loader):
            # images = images.cuda()
            # labels = labels.cuda()
            outputs = model(images)
            loss  = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs.data, 1)
            running_loss     += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects.double() / (len(train_loader.dataset))
        Losshistory.append(running_loss)
        Acchistory.append(epoch_acc)
        running_loss = 0.0
        running_corrects = 0
        torch.save(model.state_dict(), './modelsave/mOBInET.pth')
        print("Trainning Information: Loss{:.3f}, Accuracy {:.3f}".format(epoch_loss,epoch_acc))
    print(Losshistory)
    print(Acchistory)

                




