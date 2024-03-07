from torchvision import datasets, transforms
import torch
from PIL import Image
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
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
    model = Mobinet(num_classes=7, softmax=False)
    model.cuda()
    print(device)
    test_data  = datasets.ImageFolder(root = "./Test",transform=data_transform)
    test_loader  = DataLoader(test_data,batch_size=8,num_workers=4,shuffle=False)
    optimizer = optim.SGD(model.parameters(),lr=1e-3)
    loss = 0
    print(len(test_loader.dataset))
    model.load_state_dict(torch.load('E:\\ICCE\\modelsave\\mOBInET.pth'))
    model.eval()
    class_correct = torch.zeros(7)
    class_total = torch.zeros(7)
    class_correct = class_correct.double()
    # with torch.no_grad():
    #         for images, labels in test_loader:
    #             images = images.cuda()
    #             labels = labels.cuda()
    #             outputs = model(images)
    #             _, preds = torch.max(outputs.data, 1)
    #             correct = (preds == labels).squeeze()
    #             for i in range(7):
    #                 label = labels[i]
    #                 class_correct[label] += correct[i].item()
    #                 class_total[label] += 1
    # class_acc = class_correct / class_total
    # print(class_acc)
    predictions = []
    targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            images=images.cuda()
            labels=labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
            targets.extend(labels.tolist())
    print(predictions)
    print(targets)
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    cm = confusion_matrix(targets, predictions)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues', xticklabels=range(7), yticklabels=range(7))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix with Mobinet ArchiTech')
    plt.show()
    #######Grad-cam





