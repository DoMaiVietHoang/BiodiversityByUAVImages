import torch
import torch.nn as nn
from torchview import draw_graph
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride, padding,bias = False):
        super(ConvBlock,self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size= kernel_size,stride=stride,padding=padding, bias = False )
        self.batchnorm2d = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        return self.relu(self.batchnorm2d(self.conv2d(x)))
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_pooling):
        super(InceptionBlock,self).__init__()
        #brach 1
        self.branch1 = ConvBlock(in_channels,out_1x1,1,1,0)
        #branch2
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels,red_3x3,1,1,0),
            ConvBlock(red_3x3,out_3x3,3,1,1)
        )
        #branch3
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels,red_5x5,1,1,0),
            ConvBlock(red_5x5,out_5x5,3,1,2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            ConvBlock(in_channels,out_1x1_pooling,1,1,0)
        )
    def forward(self,x):
        return torch.cat([self.branch1(x),self.branch2(x), self.branch3(x),self.branch4(x)],dim=1)
    
def testInceptionBlock():
    x = torch.randn((32,192,28,28))
    model = InceptionBlock(192,64,96,128,16,32,32)
    print(model(x).shape)
    return model

model = testInceptionBlock()

