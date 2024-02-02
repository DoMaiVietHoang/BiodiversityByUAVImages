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
    '''

    building block of inception-v1 architecture. creates following 4 branches and concatenate them
    (a) branch1: 1x1 conv
    (b) branch2: 1x1 conv followed by 3x3 conv
    (c) branch3: 1x1 conv followed by 5x5 conv
    (d) branch4: Maxpool2d followed by 1x1 conv

        Note:
            1. output and input feature map height and width should remain the same. Only the channel output should change. eg. 28x28x192 -> 28x28x256
            2. To generate same height and width of output feature map as the input feature map, following should be padding for
                * 1x1 conv : p=0
                * 3x3 conv : p=1
                * 5x5 conv : p=2


    Args:
       in_channels (int) : # of input channels
       out_1x1 (int) : number of output channels for branch 1
       red_3x3 (int) : reduced 3x3 referring to output channels of 1x1 conv just before 3x3 in branch2
       out_3x3 (int) : number of output channels for branch 2
       red_5x5 (int) : reduced 5x5 referring to output channels of 1x1 conv just before 5x5 in branch3
       out_5x5 (int) : number of output channels for branch 3
       out_1x1_pooling (int) : number of output channels for branch 4

    Attributes:
        concatenated feature maps from all 4 branches constituiting output of Inception module.

    '''
    def __init__(self , in_channels , out_1x1 , red_3x3 , out_3x3 , red_5x5 , out_5x5 , out_1x1_pooling):
        super(InceptionBlock,self).__init__()

        # branch1 : k=1,s=1,p=0
        self.branch1 = ConvBlock(in_channels,out_1x1,1,1,0)

        # branch2 : k=1,s=1,p=0 -> k=3,s=1,p=1
        self.branch2 = nn.Sequential(ConvBlock(in_channels,red_3x3,1,1,0),ConvBlock(red_3x3,out_3x3,3,1,1))

        # branch3 : k=1,s=1,p=0 -> k=5,s=1,p=2
        self.branch3 = nn.Sequential(ConvBlock(in_channels,red_5x5,1,1,0),ConvBlock(red_5x5,out_5x5,5,1,2))

        # branch4 : pool(k=3,s=1,p=1) -> k=1,s=1,p=0
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3,stride=1,padding=1),ConvBlock(in_channels,out_1x1_pooling,1,1,0))
    def forward(self,x):

        # concatenation from dim=1 as dim=0 represents batchsize
        return torch.cat([self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x)],dim=1)


class Inceptionv1(nn.Module):
    '''
    step-by-step building the inceptionv1 architecture. Using testInceptionv1 to evaluate the dimensions of output after each layer and deciding the padding number.

    Args:
        in_channels (int) : input channels. 3 for RGB image
        num_classes : number of classes of training dataset

    Attributes:
        inceptionv1 model

    For conv2 2 layers with first having 1x1 conv
    '''

    def __init__(self , in_channels , num_classes ):
        super(Inceptionv1,self).__init__()

        self.conv1 =  ConvBlock(in_channels,64,7,2,3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.conv2 =  nn.Sequential(ConvBlock(64,64,1,1,0),ConvBlock(64,192,3,1,1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        # in_channels , out_1x1 , red_3x3 , out_3x3 , red_5x5 , out_5x5 , out_1x1_pooling
        self.inception3a = InceptionBlock(192,64,96,128,16,32,32)
        self.inception3b = InceptionBlock(256,128,128,192,32,96,64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception4a = InceptionBlock(480,192,96,208,16,48,64)
        self.inception4b = InceptionBlock(512,160,112,224,24,64,64)
        self.inception4c = InceptionBlock(512,128,128,256,24,64,64)
        self.inception4d = InceptionBlock(512,112,144,288,32,64,64)
        self.inception4e = InceptionBlock(528,256,160,320,32,128,128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception5a = InceptionBlock(832,256,160,320,32,128,128)
        self.inception5b = InceptionBlock(832,384,192,384,48,128,128)

        self.avgpool = nn.AvgPool2d(kernel_size = 7 , stride = 1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear( 1024 , num_classes)


    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x


