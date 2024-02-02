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


def testInceptionBlock():
    x = torch.randn((32,192,28,28))
    model = InceptionBlock(192,64,96,128,16,32,32)
    print(model(x).shape)
    return model

model = testInceptionBlock()
