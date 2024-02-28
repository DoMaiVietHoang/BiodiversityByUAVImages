from Model import EfficientNet
from torchvision import datasets, transforms
###defineMetrics
data_transfor, = transforms.Compose()
model = EfficientNet( num_classes = 10, softmax = True)



