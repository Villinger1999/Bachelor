import torch.nn as nn
from torchvision import models
import sys

def get_model(num_classes=10):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def shallow_resnet(num_classes=10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    model.layer3 = nn.Identity()
    model.layer4 = nn.Identity()

    # after layer2, channels = 128
    model.fc = nn.Linear(128, num_classes)
    return model


# activation = sys.argv[2]
# if activation == "relu":
#     activation = nn.ReLU
# else:
#     activation = nn.Sigmoid

class LeNet(nn.Module):
    def __init__(self, channel: int = 3, hidden: int = 768, num_classes: int = 10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    # # https://www.kaggle.com/code/shravankumar147/lenet-pytorch-implementation

# class LeNet(nn.Module):
    
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2, stride=2),
#         )
       
#         self.classifier = nn.Sequential(
#             nn.Linear(400,120),  #in_features = 16 x5x5 
#             nn.ReLU(),
#             nn.Linear(120,84),
#             nn.ReLU(),
#             nn.Linear(84,10),
#         )
        
#     def forward(self,x): 
#         a1=self.feature_extractor(x)
#         # print(a1.shape)
#         a1 = torch.flatten(a1,1)
#         a2=self.classifier(a1)
#         return a2