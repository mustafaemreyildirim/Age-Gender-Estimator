import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from dataloader import *
import torchvision.transforms as transforms
import torchvision



class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, 2, 0)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.bn1(self.conv1(x))
        z = self.bn2(self.conv2(x))
        return F.relu_(y + z)

class Model(nn.Module):
    def __init__(self, channels):
        super(Model, self).__init__()
        in_channels, out_channels = 3, channels

        modules = []
        for i in range(4):
            modules.append(Block(in_channels, out_channels))
            in_channels, out_channels = out_channels, out_channels * 2
        
        modules.append(nn.AvgPool2d(4))
        modules.append(nn.Conv2d(64, 2, 1, 1,0))
 
        self.model = nn.Sequential(*modules)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,0)

    def forward(self, x):
        return torch.sigmoid(self.model.forward(x).reshape(-1, 2))
