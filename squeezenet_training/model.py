import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torchvision.transforms as transforms
import torchvision



class Model(nn.Module):
    def __init__(self, extractor):
        super(Model, self).__init__()

        self.final_final_conv = nn.Conv2d(512,2, kernel_size=1) 
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.extractor = extractor

    def forward(self, x):
        x = self.pool(self.final_final_conv(self.extractor(x)))
        return torch.sigmoid(x.reshape(-1, 2))



