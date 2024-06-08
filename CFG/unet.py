import torch
import torch.nn as nn
import torch.functional as F


class doubleconv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel,out_channel,kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(out_channel)
    def forward(self,x): 
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        out = self.relu(x)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.doubleconv_1 = doubleconv(in_channel,64)
        self.doubleconv_2 = doubleconv(64,128)
        self.doubleconv_3 = doubleconv(128,256)
        self.doubleconv_4 = doubleconv(256,512)
        self.doubleconv_5 = doubleconv(512,1024)
        self.pool = nn.MaxPool2d(kernel_size=2 , stride=2)
    def forward(self,x):
        x = self.doubleconv_1(x)
        x = self.pool(x)
        x = self.doubleconv_2(x)
        x = self.pool(x)
        x = self.doubleconv_3(x)
        x = self.pool(x)
        x = self.doubleconv_4(x)
        x = self.pool(x)
        out = self.doubleconv_5(x)
        return out


model = Encoder(3)
x = torch.randn(1, 3, 572, 572)
y = model(x)
print(y.shape)



        