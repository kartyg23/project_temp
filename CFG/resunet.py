import torch
import torch.nn as nn

class Res_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_channel)
        self.conv_skip = nn.Conv2d(in_channel, out_channel , kernel_size=3, stride=1, padding=1)
    def forward(self,x):
        skip = self.conv_skip(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x) 
        x = x + skip
        out = self.relu(x)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.Res_Block_1 = Res_Block(in_channel,64)
        self.Res_Block_2 = Res_Block(64,128)
        self.Res_Block_3 = Res_Block(128,256)
        self.Res_Block_4 = Res_Block(256,512)
        self.Res_Block_5 = Res_Block(512,1024)
        self.pool = nn.MaxPool2d(kernel_size=2 , stride=2)
    def forward(self,x):
        x = self.Res_Block_1(x)
        x = self.pool(x)
        x = self.Res_Block_2(x)
        x = self.pool(x)
        x = self.Res_Block_3(x)
        x = self.pool(x)
        x = self.Res_Block_4(x)
        x = self.pool(x)
        out = self.Res_Block_5(x)
        return out


model = Encoder(in_channel=3)
x = torch.randn(1, 3, 128, 128)
y = model(x)
print(y.shape)



        