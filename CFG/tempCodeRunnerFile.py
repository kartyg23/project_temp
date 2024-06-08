class Encoder(nn.Module):
#     def __init__(self, in_channel):
#         super().__init__()
#         self.doubleconv_1 = doubleconv(in_channel,32)
#         self.doubleconv_2 = doubleconv(32,64)
#         self.doubleconv_3 = doubleconv(64,128)
#         self.doubleconv_4 = doubleconv(128,256)
#         self.doubleconv_5 = doubleconv(256,512)
#         self.pool = nn.MaxPool2d(kernel_size=2 , stride=2)
#     def forward(self,x):
#         x = self.doubleconv_1(x)
#         x = self.pool(x)
#         x = self.doubleconv_2(x)
#         x = self.pool(x)
#         x = self.doubleconv_3(x)
#         x = self.pool(x)
#         x = self.doubleconv_4(x)
#         x = self.pool(x)
#         out = self.doubleconv_5(x)
#         return x