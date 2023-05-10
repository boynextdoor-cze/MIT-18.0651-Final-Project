import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSample(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel, out_channel,3,2,1),
            nn.BatchNorm2d(out_channel),
            # nn.Dropout2d(0.3),
            nn.LeakyReLU(),
        )
    def forward(self,x):
  	    return self.layer(x)

class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample, self).__init__()
        # self.layer=nn.Conv2d(channel,channel//2,1,1)
        self.layer = nn.ConvTranspose2d(channel, channel//2, kernel_size = (3,3), stride = 2, padding = 1,output_padding=1)

    def forward(self,x):
        # up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(x)
        return out

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            DownSample(3,64),
            DownSample(64,128),
            DownSample(128,256),
            DownSample(256,512),
        )
        self.decoder = nn.Sequential(
            UpSample(512),
            UpSample(256),
            UpSample(128),
            UpSample(64),
            nn.Conv2d(32,3,3,1,1)
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode
    
    
# class DownSample(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(DownSample, self).__init__()
#         self.layer = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, 3, 1, 1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#     def forward(self, x):
#         return self.layer(x)


# class UpSample(nn.Module):
#     def __init__(self, channel):
#         super(UpSample, self).__init__()
#         self.layer = nn.Sequential(
#             nn.ConvTranspose2d(channel, channel//2, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         return self.layer(x)


# class Autoencoder(nn.Module):

#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             DownSample(3, 64),
#             DownSample(64, 128),
#             DownSample(128, 256),
#             DownSample(256, 512),
#         )
#         self.decoder = nn.Sequential(
#             UpSample(512),
#             UpSample(256),
#             UpSample(128),
#             UpSample(64),
#             nn.Conv2d(32, 3, 3, 1, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         encode = self.encoder(x)
#         decode = self.decoder(encode)
#         return decode