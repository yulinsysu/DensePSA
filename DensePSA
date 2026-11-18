import torch
import torch.nn as nn

class SimpleSABlock(nn.Module):
    def __init__(self, ch):
        super(SimpleSABlock, self).__init__()
        self.query = nn.Conv2d(ch, ch//8, 1)
        self.key   = nn.Conv2d(ch, ch//8, 1)
        self.softmax  = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        B, C, H, W = x.shape
        query = self.query(x).view(B,C//8,-1).permute(0,2,1)
        key = self.key(x).view(B,C//8,-1)
        attention = self.softmax(torch.bmm(query, key))
        value = torch.bmm(x.view(B,C,-1), attention).view(B,C,H,W)
        return x + self.gamma * value

class PSABlock(nn.Module):
    def __init__(self, ch, b):
        super(PSABlock, self).__init__()
        self.avg_pool = nn.AvgPool2d((b,b))
        self.max_pool = nn.MaxPool2d((b,b))
        self.query = nn.Conv2d(ch*2, ch//8, 1)
        self.key   = nn.Conv2d(ch*2, ch//8, 1)
        self.softmax  = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.up = nn.Upsample(scale_factor=b, mode='nearest')
    def forward(self, x):        
        avg_x = self.avg_pool(x)
        con = torch.cat([avg_x, self.max_pool(x)], dim=1)
        B, C, H, W = avg_x.shape
        query = self.query(con).view(B,C//8,-1).permute(0,2,1)
        key = self.key(con).view(B,C//8,-1)
        attention = self.softmax(torch.bmm(query, key))
        value = torch.bmm(avg_x.view(B,C,-1), attention).view(B,C,H,W)
        return x + self.up(self.gamma * value)

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride, scale):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)
        )
        self.att = nn.Sequential(PSABlock(ch_out, scale)) if scale > 1 else SimpleSABlock(ch_out)
    def forward(self, x):
        return self.att(self.conv(x))

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Sequential(ConvBlock(3+1, 32, 2, scale=4))
        self.conv2 = nn.Sequential(ConvBlock(32, 64, 2, scale=2))
        self.conv3 = nn.Sequential(ConvBlock(64, 128, 2, scale=1))
        self.conv4 = nn.Sequential(ConvBlock(128, 64, 1, scale=1), nn.Upsample(scale_factor=2, mode='nearest'))
        self.conv5 = nn.Sequential(ConvBlock(64+64, 60, 1, scale=2), nn.Upsample(scale_factor=2, mode='nearest'))
        self.conv6 = nn.Sequential(ConvBlock(64+60+32, 80, 1, scale=4), nn.Upsample(scale_factor=2, mode='nearest'))
        self.conv7 = nn.Sequential(
            nn.Conv2d(80+3+60+64, 32, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
        )
    def forward(self, image, message):
        mp = torch.nn.functional.interpolate(message.reshape(-1,1,8,8), scale_factor=16, mode='nearest')
        feat1 = self.conv1(torch.cat([image, mp], dim=1))
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        feat4 = self.conv4(feat3)
        feat5 = self.conv5(torch.cat([feat4, feat2], dim=1))
        feat6 = self.conv6(torch.cat([feat5, feat1, self.up(feat4)], dim=1))
        feat7 = self.conv7(torch.cat([feat6, image, self.up(feat5), self.up(self.up(feat4))], dim=1))
        return image + feat7

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.sh = nn.MaxPool2d((2,2))
        self.conv1 = nn.Sequential(
            ConvBlock(3, 16, 1, scale=8),
            ConvBlock(16, 32, 2, scale=4)
        )
        self.conv2 = nn.Sequential(ConvBlock(32, 64, 2, scale=2))
        self.conv3 = nn.Sequential(ConvBlock(32+64, 128, 2, scale=1))
        self.conv4 = nn.Sequential(ConvBlock(32+64+128, 128, 2, scale=1))
        self.conv5 = nn.Sequential(nn.Conv2d(128, 1, 3, stride=1, padding=1))
    def forward(self, image):
        feat0 = self.conv1(image)
        feat1 = self.conv2(feat0)
        feat2 = self.conv3(torch.cat([self.sh(feat0), feat1], dim=1))
        feat3 = self.conv4(torch.cat([self.sh(self.sh(feat0)), self.sh(feat1), feat2], dim=1))
        return self.conv5(feat3).reshape(image.shape[0],-1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1, stride=1, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.out = nn.Linear(32, 1)
    
    def forward(self, image):
        x = self.net(image)
        x = x.reshape(x.shape[0], -1)
        x = self.out(x)
        return x
