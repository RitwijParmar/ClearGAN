# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
          
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
          
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out

class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        self.enc5 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        
        # Bottleneck
        self.bottleneck_res = nn.Sequential(ResidualBlock(512), ResidualBlock(512), ResidualBlock(512))
        self.bottleneck_attn = SelfAttention(512)
        
        # Decoder
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.dec1_res = ResidualBlock(1024)
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.dec2_res = ResidualBlock(512)
        self.dec2_attention = SelfAttention(512)
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.dec3_res = ResidualBlock(256)
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dec4_res = ResidualBlock(128)
        self.dec5 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        # Output
        self.out = nn.Sequential(nn.Conv2d(64, 3, 1), nn.Tanh())
          
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        bott = self.bottleneck_attn(self.bottleneck_res(e5))
        d1 = self.dec1_res(torch.cat([self.dec1(bott), e4], dim=1))
        d2 = self.dec2_attention(self.dec2_res(torch.cat([self.dec2(d1), e3], dim=1)))
        d3 = self.dec3_res(torch.cat([self.dec3(d2), e2], dim=1))
        d4 = self.dec4_res(torch.cat([self.dec4(d3), e1], dim=1))
        d5 = self.dec5(d4)
        return self.out(d5)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=6, ndf=32, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        layers = [nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, 4, 2, 1)), nn.Dropout2d(0.3), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers + 1):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            stride = 2 if n <= 2 else 1
            layers += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride, 1)),
                nn.Dropout2d(0.4),
                nn.LeakyReLU(0.2, True)
            ]
        layers += [nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1))]
        self.model = nn.Sequential(*layers)
          
    def forward(self, x):
        return self.model(x)

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc=6, ndf=16, n_layers=2, num_d=2):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_d = num_d
        for i in range(num_d):
            net_d = NLayerDiscriminator(input_nc, ndf, n_layers)
            setattr(self, 'discriminator_' + str(i), net_d)
              
    def forward(self, blurry, sharp):
        x = torch.cat([blurry, sharp], dim=1)
        outputs = []
        for i in range(self.num_d):
            net_d = getattr(self, 'discriminator_' + str(i))
            outputs.append(net_d(x))
            x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)
        return outputs# Add Discriminator
