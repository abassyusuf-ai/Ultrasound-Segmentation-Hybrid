import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128) # 128 from up1 + 128 from skip connection
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64) # 64 from up2 + 64 from skip connection
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)           # Output: 64 channels
        x2 = self.down1(x1)        # Output: 128 channels
        x3 = self.down2(x2)        # Output: 256 channels
        
        # Up-sampling + Skip connections
        x = self.up1(x3)           # 256 -> 128 channels
        x = torch.cat([x, x2], dim=1) # Concatenate with x2 (128+128=256)
        x = self.conv_up1(x)
        
        x = self.up2(x)           # 128 -> 64 channels
        x = torch.cat([x, x1], dim=1) # Concatenate with x1 (64+64=128)
        x = self.conv_up2(x)
        
        return self.sigmoid(self.outc(x))
