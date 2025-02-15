import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        x_res = torch.relu(self.bn1(self.conv1(x)))
        x_res = self.bn2(self.conv2(x_res))
        return x + x_res
    
class Decoder(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        
        self.resnet_blocks = nn.Sequential( *[ ResBlock(base_channels*8) for _ in range(6) ] )
        
        self.upconv1 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels*4)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels*2)

        self.conv3 = nn.Conv2d(base_channels*2, 1, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, style_features, content_features):
        # style_features: (B, base_channels*4)
        # content_features: (B, base_channels*4, H, W)
        B, C, H, W = content_features.shape

        # Expanding style features to match content features
        style_features = style_features.view(B, C, 1, 1) # (B, base_channels*4, 1, 1)
        style_features = style_features.expand(B, C, H, W) # (B, base_channels*4, H, W)

        # Concating style and content features across the C dim
        x = torch.cat([content_features, style_features], dim=1) # (B, base_channels*8, H, W)
        
        # ResNet
        x = self.resnet_blocks(x) # (B, base_channels*8, H, W)
        
        # Upsampling
        x = torch.relu(self.bn1(self.upconv1(x))) # (B, base_channels*4, H', W')
        x = torch.relu(self.bn2(self.upconv2(x))) # (B, base_channels*2, H', W')
        x = torch.tanh(self.conv3(x)) # (B, 1, 64, 64)
        
        return x