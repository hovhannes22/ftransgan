import torch
import torch.nn as nn

import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, in_channels=2, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(base_channels)

        self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels*2)

        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(base_channels*4)

        self.conv4 = nn.Conv2d(base_channels*4, base_channels*8, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(base_channels*8)

        self.conv5 = nn.Conv2d(base_channels*8, 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, images):
        # image: (B, 2, 64, 64)
        images = F.leaky_relu(self.conv1(images), 0.2) # (B, 64, H, W)
        images = F.leaky_relu(self.bn2(self.conv2(images)), 0.2) # (B, 128, H, W)
        images = F.leaky_relu(self.bn3(self.conv3(images)), 0.2) # (B, 256, H, W)
        images = F.leaky_relu(self.bn4(self.conv4(images)), 0.2) # (B, 512, H, W)
        images = self.conv5(images) # (B, 1, H, W)
        return images