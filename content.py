import torch
import torch.nn as nn

class ContentEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3, padding_mode='reflect', bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels*2)

        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_channels*4)
        
    def forward(self, content_image):
        # content_image: (B, 1, 64, 64)
        content_image = torch.relu(self.bn1(self.conv1(content_image))) # (B, base_channels, H', W')
        content_image = torch.relu(self.bn2(self.conv2(content_image))) # (B, base_channels*2, H', W')
        content_image = torch.relu(self.bn3(self.conv3(content_image))) # (B, base_channels*4, H', W')
        return content_image