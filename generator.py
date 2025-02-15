import torch
import torch.nn as nn

from style import StyleEncoder
from content import ContentEncoder
from decoder import Decoder

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.style_encoder = StyleEncoder()
        self.content_encoder = ContentEncoder()
        self.decoder = Decoder()
        
    def forward(self, style_images, content_image):
        # style_images: (B, K, 1, 64, 64)
        # content_image: (B, 1, 64, 64)
        style_encoded = self.style_encoder(style_images) # (B, base_channels*4)
        content_encoded = self.content_encoder(content_image) # (B, base_channels*4, H, W)
        image = self.decoder(style_encoded, content_encoded) # (B, 1, 64, 64)
        return image