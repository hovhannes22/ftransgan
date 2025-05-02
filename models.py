import torch
import torch.nn as nn

import torch.nn.functional as F

import config

# ----------------------------
# Attention
# ----------------------------

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()

        q = self.query(x).view(B, -1, H*W) # (B, C, H*W)
        k = self.key(x).view(B, -1, H*W) # (B, C, H*W)
        attention_scores = torch.softmax(q.transpose(-2, -1) @ k, dim=-1) # (B, H*W, C) @ (B, C, H*W) -> (B, H*W, H*W)
        v = self.value(x).view(B, -1, H*W) # (B, C, H*W)

        out = v @ attention_scores.transpose(-2, -1) # (B, C, H*W) @ (B, H*W, H*W) -> (B, C, H*W) 
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class ContextAttentionBlock(nn.Module):
    def __init__(self, in_channels, hidden_size=100):
        super().__init__()

        self.self_attn = SelfAttentionBlock(in_channels)
        self.fc = nn.Linear(in_channels, hidden_size)
        self.context_vector = nn.Parameter(torch.randn(hidden_size, 1))

    def forward(self, style_features):
        # feature_map: (B*K, C, H, W)
        B, C, H, W = style_features.shape
        
        # Self-Attention
        attention_output = self.self_attn(style_features) # (B*K, C, H, W)
        attention_output = attention_output.view(B, C, H*W).permute(2, 0, 1) # (H*W, B*K, C)

        # Linear layer
        latent_features = torch.tanh(self.fc(attention_output)) # (H*W, B*K, hidden_size)
        
        # Context vector
        scores = latent_features @ self.context_vector # (H*W, B*K, hidden_size) @ (hidden_size, 1) -> (H*W, B*K, 1)
        scores = scores.permute(1, 2, 0) # (B, 1, H*W)
        attention_score = torch.softmax(scores.view(B, H*W), dim=1).view(B, 1, H, W) # (B*K, 1, H, W)

        # Multiplying by original feature_map
        out = torch.sum(style_features * attention_score, dim=[-2, -1]) # (B*K, C)
        return out

class LayerAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.fc = nn.Linear(in_channels, 3)
        
    def forward(self, feature_map, style_features):
        # feature_map: (B, K, C, H, W) 
        # style_features: list of 3 (B, K, C')
        B, K, _, H, W = feature_map.shape

        context = feature_map.mean(dim=[1]) # (B, C, H, W)
        context = context.view(B, -1) # (B, C*H*W)

        style_features = [ feature.mean(dim=1) for feature in style_features ] # List of (B, C')
        
        weights = torch.softmax(torch.tanh(self.fc(context)), dim=1) # (B, 3) 
        
        # Combine features: w1*f1 + w2*f2 + w3*f3
        return sum(w.unsqueeze(1) * f for w, f in zip(weights.unbind(1), style_features)) # (B, C')

# ----------------------------
# Encoder
# ----------------------------

class Encoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=config.embedding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3, padding_mode='reflect', bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels*2)

        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_channels*4)
        
    def forward(self, image):
        # image: (B, 1, 64, 64)
        features = torch.relu(self.bn1(self.conv1(image))) # (B, base_channels, H', W')
        features = torch.relu(self.bn2(self.conv2(features))) # (B, base_channels*2, H', W')
        features = torch.relu(self.bn3(self.conv3(features))) # (B, base_channels*4, H', W')
        return features

class StyleEncoder(nn.Module):
    def __init__(self, base_channels=config.embedding_dim):
        super().__init__()

        self.encoder = Encoder()

        self.conv1 = nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, stride=2, padding=1, bias=False) # 13x13
        self.bn1 = nn.BatchNorm2d(base_channels*4)

        self.conv2 = nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, stride=2, padding=1, bias=False) # 21x21
        self.bn2 = nn.BatchNorm2d(base_channels*4)

        self.context_attention1 = ContextAttentionBlock(base_channels*4)
        self.context_attention2 = ContextAttentionBlock(base_channels*4)
        self.context_attention3 = ContextAttentionBlock(base_channels*4)

        self.layer_attention = LayerAttention(base_channels*64)
        
    def forward(self, style_images):
        # style_images: (B, K, 1, 64, 64)
        B, K, C, H, W = style_images.shape
        style_images = style_images.view(B*K, C, H, W) # (B*K, 1, 64, 64)

        # Convolution blocks
        feature_map1 = self.encoder(style_images) # (B, base_channels*4, H', W')
        feature_map2 = torch.relu(self.bn1(self.conv1(feature_map1))) # (B*K, base_channels*4, H, W)
        feature_map3 = torch.relu(self.bn2(self.conv2(feature_map2))) # (B*K, base_channels*4, H, W)

        # Context-aware Attention
        context_attention1 = self.context_attention1(feature_map1).view(B, K, -1) # (B, K, base_channels*4)
        context_attention2 = self.context_attention2(feature_map2).view(B, K, -1) # (B, K, base_channels*4)
        context_attention3 = self.context_attention3(feature_map3).view(B, K, -1) # (B, K, base_channels*4)

        features = [context_attention1, context_attention2, context_attention3] # List of 3 (B, K, C')

        feature_map3 = feature_map3.view(B, K, -1, H, W)

        # Layer Attention
        layer_attention_scores = self.layer_attention(feature_map3, features) # (B, C)

        # Mean across K
        # style_features = layer_attention_scores.view(B, K, -1).mean(dim=1) # (B, base_channels*4)
        style_features = layer_attention_scores

        return style_features

class ContentEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=config.embedding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3, padding_mode='reflect', bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels*2)

        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_channels*4)
        
    def forward(self, content_image):
        # content_image: (B, 1, 64, 64)
        content_features = torch.relu(self.bn1(self.conv1(content_image))) # (B, base_channels, H', W')
        content_features = torch.relu(self.bn2(self.conv2(content_features))) # (B, base_channels*2, H', W')
        content_features = torch.relu(self.bn3(self.conv3(content_features))) # (B, base_channels*4, H', W')
        return content_features

# ----------------------------
# Generator
# ----------------------------

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
    
    @torch.inference_mode()
    def infer(self, style_images, content_image):
        # style_images: (B, K, 1, 64, 64)
        # content_image: (B, 1, 64, 64)
        self.eval()  # Ensure that layers like dropout and BN behave in evaluation mode.
        generated_image = self.forward(style_images, content_image)
        self.train()  # Back to train mode.
        return generated_image
    
# ----------------------------
# Discriminator
# ----------------------------

class Discriminator(nn.Module):
    def __init__(self, in_channels=2, base_channels=config.embedding_dim):
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
    
# ----------------------------
# Decoder
# ----------------------------

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
    def __init__(self, base_channels=config.embedding_dim):
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