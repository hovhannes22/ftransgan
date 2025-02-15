import torch

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import numpy as np

class FontDataset(Dataset):
    def __init__(self, root_dir='./fonts/', K=6, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        self.fonts = [ f for f in os.listdir(os.path.join(root_dir, 'base' if self.mode == 'train' else 'test')) if not f.startswith('.') ]
        self.K = K
        self.transform = transforms.ToTensor()
        self.support_font = 'arial'
        self.base_letters = [ f for f in os.listdir(os.path.join(self.root_dir, 'base' if self.mode == 'train' else 'test', self.fonts[0])) if not f.startswith('.') ]
        self.final_letters = [ f for f in  os.listdir(os.path.join(self.root_dir, 'final', self.support_font)) if not f.startswith('.') ]

    def __len__(self):
        return len(self.fonts)

    def __getitem__(self, idx):
        # Select random letter
        final_letter = random.choice(self.final_letters)
        base_letters = random.sample(self.base_letters, self.K)

        # Select the content image
        content_image = self.transform(Image.open(os.path.join(self.root_dir, 'final', self.support_font, final_letter)).convert('L'))
        
        # Select target image
        target_image = self.transform(Image.open(os.path.join(self.root_dir, 'final', self.fonts[idx], final_letter)).convert('L')) if self.mode == 'train' else torch.tensor([0])
        
        # Select base images
        style_images = torch.stack([ self.transform(Image.open(os.path.join(self.root_dir, 'base' if self.mode == 'train' else 'test', self.fonts[idx], letter))) for letter in base_letters ])

        return {
                'style_images': style_images,
                'content_image': content_image,
                'target_image': target_image
            }
    
def show_images(style_images=None, support_images=None, content_image=None, target_image=None, generated_image=None, d_content=None, d_style=None, filename="sample.png"):
    os.makedirs("samples", exist_ok=True)
    
    # Arrange images in order. Note that style_images is assumed to have shape (B, K, 1, H, W)
    # while the others (including discriminator outputs) have shape (B, 1, H, W).
    images = [style_images, support_images, content_image, target_image, generated_image, d_content, d_style]
    labels = ["Style", "Support", "Content", "Target", "Generated", "Content Disc", "Style Disc"]
    
    # Move all tensors to CPU and detach if they exist
    images = [img.detach().cpu() if isinstance(img, torch.Tensor) else None for img in images]
    
    # Determine batch size (B) and number of style images (K)
    if images[0] is not None:
        B, K = images[0].shape[:2]
    else:
        B = content_image.shape[0]
        K = 0

    # Count the remaining columns (each non-style image counts as one column)
    ncols = K + sum(img is not None for img in images[1:])
    
    # Create subplots: one row per sample and one column per image
    fig, axes = plt.subplots(B, ncols, figsize=(2 * ncols, 2 * B))
    
    # If only one row, ensure axes is a 2D array for consistent indexing
    if B == 1:
        axes = axes[None, :]
    
    for i in range(B):
        col = 0
        # Plot style images (assumed to be a multi-channel tensor: (B, K, 1, H, W))
        if images[0] is not None:
            for j in range(K):
                axes[i, col].imshow(images[0][i, j, 0].numpy(), cmap="gray")
                axes[i, col].axis("off")
                if i == 0:
                    axes[i, col].set_title(f"{labels[0]} {j+1}")
                col += 1
        
        # Plot remaining images (each expected to have shape: (B, 1, H, W))
        for idx, img in enumerate(images[1:], start=1):
            if img is not None:
                axes[i, col].imshow(img[i, 0].numpy(), cmap="gray")
                axes[i, col].axis("off")
                if i == 0:
                    axes[i, col].set_title(labels[idx])
                col += 1
    
    plt.tight_layout()
    plt.savefig(os.path.join("samples", filename), bbox_inches="tight")
    plt.close(fig)

def show_losses(d_losses, g_losses, filename='result.png'):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    d_losses = np.array(d_losses[:len(d_losses) - len(d_losses) % 10])
    g_losses = np.array(g_losses[:len(g_losses) - len(g_losses) % 10])
    ax.plot(d_losses.reshape(-1, 10).mean(axis=1), label='$Loss_{D}$')
    ax.plot(g_losses.reshape(-1, 10).mean(axis=1), label='$Loss_{G}$')

    plt.tight_layout()
    plt.legend()
    plt.grid(0.2)
    plt.savefig(os.path.join("losses", filename), bbox_inches="tight")
    plt.close(fig)