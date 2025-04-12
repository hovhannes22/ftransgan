import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import numpy as np

import config

class FontDataset(Dataset):
    def __init__(self, K=6, mode='train'):
        self.root_dir = config.root_dir  # Root directory
        self.mode = mode
        self.K = K  # Number of samples to pick for style images
        self.embedding_dim = config.embedding_dim

        # For training, base images are in 'base'; for inference, they are in 'test'
        self.fonts_path = os.path.join(self.root_dir, 'base' if self.mode == 'train' else 'test')
        # List available fonts (subdirectories in the fonts_path)
        self.fonts = sorted([f for f in os.listdir(self.fonts_path) if not f.startswith('.') and os.path.isdir(os.path.join(self.fonts_path, f))])
        
        # Build the list of base letters from the first font folder as a template.
        base_letters_path = os.path.join(self.fonts_path, self.fonts[0])
        self.base_letters = sorted([f for f in os.listdir(base_letters_path) if f.endswith('.png') and not f.startswith('.')], 
                                    key=lambda x: int(os.path.splitext(x)[0]))
        # Keep only the numeric part (as string) for later loading (we'll append ".png" when needed)
        self.base_letters = [os.path.splitext(letter)[0] for letter in self.base_letters]

        # Build the list of content letters from the arial font.
        content_letters_path = os.path.join(self.root_dir, 'content', 'arial')
        self.content_letters = sorted([f for f in os.listdir(content_letters_path) if f.endswith('.png') and not f.startswith('.')], 
                                       key=lambda x: int(os.path.splitext(x)[0]))
        self.content_letters = [os.path.splitext(letter)[0] for letter in self.content_letters]

        # Transform: Convert PIL images to tensor
        self.transform = transforms.ToTensor()

        # Embeddings for base and content letters (indexed by their numerical order)
        self.base_stoi = { letter: idx for idx, letter in enumerate(self.base_letters) }
        self.base_letter_embedding = nn.Embedding(len(self.base_letters), self.embedding_dim)

        self.content_stoi = { letter: idx for idx, letter in enumerate(self.content_letters) }
        self.content_letter_embedding = nn.Embedding(len(self.content_letters), self.embedding_dim)

    def __len__(self):
        return len(self.fonts)

    def __getitem__(self, idx):
        font_name = self.fonts[idx]

        # ----------------------
        # Load Style (Base) Images from the custom font
        # ----------------------
        base_folder = os.path.join(self.root_dir, 'base' if self.mode == 'train' else 'test', font_name)
        available_base_files = sorted(
            [f for f in os.listdir(base_folder) if f.endswith('.png') and not f.startswith('.')],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        available_base_letters = [os.path.splitext(f)[0] for f in available_base_files]

        if len(available_base_letters) < self.K:
            raise RuntimeError(f"Font '{font_name}' has only {len(available_base_letters)} base letters, but K={self.K} required.")

        sampled_letters = random.sample(available_base_letters, self.K)
        style_images = torch.stack([
            self.transform(
                Image.open(os.path.join(base_folder, f"{letter}.png")).convert('L')
            ) for letter in sampled_letters
        ])

        # ----------------------
        # Select a valid content/target letter from available target images
        # ----------------------
        content_font_folder = os.path.join(self.root_dir, 'content' if self.mode == 'train' else 'test', font_name)
        available_target_files = sorted(
            [f for f in os.listdir(content_font_folder) if f.endswith('.png') and not f.startswith('.')],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        available_letters = [os.path.splitext(f)[0] for f in available_target_files]

        if not available_letters:
            raise RuntimeError(f"Font '{font_name}' has no valid content/target letters.")

        chosen_letter = random.choice(available_letters)

        # Load content image from Arial
        arial_folder = os.path.join(self.root_dir, 'content', 'arial')
        content_image_path = os.path.join(arial_folder, f"{chosen_letter}.png")
        content_image = self.transform(Image.open(content_image_path).convert('L'))

        # ----------------------
        # Load Target Image (from current font)
        # ----------------------
        target_image_path = os.path.join(content_font_folder, f"{chosen_letter}.png")
        target_image = self.transform(Image.open(target_image_path).convert('L')) if self.mode == 'train' else 0

        return {
            'font_name': font_name,
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

def show_losses(d_losses, g_losses, epoch, step):
    os.makedirs("losses", exist_ok=True)
    filename=f'loss_{epoch}_{step}.png'

    if len(d_losses) > 1000 and len(g_losses) > 1000:
        d_losses = d_losses[-1000:]
        g_losses = g_losses[-1000:]

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