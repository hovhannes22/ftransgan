import torch
import glob

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from config import device, learning_rate, epochs

from models import Generator
from models import Discriminator

# ----------------------------
# Directories
# ----------------------------

def get_output_dirs(output_folder="outputs"):
    root = Path(__file__).resolve().parents[1]

    out_dir = root / output_folder
    checkpoints_dir = out_dir / "checkpoints"
    samples_dir = out_dir / "samples"
    loss_dir = out_dir / "loss"
    test_dir = out_dir / "test"

    for d in [out_dir, checkpoints_dir, samples_dir, loss_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "outputs": out_dir,
        "checkpoints": checkpoints_dir,
        "samples": samples_dir,
        "loss": loss_dir,
        "test": test_dir,
    }

# ----------------------------
# Training utils
# ----------------------------

def save_checkpoint(generator, content_discriminator, style_discriminator, g_optimizer, d_optimizer, epoch, path):
    torch.save({
        'generator': generator.state_dict(),
        'content_discriminator': content_discriminator.state_dict(),
        'style_discriminator': style_discriminator.state_dict(),
        'g_optimizer': g_optimizer.state_dict(),
        'd_optimizer': d_optimizer.state_dict(),
        'epoch': epoch
    }, path)

def load_checkpoint(generator, content_discriminator, style_discriminator, g_optimizer, d_optimizer, path='checkpoints'):
    folder = Path(path)
    checkpoint_files = list(folder.glob("*.pth"))
    if not checkpoint_files:
        print("No checkpoint found.")
        return -1

    # Get the most recently modified file
    latest_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
    print(f"Loading checkpoint from {latest_file}")

    checkpoint = torch.load(latest_file, map_location=device)
    if checkpoint:
        generator.load_state_dict(checkpoint['generator'])
        content_discriminator.load_state_dict(checkpoint['content_discriminator'])
        style_discriminator.load_state_dict(checkpoint['style_discriminator'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        epoch = checkpoint['epoch']
        print(f"Resuming from epoch {epoch}")
        return epoch

def create_model():
    generator = Generator().to(device)
    content_discriminator = Discriminator(in_channels=2).to(device)
    style_discriminator = Discriminator(in_channels=2).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    d_optimizer = torch.optim.Adam(list(content_discriminator.parameters()) + list(style_discriminator.parameters()), lr=learning_rate)

    return generator, content_discriminator, style_discriminator, g_optimizer, d_optimizer

# ----------------------------
# Visualization
# ----------------------------

def print_loss(epoch, step, dataloader_len, d_content_loss, d_style_loss, g_content_loss, g_style_loss, l1_loss):
    print(
        f"Epoch [{epoch+1}/{epochs}] Batch [{step}/{dataloader_len}] | "
        f"D_c L: {d_content_loss.item():.4f} | "
        f"D_s L: {d_style_loss.item():.4f} | "
        f"G_c L: {g_content_loss.item():.4f} | "
        f"G_s L: {g_style_loss.item():.4f} | "
        f"L1: {l1_loss.item():.4f}"
    )

def save_samples(style_images, content_image, generated_image, target_image=None, filename='sample.png'):
    # style_images: (B, K, 1, H, W)
    # content_image: (B, 1, H, W)
    # generated_image: (B, 1, H, W)
    # target_image: (B, 1, H, W) or None

    B, K = style_images.shape[0:2] # B, K
    num_columns = K + 2 + (1 if target_image is not None else 0)

    fig, axes = plt.subplots(B, num_columns, figsize=(2 * num_columns, 2 * B))

    if B == 1:
        axes = axes[None, :]  # Ensure 2D array for consistency

    for b in range(B): # For each sample in batch
        # Style images
        for k in range(K): # For each K style image
            img = style_images[b, k, 0].cpu().detach().numpy()
            axes[b, k].imshow(img, cmap='gray')
            axes[b, k].axis('off')
                
        # Content
        img = content_image[b, 0].cpu().detach().numpy()
        axes[b, K].imshow(img, cmap='gray')

        # Generated
        img = generated_image[b, 0].cpu().detach().numpy()
        axes[b, K+1].imshow(img, cmap='gray')

        # Target
        if target_image is not None:
            img = target_image[b, 0].cpu().detach().numpy()
            axes[b, K+2].imshow(img, cmap='gray')

    # Add titles
    axes[0, 0].set_title("Style")
    axes[0, K].set_title("Content")
    axes[0, K+1].set_title("Generated")
    if target_image is not None:
        axes[0, K+2].set_title("Target")

    # Hide axis
    for ax in axes.flat:
        ax.axis('off')

    # Save
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_loss(d_losses, g_losses, filename, window=100, max_steps=2000):
    d_losses = np.array(d_losses[-max_steps:])
    g_losses = np.array(g_losses[-max_steps:])

    # Compute moving averages and corresponding x-values
    if len(d_losses) >= window:
        d_losses_moving_avg = np.convolve(d_losses, np.ones(window)/window, mode='valid')
        d_x = np.arange(len(d_losses) - len(d_losses_moving_avg), len(d_losses))
    else:
        d_losses_moving_avg = d_losses
        d_x = np.arange(len(d_losses))

    if len(g_losses) >= window:
        g_losses_moving_avg = np.convolve(g_losses, np.ones(window)/window, mode='valid')
        g_x = np.arange(len(g_losses) - len(g_losses_moving_avg), len(g_losses))
    else:
        g_losses_moving_avg = g_losses
        g_x = np.arange(len(g_losses))

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(d_x, d_losses_moving_avg, label='$Loss_{D}(Avg)$')
    ax.plot(g_x, g_losses_moving_avg, label='$Loss_{G}(Avg)$')

    ax.legend()
    ax.grid(alpha=0.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss (Last {max_steps} Steps, Smoothed)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate(dataloader, generator):
    batch = next(iter(dataloader))
    style_images = batch['style_images'].to(device) # (B, K, 1, 64, 64)
    content_image = batch['content_image'].to(device) # (B, 1, 64, 64)

    generated_image = generator.infer(style_images, content_image) # (B, 1, 64, 64)
    return style_images, content_image, generated_image