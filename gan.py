import torch
import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from dataloader import FontDataset, show_images, show_losses

import config

from generator import Generator
from discriminator import Discriminator

import os
import glob

# Parameters
epochs = config.epochs
batch_size = config.batch_size # B
num_samples = config.num_samples # K
learning_rate = config.learning_rate

lambda_content = config.lambda_content
lambda_style = config.lambda_style
lambda_l1 = config.lambda_l1

save_every = config.save_every
sample_every = config.sample_every

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(generator, content_discriminator, style_discriminator, g_optimizer, d_optimizer, epoch, path):
    torch.save({
        'generator': generator.state_dict(),
        'content_discriminator': content_discriminator.state_dict(),
        'style_discriminator': style_discriminator.state_dict(),
        'g_optimizer': g_optimizer.state_dict(),
        'd_optimizer': d_optimizer.state_dict(),
        'epoch': epoch
    }, path)

def load_checkpoint(generator, content_discriminator, style_discriminator, g_optimizer, d_optimizer, folder='models'):
    checkpoint_files = glob.glob(os.path.join(folder, "*.pth"))
    if not checkpoint_files:
        print("No checkpoint files found.")
        return -1

    # Get the most recently modified file
    latest_file = max(checkpoint_files, key=os.path.getmtime)
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

def hinge_loss(real_pred, fake_pred, mode='D'):
    if mode == 'D':
        return (torch.relu(1 - real_pred) + torch.relu(1 + fake_pred))
    else: # G
        return -fake_pred

def print_losses(epoch, step, dataloader_len, d_content_loss, d_style_loss, g_content_loss, g_style_loss, l1_loss):
    print(
        f"Epoch [{epoch+1}/{epochs}] Batch [{step}/{dataloader_len}] | "
        f"D_c L: {d_content_loss.item():.4f} | "
        f"D_s L: {d_style_loss.item():.4f} | "
        f"G_c L: {g_content_loss.item():.4f} | "
        f"G_s L: {g_style_loss.item():.4f} | "
        f"L1: {l1_loss.item():.4f}"
    )
    return 1

def train_model():
    # Create model
    generator, content_discriminator, style_discriminator, g_optimizer, d_optimizer = create_model()
    
    # Load if checkpoints available
    start_epoch = load_checkpoint(generator, content_discriminator, style_discriminator, g_optimizer, d_optimizer) + 1

    # Create dataset
    dataset = FontDataset(K=num_samples, mode='train')
    dataloader = DataLoader(dataset, batch_size=batch_size)

    dataset_test = FontDataset(K=6, mode='test')
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    d_losses = []
    g_losses = []

    for epoch in range(start_epoch, epochs):
        for step, batch in enumerate(dataloader):
            style_images = batch['style_images'].to(device) # (B, K, 1, 64, 64)
            content_image = batch['content_image'].to(device) # (B, 1, 64, 64)
            target_image = batch['target_image'].to(device) # (B, 1, 64, 64)
            
            # -------------------------
            # Train Discriminators
            # -------------------------
            d_optimizer.zero_grad()
            
            # Generate image
            with torch.no_grad():
                generated_image = generator(style_images, content_image) # (B, 1, 64, 64)
            
            # Content discrimination
            real_content = torch.cat([target_image, content_image], dim=1) # (B, 2, 64, 64)
            fake_content = torch.cat([generated_image, content_image], dim=1) # (B, 2, 64, 64)
            d_content_real = content_discriminator(real_content) # (B, 1, H, W)
            d_content_fake = content_discriminator(fake_content) # (B, 1, H, W)
            
            # Style discrimination
            real_style = torch.cat([target_image, style_images[:, 0]], dim=1) # (B, 2, 64, 64)
            fake_style = torch.cat([generated_image, style_images[:, 0]], dim=1) # (B, 2, 64, 64)
            d_style_real = style_discriminator(real_style) # (B, 1, H, W)
            d_style_fake = style_discriminator(fake_style) # (B, 1, H, W)
            
            # Calculate discriminator loss
            d_content = hinge_loss(d_content_real, d_content_fake, mode='D') # Content loss
            d_style = hinge_loss(d_style_real, d_style_fake, mode='D') # Style loss
            d_content_loss = d_content.mean()
            d_style_loss = d_style.mean()
            d_total_loss = d_content_loss + d_style_loss # Total discriminator loss
            
            # Update discriminators
            d_total_loss.backward()
            d_optimizer.step()

            # Update loss
            d_losses.append(d_total_loss.item())
            
            # -------------------------
            # Train Generator
            # -------------------------
            g_optimizer.zero_grad()
            
            # Generate image
            generated_image = generator(style_images, content_image) # (B, 1, 64, 64)
            
            # Content loss
            fake_content = torch.cat([generated_image, content_image], dim=1) # (B, 2, 64, 64)
            g_content_loss = -content_discriminator(fake_content).mean()
            
            # Style loss
            fake_style = torch.cat([generated_image, target_image], dim=1) # (B, K*C+1, 64, 64)
            g_style_loss = -style_discriminator(fake_style).mean()
            
            # L1 loss
            l1_loss = nn.L1Loss()(generated_image, target_image)
            
            # Total generator loss
            g_total_loss = (lambda_content * g_content_loss + lambda_style * g_style_loss + lambda_l1 * l1_loss)
            
            # Update generator
            g_total_loss.backward()
            g_optimizer.step()
            g_losses.append(g_total_loss.item())
            
            # -------------------------
            # Training Monitoring
            # -------------------------
            if step % 50 == 0:
                print_losses(epoch, step, len(dataloader), d_content_loss, d_style_loss, g_content_loss, g_style_loss, l1_loss)

            if step % sample_every == 0:
                generate(dataloader_test, generator, epoch, step)
                show_images(style_images=style_images,
                    content_image=content_image,
                    target_image=target_image,
                    generated_image=generated_image,
                    # d_content=d_content,
                    # d_style=d_style,
                    filename=f'train_{epoch}_{step+1}.png')
                show_losses(d_losses, g_losses, epoch, step)

        print_losses(epoch, step, len(dataloader), d_content_loss, d_style_loss, g_content_loss, g_style_loss, l1_loss)

        # Save checkpoint every N steps
        if (epoch + 1) % save_every == 0:
            checkpoint_path = f'./models/state_{epoch}.pth'
            save_checkpoint(
                generator, content_discriminator, style_discriminator,
                g_optimizer, d_optimizer, epoch,
                checkpoint_path
            )
            print(f"Saved checkpoint at epoch {epoch}")

def generate(dataloader, generator, epoch='x', step='x'):
    for _, batch in enumerate(dataloader):
        style_images = batch['style_images'].to(device) # (B, K, 1, 64, 64)
        content_image = batch['content_image'].to(device) # (B, 1, 64, 64)

        generated_image = generator.infer(style_images, content_image) # (B, 1, 64, 64)
        show_images(style_images=style_images, content_image=content_image, generated_image=generated_image, filename=f'test/test_{epoch}_{step}.png')


# torch.cuda.empty_cache()
os.makedirs('./samples/test', exist_ok=True)
train_model()

# Create dataset
# dataset_test = FontDataset("./fonts", K=num_samples, mode='test')
# dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
# generate(dataloader_test)