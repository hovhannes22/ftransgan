import torch
import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from dataloader import FontDataset, show_images, show_losses

from generator import Generator
from discriminator import Discriminator

import os

# Parameters
epochs = 600
batch_size = 4 # B
num_samples = 6 # K
learning_rate = 0.00002

lambda_content = 1
lambda_style = 1
lambda_l1 = 100

checkpoint_path = os.path.join("models", "latest_checkpoint.pth")
save_every = 10
sample_every = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(generator, content_discriminator, style_discriminator, g_optimizer, d_optimizer, epoch, path=checkpoint_path):
    torch.save({
        'generator': generator.state_dict(),
        'content_discriminator': content_discriminator.state_dict(),
        'style_discriminator': style_discriminator.state_dict(),
        'g_optimizer': g_optimizer.state_dict(),
        'd_optimizer': d_optimizer.state_dict(),
        'epoch': epoch
    }, path)

def load_checkpoint(generator, content_discriminator, style_discriminator, g_optimizer, d_optimizer):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint:
            generator.load_state_dict(checkpoint['generator'])
            content_discriminator.load_state_dict(checkpoint['content_discriminator'])
            style_discriminator.load_state_dict(checkpoint['style_discriminator'])
            g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            epoch = checkpoint['epoch']
            print(f"Loaded a checkpoint, resuming from epoch {epoch}")
    
            return epoch
    return -1

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

def train_model():
    # Create model
    generator, content_discriminator, style_discriminator, g_optimizer, d_optimizer = create_model()
    
    # Load if checkpoints available
    start_epoch = load_checkpoint(generator, content_discriminator, style_discriminator, g_optimizer, d_optimizer) + 1

    # Create dataset
    dataset = FontDataset("./fonts", K=num_samples, mode='train')
    dataloader = DataLoader(dataset, batch_size=batch_size)

    dataset_test = FontDataset("./fonts", K=num_samples, mode='test')
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
            # Progress Monitoring
            # -------------------------
            if step % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] Batch [{step}/{len(dataloader)}] | "
                    f"D_c L: {d_content_loss.item():.4f} | "
                    f"D_s L: {d_style_loss.item():.4f} | "
                    f"G_c L: {g_content_loss.item():.4f} | "
                    f"G_s L: {g_style_loss.item():.4f} | "
                    f"L1: {l1_loss.item():.4f}"
                )

            if (step + 1) % sample_every == 0:
                generate(dataloader_test)
                show_images(style_images=style_images,
                    content_image=content_image,
                    target_image=target_image,
                    generated_image=generated_image,
                    # d_content=d_content,
                    # d_style=d_style,
                    filename=f'{epoch}_{step+1}.png')
                show_losses(d_losses, g_losses)

        # Save checkpoint every N steps
        if (epoch + 1) % save_every == 0:
            checkpoint_path = f'./models/state_{epoch}.pth'
            save_checkpoint(
                generator, content_discriminator, style_discriminator,
                g_optimizer, d_optimizer, epoch,
                checkpoint_path
            )
            print(f"Saved checkpoint at epoch {epoch}")

def generate(dataloader):
    # Create model
    generator, content_discriminator, style_discriminator, g_optimizer, d_optimizer = create_model()
    
    # Load if checkpoints available
    start_epoch = load_checkpoint(generator, content_discriminator, style_discriminator, g_optimizer, d_optimizer) + 1

    for step, batch in enumerate(dataloader):
        style_images = batch['style_images'].to(device) # (B, K, 1, 64, 64)
        content_image = batch['content_image'].to(device) # (B, 1, 64, 64)

        with torch.no_grad():
            generated_image = generator(style_images, content_image) # (B, 1, 64, 64)
        show_images(style_images=style_images, content_image=content_image, generated_image=generated_image, filename=f'test_{step}.png')


# torch.cuda.empty_cache()
train_model()

# Create dataset
# dataset_test = FontDataset("./fonts", K=num_samples, mode='test')
# dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
# generate(dataloader_test)