import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FontDataset, FontDatasetTest
from utils.utils import get_output_dirs, print_loss, plot_loss, save_samples, create_model, save_checkpoint, load_checkpoint, generate

from pathlib import Path

import config

# ----------------------------
# Parameters
# ----------------------------
epochs = config.epochs
batch_size = config.batch_size # B
num_samples = config.num_samples # K
learning_rate = config.learning_rate

lambda_content = config.lambda_content
lambda_style = config.lambda_style
lambda_l1 = config.lambda_l1

save_every = config.save_every
sample_every = config.sample_every

device = config.device

base_dir = Path(__file__).resolve().parent # Root folder
dirs = get_output_dirs()

# ----------------------------
# Loss Function
# ----------------------------

def hinge_loss(real_pred, fake_pred, mode='D'):
    if mode == 'D':
        return (torch.relu(1 - real_pred) + torch.relu(1 + fake_pred))
    else: # G
        return -fake_pred

# ----------------------------
# Training
# ----------------------------

def train_model():
    # Create model
    generator, content_discriminator, style_discriminator, g_optimizer, d_optimizer = create_model()
    
    # Load if checkpoints available
    start_epoch = load_checkpoint(generator, content_discriminator, style_discriminator, g_optimizer, d_optimizer, dirs['checkpoints']) + 1

    # Create dataset
    dataset = FontDataset(K=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataset_test = FontDatasetTest(K=num_samples)
    dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=True)

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
            fake_style = torch.cat([generated_image, style_images[:, 0]], dim=1) # (B, K*C+1, 64, 64)
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

            if step % sample_every == 0:
                save_samples(style_images, content_image, generated_image, target_image, filename=dirs["samples"]/f'train_{epoch}_{step}.png') # Save sample
                print_loss(epoch, step, len(dataloader), d_content_loss, d_style_loss, g_content_loss, g_style_loss, l1_loss) # Print loss
                plot_loss(d_losses, g_losses, filename=dirs["loss"]/f'loss_{epoch}_{step}.png') # Plot loss

                # Generate test
                style_images, content_image, generated_image = generate(dataloader_test, generator)
                save_samples(style_images, content_image, generated_image, filename=dirs["test"]/f'test_{epoch}_{step}.png') # Save sample

            # Save checkpoint every N steps
            if step % save_every == 0:
                save_checkpoint(
                    generator, content_discriminator, style_discriminator,
                    g_optimizer, d_optimizer, epoch,
                    dirs["checkpoints"]/f'state_{epoch}_{step}.pth'
                )
                print(f"Saved checkpoint at epoch {epoch}")

        print_loss(epoch, step, len(dataloader), d_content_loss, d_style_loss, g_content_loss, g_style_loss, l1_loss)

train_model()