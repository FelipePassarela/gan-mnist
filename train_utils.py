import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from gan import Discriminator, Generator


with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    train_config = config["training"]


def set_seed(seed=train_config["seed"]):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train_step(
        generator: Generator,
        discriminator: Discriminator,
        dataloader,
        g_optimizer,
        d_optimizer,
        criterion,
        device: str,
):
    n_batches = len(dataloader)
    progress_bar = tqdm(enumerate(dataloader), desc="Training", total=n_batches, unit="batch", colour="green")

    running_losses = {"d_loss": 0, "g_loss": 0}

    generator.train()
    discriminator.train()

    for i, (real_images, _) in progress_bar:
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Train discriminator on real images
        discriminator.zero_grad()
        output = discriminator(real_images)
        loss_real = criterion(output, torch.ones_like(output, device=device))
        
        # Train discriminator on fake images
        noise_shape = (batch_size, train_config["latent_dim"], 1, 1)
        noise = torch.randn(noise_shape, device=device)
        fake_images = generator(noise).detach()

        output = discriminator(fake_images)
        loss_fake = criterion(output, torch.zeros_like(output, device=device))

        d_loss = loss_real + loss_fake
        d_loss.backward()
        d_optimizer.step()

        # Train generator
        generator.zero_grad()

        noise = torch.randn(noise_shape, device=device)
        fake_images = generator(noise)
        output = discriminator(fake_images)

        g_loss = criterion(output, torch.ones_like(output, device=device))
        g_loss.backward()
        g_optimizer.step()

        running_losses["d_loss"] += (loss_real + loss_fake).item()
        running_losses["g_loss"] += g_loss.item()

        progress_bar.set_postfix({
            "D Loss": f"{running_losses["d_loss"] / (i + 1):.4f}",
            "G Loss": f"{running_losses["g_loss"] / (i + 1):.4f}",
        })

    avg_losses = {k: v / n_batches for k, v in running_losses.items()}
    return avg_losses


def test_step(
        generator: Generator,
        discriminator: Discriminator,
        dataloader,
        criterion,
        device,
        save_image_freq = 30,
):
    n_batches = len(dataloader)
    progress_bar = tqdm(enumerate(dataloader), desc="Testing", total=n_batches, unit="batch", colour="blue")
    
    running_losses = {"d_loss": 0, "g_loss": 0}
    saved_fake_images = []
    
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        for i, (real_images, _) in progress_bar:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            noise = torch.randn(batch_size, train_config["latent_dim"], 1, 1, device=device)
            fake_images = generator(noise)

            output_real = discriminator(real_images)
            output_fake = discriminator(fake_images)

            loss_real = criterion(output_real, torch.ones_like(output_real, device=device))
            loss_fake = criterion(output_fake, torch.zeros_like(output_fake, device=device))
            g_lossen = criterion(output_fake, torch.ones_like(output_fake, device=device))
            
            running_losses["d_loss"] += (loss_real + loss_fake).item()
            running_losses["g_loss"] += g_lossen.item()
            
            if i % (n_batches // save_image_freq) == 0:
                saved_fake_images.append(fake_images[:5].cpu())

            progress_bar.set_postfix({
                "D Loss": f"{running_losses["d_loss"] / (i + 1):.4f}",
                "G Loss": f"{running_losses["g_loss"] / (i + 1):.4f}",
            })
    
    avg_losses = {k: v / n_batches for k, v in running_losses.items()}
    return avg_losses, saved_fake_images
