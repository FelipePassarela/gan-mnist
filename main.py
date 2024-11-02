import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torchvision.utils import make_grid

from gan import Discriminator, Generator
from train_utils import set_seed, test_step, train_step


with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    train_config = config["training"]
    dirs_config = config["dirs"]


def main():
    wandb.init(project="gan-mnist", config=train_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(train_config["seed"])
    print(f"Using {device} device")
    print(f"Seed set to {train_config['seed']}\n")

    train_loader, test_loader = load_data()

    generator = Generator(train_config["latent_dim"], 1).to(device)
    discriminator = Discriminator(1).to(device)

    criterion = nn.BCELoss()
    optimizer_g = optim.AdamW(generator.parameters(), lr=train_config["learning_rate"])
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=train_config["learning_rate"])

    epochs = train_config["epochs"]

    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")
        print("-" * 30)
        
        train_losses = train_step(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion, device)
        test_losses = test_step(generator, discriminator, test_loader, criterion, device)

        wandb.log({
            "train_generator_loss": train_losses["g_loss"],
            "train_discriminator_loss": train_losses["d_loss"],
            "test_generator_loss": test_losses["g_loss"],
            "test_discriminator_loss": test_losses["d_loss"],
            "generated_images": [wandb.Image(generate_images_grid(device, generator))],
        })

        print(f"G Train Loss: {train_losses['g_loss']:.4f} - D Train Loss: {train_losses['d_loss']:.4f}")
        print(f"G Test Loss: {test_losses['g_loss']:.4f} - D Test Loss: {test_losses['d_loss']:.4f}")
        print()

    save_models(generator, discriminator)
    wandb.finish()


def generate_images_grid(device, generator):
    noise = torch.randn(config["n_imgs_to_gen"], train_config["latent_dim"], 1, 1, device=device)
    generated_images = generator(noise).detach()
    nrow = int(math.sqrt(config["n_imgs_to_gen"]))
    return make_grid(generated_images, nrow=nrow, normalize=True)


def load_data():
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_set = datasets.MNIST(
        dirs_config["data_dir"], 
        train=True, 
        transform=transform, 
        download=True, 
    )
    test_set = datasets.MNIST(
        dirs_config["data_dir"], 
        train=False, 
        transform=transform, 
        download=True, 
    )

    train_loader = DataLoader(train_set, batch_size=train_config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=train_config["batch_size"], shuffle=False)
    return train_loader, test_loader


def save_models(generator, discriminator):
    gen_name = "generator_" + time.strftime("%d%m%Y-%H%M%S") + ".pt"
    disc_name = "discriminator_" + time.strftime("%d%m%Y-%H%M%S") + ".pt"
    
    os.makedirs(dirs_config["models_dir"], exist_ok=True)
    torch.save(generator.state_dict(), str(Path(dirs_config["models_dir"]) / gen_name))
    torch.save(discriminator.state_dict(), str(Path(dirs_config["models_dir"]) / disc_name))


if __name__ == '__main__':
    main()
