import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, 3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),

            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.in_channels = in_channels

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),

            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 256),
            nn.GELU(),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def test_dimensions():
    batch_size = 4

    # Generator test
    latent_dim = 128
    z = torch.randn(batch_size, latent_dim, 1, 1)

    generator = Generator(latent_dim, 3)
    fake_images = generator(z)
    assert fake_images.shape == (batch_size, 3, 28, 28), f"Shape was {fake_images.shape}"
    
    # Discriminator test
    discriminator = Discriminator(3)
    output = discriminator(fake_images)
    assert output.shape == (batch_size, 1), f"Shape was {output.shape}"


if __name__ == "__main__":
    test_dimensions()
    print("All tests passed!")
