"""
Deep Convolutional GAN (DCGAN) for Construction Site Image Synthesis.

Module Architecture:
- Generator: Progressive transposed convolution upsampler (1×1 -> 128×128) with BatchNorm + ReLU.
  Generates [B, 3, 128, 128] images from a latent vector z ~ N(0, I) of shape [B, latent_dim, 1, 1].
- Discriminator: Progressive convolution downsampler (128×128 -> 1×1) with BatchNorm + LeakyReLU.
  Classifies images [B, 3, 128, 128] as real (1) or fake (0) via Sigmoid.

Training Objective (Minimax Game):
    min_G max_D  V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN Generator: Maps latent noise z ~ N(0, I) to synthetic construction site images.

    Inputs:
        x: Tensor of shape [B, latent_dim, 1, 1] (random noise vector)
    Outputs:
        img: Tensor of shape [B, 3, 128, 128] (generated RGB image in [-1, 1] via Tanh)
    """
    def __init__(self, latent_dim=100, out_channels=3, features=64):
        super(Generator, self).__init__()
        
        self.net = nn.Sequential(
            # Input: [B, latent_dim, 1, 1]
            nn.ConvTranspose2d(latent_dim, features * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.ReLU(True),
            # Size: [B, features*16, 4, 4]
            
            nn.ConvTranspose2d(features * 16, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # Size: [B, features*8, 8, 8]
            
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # Size: [B, features*4, 16, 16]
            
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # Size: [B, features*2, 32, 32]
            
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # Size: [B, features, 64, 64]
            
            nn.ConvTranspose2d(features, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Size: [B, 3, 128, 128]
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator: Classifies construction site images as real or generated.

    Inputs:
        x: Tensor of shape [B, 3, 128, 128] (RGB image)
    Outputs:
        validity: Tensor of shape [B, 1, 1, 1] (probability of being real, via Sigmoid)
    """
    def __init__(self, in_channels=3, features=64):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            # Input: [B, 3, 128, 128]
            nn.Conv2d(in_channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: [B, features, 64, 64]

            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: [B, features*2, 32, 32]

            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: [B, features*4, 16, 16]

            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: [B, features*8, 8, 8]

            nn.Conv2d(features * 8, features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: [B, features*16, 4, 4]

            nn.Conv2d(features * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Size: [B, 1, 1, 1]
        )

    def forward(self, x):
        return self.net(x)
