"""
Convolutional Autoencoder (AE) for Construction Site Digital Twin Feature Extraction.

Module Architecture:
- ConvEncoder: 4-layer CNN with progressive downsampling (128x128 -> 8x8) + Linear bottleneck.
- ConvDecoder: Linear expansion + 4-layer Transposed CNN upsampling (8x8 -> 128x128) + Sigmoid.
- ConvAutoencoder: End-to-end model with separate encode(x), decode(z), and forward(x) paths.
"""
import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """
    Downsamples RGB construction site image [B, 3, 128, 128] 
    into a compact latent bottleneck representation [B, latent_dim].
    
    Inputs:
        x: Tensor of shape [B, in_channels, 128, 128]
    Outputs:
        z: Tensor of shape [B, latent_dim] (Latent embedding)
    """
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            # 3 x 128 x 128 -> 32 x 64 x 64
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32 x 64 x 64 -> 64 x 32 x 32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 x 32 x 32 -> 128 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 x 16 x 16 -> 256 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        features = self.conv(x)
        z = self.fc(features)
        return z


class ConvDecoder(nn.Module):
    """
    Upsamples latent bottleneck representation [B, latent_dim] 
    back to reconstructed RGB construction site image [B, 3, 128, 128].
    
    Inputs:
        z: Tensor of shape [B, latent_dim]
    Outputs:
        recon: Tensor of shape [B, out_channels, 128, 128]
    """
    def __init__(self, out_channels=3, latent_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(inplace=True)
        )
        self.deconv = nn.Sequential(
            # 256 x 8 x 8 -> 128 x 16 x 16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 128 x 16 x 16 -> 64 x 32 x 32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64 x 32 x 32 -> 32 x 64 x 64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 32 x 64 x 64 -> 3 x 128 x 128
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 8, 8)
        recon = self.deconv(x)
        return recon


class ConvAutoencoder(nn.Module):
    """
    Complete Convolutional Autoencoder for Construction Site AI.
    
    Contract:
        Input:  x in R^[B, 3, 128, 128]
        Latent: z in R^[B, 256]
        Output: recon in R^[B, 3, 128, 128]
    """
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = ConvDecoder(out_channels=in_channels, latent_dim=latent_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
