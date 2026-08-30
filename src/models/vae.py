"""
Convolutional Variational Autoencoder (VAE) for Construction Site Digital Twin Modeling.

Module Architecture:
- VAEEncoder: 4-layer CNN feature extractor + Dual linear projection heads (mu, logvar).
- VAEDecoder: Linear expansion + 4-layer Transposed CNN upsampler + Sigmoid.
- ConvVAE: Reparameterization trick (z = mu + eps * std), sampling from prior z ~ N(0, I), and ELBO loss calculation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    """
    Encodes image [B, 3, 128, 128] into Gaussian parameter vectors mu and logvar [B, latent_dim].
    
    Inputs:
        x: Tensor of shape [B, in_channels, 128, 128]
    Outputs:
        mu: Tensor of shape [B, latent_dim] (Mean vector)
        logvar: Tensor of shape [B, latent_dim] (Log-variance vector)
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
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):
        features = self.conv(x)
        flat = self.flatten(features)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    Reconstructs image [B, 3, 128, 128] from sampled latent state z [B, latent_dim].
    
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


class ConvVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for Construction Site Images.
    
    Contract:
        Input:       x in R^[B, 3, 128, 128]
        Latent:      mu, logvar in R^[B, 256]
        Sampled:     z in R^[B, 256]
        Output:      recon in R^[B, 3, 128, 128]
        Generative:  sample(N) -> R^[N, 3, 128, 128]
    """
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = VAEDecoder(out_channels=in_channels, latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + eps * exp(0.5 * logvar), where eps ~ N(0, I)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, num_samples, device="cpu"):
        """
        Generates novel construction site states by sampling directly from standard Gaussian prior z ~ N(0, I).
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=1.0):
        """
        Evidence Lower Bound (ELBO) Loss:
            L_total = MSE_Reconstruction(recon_x, x) + beta * KL_Divergence(q(z|x) || p(z))
        """
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kld_loss
        return total_loss, recon_loss, kld_loss
