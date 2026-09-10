"""
Conditional Variational Autoencoder (CVAE) for Construction Site Scene Generation.

The CVAE extends the standard VAE by conditioning both the encoder and decoder
on a class label c (one-hot encoded over the 10 safety classes).

Architecture:
  Encoder: Image x + condition c → μ(x,c), σ²(x,c)
  Latent:  z = μ + ε·σ   where ε ~ N(0, I)
  Decoder: z + condition c → x̂

Conditioning is achieved by concatenating the one-hot label embedding
to the flattened feature vector before the μ/σ heads, and to the
latent z before the decoder FC layer.

Safety classes (10):
  0: Hardhat          5: Person
  1: Mask             6: Safety Cone
  2: NO-Hardhat       7: Safety Vest
  3: NO-Mask          8: machinery
  4: NO-Safety Vest   9: vehicle
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 10 safety classes in the Roboflow Construction Safety dataset
NUM_CLASSES = 10
SAFETY_CLASSES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
]


class CVAEEncoder(nn.Module):
    """
    Encodes (image x, condition c) → (μ, log σ²) in latent space.

    Condition injection: the one-hot label is projected to `cond_dim` features
    and concatenated to the flattened CNN output before the μ/σ linear heads.

    Args:
        in_channels (int): Input image channels (default 3 for RGB).
        latent_dim  (int): Dimension of latent space z.
        num_classes (int): Number of condition classes.
        cond_dim    (int): Dimension of the learned condition embedding.
    """
    def __init__(self, in_channels=3, latent_dim=256, num_classes=NUM_CLASSES, cond_dim=64):
        super().__init__()
        self.cond_dim = cond_dim

        # --- CNN backbone (same as standard VAE) ---
        self.conv = nn.Sequential(
            # [B, 3, 128, 128] → [B, 32, 64, 64]
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # [B, 32, 64, 64] → [B, 64, 32, 32]
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # [B, 64, 32, 32] → [B, 128, 16, 16]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # [B, 128, 16, 16] → [B, 256, 8, 8]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flatten = nn.Flatten()  # → [B, 256*8*8] = [B, 16384]

        # --- Condition embedding: one-hot → dense ---
        self.cond_embed = nn.Sequential(
            nn.Linear(num_classes, cond_dim),
            nn.ReLU(inplace=True),
        )

        # --- μ and log σ² heads take (image_feat || cond_feat) ---
        feat_dim = 256 * 8 * 8 + cond_dim  # 16384 + 64 = 16448
        self.fc_mu     = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)

    def forward(self, x, c_onehot):
        """
        Args:
            x        : [B, 3, 128, 128] image tensor
            c_onehot : [B, num_classes] one-hot condition tensor
        Returns:
            mu     : [B, latent_dim]
            logvar : [B, latent_dim]
        """
        feat = self.flatten(self.conv(x))          # [B, 16384]
        cond = self.cond_embed(c_onehot)           # [B, cond_dim]
        combined = torch.cat([feat, cond], dim=1)  # [B, 16448]
        mu     = self.fc_mu(combined)
        logvar = torch.clamp(self.fc_logvar(combined), min=-10.0, max=10.0)
        return mu, logvar


class CVAEDecoder(nn.Module):
    """
    Decodes (z, condition c) → reconstructed image x̂.

    Condition injection: the one-hot label embedding is concatenated to z
    before the fully-connected expansion layer.

    Args:
        out_channels (int): Output image channels.
        latent_dim   (int): Dimension of latent z.
        num_classes  (int): Number of condition classes.
        cond_dim     (int): Dimension of the learned condition embedding.
    """
    def __init__(self, out_channels=3, latent_dim=256, num_classes=NUM_CLASSES, cond_dim=64):
        super().__init__()

        # --- Condition embedding ---
        self.cond_embed = nn.Sequential(
            nn.Linear(num_classes, cond_dim),
            nn.ReLU(inplace=True),
        )

        # --- FC expansion: (z || cond) → spatial feature map ---
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256 * 8 * 8),
            nn.ReLU(inplace=True),
        )

        # --- Transposed CNN upsampler (mirrors encoder) ---
        self.deconv = nn.Sequential(
            # [B, 256, 8, 8] → [B, 128, 16, 16]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # [B, 128, 16, 16] → [B, 64, 32, 32]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # [B, 64, 32, 32] → [B, 32, 64, 64]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # [B, 32, 64, 64] → [B, 3, 128, 128]
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, c_onehot):
        """
        Args:
            z        : [B, latent_dim] sampled latent vector
            c_onehot : [B, num_classes] one-hot condition tensor
        Returns:
            recon : [B, out_channels, 128, 128]
        """
        cond = self.cond_embed(c_onehot)           # [B, cond_dim]
        combined = torch.cat([z, cond], dim=1)     # [B, latent_dim + cond_dim]
        x = self.fc(combined)                      # [B, 256*8*8]
        x = x.view(-1, 256, 8, 8)                  # [B, 256, 8, 8]
        recon = self.deconv(x)                     # [B, 3, 128, 128]
        return recon


class ConvCVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE).

    Extends ConvVAE by conditioning both encoder and decoder on a safety class
    label provided as a one-hot vector.

    Usage:
        model = ConvCVAE(latent_dim=256, num_classes=10)

        # Reconstruction (training / inference with input image):
        recon, mu, logvar = model(x, c_onehot)

        # Conditional generation (pure generation, no input image needed):
        generated = model.sample(n=4, class_idx=2, device='cuda')
        # class_idx=2 → 'NO-Hardhat' scenario
    """
    def __init__(self, in_channels=3, latent_dim=256, num_classes=NUM_CLASSES, cond_dim=64):
        super().__init__()
        self.latent_dim  = latent_dim
        self.num_classes = num_classes

        self.encoder = CVAEEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            num_classes=num_classes,
            cond_dim=cond_dim,
        )
        self.decoder = CVAEDecoder(
            out_channels=in_channels,
            latent_dim=latent_dim,
            num_classes=num_classes,
            cond_dim=cond_dim,
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def reparameterize(self, mu, logvar):
        """z = μ + ε·σ  (ε ~ N(0,I)) — only adds noise during training."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x, c_onehot):
        return self.encoder(x, c_onehot)

    def decode(self, z, c_onehot):
        return self.decoder(z, c_onehot)

    def forward(self, x, c_onehot):
        """
        Full CVAE forward pass.

        Args:
            x        : [B, 3, 128, 128]  input image
            c_onehot : [B, num_classes]  one-hot class condition

        Returns:
            recon  : [B, 3, 128, 128]   reconstructed image
            mu     : [B, latent_dim]
            logvar : [B, latent_dim]
        """
        mu, logvar = self.encode(x, c_onehot)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c_onehot)
        return recon, mu, logvar

    def sample(self, n: int, class_idx: int, device: str = "cpu", temperature: float = 1.0):
        """
        Generate `n` images conditioned on class `class_idx`.

        Args:
            n           : Number of images to generate.
            class_idx   : Integer class index (0–9).
            device      : Torch device string.
            temperature : Scale factor for noise (>1 → more diverse, <1 → more focused).

        Returns:
            images : [n, 3, 128, 128] generated images, values in [0, 1].
        """
        # Build one-hot condition for the whole batch
        c = torch.zeros(n, self.num_classes, device=device)
        c[:, class_idx] = 1.0

        # Sample from standard Gaussian prior, scaled by temperature
        z = torch.randn(n, self.latent_dim, device=device) * temperature

        with torch.no_grad():
            images = self.decode(z, c)
        return images

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta: float = 1.0):
        """
        Conditional ELBO Loss (identical form to standard VAE ELBO):
            L = MSE(recon_x, x) + β · KL[q(z|x,c) ‖ p(z)]

        Args:
            recon_x : Reconstructed image [B, 3, 128, 128]
            x       : Original image      [B, 3, 128, 128]
            mu      : Encoded mean        [B, latent_dim]
            logvar  : Encoded log-variance[B, latent_dim]
            beta    : KL weight (β-VAE style)

        Returns:
            total_loss, recon_loss, kld_loss
        """
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld_loss, recon_loss, kld_loss
