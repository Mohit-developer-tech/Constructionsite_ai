"""
Weight Export Script for Construction Site AI Models.

Trains all 4 models briefly on synthetic data and exports state_dict() as .pkl files
for the backend inference server. For production, replace with full training on the
Roboflow Construction Safety dataset.

Usage:
    python weights/export_weights.py
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.autoencoder import ConvAutoencoder
from src.models.vae import ConvVAE
from src.models.gan import Generator, Discriminator
from src.models.transformer_models import ConstructionProgressTransformer

WEIGHTS_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters for demo (short training for functional weights)
DEMO_EPOCHS = 5
DEMO_BATCH = 8
IMG_SIZE = 128
LATENT_AE = 256
LATENT_GAN = 100


def create_synthetic_batch(batch_size=DEMO_BATCH):
    """Generate a batch of synthetic 'construction-like' images for demo training."""
    images = torch.rand(batch_size, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    # Add some structure: horizontal gradient (sky→ground), random rectangles
    for i in range(batch_size):
        grad = torch.linspace(0.6, 0.2, IMG_SIZE, device=DEVICE).unsqueeze(0).unsqueeze(0).expand(3, -1, IMG_SIZE)
        images[i] = images[i] * 0.3 + grad * 0.7
        # Add random 'structures'
        for _ in range(3):
            x, y = torch.randint(20, 100, (2,))
            w, h = torch.randint(10, 40, (2,))
            color = torch.rand(3, device=DEVICE) * 0.5
            images[i, :, y:y+h, x:x+w] = color.view(3, 1, 1)
    return images


def train_autoencoder():
    print("\n[1/4] Training Autoencoder...")
    model = ConvAutoencoder(in_channels=3, latent_dim=LATENT_AE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(DEMO_EPOCHS):
        images = create_synthetic_batch()
        recon, _ = model(images)
        loss = criterion(recon, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch+1}/{DEMO_EPOCHS} — Loss: {loss.item():.6f}")
    
    path = os.path.join(WEIGHTS_DIR, "ae_weights.pkl")
    torch.save(model.state_dict(), path)
    print(f"  [OK] Saved: {path}")
    return model


def train_vae():
    print("\n[2/4] Training VAE...")
    model = ConvVAE(in_channels=3, latent_dim=LATENT_AE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(DEMO_EPOCHS):
        images = create_synthetic_batch()
        recon, mu, logvar = model(images)
        total_loss, recon_loss, kld_loss = ConvVAE.loss_function(recon, images, mu, logvar, beta=1.0)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch+1}/{DEMO_EPOCHS} — Total: {total_loss.item():.2f}  Recon: {recon_loss.item():.2f}  KL: {kld_loss.item():.2f}")
    
    path = os.path.join(WEIGHTS_DIR, "vae_weights.pkl")
    torch.save(model.state_dict(), path)
    print(f"  [OK] Saved: {path}")
    return model


def train_gan():
    print("\n[3/4] Training GAN...")
    gen = Generator(latent_dim=LATENT_GAN, out_channels=3, features=64).to(DEVICE)
    disc = Discriminator(in_channels=3, features=64).to(DEVICE)
    
    opt_g = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    gen.train()
    disc.train()
    for epoch in range(DEMO_EPOCHS):
        # Real images normalized to [-1, 1] for Tanh output
        real_images = create_synthetic_batch() * 2 - 1
        batch_size = real_images.size(0)
        
        real_labels = torch.ones(batch_size, 1, 1, 1, device=DEVICE)
        fake_labels = torch.zeros(batch_size, 1, 1, 1, device=DEVICE)
        
        # Train Discriminator
        z = torch.randn(batch_size, LATENT_GAN, 1, 1, device=DEVICE)
        fake_images = gen(z)
        d_real = disc(real_images)
        d_fake = disc(fake_images.detach())
        d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        
        # Train Generator
        z = torch.randn(batch_size, LATENT_GAN, 1, 1, device=DEVICE)
        fake_images = gen(z)
        d_fake = disc(fake_images)
        g_loss = criterion(d_fake, real_labels)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        
        print(f"  Epoch {epoch+1}/{DEMO_EPOCHS} — D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}")
    
    path = os.path.join(WEIGHTS_DIR, "gan_weights.pkl")
    torch.save(gen.state_dict(), path)
    print(f"  [OK] Saved Generator: {path}")
    return gen


def train_transformer():
    print("\n[4/4] Training Progress Transformer...")
    model = ConstructionProgressTransformer(
        img_size=128, patch_size=16, in_channels=3, num_stages=5,
        embed_dim=256, depth=4, num_heads=8
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(DEMO_EPOCHS):
        current_images = create_synthetic_batch()
        future_images = create_synthetic_batch()  # Target future state
        risk_maps = torch.rand(DEMO_BATCH, 1, IMG_SIZE, IMG_SIZE, device=DEVICE) * 0.3  # Low risk baseline
        
        stage_idx = torch.randint(0, 5, (DEMO_BATCH,), device=DEVICE)
        delta_days = torch.rand(DEMO_BATCH, 1, device=DEVICE) * 90
        
        future_pred, risk_pred = model(current_images, stage_idx, delta_days)
        loss = criterion(future_pred, future_images) + 0.5 * criterion(risk_pred, risk_maps)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch+1}/{DEMO_EPOCHS} — Loss: {loss.item():.6f}")
    
    path = os.path.join(WEIGHTS_DIR, "transformer_weights.pkl")
    torch.save(model.state_dict(), path)
    print(f"  [OK] Saved: {path}")
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("  Construction Site AI — Weight Export Script")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {WEIGHTS_DIR}")
    print("=" * 60)
    
    train_autoencoder()
    train_vae()
    train_gan()
    train_transformer()
    
    print("\n" + "=" * 60)
    print("  All weights exported successfully!")
    print(f"  Files saved to: {WEIGHTS_DIR}")
    print("=" * 60)
