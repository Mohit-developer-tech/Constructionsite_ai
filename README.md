# 🏗️ Construction Site AI — Digital Twin & Generative Modeling

An end-to-end AI framework for **Construction Site Safety, Scene Reconstruction, and Generative Modeling** using Deep Learning and PyTorch.

---

## 📌 Project Overview

This repository explores unsupervised representation learning, generative modeling, and temporal risk forecasting for construction site safety imagery:

1. **Exploratory Data Analysis (EDA)**: Dataset distribution, bounding box density, PPE compliance annotations, and spatial heatmap analytics.
2. **Convolutional Autoencoder (CAE)**: Compressed latent representation learning and image reconstruction.
3. **Variational Autoencoder (VAE)**: Probabilistic latent space modeling and generative synthetic scene sampling from prior $\mathcal{N}(0, I)$.
4. **Deep Convolutional GAN (DCGAN)**: Adversarial synthetic scene generation and latent vector interpolation.
5. **Transformer Architecture**: Sequence modeling and risk transition prediction across construction digital twin states.

---

## 📂 Project Structure

```
Construction_Site_AI/
├── data/
│   ├── construction_safety.yaml    # Dataset configuration & class definitions
│   ├── processed/
│   │   ├── features/               # Loss curves, t-SNE plots, metrics
│   │   └── sequences/              # Processed temporal state sequences
│   └── synthetic/                  # GAN/VAE generated synthetic scenes
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis & visual analytics
│   ├── 02_Autoencoder.ipynb        # Convolutional Autoencoder pipeline
│   ├── 03_VAE.ipynb                # Variational Autoencoder & generative sampling
│   ├── 04_GAN.ipynb                # DCGAN adversarial training
│   └── 05_Transformer_Architecture.ipynb # Sequence modeling & risk prediction
├── src/
│   ├── models/
│   │   ├── autoencoder.py          # Autoencoder architecture
│   │   ├── vae.py                  # VAE architecture & ELBO loss
│   │   └── transformer_models.py   # Transformer models
│   └── utils/
│       └── metrics.py              # SSIM, PSNR, and reconstruction metrics
├── requirements.txt                # Python dependencies
└── .gitignore                      # Git ignore rules for datasets & model checkpoints
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd Construction_Site_AI
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\activate
# On Windows CMD:
.venv\Scripts\activate.bat

# Install PyTorch with CUDA (for GPU acceleration)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install required dependencies
pip install -r requirements.txt
```

### 3. Register Jupyter Kernel
```bash
pip install ipykernel
python -m ipykernel install --user --name=construction-ai --display-name="Python 3.12 (Construction AI)"
```

---

## 📊 Models & Experiments

| Model | Objective | Architecture | Key Metric / Feature |
|---|---|---|---|
| **Autoencoder** | Image Compression & Reconstruction | 4-layer CNN Encoder + 4-layer Decoder | SSIM: `~0.75+`, PSNR: `~22dB+` |
| **VAE** | Probabilistic Modeling & Generation | 4-layer CNN + Reparameterization Trick | ELBO Loss, $z \sim \mathcal{N}(0, I)$ Sampling |
| **DCGAN** | Synthetic Image Generation | Generator & Discriminator CNN | Adversarial Minimax, Latent Interpolation |
| **Transformer** | Risk & Digital Twin Forecasting | Self-Attention Encoder-Decoder | Multi-step Temporal Forecasting |

---

## 📜 License
MIT License.
