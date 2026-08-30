"""
Transformer-based Architectures for Construction Site Digital Twins & Progress Simulation.

Includes:
1. PatchEmbedding: 2D Image Patch Partitioner (converts [B, C, H, W] -> [B, N, D]).
2. TransformerEncoderBlock: Multi-Head Self-Attention (MHSA) + MLP with Pre-LayerNorm & Residuals.
3. CrossAttentionBlock: Multi-Head Cross-Attention for multi-modal guidance (BIM/stage tokens).
4. VisionTransformerAutoencoder (ViT-AE): Self-attention based autoencoder for high-fidelity digital twin representations.
5. ConstructionProgressTransformer: Conditioned transformer for simulating future site progress and forecasting spatial safety risk.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Splits image [B, C, H, W] into non-overlapping patches of size [P, P]
    and linearly projects them to embedding dimension D.
    
    Output shape: [B, num_patches, embed_dim]
    """
    def __init__(self, img_size=128, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W] -> [B, embed_dim, H/P, W/P] -> [B, num_patches, embed_dim]
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, (
            f"Input spatial resolution ({H}x{W}) does not match model resolution ({self.img_size}x{self.img_size})"
        )
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block with Multi-Head Self-Attention (MHSA) and MLP.
    """
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerAutoencoder(nn.Module):
    """
    Vision Transformer Autoencoder (ViT-AE) for Construction Site Images.
    
    Inputs:
        x: Tensor of shape [B, 3, 128, 128]
    Outputs:
        recon: Tensor of shape [B, 3, 128, 128] (Reconstructed image)
        latent_tokens: Tensor of shape [B, num_patches, embed_dim] (Spatial patch embeddings)
    """
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_channels=3,
        embed_dim=256,
        depth=6,
        num_heads=8,
        decoder_embed_dim=128,
        decoder_depth=4,
        decoder_num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ViT Encoder Blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ViT Decoder
        self.decoder_proj = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        self.decoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, dropout)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Pixel Reconstruction Head (projects back to patch pixels)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_channels)

    def unpatchify(self, patch_preds):
        """
        Converts patch predictions [B, num_patches, P*P*C] back to image [B, C, H, W].
        """
        p = self.patch_size
        h = w = self.img_size // p
        B = patch_preds.shape[0]
        # [B, h*w, p*p*c] -> [B, h, w, p, p, c] -> [B, c, h, p, w, p] -> [B, c, H, W]
        x = patch_preds.reshape(shape=(B, h, w, p, p, self.in_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(B, self.in_channels, h * p, w * p))
        return imgs

    def forward(self, x):
        # 1. Patch partition & Positional Encoding
        tokens = self.patch_embed(x) + self.pos_embed
        
        # 2. Encoder forward pass
        for block in self.encoder_blocks:
            tokens = block(tokens)
        latent_tokens = self.encoder_norm(tokens)

        # 3. Decoder projection & forward pass
        dec_tokens = self.decoder_proj(latent_tokens) + self.decoder_pos_embed
        for block in self.decoder_blocks:
            dec_tokens = block(dec_tokens)
        dec_tokens = self.decoder_norm(dec_tokens)

        # 4. Unpatchify to reconstructed image
        patch_preds = self.decoder_pred(dec_tokens)
        recon = torch.sigmoid(self.unpatchify(patch_preds))

        return recon, latent_tokens


class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention block where site visual tokens attend to conditioning stage/schedule tokens.
    """
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm_self = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context):
        # Self-attention among visual patches
        x = x + self.self_attn(self.norm_self(x), self.norm_self(x), self.norm_self(x))[0]
        # Cross-attention with conditioning context (BIM / Stage / Schedule)
        q = self.norm_q(x)
        kv = self.norm_kv(context)
        x = x + self.cross_attn(q, kv, kv)[0]
        # Feed-forward
        x = x + self.mlp(self.norm_mlp(x))
        return x


class ConstructionProgressTransformer(nn.Module):
    """
    Multi-Modal Transformer for Construction Site Progress Simulation and Risk Intelligence.
    
    Inputs:
        x_current: Tensor of shape [B, 3, 128, 128] (Current visual state)
        stage_idx: Tensor of shape [B] (Target construction stage ID: 0=Excavation, 1=Substructure, 2=Framing, 3=Facade, 4=Finishing)
        delta_days: Tensor of shape [B, 1] (Time horizon in days into the future)
    
    Outputs:
        x_future_sim: Tensor of shape [B, 3, 128, 128] (Simulated future site state)
        risk_map: Tensor of shape [B, 1, 128, 128] (Predicted spatial risk hazard map)
    """
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_channels=3,
        num_stages=6,
        embed_dim=256,
        depth=4,
        num_heads=8
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # Visual Tokenizer
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Multi-Modal Conditioning Tokenizers
        self.stage_embedding = nn.Embedding(num_stages, embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Cross-Attention Progress Simulation Transformer
        self.sim_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Dual Prediction Heads: Future Visual State + Risk Map
        self.future_patch_pred = nn.Linear(embed_dim, patch_size * patch_size * in_channels)
        self.risk_patch_pred = nn.Linear(embed_dim, patch_size * patch_size * 1)

    def unpatchify(self, patch_preds, out_channels):
        p = self.patch_size
        h = w = self.img_size // p
        B = patch_preds.shape[0]
        x = patch_preds.reshape(shape=(B, h, w, p, p, out_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(B, out_channels, h * p, w * p))

    def forward(self, x_current, stage_idx, delta_days):
        # 1. Visual Tokenization
        vis_tokens = self.patch_embed(x_current) + self.pos_embed

        # 2. Multi-Modal Context Tokenization
        stage_token = self.stage_embedding(stage_idx).unsqueeze(1) # [B, 1, embed_dim]
        time_token = self.time_mlp(delta_days).unsqueeze(1)         # [B, 1, embed_dim]
        context = torch.cat([stage_token, time_token], dim=1)       # [B, 2, embed_dim]

        # 3. Progress Simulation via Cross-Attention
        tokens = vis_tokens
        for block in self.sim_blocks:
            tokens = block(tokens, context)
        tokens = self.norm(tokens)

        # 4. Predict Future Image & Risk Map
        future_patches = self.future_patch_pred(tokens)
        risk_patches = self.risk_patch_pred(tokens)

        x_future_sim = torch.sigmoid(self.unpatchify(future_patches, self.in_channels))
        risk_map = torch.sigmoid(self.unpatchify(risk_patches, 1))

        return x_future_sim, risk_map
