"""
Transformer-based Best Move Decoder.

Architecture:
16×256 patches
      ↓
prepend [CLS] token → 17×256
      ↓
2 transformer blocks (ff_dim=512)
      ↓
extract [CLS] → (B, 256)
      ↓
shared MLP trunk
      ↓
from_head (64) ‖ to_head (64)
      ↓
outer sum → move logits (4096)
"""

import torch
import torch.nn as nn

NUM_MOVES = 4096  # 64 from-squares × 64 to-squares


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class TransformerMoveDecoder(nn.Module):
    def __init__(self, embed_dim=256, num_patches=16, num_heads=8, ff_dim=512, num_layers=2, mlp_hidden=256, dropout=0.1, head_dropout=0.3, latent_dropout=0.1):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Latent dropout on frozen JEPA embeddings (train-only)
        self.latent_drop = nn.Dropout(latent_dropout)

        # Learnable [CLS] token — aggregates spatial info via attention
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional embeddings for the 16 patch tokens (not CLS)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Shared MLP trunk: CLS → hidden representation
        self.mlp = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),
            nn.Dropout(head_dropout),
        )

        # Factorized heads: predict from/to squares independently
        # Combined via outer sum → (B, 64, 64) → flatten → (B, 4096)
        self.from_head = nn.Linear(mlp_hidden, 64)
        self.to_head   = nn.Linear(mlp_hidden, 64)

        # Value head: predicts game outcome from current player's perspective
        # Branches from cls_out (before policy MLP) to avoid conflation
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # output in (-1, 1): -1 = loss, 0 = draw, +1 = win
        )

    def forward(self, x):
        # x: (B, P, D) or (B, T, P, D) — take last timestep if sequence
        if x.ndim == 4:
            x = x[:, -1]  # (B, P, D)

        assert x.shape[1] == self.num_patches and x.shape[2] == self.embed_dim, \
            f"Expected (B, {self.num_patches}, {self.embed_dim}), got {x.shape}"

        B = x.shape[0]

        x = self.latent_drop(x)

        # Add positional embeddings to patch tokens, then prepend [CLS]
        x = x + self.pos_embed
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        x = torch.cat([cls, x], dim=1)            # (B, 17, D)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)

        # Extract [CLS] token — has aggregated all spatial patch info via attention
        cls_out = x[:, 0]  # (B, D)

        # Value head branches from cls_out before the policy MLP
        value = self.value_head(cls_out).squeeze(-1)  # (B,)

        feat = self.mlp(cls_out)                              # (B, mlp_hidden)
        from_logits = self.from_head(feat)                    # (B, 64)
        to_logits   = self.to_head(feat)                      # (B, 64)

        # Outer sum in log space: treats from/to as (approximately) independent
        logits = (from_logits.unsqueeze(2) + to_logits.unsqueeze(1)).view(B, NUM_MOVES)
        return logits, value
