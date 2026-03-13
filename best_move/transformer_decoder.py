"""
Transformer-based Best Move Decoder.

Architecture:
16×256 patches
      ↓
2 transformer blocks
      ↓
flatten
      ↓
MLP
      ↓
move logits
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
        # x: (B, seq_len, embed_dim)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class TransformerMoveDecoder(nn.Module):
    def __init__(self, embed_dim=256, num_patches=16, num_heads=8, ff_dim=512, num_layers=2, mlp_hidden=512, dropout=0.1):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Positional embeddings for the patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(num_patches * embed_dim, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),
            nn.Linear(mlp_hidden, NUM_MOVES)
        )

    def forward(self, x):
        # x can be (B, P, D) or (B, T, P, D) - take last timestep if sequence
        if x.ndim == 4:
            x = x[:, -1]  # (B, P, D)

        assert x.shape[1] == self.num_patches and x.shape[2] == self.embed_dim, f"Expected (B, {self.num_patches}, {self.embed_dim}), got {x.shape}"

        # Add positional embeddings
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Flatten
        x = x.view(x.shape[0], -1)  # (B, P*D)

        # MLP to move logits
        logits = self.mlp(x)
        return logits