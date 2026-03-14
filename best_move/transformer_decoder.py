"""
Transformer-based Best Move Decoder.

Architecture:
16×256 patches
      ↓
2 transformer blocks (ff_dim=512, wider to do the reasoning)
      ↓
GAP ‖ GMP → concat → (B, 512)
      ↓
MLP with head dropout (0.3)
      ↓
move logits / value
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
        # Pre-LayerNorm: normalize before attention and FFN (more stable)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class TransformerMoveDecoder(nn.Module):
    def __init__(self, embed_dim=256, num_patches=16, num_heads=8, ff_dim=512, num_layers=2, mlp_hidden=256, dropout=0.1, head_dropout=0.3):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Positional embeddings for the patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        # Transformer blocks (wider ff_dim — let the transformer do the reasoning)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        pool_dim = embed_dim * 2  # GAP ‖ GMP concatenated

        # Policy head: pool_dim → move logits
        self.mlp = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(pool_dim, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),
            nn.Dropout(head_dropout),
            nn.Linear(mlp_hidden, NUM_MOVES)
        )

        # Value head: pool_dim → scalar evaluation in (-1, 1)
        self.value_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(pool_dim, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),
            nn.Dropout(head_dropout),
            nn.Linear(mlp_hidden, 1),
            nn.Tanh()
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

        x = self.norm(x)

        # GAP ‖ GMP: concat avg and max over patch dimension → (B, 2D)
        # Max captures salient piece signals; avg captures global board context
        gap = x.mean(dim=1)
        gmp = x.max(dim=1).values
        x = torch.cat([gap, gmp], dim=-1)

        # Policy logits and value estimate
        logits = self.mlp(x)
        value = self.value_head(x).squeeze(-1)  # (B,)
        return logits, value