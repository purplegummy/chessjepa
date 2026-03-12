"""
Chess V-JEPA — Board Encoder

Transforms a sequence of board states into latent representations using a
Vision-Transformer-style architecture with full spatiotemporal attention.

Input:  (B, T, 17, 8, 8)   — B batches of T board positions
Output: (B, T, embed_dim)  — one latent vector per time step

Pipeline per board position:
    (17, 8, 8) board
        → 2×2 patchify → (num_patches, patch_dim)     [num_patches = 16]
        → linear projection → (num_patches, embed_dim)
        → add spatial pos encoding
    Then across all T positions:
        → add temporal pos encoding
        → concatenate all patches from all time steps → (T*num_patches, embed_dim)
        → Transformer encoder blocks (full spatiotemporal attention)
        → mean-pool patches within each time step → (T, embed_dim)
"""

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# 1) Patch Embedding
# ─────────────────────────────────────────────────────────────────────────────
#
# The board is (17, 8, 8).  We split the 8×8 spatial grid into 2×2 patches,
# giving us 4×4 = 16 patches.  Each patch has shape (17, 2, 2), which we
# flatten to a vector of length 17*2*2 = 68 and linearly project to embed_dim.
#
# Why 2×2?  It gives 16 patches — enough spatial resolution to distinguish
# kingside from queenside, while keeping the sequence length manageable.
# ─────────────────────────────────────────────────────────────────────────────


class PatchEmbedding(nn.Module):
    """Patchify an (C, H, W) board and project each patch to embed_dim."""

    def __init__(
        self,
        in_channels: int = 17,
        board_size: int = 8,
        patch_size: int = 2,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (board_size // patch_size) ** 2          # 16
        patch_dim = in_channels * patch_size * patch_size           # 68

        # A single linear layer: flatten patch → embed_dim
        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)  — a single board state, e.g. (B, 17, 8, 8)
        returns: (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # Reshape into patches: (B, C, H//p, p, W//p, p)
        x = x.reshape(B, C, H // p, p, W // p, p)
        # → (B, H//p, W//p, C, p, p)  — group spatial patches together
        x = x.permute(0, 2, 4, 1, 3, 5)
        # → (B, num_patches, patch_dim)  — flatten each patch
        x = x.reshape(B, self.num_patches, -1)

        return self.proj(x)  # (B, num_patches, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Transformer Block
# ─────────────────────────────────────────────────────────────────────────────
#
# Standard pre-norm Transformer block:
#   x → LayerNorm → MultiHeadAttention → residual
#   x → LayerNorm → FFN → residual
#
# Pre-norm (instead of post-norm) is what modern ViTs and JEPA use because
# it leads to more stable training.
# ─────────────────────────────────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with multi-head self-attention and FFN."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, N, D) — N tokens, D embed_dim."""
        # Self-attention with pre-norm
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]

        # Feed-forward with pre-norm
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3) Full Encoder: ChessBoardEncoder
# ─────────────────────────────────────────────────────────────────────────────
#
# Processes a sequence of T board states:
#   1. Patch-embed each board independently        → (B, T, P, D)
#   2. Add learnable spatial pos encoding          → (B, T, P, D)
#   3. Add learnable temporal pos encoding         → (B, T, P, D)
#   4. Flatten to (B, T*P, D) and run Transformer  → (B, T*P, D)
#   5. Reshape back to (B, T, P, D)
#   6. Mean-pool over patches per time step        → (B, T, D)
#
# The full spatiotemporal attention in step 4 lets the model learn that e.g.
# a knight on f3 at move 4 is related to a pawn on e4 at move 2.
# ─────────────────────────────────────────────────────────────────────────────


class ChessBoardEncoder(nn.Module):
    """
    V-JEPA-style spatiotemporal Transformer encoder for chess board sequences.

    Args:
        in_channels:  number of input planes per board (default 17)
        board_size:   spatial dimension (default 8)
        patch_size:   spatial patch size (default 2 → 16 patches)
        embed_dim:    transformer hidden dim (default 256)
        depth:        number of transformer blocks (default 6)
        num_heads:    attention heads (default 8)
        mlp_ratio:    FFN expansion ratio (default 4.0)
        max_seq_len:  maximum number of board positions in a sequence (default 16)
        dropout:      dropout rate (default 0.0)
    """

    def __init__(
        self,
        in_channels: int = 17,
        board_size: int = 8,
        patch_size: int = 2,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        num_patches = (board_size // patch_size) ** 2  # 16

        # ── Patch embedding ──────────────────────────────────────────────
        self.patch_embed = PatchEmbedding(
            in_channels, board_size, patch_size, embed_dim
        )

        # ── Positional encodings (both learnable) ────────────────────────
        #   spatial:  one embedding per patch position  (num_patches, D)
        #   temporal: one embedding per time step       (max_seq_len, D)
        self.spatial_pos = nn.Parameter(
            torch.randn(1, 1, num_patches, embed_dim) * 0.02
        )
        self.temporal_pos = nn.Parameter(
            torch.randn(1, max_seq_len, 1, embed_dim) * 0.02
        )

        # ── Transformer blocks ───────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # ── Final layer norm ─────────────────────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)

        self._num_patches = num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, C, H, W) — batch of board-state sequences
                e.g. (B, 16, 17, 8, 8)

        Returns:
            (B, T, embed_dim) — one latent per time step
        """
        B, T, C, H, W = x.shape
        P = self._num_patches  # 16

        # ── 1. Patch-embed each board independently ──────────────────────
        # Merge batch and time: (B*T, C, H, W) → (B*T, P, D)
        x = x.reshape(B * T, C, H, W)
        x = self.patch_embed(x)               # (B*T, P, D)
        x = x.reshape(B, T, P, self.embed_dim)  # (B, T, P, D)

        # ── 2. Add spatial positional encoding ───────────────────────────
        #   self.spatial_pos is (1, 1, P, D) — broadcast over B and T
        x = x + self.spatial_pos[:, :, :P, :]

        # ── 3. Add temporal positional encoding ──────────────────────────
        #   self.temporal_pos is (1, max_T, 1, D) — broadcast over B and P
        x = x + self.temporal_pos[:, :T, :, :]

        # ── 4. Flatten to token sequence and run Transformer ─────────────
        #   (B, T, P, D) → (B, T*P, D)
        x = x.reshape(B, T * P, self.embed_dim)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # ── 5. Reshape back and mean-pool over patches ───────────────────
        #   (B, T*P, D) → (B, T, P, D) → mean over P → (B, T, D)
        x = x.reshape(B, T, P, self.embed_dim)
        # average patches, one vector per time step
        x = x.mean(dim=2)  # (B, T, D) #hmm should this be fixed?

        return x
