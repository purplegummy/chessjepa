"""
Chess AC-JEPA — Action-Conditioned Predictor  (patch-aware version)

Extends the patch-aware Predictor with chess move conditioning.

Changes vs. standard Predictor
--------------------------------
  • Each target mask token is conditioned on the action (from_sq, to_sq)
    that produced that board state.
  • The action is per *timestep*, not per patch — so we embed once per
    timestep and broadcast the same action embedding across all P patches
    at that timestep.

Input / Output
--------------
  context_latents : (B, T_ctx * P, encoder_dim)
  context_positions: list[int]  length T_ctx
  target_positions : list[int]  length T_tgt
  target_actions   : (B, T_tgt, 2) int64   [from_sq, to_sq]  values 0–64

  → predicted_latents : (B, T_tgt * P, encoder_dim)
"""

import torch
import torch.nn as nn

from model.encoder import TransformerBlock


class ActionConditionedPredictor(nn.Module):
    """
    Patch-aware V-JEPA predictor conditioned on chess move actions.

    Args:
        encoder_dim   : dimension of encoder output (default 256)
        predictor_dim : internal bottleneck dimension (default 128)
        depth         : number of transformer blocks (default 4)
        num_heads     : attention heads (default 4)
        mlp_ratio     : FFN expansion ratio (default 4.0)
        max_seq_len   : maximum number of time steps (default 16)
        num_patches   : spatial patches per board (default 16 = 4×4 grid)
        dropout       : dropout rate (default 0.0)
        num_squares   : square embedding vocab size (65 = 64 squares + 1 null)
    """

    def __init__(
        self,
        encoder_dim: int = 256,
        predictor_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 16,
        num_patches: int = 16,
        dropout: float = 0.0,
        num_squares: int = 65,
    ):
        super().__init__()
        self.encoder_dim   = encoder_dim
        self.predictor_dim = predictor_dim
        self.num_patches   = num_patches

        # ── 1. Input projection ──────────────────────────────────────────
        self.input_proj = nn.Linear(encoder_dim, predictor_dim)

        # ── 2. Action embedding ──────────────────────────────────────────
        #   Embed from_sq and to_sq independently (predictor_dim // 2 each)
        #   so their concatenation fits in predictor_dim.
        sq_dim = predictor_dim // 2
        self.square_embed = nn.Embedding(num_squares, sq_dim)
        self.action_proj  = nn.Linear(predictor_dim, predictor_dim)

        # ── 3. Mask token ─────────────────────────────────────────────────
        self.mask_token = nn.Parameter(torch.randn(1, 1, predictor_dim) * 0.02)

        # ── 4. Positional encodings (separate spatial + temporal) ─────────
        self.temporal_pos = nn.Parameter(
            torch.randn(1, max_seq_len, predictor_dim) * 0.02
        )
        self.spatial_pos = nn.Parameter(
            torch.randn(1, num_patches, predictor_dim) * 0.02
        )

        # ── 5. Transformer blocks ─────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # ── 6. Output ─────────────────────────────────────────────────────
        self.norm        = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

    def _make_pos(self, time_indices: list[int]) -> torch.Tensor:
        """
        Combined spatial + temporal positional encoding.

        Returns (1, T * P, predictor_dim) where T = len(time_indices).
        Layout: [t0p0, t0p1, ..., t0p(P-1), t1p0, ..., t(T-1)p(P-1)]
        """
        T = len(time_indices)
        P = self.num_patches
        t_pos = self.temporal_pos[:, time_indices, :].unsqueeze(2)  # (1, T, 1, D)
        s_pos = self.spatial_pos.unsqueeze(1)                        # (1, 1, P, D)
        pos   = (t_pos + s_pos).reshape(1, T * P, self.predictor_dim)
        return pos

    def _embed_actions(self, target_actions: torch.Tensor) -> torch.Tensor:
        """
        Embed (from_sq, to_sq) and expand across P patches per timestep.

        Args:
            target_actions : (B, T_tgt, 2) int64

        Returns:
            (B, T_tgt * P, predictor_dim)
        """
        P = self.num_patches

        from_emb = self.square_embed(target_actions[..., 0])  # (B, T_tgt, sq_dim)
        to_emb   = self.square_embed(target_actions[..., 1])  # (B, T_tgt, sq_dim)

        action_raw     = torch.cat([from_emb, to_emb], dim=-1)  # (B, T_tgt, predictor_dim)
        action_per_t   = self.action_proj(action_raw)            # (B, T_tgt, predictor_dim)

        # Broadcast the same action embedding to all P patches at each timestep
        # (B, T_tgt, D) → (B, T_tgt, P, D) → (B, T_tgt*P, D)
        action_expanded = action_per_t.unsqueeze(2).expand(-1, -1, P, -1)
        return action_expanded.reshape(action_expanded.shape[0], -1, self.predictor_dim)

    def forward(
        self,
        context_latents: torch.Tensor,
        context_positions: list[int],
        target_positions: list[int],
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_latents  : (B, T_ctx * P, encoder_dim)
            context_positions: list[int] — time indices of context frames
            target_positions : list[int] — time indices to predict
            target_actions   : (B, T_tgt, 2) int64 — [from_sq, to_sq] per frame
                               Use index 64 for null / unknown moves.

        Returns:
            predicted_latents : (B, T_tgt * P, encoder_dim)
        """
        B     = context_latents.shape[0]
        T_tgt = len(target_positions)
        P     = self.num_patches

        # ── 1. Project and position-encode context tokens ─────────────────
        ctx = self.input_proj(context_latents)        # (B, T_ctx*P, predictor_dim)
        ctx = ctx + self._make_pos(context_positions)

        # ── 2. Build action-conditioned mask tokens ───────────────────────
        mask_tokens = self.mask_token.expand(B, T_tgt * P, -1)   # (B, T_tgt*P, D)

        tgt_pos    = self._make_pos(target_positions)             # (1, T_tgt*P, D)
        action_emb = self._embed_actions(target_actions)          # (B, T_tgt*P, D)

        # Fuse: positional location + action physics
        mask_tokens = mask_tokens + tgt_pos + action_emb

        # ── 3. Transformer forward ────────────────────────────────────────
        x = torch.cat([ctx, mask_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # ── 4. Extract and project target tokens ──────────────────────────
        target_tokens = x[:, -T_tgt * P:, :]
        return self.output_proj(target_tokens)         # (B, T_tgt*P, encoder_dim)