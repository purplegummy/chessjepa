"""
Chess V-JEPA — Predictor Network  (patch-aware version)

The predictor operates over *per-patch* latents produced by the encoder.

Key changes vs. the original mean-pooled predictor
----------------------------------------------------
  • The encoder now returns (B, T, P, D) instead of (B, T, D).
  • jepa.py flattens this to (B, T*P, D) before passing here.
  • The predictor receives T_ctx*P context tokens and produces T_tgt*P
    predicted tokens — one per (timestep, patch) pair.
  • Positional encodings are SPLIT into:
      - temporal : (1, max_seq_len, predictor_dim) — which time step
      - spatial  : (1, num_patches, predictor_dim) — which patch on the board
    Combined per token: pos = temporal[t] + spatial[p]
    This lets the predictor attend to both when and where.

Input / Output shapes
---------------------
  context_latents  : (B, T_ctx * P, encoder_dim)   — from context encoder
  context_positions: list[int]   length T_ctx       — time indices
  target_positions : list[int]   length T_tgt       — time indices to predict

  → predicted_latents : (B, T_tgt * P, encoder_dim)
"""

import torch
import torch.nn as nn

from model.encoder import TransformerBlock


class Predictor(nn.Module):
    """
    V-JEPA-style predictor that operates on full per-patch latents.

    Args:
        encoder_dim   : dimension of encoder output (default 256)
        predictor_dim : internal bottleneck dimension (default 128)
        depth         : number of transformer blocks (default 4)
        num_heads     : attention heads (default 4)
        mlp_ratio     : FFN expansion ratio (default 4.0)
        max_seq_len   : maximum number of time steps (default 16)
        num_patches   : number of spatial patches per board (default 16 = 4×4)
        dropout       : dropout rate (default 0.0)
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
    ):
        super().__init__()
        self.encoder_dim   = encoder_dim
        self.predictor_dim = predictor_dim
        self.num_patches   = num_patches

        # ── 1. Input projection: encoder space → predictor space ─────────
        self.input_proj = nn.Linear(encoder_dim, predictor_dim)

        # ── 2. Learnable mask token ──────────────────────────────────────
        #   One shared token; differentiated by spatial + temporal pos encoding.
        self.mask_token = nn.Parameter(torch.randn(1, 1, predictor_dim) * 0.02)

        # ── 3. Split positional encodings ────────────────────────────────
        #   temporal: one embedding per time step
        #   spatial:  one embedding per patch position on the board
        #   Combined: temporal[t] + spatial[p] for token at (time t, patch p)
        self.temporal_pos = nn.Parameter(
            torch.randn(1, max_seq_len, predictor_dim) * 0.02
        )
        self.spatial_pos = nn.Parameter(
            torch.randn(1, num_patches, predictor_dim) * 0.02
        )

        # ── 4. Transformer blocks ─────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # ── 5. Layer norm + output projection ────────────────────────────
        self.norm        = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

    def _make_pos(self, time_indices: list[int]) -> torch.Tensor:
        """
        Build combined positional encodings for a set of time steps.

        Returns a tensor of shape (1, len(time_indices)*P, predictor_dim)
        where P = num_patches.  The ordering is:
            [t0p0, t0p1, ..., t0p(P-1), t1p0, ..., t(T-1)p(P-1)]
        which matches the flat layout (B, T*P, D) from the encoder.

        Args:
            time_indices : list of timestep indices

        Returns:
            pos : (1, T * P, predictor_dim)
        """
        T = len(time_indices)
        P = self.num_patches

        # temporal: (1, T, D) → (1, T, 1, D)
        t_pos = self.temporal_pos[:, time_indices, :].unsqueeze(2)
        # spatial: (1, P, D) → (1, 1, P, D)
        s_pos = self.spatial_pos.unsqueeze(1)

        # Broadcast add: (1, T, P, D) → flatten → (1, T*P, D)
        pos = (t_pos + s_pos).reshape(1, T * P, self.predictor_dim)
        return pos

    def forward(
        self,
        context_latents: torch.Tensor,
        context_positions: list[int],
        target_positions: list[int],
    ) -> torch.Tensor:
        """
        Args:
            context_latents  : (B, T_ctx * P, encoder_dim)
            context_positions: list[int] — time indices of context frames
            target_positions : list[int] — time indices to predict

        Returns:
            predicted_latents : (B, T_tgt * P, encoder_dim)
        """
        B    = context_latents.shape[0]
        T_tgt = len(target_positions)
        P     = self.num_patches

        # ── 1. Project context to predictor dim ──────────────────────────
        ctx = self.input_proj(context_latents)        # (B, T_ctx*P, predictor_dim)

        # ── 2. Add combined spatial+temporal pos to context tokens ────────
        ctx = ctx + self._make_pos(context_positions) # (1, T_ctx*P, D) broadcast

        # ── 3. Build mask tokens for target positions ─────────────────────
        #   One mask token per (timestep, patch) pair
        mask_tokens = self.mask_token.expand(B, T_tgt * P, -1)  # (B, T_tgt*P, D)
        mask_tokens = mask_tokens + self._make_pos(target_positions)

        # ── 4. Concatenate context + mask tokens and run transformer ──────
        x = torch.cat([ctx, mask_tokens], dim=1)      # (B,(T_ctx+T_tgt)*P, D)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # ── 5. Extract target patch tokens and project to encoder dim ─────
        target_tokens = x[:, -T_tgt * P:, :]          # (B, T_tgt*P, predictor_dim)
        predicted     = self.output_proj(target_tokens) # (B, T_tgt*P, encoder_dim)

        return predicted
