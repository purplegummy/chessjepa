"""
Chess V-JEPA — Predictor Network

The predictor is a *lightweight* transformer that takes context latents from
the context encoder and predicts what the target encoder would output for the
unseen (future) board positions.

Key V-JEPA design choices:
  • The predictor is NARROWER than the encoder (predictor_dim < embed_dim).
    This is intentional — it acts as an information bottleneck that forces the
    encoder to learn rich representations (if the predictor were too powerful,
    the encoder could be lazy and let the predictor do all the work).
  • Learnable MASK TOKENS stand in for the target positions. The predictor
    sees these alongside the context latents and must "fill in" the targets.
  • Positional encodings tell the predictor WHICH time steps are targets
    so it knows what to predict.

Input:
    context_latents : (B, T_ctx, encoder_dim)   — from the context encoder
    target_positions: list[int]                  — which time indices to predict

Output:
    predicted_latents : (B, T_tgt, encoder_dim)  — predictions in encoder space
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Re-use the same TransformerBlock from the encoder
# ─────────────────────────────────────────────────────────────────────────────
from model.encoder import TransformerBlock


class Predictor(nn.Module):
    """
    V-JEPA-style predictor.

    Takes context latents, appends mask tokens for target positions,
    runs a shallow transformer, and outputs predicted target latents.

    Args:
        encoder_dim:    dimension of the encoder output (default 256)
        predictor_dim:  internal dimension of the predictor — intentionally
                        smaller than encoder_dim to act as a bottleneck
                        (default 128)
        depth:          number of transformer blocks (default 4)
        num_heads:      attention heads (default 4)
        mlp_ratio:      FFN expansion ratio (default 4.0)
        max_seq_len:    maximum number of time steps (default 16)
        dropout:        dropout rate (default 0.0)
    """

    def __init__(
        self,
        encoder_dim: int = 256,
        predictor_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim

        # ── 1. Input projection: encoder space → predictor space ─────────
        #   Narrows the representation.  This bottleneck is crucial:
        #   it prevents the predictor from being too expressive, which
        #   would let the encoder get away with learning poor features.
        self.input_proj = nn.Linear(encoder_dim, predictor_dim)

        # ── 2. Learnable mask token ──────────────────────────────────────
        #   One shared mask token vector that gets placed at each target
        #   position.  The predictor must transform these identical tokens
        #   into distinct predictions — the only differentiator is the
        #   positional encoding added to each one.
        self.mask_token = nn.Parameter(torch.randn(1, 1, predictor_dim) * 0.02)

        # ── 3. Positional encoding for the predictor ─────────────────────
        #   Tells the predictor which time step each token corresponds to.
        #   Both context and target tokens get their respective time-step
        #   encodings, so the predictor can reason about temporal order.
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, predictor_dim) * 0.02
        )

        # ── 4. Transformer blocks ───────────────────────────────────────
        #   Shallow and narrow — this is intentional.
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # ── 5. Layer norm + output projection ────────────────────────────
        #   Projects back from predictor space → encoder space so that the
        #   predicted latents can be compared with the target encoder output.
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

    def forward(
        self,
        context_latents: torch.Tensor,
        context_positions: list[int],
        target_positions: list[int],
    ) -> torch.Tensor:
        """
        Args:
            context_latents  : (B, T_ctx, encoder_dim) — from context encoder
            context_positions: list of int — time indices of context tokens
                               e.g. [0, 1, 2, ..., 9]
            target_positions : list of int — time indices to predict
                               e.g. [10, 11, 12, 13, 14, 15]

        Returns:
            predicted_latents : (B, T_tgt, encoder_dim)
        """
        B = context_latents.shape[0]
        T_tgt = len(target_positions)

        # ── 1. Project context latents to predictor dim ──────────────────
        ctx = self.input_proj(context_latents)  # (B, T_ctx, predictor_dim)

        # ── 2. Create mask tokens for target positions ───────────────────
        #   Expand the single mask token to (B, T_tgt, predictor_dim)
        mask_tokens = self.mask_token.expand(B, T_tgt, -1)

        # ── 3. Add positional encodings ──────────────────────────────────
        #   Context tokens get their actual time-step positions
        ctx_pos = self.pos_embed[:, context_positions, :]   # (1, T_ctx, D)
        ctx = ctx + ctx_pos

        #   Mask tokens get the target time-step positions — this is the
        #   ONLY information the predictor has about what to predict
        tgt_pos = self.pos_embed[:, target_positions, :]    # (1, T_tgt, D)
        mask_tokens = mask_tokens + tgt_pos

        # ── 4. Concatenate context + mask tokens ─────────────────────────
        #   The predictor attends over everything: real context info +
        #   placeholder tokens at target positions.
        #   Context first, then targets (order doesn't matter for attention,
        #   but it makes extraction easier).
        x = torch.cat([ctx, mask_tokens], dim=1)  # (B, T_ctx + T_tgt, D)

        # ── 5. Run through transformer blocks ────────────────────────────
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # ── 6. Extract only the target tokens and project to encoder dim ─
        #   The last T_tgt tokens correspond to target positions
        target_tokens = x[:, -T_tgt:, :]          # (B, T_tgt, predictor_dim)

        predicted = self.output_proj(target_tokens)  # (B, T_tgt, encoder_dim)

        return predicted
