"""
Chess AC-JEPA — Action-Conditioned Predictor Network

An extension of the standard V-JEPA Predictor that incorporates the chess
move (action) that *caused* each target board state.

Key idea
--------
In standard V-JEPA the predictor sees:
  • Context latents at known time steps
  • Learnable mask tokens at target positions (differentiated only by pos. encoding)

Here we add one more signal to each mask token:
  • The chess move (from_square, to_square) that led to this board state

This gives the predictor the *physics* of what happened — not just "predict the
future" but "predict the future *given that this specific move was played*".

Action encoding
---------------
A chess move consists of two squares on an 8×8 board (0–63 each).
We embed them independently with a shared Embedding table of size 64.
If predictor_dim = D, each square → D//2; concatenated → D; projected → D.
A special "null move" index (64) is supported so the class can handle padding
or positions with no action gracefully.

Input / Output shapes
---------------------
  context_latents : (B, T_ctx, encoder_dim)
  context_positions: list[int]              length T_ctx
  target_positions : list[int]              length T_tgt
  target_actions   : (B, T_tgt, 2) int64   [from_sq, to_sq]  values 0–64

  → predicted_latents : (B, T_tgt, encoder_dim)
"""

import torch
import torch.nn as nn

from model.encoder import TransformerBlock


class ActionConditionedPredictor(nn.Module):
    """
    V-JEPA-style predictor conditioned on chess move actions.

    Compared to the standard Predictor, this inserts an action embedding
    at each target mask token so the predictor knows *which move* to
    extrapolate from.

    Args:
        encoder_dim    : dimension of the encoder output (default 256)
        predictor_dim  : internal dimension of the predictor — intentionally
                         narrower than encoder_dim to act as a bottleneck
                         (default 128)
        depth          : number of transformer blocks (default 4)
        num_heads      : attention heads (default 4)
        mlp_ratio      : FFN expansion ratio (default 4.0)
        max_seq_len    : maximum number of time steps (default 16)
        dropout        : dropout rate (default 0.0)
        num_squares    : vocabulary size for square embeddings (65 = 64 squares +
                         1 null token for padding; default 65)
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
        num_squares: int = 65,        # 64 board squares + 1 null / padding index
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim

        # ── 1. Input projection: encoder space → predictor space ─────────
        #   Intentional bottleneck — prevents the predictor from being too
        #   expressive, which would let the encoder be lazy.
        self.input_proj = nn.Linear(encoder_dim, predictor_dim)

        # ── 2. Action embedding ──────────────────────────────────────────
        #   Embed each square independently (shared table for from/to).
        #   We use predictor_dim // 2 per square so that concatenating
        #   from_emb + to_emb gives a vector of exactly predictor_dim.
        #   Index 64 is reserved as a null/padding token.
        sq_dim = predictor_dim // 2
        self.square_embed = nn.Embedding(num_squares, sq_dim)
        # Project the raw concatenated action (pred_dim) → pred_dim,
        # allowing the network to mix from/to information nonlinearly.
        self.action_proj = nn.Linear(predictor_dim, predictor_dim)

        # ── 3. Learnable mask token ──────────────────────────────────────
        #   One shared vector per target slot; differentiated by positional
        #   encoding and by the injected action embedding below.
        self.mask_token = nn.Parameter(torch.randn(1, 1, predictor_dim) * 0.02)

        # ── 4. Positional encoding ────────────────────────────────────────
        #   Tells the predictor which time step each token belongs to.
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, predictor_dim) * 0.02
        )

        # ── 5. Transformer blocks ─────────────────────────────────────────
        #   Shallow and narrow — intentional information bottleneck.
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # ── 6. Layer norm + output projection ────────────────────────────
        #   Projects back from predictor space → encoder space so predictions
        #   can be compared to target encoder outputs via MSE loss.
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

    def _embed_actions(self, target_actions: torch.Tensor) -> torch.Tensor:
        """
        Embed (from_sq, to_sq) pairs into predictor_dim vectors.

        Args:
            target_actions : (B, T_tgt, 2)  int64 — values in [0, 64]

        Returns:
            action_tokens : (B, T_tgt, predictor_dim)
        """
        from_sq = target_actions[..., 0]   # (B, T_tgt)
        to_sq   = target_actions[..., 1]   # (B, T_tgt)

        from_emb = self.square_embed(from_sq)   # (B, T_tgt, sq_dim)
        to_emb   = self.square_embed(to_sq)     # (B, T_tgt, sq_dim)

        # Concatenate: (B, T_tgt, predictor_dim)
        action_raw = torch.cat([from_emb, to_emb], dim=-1)

        # Linear mix of the from/to information
        return self.action_proj(action_raw)     # (B, T_tgt, predictor_dim)

    def forward(
        self,
        context_latents: torch.Tensor,
        context_positions: list[int],
        target_positions: list[int],
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_latents  : (B, T_ctx, encoder_dim) — from context encoder
            context_positions: list of int — time indices of context tokens
                               e.g. [0, 1, 2, ..., 9]
            target_positions : list of int — time indices to predict
                               e.g. [10, 11, 12, 13, 14, 15]
            target_actions   : (B, T_tgt, 2) int64 — [from_sq, to_sq]
                               The move that was played to reach each target board.
                               Use index 64 for null / unknown moves.

        Returns:
            predicted_latents : (B, T_tgt, encoder_dim)
        """
        B = context_latents.shape[0]
        T_tgt = len(target_positions)

        # ── 1. Project context latents to predictor dim ──────────────────
        ctx = self.input_proj(context_latents)          # (B, T_ctx, predictor_dim)

        # ── 2. Add positional encodings to context tokens ────────────────
        ctx = ctx + self.pos_embed[:, context_positions, :]   # (1, T_ctx, D)

        # ── 3. Build action-conditioned mask tokens ───────────────────────
        #   Start from the shared mask token, then add:
        #     a) temporal positional encoding  (WHERE in time to predict)
        #     b) action embedding              (WHAT move was played)
        mask_tokens = self.mask_token.expand(B, T_tgt, -1)  # (B, T_tgt, D)

        tgt_pos     = self.pos_embed[:, target_positions, :]  # (1, T_tgt, D)
        action_emb  = self._embed_actions(target_actions)     # (B, T_tgt, D)

        # Injection: position + action physics are fused into each mask token
        mask_tokens = mask_tokens + tgt_pos + action_emb     # (B, T_tgt, D)

        # ── 4. Concatenate context + mask tokens ─────────────────────────
        #   Full sequence the predictor attends over:
        #   [real context tokens | action-conditioned target placeholders]
        x = torch.cat([ctx, mask_tokens], dim=1)  # (B, T_ctx + T_tgt, D)

        # ── 5. Run through transformer blocks ────────────────────────────
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # ── 6. Extract target tokens and project to encoder dim ──────────
        #   The last T_tgt tokens correspond to target positions.
        target_tokens = x[:, -T_tgt:, :]               # (B, T_tgt, predictor_dim)
        predicted = self.output_proj(target_tokens)     # (B, T_tgt, encoder_dim)

        return predicted