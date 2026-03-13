"""
Chess AC-JEPA — Action-Conditioned Full JEPA Module

Extends ChessJEPA with move-action conditioning.

Architecture
------------
  • Context Encoder     — same trainable encoder as base JEPA
  • Target Encoder      — same EMA-frozen copy as base JEPA
  • AC Predictor        — ActionConditionedPredictor instead of Predictor
                          receives the *move* that led to each target board

Training signal
---------------
  L2 (MSE) loss between predicted latents and target encoder latents,
  identical to the base JEPA — the only change is that the predictor now
  has access to action information which helps it make sharper predictions.

Action format
-------------
  actions : (B, T, 2) int64 — full sequence of moves aligned to the board
            sequence.  actions[:, t] = (from_sq, to_sq) of the move played
            to produce board state boards[:, t].  Use index 64 for padding
            or unknown moves (e.g. the very first board in a game chunk).

Usage example
-------------
    model = ActionConditionedChessJEPA(
        encoder_kwargs=cfg.encoder_kwargs,
        predictor_kwargs=cfg.predictor_kwargs,
    ).to(device)

    predicted, targets = model(boards, actions, ctx_idx, tgt_idx)
    loss = ActionConditionedChessJEPA.compute_loss(predicted, targets)
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import ChessBoardEncoder
from model.acpredictor import ActionConditionedPredictor


class ActionConditionedChessJEPA(nn.Module):
    """
    Action-conditioned V-JEPA model for chess board sequences.

    Identical to ChessJEPA in encoder / EMA structure; swaps the standard
    Predictor for an ActionConditionedPredictor.

    Args:
        encoder_kwargs   : dict of kwargs passed to ChessBoardEncoder
        predictor_kwargs : dict of kwargs passed to ActionConditionedPredictor
        ema_momentum_start : initial EMA momentum τ (default 0.996)
        ema_momentum_end   : final EMA momentum τ   (default 1.0)
    """

    def __init__(
        self,
        encoder_kwargs: dict | None = None,
        predictor_kwargs: dict | None = None,
        ema_momentum_start: float = 0.996,
        ema_momentum_end: float = 1.0,
    ):
        super().__init__()
        encoder_kwargs = encoder_kwargs or {}
        predictor_kwargs = predictor_kwargs or {}

        # ── 1. Context Encoder (trainable) ───────────────────────────────
        self.context_encoder = ChessBoardEncoder(**encoder_kwargs)

        # ── 2. Target Encoder (EMA copy, frozen) ────────────────────────
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # ── 3. Action-Conditioned Predictor ──────────────────────────────
        self.predictor = ActionConditionedPredictor(**predictor_kwargs)

        # ── 4. EMA schedule parameters ──────────────────────────────────
        self.ema_momentum_start = ema_momentum_start
        self.ema_momentum_end = ema_momentum_end

    # ─────────────────────────────────────────────────────────────────────
    # EMA helpers  (identical to ChessJEPA)
    # ─────────────────────────────────────────────────────────────────────

    def get_ema_momentum(self, step: int, total_steps: int) -> float:
        """
        Cosine schedule for EMA momentum.

        Starts at τ_start and anneals toward τ_end (typically 1.0).
        As τ → 1.0 the target encoder changes more slowly, providing
        increasingly stable prediction targets later in training.
        """
        ratio = step / max(total_steps, 1)
        return (
            self.ema_momentum_end
            - (self.ema_momentum_end - self.ema_momentum_start)
            * (math.cos(math.pi * ratio) + 1) / 2
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """
        EMA update: target_param = τ·target_param + (1−τ)·context_param

        Called once per training step, AFTER the optimizer step.
        No gradients — this is a pure parameter copy operation.
        """
        for target_param, context_param in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):
            target_param.data.mul_(momentum).add_(
                context_param.data, alpha=1.0 - momentum
            )

    # ─────────────────────────────────────────────────────────────────────
    # Forward Pass
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        boards: torch.Tensor,
        actions: torch.Tensor,
        context_indices: list[int],
        target_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            boards          : (B, T, 17, 8, 8)  — full chunk of board states
            actions         : (B, T, 2)  int64   — full sequence of moves;
                              actions[:, t] = (from_sq, to_sq) of the move
                              that *produced* boards[:, t].
                              Use square index 64 for null / padding moves.
            context_indices : list[int] — observed time steps
            target_indices  : list[int] — time steps to predict

        Returns:
            predicted : (B, T_tgt * P, encoder_dim) — predictor output
            targets   : (B, T_tgt * P, encoder_dim) — target encoder output
                        P = num_patches = 16
        """
        # ── Split boards into context / target ────────────────────────────
        context_boards = boards[:, context_indices]   # (B, T_ctx, 17, 8, 8)
        target_boards  = boards[:, target_indices]    # (B, T_tgt, 17, 8, 8)

        # Extract the actions that caused the target board states
        target_actions = actions[:, target_indices]   # (B, T_tgt, 2)

        # ── Context encoder → (B, T_ctx, P, D) ───────────────────────────
        context_latents = self.context_encoder(context_boards)
        B, T_ctx, P, D = context_latents.shape
        context_latents_flat = context_latents.reshape(B, T_ctx * P, D)

        # ── Target encoder (frozen) → (B, T_tgt, P, D) ───────────────────
        with torch.no_grad():
            target_latents = self.target_encoder(target_boards)
            T_tgt = target_latents.shape[1]
            target_latents_flat = target_latents.reshape(B, T_tgt * P, D)

        # ── AC Predictor: context patches + actions → predicted patches ───
        predicted_latents = self.predictor(
            context_latents_flat,
            context_indices,
            target_indices,
            target_actions,
        )
        # → (B, T_tgt*P, encoder_dim)

        return predicted_latents, target_latents_flat

    # ─────────────────────────────────────────────────────────────────────
    # Loss  (identical to ChessJEPA)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_loss(
        predicted: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Variance-normalized L2 loss between predicted and target latents.

        Layer-normalizing both tensors before MSE prevents the model from
        collapsing to a constant output (a common failure mode in self-
        supervised learning without explicit contrastive negatives).

        Args:
            predicted : (B, T_tgt, encoder_dim)
            targets   : (B, T_tgt, encoder_dim)  — already detached

        Returns:
            scalar loss tensor
        """
        predicted = F.layer_norm(predicted, predicted.shape[-1:])
        targets   = F.layer_norm(targets,   targets.shape[-1:])
        return F.mse_loss(predicted, targets)