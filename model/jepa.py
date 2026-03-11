"""
Chess V-JEPA — Full JEPA Module

Wires together:
  • Context Encoder  — trainable, produces latents for observed positions
  • Target Encoder   — EMA copy of context encoder, frozen (no gradients)
  • Predictor        — maps context latents → predicted target latents

Training signal:
  L2 loss between predictor output and target encoder output (stop-gradient).

EMA (Exponential Moving Average) update:
  After each training step, the target encoder's weights are updated as:
      target_param = τ · target_param + (1 − τ) · context_param
  τ starts high (0.996) and anneals toward 1.0 via a cosine schedule.
  This makes the target encoder a slowly-evolving version of the context
  encoder — providing stable, consistent targets for the predictor.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import ChessBoardEncoder
from model.predictor import Predictor


class ChessJEPA(nn.Module):
    """
    Full V-JEPA model for chess board sequences.

    Args:
        encoder_kwargs: dict of kwargs for ChessBoardEncoder
        predictor_kwargs: dict of kwargs for Predictor
        ema_momentum_start: initial EMA momentum τ (default 0.996)
        ema_momentum_end:   final EMA momentum τ (default 1.0)
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
        #   This is the main encoder that receives gradients during training.
        self.context_encoder = ChessBoardEncoder(**encoder_kwargs)

        # ── 2. Target Encoder (EMA copy, frozen) ────────────────────────
        #   A deep copy of the context encoder.  We immediately freeze it
        #   (requires_grad=False) because it's updated ONLY through EMA,
        #   never through backprop.
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # ── 3. Predictor ────────────────────────────────────────────────
        self.predictor = Predictor(**predictor_kwargs)

        # ── 4. EMA schedule parameters ──────────────────────────────────
        self.ema_momentum_start = ema_momentum_start
        self.ema_momentum_end = ema_momentum_end

    # ─────────────────────────────────────────────────────────────────────
    # EMA Update
    # ─────────────────────────────────────────────────────────────────────

    def get_ema_momentum(self, step: int, total_steps: int) -> float:
        """
        Cosine schedule for EMA momentum.

        Starts at τ_start (0.996) and anneals to τ_end (1.0).
        As τ → 1.0 the target encoder changes more and more slowly,
        providing increasingly stable targets late in training.
        """
        ratio = step / max(total_steps, 1)
        return (
            self.ema_momentum_end
            - (self.ema_momentum_end - self.ema_momentum_start)
            * (math.cos(math.pi * ratio) + 1) / 2
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum: float):
        """
        EMA update: target = τ·target + (1−τ)·context

        Called once per training step, AFTER the optimizer step.
        Must be inside torch.no_grad() since we don't want this
        to participate in the computational graph.
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
        context_indices: list[int],
        target_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            boards          : (B, T, 17, 8, 8) — full chunk of board states
            context_indices : list of int — which time steps are observed
                              e.g. [0, 1, 2, ..., 9]
            target_indices  : list of int — which time steps to predict
                              e.g. [10, 11, 12, 13, 14, 15]

        Returns:
            predicted : (B, T_tgt, encoder_dim) — predictor output
            targets   : (B, T_tgt, encoder_dim) — target encoder output
                        (detached, no gradients)
        """
        # ── Split the chunk into context and target boards ───────────────
        context_boards = boards[:, context_indices]  # (B, T_ctx, 17, 8, 8)
        target_boards = boards[:, target_indices]    # (B, T_tgt, 17, 8, 8)

        # ── Context encoder (trainable, receives gradients) ──────────────
        context_latents = self.context_encoder(context_boards)
        # → (B, T_ctx, encoder_dim)

        # ── Target encoder (frozen, no gradients) ────────────────────────
        with torch.no_grad():
            target_latents = self.target_encoder(target_boards)
            # → (B, T_tgt, encoder_dim)

        # ── Predictor: context latents → predicted target latents ────────
        predicted_latents = self.predictor(
            context_latents, context_indices, target_indices
        )
        # → (B, T_tgt, encoder_dim)

        return predicted_latents, target_latents

    # ─────────────────────────────────────────────────────────────────────
    # Loss
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_loss(
        predicted: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        L2 (MSE) loss between predicted and target latents.

        The targets are already detached (no grad flows to target encoder).
        We normalize both predicted and target vectors before computing MSE.
        This variance-normalization trick (from V-JEPA) prevents
        representation collapse and stabilizes training.
        """
        # ── Normalize along the feature dimension ────────────────────────
        #   After normalization, each latent vector has unit variance.
        #   This prevents the model from collapsing to a constant output.
        predicted = F.layer_norm(predicted, predicted.shape[-1:])
        targets = F.layer_norm(targets, targets.shape[-1:])

        # ── MSE loss ─────────────────────────────────────────────────────
        loss = F.mse_loss(predicted, targets)

        return loss
