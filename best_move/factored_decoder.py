"""
Factored Move Decoder.

Instead of 4096-class classification (from_sq * 64 + to_sq), this decoder
treats the move as two sequential 64-class problems:

  1. from-square head : board_embedding → 64 logits
  2. to-square head   : [board_embedding ‖ from_sq_embedding] → 64 logits

Training uses teacher forcing on the from-square so both heads receive
clean gradients simultaneously.

Inference uses score_all() which computes a full (64 × 64) score matrix
in a single batched forward pass by broadcasting the board embedding against
all 64 learnable from-square embeddings, then slicing out legal moves.
"""

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_hidden: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = in_dim
    for _ in range(num_hidden):
        layers += [nn.Linear(current, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
        current = hidden_dim
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


class FactoredMoveDecoder(nn.Module):
    """
    Args:
        in_features : dimension of the board embedding from the JEPA encoder (256)
        hidden      : hidden width for both MLP heads
        num_hidden  : number of hidden layers in each head

    This decoder has three outputs now: from-square logits, to-square logits and
    a scalar value prediction. The value head looks only at the board embedding
    (not conditioned on the chosen from-square), similar to AlphaZero-style
    dual‑head architectures.
    """

    def __init__(self, in_features: int = 256, hidden: int = 512, num_hidden: int = 2):
        super().__init__()

        # Learnable embedding for each of the 64 squares (used to condition to-head)
        self.from_sq_embed = nn.Embedding(64, in_features)

        # Head 1 — from-square: board_emb → 64
        self.from_head = _mlp(in_features, hidden, 64, num_hidden)

        # Head 2 — to-square: [board_emb ‖ from_sq_emb] → 64
        self.to_head = _mlp(in_features * 2, hidden, 64, num_hidden)

        # Value head — board_emb → 1 scalar
        self.value_head = _mlp(in_features, hidden, 1, num_hidden)

    # ── training forward ──────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        from_sq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x        : (B, C) board embedding  [if (B, SeqLen, C) takes last step]
            from_sq  : (B,) ground-truth from-squares for teacher forcing.
                       Pass None at inference time — argmax is used instead.

        Returns:
            from_logits : (B, 64)
            to_logits   : (B, 64)
            value       : (B,)
        """
        if x.ndim == 3:
            x = x[:, -1, :]                           # (B, C)

        from_logits = self.from_head(x)                # (B, 64)

        if from_sq is None:
            # Inference path — use the decoder's own prediction
            from_sq = from_logits.argmax(dim=1)        # (B,)

        sq_emb  = self.from_sq_embed(from_sq)          # (B, C)
        to_inp  = torch.cat([x, sq_emb], dim=1)        # (B, 2C)
        to_logits = self.to_head(to_inp)               # (B, 64)

        # value prediction uses only the board embedding
        value = self.value_head(x).squeeze(-1)         # (B,)

        return from_logits, to_logits, value

    # ── inference helper ──────────────────────────────────────────────────────

    def score_all(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a (B, 64, 64) score matrix for every (from_sq, to_sq) pair
        in a single batched forward pass.

        score[b, f, t] = from_logit[b, f] + to_logit[b, f, t]

        This lets callers filter to legal moves and pick the argmax efficiently
        without any Python loop over squares.

        Args:
            x : (B, C) or (B, SeqLen, C) board embedding

        Returns:
            scores : (B, 64, 64)  — from-square × to-square score matrix
            value  : (B,)          — scalar evaluation for each board
        """
        if x.ndim == 3:
            x = x[:, -1, :]                            # (B, C)

        B, C = x.shape

        # ── from-square scores ────────────────────────────────────────────────
        from_logits = self.from_head(x)                # (B, 64)

        # ── to-square scores for every from-square ────────────────────────────
        # Expand x to (B, 64, C) and from_sq_embed to (B, 64, C), then concat
        x_expanded  = x.unsqueeze(1).expand(B, 64, C)                          # (B, 64, C)
        sq_embs     = self.from_sq_embed.weight.unsqueeze(0).expand(B, 64, C)  # (B, 64, C)
        to_inp      = torch.cat([x_expanded, sq_embs], dim=2)                  # (B, 64, 2C)

        # Flatten batch+from_sq, run to_head, reshape back
        to_logits   = self.to_head(to_inp.view(B * 64, 2 * C))                 # (B*64, 64)
        to_logits   = to_logits.view(B, 64, 64)                                # (B, 64, 64)

        # Combined score: broadcast from_logits over to-square axis
        scores = from_logits.unsqueeze(2) + to_logits                          # (B, 64, 64)
        # compute value for convenience too
        value = self.value_head(x).squeeze(-1)                                   # (B,)
        return scores, value
