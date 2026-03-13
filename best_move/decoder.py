"""
Best Move Decoder.
Maps per-patch board embeddings to a move logit over 4096 (from_sq * 64 + to_sq) classes.

Input shapes accepted:
  (B, P, D)    — single board, P patches of dim D  (e.g. 16 × 256)
  (B, T, P, D) — sequence of boards; uses the last timestep
  (B, D)       — already pooled embedding (legacy)

The forward pass flattens all P patches into one vector (B, P*D) so the
MLP sees the entire spatial grid rather than a single corner.
"""

import torch
import torch.nn as nn

NUM_MOVES = 4096  # 64 from-squares × 64 to-squares


class BestMoveDecoder(nn.Module):
    def __init__(
        self,
        in_features: int = 4096,   # encoder_dim * num_patches = 256 * 16
        hidden_features: int = 512,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        current_features = in_features
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_features, hidden_features))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(hidden_features))
            current_features = hidden_features
        # shared trunk (everything except final policy linear)
        # we'll build heads after constructing the trunk so we can reuse
        # the `current_features` value for both policy and value outputs.
        self.trunk = nn.Sequential(*layers)

        # policy head produces logits over the 4096 move classes
        self.policy_head = nn.Linear(current_features, NUM_MOVES)
        # value head predicts a single scalar evaluation
        # (we'll squeeze the last dimension in forward)
        self.value_head = nn.Linear(current_features, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x : (B, T, P, D) | (B, P, D) | (B, D)

        Returns:
            policy_logits : (B, NUM_MOVES)  — raw move logits
            value         : (B,)             — scalar evaluation
        """
        if x.ndim == 4:
            # Sequence of boards — use the last timestep's patches
            x = x[:, -1]          # (B, P, D)
        if x.ndim == 3:
            # Flatten all spatial patches into one vector
            B = x.shape[0]
            x = x.reshape(B, -1)  # (B, P*D)
        # x is now (B, P*D) or (B, D) — run through shared trunk
        x = self.trunk(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy_logits, value
