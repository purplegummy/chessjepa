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
    def __init__(self, in_features=4096, hidden_features=512, num_layers=3, dropout=0.2):
        super().__init__()
        # 1. Shared features are minimal
        self.initial_layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.LayerNorm(hidden_features)
        )

        # 2. Independent Policy Trunk
        self.policy_trunk = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_features),
            nn.Linear(hidden_features, 4096) # NUM_MOVES
        )

        # 3. Independent Value Trunk
        self.value_trunk = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, 1)
        )

        # CRITICAL: Force the value head to start at ZERO
        torch.nn.init.zeros_(self.value_trunk[-1].weight)
        torch.nn.init.zeros_(self.value_trunk[-1].bias)

    def forward(self, x):
        if x.ndim == 4: x = x[:, -1]
        if x.ndim == 3: x = x.reshape(x.shape[0], -1)
        
        shared = self.initial_layer(x)
        policy_logits = self.policy_trunk(shared)
        value = self.value_trunk(shared).squeeze(-1)
        return policy_logits, value