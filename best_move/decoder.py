"""
Best Move Decoder.
Maps 256-D board embeddings to a move logit over 4096 (from_sq * 64 + to_sq) classes.
"""

import torch
import torch.nn as nn

NUM_MOVES = 4096  # 64 from-squares × 64 to-squares


class BestMoveDecoder(nn.Module):
    def __init__(self, in_features: int = 256, hidden_features: int = 512, num_layers: int = 3):
        super().__init__()

        layers = []
        current_features = in_features

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_features, hidden_features))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_features))
            current_features = hidden_features

        layers.append(nn.Linear(current_features, NUM_MOVES))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, SeqLen, C) or (B, C) latent from context encoder
        Returns: (B, NUM_MOVES) logits
        """
        if x.ndim == 3:
            x = x[:, -1, :]  # use the current (last) position's embedding, not the sequence mean
        return self.net(x)
