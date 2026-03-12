"""
Win Probability Decoder.
Maps 256-D board embeddings to a single scalar representing White's Win Probability.
"""

import torch
import torch.nn as nn

class WinProbabilityDecoder(nn.Module):
    def __init__(self, in_features: int = 256, hidden_features: int = 512, num_layers: int = 3):
        super().__init__()
        
        layers = []
        current_features = in_features
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_features, hidden_features))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_features))
            current_features = hidden_features
            
        # Final layer outputs a single scalar (win probability between 0 and 1)
        layers.append(nn.Linear(current_features, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C) latent embedding from context encoder
        Returns: (B, 1) probability of win
        """
        # JEPA context encoder usually outputs (B, L, C). 
        # If it's a sequence, we probably want to pool it first.
        # However, for a single board position passed as (1, 1, 17, 8, 8), 
        # the output is likely (B, 1, C). We'll handle both shapes by flattening/pooling
        
        if x.ndim == 3: # (B, SeqLen, C)
            # Pool across sequence dimension (average pooling)
            x = x.mean(dim=1)
            
        return self.net(x)
