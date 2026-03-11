"""
Chess V-JEPA — Temporal Masking Strategy

Decides which board positions in a chunk are CONTEXT (observed by the context
encoder) and which are TARGETS (to be predicted by the predictor).

V-JEPA uses contiguous block masking in the temporal dimension.  For chess
this means: "observe the first N moves, predict the next M moves."

Two masking modes:

1. CAUSAL (default) — Context is always the EARLIER positions, targets are
   the LATER ones.  This mirrors how chess actually works: you see the past
   and predict the future.

   Example (seq_len=16, target_len=6):
       context = [0, 1, 2, ..., 9]     (first 10)
       target  = [10, 11, 12, ..., 15] (last 6)

2. RANDOM — Target block can start anywhere, and the remaining positions
   are context.  More diverse training signal, but loses the causal
   structure.  Included for experimentation.
"""

import random


def generate_temporal_mask(
    seq_len: int = 16,
    target_ratio: float = 0.4,
    mode: str = "causal",
    min_context: int = 4,
) -> tuple[list[int], list[int]]:
    """
    Generate context and target index lists for one training sample.

    Args:
        seq_len      : total number of time steps in the chunk (default 16)
        target_ratio : approximate fraction of steps to use as targets
                       (default 0.4 → ~6 out of 16 are targets)
        mode         : "causal" or "random" (default "causal")
        min_context  : minimum number of context positions — ensures the
                       predictor always has enough history (default 4)

    Returns:
        (context_indices, target_indices) — two sorted lists of ints
    """
    # ── Compute target block length ──────────────────────────────────────
    #   Add some randomness (±1) so the model doesn't memorize a fixed split
    target_len = max(1, round(seq_len * target_ratio))
    jitter = random.randint(-1, 1)
    target_len = max(1, min(target_len + jitter, seq_len - min_context))

    if mode == "causal":
        # ── Causal: target = last target_len positions ───────────────────
        #   The most natural choice for chess: observe early game, predict
        #   later moves.  A small random offset can be added to vary the
        #   split point.
        #
        #   timeline:  [0 1 2 3 4 5 6 7 8 9 | 10 11 12 13 14 15]
        #               ←── context ──────→   ←── target ────→
        target_start = seq_len - target_len
        target_indices = list(range(target_start, seq_len))
        context_indices = list(range(0, target_start))

    elif mode == "random":
        # ── Random: target block can start anywhere ──────────────────────
        #   Pick a random start position for the target block.
        #   Everything else is context.
        max_start = seq_len - target_len
        target_start = random.randint(0, max_start)
        target_indices = list(range(target_start, target_start + target_len))
        context_indices = [i for i in range(seq_len) if i not in target_indices]

        # Ensure minimum context
        if len(context_indices) < min_context:
            # Fall back to causal
            target_start = seq_len - target_len
            target_indices = list(range(target_start, seq_len))
            context_indices = list(range(0, target_start))

    else:
        raise ValueError(f"mode must be 'causal' or 'random', got '{mode}'")

    return context_indices, target_indices


class TemporalMaskGenerator:
    """
    Callable mask generator for use in the training loop.

    Usage:
        mask_gen = TemporalMaskGenerator(seq_len=16, target_ratio=0.4)
        ctx_idx, tgt_idx = mask_gen()   # call once per batch
    """

    def __init__(
        self,
        seq_len: int = 16,
        target_ratio: float = 0.4,
        mode: str = "causal",
        min_context: int = 4,
    ):
        self.seq_len = seq_len
        self.target_ratio = target_ratio
        self.mode = mode
        self.min_context = min_context

    def __call__(self) -> tuple[list[int], list[int]]:
        """Generate a new mask (call once per batch for variety)."""
        return generate_temporal_mask(
            self.seq_len,
            self.target_ratio,
            self.mode,
            self.min_context,
        )

    def __repr__(self) -> str:
        return (
            f"TemporalMaskGenerator(seq_len={self.seq_len}, "
            f"target_ratio={self.target_ratio}, mode='{self.mode}')"
        )
