"""
Chess V-JEPA — Configuration

All hyperparameters live here in a single dataclass.  Modify these values
or override from the command line (train.py uses argparse for common ones).
"""

from dataclasses import dataclass, field


@dataclass
class JEPAConfig:
    """All hyperparameters for the Chess V-JEPA model and training."""

    resume_from: str | None = None       # path to checkpoint to resume training

    # ── Data ─────────────────────────────────────────────────────────────
    zarr_path: str = "chess_chunks.zarr"
    seq_len: int = 16                    # board positions per chunk
    val_fraction: float = 0.05           # fraction held out for validation

    # ── Encoder ──────────────────────────────────────────────────────────
    in_channels: int = 17                # piece planes + metadata
    board_size: int = 8
    patch_size: int = 2                  # 2×2 → 16 patches per board
    embed_dim: int = 256                 # encoder hidden dimension
    encoder_depth: int = 6               # transformer blocks in encoder
    encoder_heads: int = 8               # attention heads in encoder
    mlp_ratio: float = 4.0               # FFN expansion factor
    dropout: float = 0.0

    # ── Predictor ────────────────────────────────────────────────────────
    predictor_dim: int = 128             # narrower than encoder (bottleneck)
    predictor_depth: int = 4             # shallower than encoder
    predictor_heads: int = 4

    # ── Masking ──────────────────────────────────────────────────────────
    target_ratio: float = 0.4            # ~40% of positions as targets
    mask_mode: str = "causal_float"            # "causal" or "random"
    min_context: int = 4                 # minimum context positions

    # ── EMA ──────────────────────────────────────────────────────────────
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 1.0

    # ── Optimization ─────────────────────────────────────────────────────
    batch_size: int = 512
    learning_rate: float | None = None   # auto-scaled based on batch_size if None
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    max_epochs: int = 100
    max_steps: int | None = None         # optional: stop after N steps

    # ── Infrastructure ───────────────────────────────────────────────────
    num_workers: int = 14
    device: str = "cuda"
    mixed_precision: bool = True         # AMP for NVIDIA GPUs
    checkpoint_dir: str = "checkpoints"
    log_every: int = 50                  # print loss every N steps
    save_every_epochs: int = 5           # save checkpoint every N epochs
    max_checkpoints_to_keep: int = 5     # 0 = keep all

    # ── Initialization ───────────────────────────────────────────────────
    def __post_init__(self):
        # Linear scaling rule: base lr 1.5e-4 is for batch size 256
        if self.learning_rate is None:
            self.learning_rate = 1.5e-4 * (self.batch_size / 256)

    # ── Derived properties ───────────────────────────────────────────────

    @property
    def encoder_kwargs(self) -> dict:
        """Kwargs for ChessBoardEncoder."""
        return dict(
            in_channels=self.in_channels,
            board_size=self.board_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            mlp_ratio=self.mlp_ratio,
            max_seq_len=self.seq_len,
            dropout=self.dropout,
        )

    @property
    def predictor_kwargs(self) -> dict:
        """Kwargs for Predictor / ActionConditionedPredictor."""
        num_patches = (self.board_size // self.patch_size) ** 2   # 16
        return dict(
            encoder_dim=self.embed_dim,
            predictor_dim=self.predictor_dim,
            depth=self.predictor_depth,
            num_heads=self.predictor_heads,
            mlp_ratio=self.mlp_ratio,
            max_seq_len=self.seq_len,
            num_patches=num_patches,
            dropout=self.dropout,
        )
