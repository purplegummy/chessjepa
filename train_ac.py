"""
Chess AC-JEPA — Action-Conditioned Training Loop

Trains the ActionConditionedChessJEPA model, which is identical to the
base ChessJEPA except the predictor also receives the chess move (action)
that caused each target board state.

Pipeline
────────
  1. Load (boards, actions) from zarr via ActionChessChunkDataset
  2. Build ActionConditionedChessJEPA (context encoder + EMA target + AC predictor)
  3. Generate temporal masks each batch
  4. Forward: context encoder → AC predictor(latents, actions) → loss vs target encoder
  5. Backward + AdamW + cosine LR schedule
  6. EMA update of target encoder
  7. Repeat

Works with the existing zarr store
────────────────────────────────────
  If the zarr store has no 'actions' key yet, the dataset returns null moves
  (square index 64) as a fallback — training still runs, but the predictor
  receives no action signal.

  To generate proper actions first (recommended):
      python util/generate_actions.py --zarr data/chess_chunks.zarr

Usage
─────
    python train_ac.py
    python train_ac.py --batch_size 128 --learning_rate 3e-4 --max_epochs 50
    python train_ac.py --max_steps 200   # quick smoke test
    python train_ac.py --resume_from checkpoints_ac/checkpoint_epoch0005.pt
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn

from util.config import JEPAConfig
from util.dataset import build_ac_dataloaders
from util.masking import TemporalMaskGenerator, generate_temporal_mask
from model.acjepa import ActionConditionedChessJEPA


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: ActionConditionedChessJEPA, cfg: JEPAConfig):
    """
    AdamW over trainable parameters only.

    The target encoder is frozen (EMA-only), so it's automatically excluded.
    We separate weight-decay from no-decay groups (biases and LayerNorm weights
    should not be decayed).
    """
    decay_params    = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(param_groups, lr=cfg.learning_rate)


# ─────────────────────────────────────────────────────────────────────────────
# LR Scheduler
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup → cosine decay."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, cfg,
                    checkpoint_dir: str):
    """Save model + training state for resuming."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch:04d}.pt")
    torch.save({
        "epoch":     epoch,
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict(),
        "config":    cfg,
    }, path)
    print(f"  💾 Checkpoint saved: {path}")

    if cfg.max_checkpoints_to_keep > 0:
        import glob
        existing = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch*.pt")))
        for old in existing[:-cfg.max_checkpoints_to_keep]:
            os.remove(old)
            print(f"  🗑️  Deleted old checkpoint: {old}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: JEPAConfig, checkpoint_dir: str = "checkpoints_ac"):
    """Main training function for the action-conditioned JEPA."""

    print("=" * 60)
    print("  Chess AC-JEPA Training  (action-conditioned)")
    print("=" * 60)

    # ── 1. Device ─────────────────────────────────────────────────────────
    device = torch.device(cfg.device)
    print(f"  Device         : {device}")

    # ── 2. Data ──────────────────────────────────────────────────────────
    train_loader, val_loader = build_ac_dataloaders(
        cfg.zarr_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_fraction=cfg.val_fraction,
    )
    print(f"  Train samples  : {len(train_loader.dataset):,}")
    print(f"  Val samples    : {len(val_loader.dataset):,}")
    print(f"  Batch size     : {cfg.batch_size}")

    # Report whether the dataset has real action data
    has_real_actions = train_loader.dataset.has_actions
    if has_real_actions:
        print("  Actions        : ✅ loaded from zarr 'actions' array")
    else:
        print("  Actions        : ⚠️  zarr has no 'actions' array — using null moves")
        print("                   Run: python util/generate_actions.py --zarr <path>")

    # ── 3. Model ──────────────────────────────────────────────────────────
    model = ActionConditionedChessJEPA(
        encoder_kwargs=cfg.encoder_kwargs,
        predictor_kwargs=cfg.predictor_kwargs,
        ema_momentum_start=cfg.ema_momentum_start,
        ema_momentum_end=cfg.ema_momentum_end,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params   : {total_params:,}")
    print(f"  Trainable      : {trainable:,}")
    print(f"  Target encoder : {total_params - trainable:,} (frozen, EMA)")

    # ── 4. Optimizer + Scheduler ──────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * cfg.max_epochs
    warmup_steps    = steps_per_epoch * cfg.warmup_epochs

    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    print(f"  Steps/epoch    : {steps_per_epoch:,}")
    print(f"  Total steps    : {total_steps:,}")
    print(f"  Warmup steps   : {warmup_steps:,}")

    # ── 5. Mixed-precision scaler ─────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision)

    # ── 6. Mask generator ─────────────────────────────────────────────────
    mask_gen = TemporalMaskGenerator(
        seq_len=cfg.seq_len,
        target_ratio=cfg.target_ratio,
        mode=cfg.mask_mode,
        min_context=cfg.min_context,
    )
    print(f"  Masking        : {mask_gen}")

    # Fixed validation mask — same split every epoch so val loss is comparable.
    # Seeded with 0 to get a stable, middle-of-the-road causal split regardless
    # of the training mask mode.
    import random as _rng
    _state = _rng.getstate()
    _rng.seed(0)
    val_ctx_idx, val_tgt_idx = generate_temporal_mask(
        seq_len=cfg.seq_len,
        target_ratio=cfg.target_ratio,
        mode="causal",
        min_context=cfg.min_context,
    )
    _rng.setstate(_state)   # restore training RNG state — seed has no side effect
    print(f"  Val mask       : ctx={val_ctx_idx}  tgt={val_tgt_idx}")
    print("=" * 60)

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 0
    global_step = 0

    if cfg.resume_from:
        print(f"  Resuming from: {cfg.resume_from}")
        ckpt = torch.load(cfg.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["step"]

        # Restore LR from config (not checkpoint) if explicitly set
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.learning_rate
            if "initial_lr" in pg:
                pg["initial_lr"] = cfg.learning_rate
        scheduler.base_lrs = [cfg.learning_rate for _ in scheduler.base_lrs]
        print(f"  Resumed at epoch {start_epoch}, step {global_step}")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        epoch_loss  = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch in train_loader:
            # Unpack (boards, actions) from ActionChessChunkDataset
            boards, actions = batch
            boards  = boards.to(device, dtype=torch.float32)  # uint8 → float32
            actions = actions.to(device)                      # (B, T, 2) int64

            # Fresh mask for this batch
            ctx_idx, tgt_idx = mask_gen()

            # ── Forward + loss ──────────────────────────────────────────
            with torch.amp.autocast("cuda", enabled=cfg.mixed_precision):
                predicted, targets = model(boards, actions, ctx_idx, tgt_idx)
                loss = ActionConditionedChessJEPA.compute_loss(predicted, targets)

            # ── Backward ────────────────────────────────────────────────
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # ── EMA update ──────────────────────────────────────────────
            momentum = model.get_ema_momentum(global_step, total_steps)
            model.update_target_encoder(momentum)

            # ── Logging ─────────────────────────────────────────────────
            epoch_loss  += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % cfg.log_every == 0:
                avg = epoch_loss / epoch_steps
                lr  = optimizer.param_groups[0]["lr"]
                print(
                    f"  step {global_step:>6d} | "
                    f"loss {loss.item():.4f} | "
                    f"avg {avg:.4f} | "
                    f"lr {lr:.2e} | "
                    f"τ {momentum:.4f}"
                )

            # Smoke-test cutoff
            if cfg.max_steps and global_step >= cfg.max_steps:
                print(f"\n  Reached max_steps={cfg.max_steps}, stopping.")
                save_checkpoint(model, optimizer, scheduler, scaler,
                                epoch, global_step, cfg, checkpoint_dir)
                return

        # ── End of epoch ──────────────────────────────────────────────────
        epoch_time = time.time() - t0
        avg_loss   = epoch_loss / max(epoch_steps, 1)
        print(
            f"\n  Epoch {epoch + 1}/{cfg.max_epochs} done in {epoch_time:.1f}s | "
            f"avg loss {avg_loss:.4f}"
        )

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss  = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                boards, actions = batch
                boards  = boards.to(device, dtype=torch.float32)
                actions = actions.to(device)

                with torch.amp.autocast("cuda", enabled=cfg.mixed_precision):
                    predicted, targets = model(boards, actions, val_ctx_idx, val_tgt_idx)
                    loss = ActionConditionedChessJEPA.compute_loss(predicted, targets)

                val_loss  += loss.item()
                val_steps += 1

        if val_steps > 0:
            print(f"  Val loss       : {val_loss / val_steps:.4f}")

        # ── Checkpoint ────────────────────────────────────────────────────
        if (epoch + 1) % cfg.save_every_epochs == 0:
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch + 1, global_step, cfg, checkpoint_dir)

    print("\n  Training complete!")
    save_checkpoint(model, optimizer, scheduler, scaler,
                    cfg.max_epochs, global_step, cfg, checkpoint_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Chess AC-JEPA")
    parser.add_argument("--zarr_path",       default=argparse.SUPPRESS)
    parser.add_argument("--batch_size",      type=int,   default=argparse.SUPPRESS)
    parser.add_argument("--learning_rate",   type=float, default=argparse.SUPPRESS)
    parser.add_argument("--max_epochs",      type=int,   default=argparse.SUPPRESS)
    parser.add_argument("--max_steps",       type=int,   default=argparse.SUPPRESS,
                        help="Stop after N steps (smoke test)")
    parser.add_argument("--device",          default=argparse.SUPPRESS)
    parser.add_argument("--num_workers",     type=int,   default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint_dir",  default="checkpoints_ac",
                        help="Directory to save AC checkpoints (default: checkpoints_ac)")
    parser.add_argument("--resume_from",     default=argparse.SUPPRESS,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    # Remove checkpoint_dir from the dict before passing to JEPAConfig
    # (JEPAConfig has its own checkpoint_dir field but we override it externally)
    args_dict = vars(args)
    args_dict.pop("checkpoint_dir", None)

    cfg = JEPAConfig(**args_dict)
    train(cfg, checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
    main()
