"""
Chess V-JEPA — Training Loop

Ties everything together:
  1. Load data from zarr via ChessChunkDataset
  2. Build ChessJEPA model (context encoder + target encoder + predictor)
  3. Generate temporal masks each batch
  4. Forward pass → L2 loss in latent space
  5. Backward pass → update context encoder + predictor (NOT target encoder)
  6. EMA update → slowly sync target encoder to context encoder
  7. Repeat

Usage:
    python train.py
    python train.py --batch_size 128 --learning_rate 3e-4 --max_epochs 50
    python train.py --max_steps 100   # quick smoke test
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn

from util.config import JEPAConfig
from util.dataset import build_dataloaders
from util.masking import TemporalMaskGenerator
from model.jepa import ChessJEPA


def build_optimizer(model: ChessJEPA, cfg: JEPAConfig):
    """
    AdamW optimizer over ONLY the trainable parameters.

    The target encoder is frozen (requires_grad=False), so it's
    automatically excluded.  Only the context encoder + predictor
    receive gradient updates.
    """
    # Separate weight-decay and no-decay parameter groups
    # (biases and LayerNorm weights should not be decayed)
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(param_groups, lr=cfg.learning_rate)


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    Linear warmup → cosine decay learning rate schedule.

    warmup:  LR ramps from 0 → lr over warmup_steps
    decay:   LR follows cosine curve from lr → 0 over remaining steps
    """

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup: 0 → 1
            return step / max(warmup_steps, 1)
        else:
            # Cosine decay: 1 → 0
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, cfg):
    """Save model + optimizer + scheduler state for resuming."""
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_epoch{epoch:04d}.pt")
    torch.save({
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": cfg,
    }, path)
    print(f"  💾 Checkpoint saved: {path}")

    # Retain only the most recent checkpoints
    if cfg.max_checkpoints_to_keep > 0:
        import glob
        existing_checkpoints = sorted(glob.glob(os.path.join(cfg.checkpoint_dir, "checkpoint_epoch*.pt")))
        if len(existing_checkpoints) > cfg.max_checkpoints_to_keep:
            for old_checkpoint in existing_checkpoints[:-cfg.max_checkpoints_to_keep]:
                os.remove(old_checkpoint)
                print(f"  🗑️ Deleted old checkpoint: {old_checkpoint}")


def train(cfg: JEPAConfig):
    """Main training function."""

    print("=" * 60)
    print("  Chess V-JEPA Training")
    print("=" * 60)

    # ── 1. Device setup ──────────────────────────────────────────────────
    device = torch.device(cfg.device)
    print(f"  Device         : {device}")

    # ── 2. Data ──────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        cfg.zarr_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_fraction=cfg.val_fraction,
    )
    print(f"  Train samples  : {len(train_loader.dataset):,}")
    print(f"  Val samples    : {len(val_loader.dataset):,}")
    print(f"  Batch size     : {cfg.batch_size}")

    # ── 3. Model ─────────────────────────────────────────────────────────
    model = ChessJEPA(
        encoder_kwargs=cfg.encoder_kwargs,
        predictor_kwargs=cfg.predictor_kwargs,
        ema_momentum_start=cfg.ema_momentum_start,
        ema_momentum_end=cfg.ema_momentum_end,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params   : {total_params:,}")
    print(f"  Trainable      : {trainable:,}")
    print(f"  Target encoder : {total_params - trainable:,} (frozen, EMA)")

    # ── 4. Optimizer + Scheduler ─────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.max_epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs

    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    print(f"  Steps/epoch    : {steps_per_epoch:,}")
    print(f"  Total steps    : {total_steps:,}")
    print(f"  Warmup steps   : {warmup_steps:,}")

    # ── 5. Mixed precision scaler ────────────────────────────────────────
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.mixed_precision)

    # ── 6. Mask generator ────────────────────────────────────────────────
    mask_gen = TemporalMaskGenerator(
        seq_len=cfg.seq_len,
        target_ratio=cfg.target_ratio,
        mode=cfg.mask_mode,
        min_context=cfg.min_context,
    )
    print(f"  Masking        : {mask_gen}")
    print("=" * 60)

    # ── Resume from checkpoint ───────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if cfg.resume_from:
        print(f"  Resuming from checkpoint: {cfg.resume_from}")
        checkpoint = torch.load(cfg.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]
        print(f"  Resumed at epoch {start_epoch}, step {global_step}")

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch in train_loader:
            boards = batch.to(device)  # (B, 16, 17, 8, 8)

            # Generate a fresh mask for this batch
            ctx_idx, tgt_idx = mask_gen()

            # ── Forward pass (with AMP) ──────────────────────────────────
            with torch.amp.autocast('cuda', enabled=cfg.mixed_precision):
                predicted, targets = model(boards, ctx_idx, tgt_idx)
                loss = ChessJEPA.compute_loss(predicted, targets)

            # ── Backward pass ────────────────────────────────────────────
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # ── EMA update (after optimizer step) ────────────────────────
            momentum = model.get_ema_momentum(global_step, total_steps)
            model.update_target_encoder(momentum)

            # ── Logging ──────────────────────────────────────────────────
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % cfg.log_every == 0:
                avg = epoch_loss / epoch_steps
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  step {global_step:>6d} | "
                    f"loss {loss.item():.4f} | "
                    f"avg {avg:.4f} | "
                    f"lr {lr:.2e} | "
                    f"τ {momentum:.4f}"
                )

            # Early stop for smoke tests
            if cfg.max_steps and global_step >= cfg.max_steps:
                print(f"\n  Reached max_steps={cfg.max_steps}, stopping.")
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, cfg)
                return

        # ── End of epoch ─────────────────────────────────────────────────
        epoch_time = time.time() - t0
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(
            f"\n  Epoch {epoch+1}/{cfg.max_epochs} done in {epoch_time:.1f}s | "
            f"avg loss {avg_loss:.4f}"
        )

        # ── Validation ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                boards = batch.to(device)
                ctx_idx, tgt_idx = mask_gen()

                with torch.amp.autocast('cuda', enabled=cfg.mixed_precision):
                    predicted, targets = model(boards, ctx_idx, tgt_idx)
                    loss = ChessJEPA.compute_loss(predicted, targets)

                val_loss += loss.item()
                val_steps += 1

        if val_steps > 0:
            print(f"  Val loss       : {val_loss / val_steps:.4f}")

        # ── Checkpoint ───────────────────────────────────────────────────
        if (epoch + 1) % cfg.save_every_epochs == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, global_step, cfg)

    print("\n  Training complete!")
    save_checkpoint(model, optimizer, scheduler, scaler, cfg.max_epochs, global_step, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Chess V-JEPA")
    parser.add_argument("--zarr_path", default="chess_chunks.zarr")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1.5e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after N steps (for smoke testing)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--resume_from", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = JEPAConfig(
        zarr_path=args.zarr_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        device=args.device,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
    )

    train(cfg)


if __name__ == "__main__":
    main()
