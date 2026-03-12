"""
Train the Best Move Decoder on top of the frozen JEPA Context Encoder.
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.jepa import ChessJEPA
from util.config import JEPAConfig
from best_move.decoder import BestMoveDecoder


def train_decoder(
    ckpt_path: str,
    dataset_path: str,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    label_smoothing: float = 0.2,
    grad_clip: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    output_model_path: str = "best_move/decoder_model.pt",
):
    print(f"Loading JEPA checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: JEPAConfig = checkpoint["config"]

    print("Loading Context Encoder...")
    jepa = ChessJEPA(
        encoder_kwargs=cfg.encoder_kwargs,
        predictor_kwargs=cfg.predictor_kwargs,
    ).to(device)
    jepa.load_state_dict(checkpoint["model"])

    encoder = jepa.context_encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    print(f"Loading Best Move Dataset: {dataset_path}")
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    boards = data["boards"]        # (N, 17, 8, 8)
    move_indices = data["move_indices"]  # (N,) long

    dataset = TensorDataset(boards, move_indices)

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    embed_dim = cfg.encoder_kwargs.get("embed_dim", 256)
    print(f"Initializing BestMoveDecoder on {device} (embed_dim={embed_dim})...")
    decoder = BestMoveDecoder(in_features=embed_dim, hidden_features=512, num_layers=3).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)

    print("-" * 50)
    print(f"Training on {train_size} samples. Validating on {val_size} samples.")
    print(f"Label smoothing: {label_smoothing}  |  Grad clip: {grad_clip}")
    print("-" * 50)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        decoder.train()
        train_loss = 0.0
        train_correct = 0

        t0 = time.time()
        for batch_boards, batch_moves in train_loader:
            b = batch_boards.unsqueeze(1).to(device)   # (B, 1, 17, 8, 8)
            targets = batch_moves.to(device)            # (B,)

            optimizer.zero_grad()

            with torch.no_grad():
                latents = encoder(b)  # (B, 1, embed_dim)

            logits = decoder(latents)  # (B, 4096)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item() * batch_boards.size(0)
            train_correct += (logits.argmax(dim=1) == targets).sum().item()

        train_loss /= train_size
        train_acc = train_correct / train_size

        decoder.eval()
        val_loss = 0.0
        val_correct = 0
        logit_stats_batch = None   # capture one batch for diagnostics
        with torch.no_grad():
            for batch_boards, batch_moves in val_loader:
                b = batch_boards.unsqueeze(1).to(device)
                targets = batch_moves.to(device)

                latents = encoder(b)
                logits = decoder(latents)
                loss = criterion(logits, targets)

                val_loss += loss.item() * batch_boards.size(0)
                val_correct += (logits.argmax(dim=1) == targets).sum().item()

                if logit_stats_batch is None:
                    logit_stats_batch = logits.float()

        val_loss /= val_size
        val_acc = val_correct / val_size
        elapsed = time.time() - t0

        # Logit diagnostics — high std signals overconfidence collapse
        lg = logit_stats_batch
        print(
            f"Epoch {epoch+1:2d}/{epochs:2d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
            f"Logits max={lg.max():.1f} min={lg.min():.1f} std={lg.std():.2f} | "
            f"Time: {elapsed:.2f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            torch.save({
                "decoder": decoder.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, output_model_path)

    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f}")
    print(f"Best decoder weights saved to {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/checkpoint_epoch0060.pt", help="Path to JEPA checkpoint")
    parser.add_argument("--dataset", default="best_move/best_move_dataset.pt", help="Path to best-move dataset")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--out", default="best_move/decoder_model.pt")
    args = parser.parse_args()

    train_decoder(args.ckpt, args.dataset, args.batch, args.epochs, args.lr,
                  args.label_smoothing, args.grad_clip, output_model_path=args.out)
