"""
Train the Factored Move Decoder (from-square + to-square heads) on top of
the frozen JEPA Context Encoder.

Loss = CrossEntropy(from_logits, true_from_sq)
     + CrossEntropy(to_logits,   true_to_sq)       ← teacher-forced from_sq
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
from best_move.factored_decoder import FactoredMoveDecoder


def train_decoder(
    ckpt_path: str,
    dataset_path: str,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 5e-4,
    label_smoothing: float = 0.1,
    value_loss_weight: float = 1.0,
    grad_clip: float = 1.0,
    from_loss_weight: float = 2.0,
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    output_model_path: str = "best_move/factored_decoder_model.pt",
):
    print(f"Loading JEPA checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: JEPAConfig = checkpoint["config"]

    print("Loading Context Encoder (frozen)...")
    jepa = ChessJEPA(
        encoder_kwargs=cfg.encoder_kwargs,
        predictor_kwargs=cfg.predictor_kwargs,
    ).to(device)
    jepa.load_state_dict(checkpoint["model"])

    encoder = jepa.context_encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    print(f"Loading dataset: {dataset_path}")
    data        = torch.load(dataset_path, map_location="cpu", weights_only=False)
    boards      = data["boards"]        # (N, 17, 8, 8)
    move_flat   = data["move_indices"]  # (N,)  flat index = from_sq*64 + to_sq
    value_labels = data.get("evals", None)

    # Split flat index into two separate targets
    from_sq = (move_flat // 64).long()  # (N,)
    to_sq   = (move_flat %  64).long()  # (N,)

    if value_labels is not None:
        value_labels = value_labels.float().view(-1)
        if value_labels.abs().max() > 20:
            print("Scaling value labels from centipawns to pawn units")
            value_labels /= 100.0
        dataset    = TensorDataset(boards, from_sq, to_sq, value_labels)
    else:
        dataset    = TensorDataset(boards, from_sq, to_sq)
    total      = len(dataset)
    train_size = int(0.8 * total)
    val_size   = total - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    embed_dim = cfg.encoder_kwargs.get("embed_dim", 256)
    print(f"Initializing FactoredMoveDecoder on {device} (embed_dim={embed_dim})...")
    decoder = FactoredMoveDecoder(in_features=embed_dim, hidden=512, num_hidden=2).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    value_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)

    print("-" * 60)
    print(f"Train: {train_size:,}  |  Val: {val_size:,}")
    print(f"Label smoothing: {label_smoothing}  |  Grad clip: {grad_clip}  |  From-loss weight: {from_loss_weight}")
    print("-" * 60)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        decoder.train()
        train_loss = train_from_loss = train_to_loss = 0.0
        train_from_acc = train_to_acc = train_move_acc = 0.0
        train_value_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            if value_labels is not None:
                batch_boards, batch_from, batch_to, batch_vals = batch
                val_targets = batch_vals.to(device)
            else:
                batch_boards, batch_from, batch_to = batch
                val_targets = None

            b         = batch_boards.unsqueeze(1).to(device)  # (B, 1, 17, 8, 8)
            tgt_from  = batch_from.to(device)                 # (B,)
            tgt_to    = batch_to.to(device)                   # (B,)

            optimizer.zero_grad()

            with torch.no_grad():
                latents = encoder(b)                          # (B, 1, embed_dim)

            # Teacher-forced: pass ground-truth from_sq to to-head
            from_logits, to_logits, pred_value = decoder(latents, from_sq=tgt_from)

            from_loss = criterion(from_logits, tgt_from)
            to_loss   = criterion(to_logits,   tgt_to)
            loss      = from_loss_weight * from_loss + to_loss
            if val_targets is not None:
                value_loss = value_criterion(pred_value, val_targets)
                loss = loss + value_loss_weight * value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            optimizer.step()

            B = batch_boards.size(0)
            train_loss      += loss.item()      * B
            train_from_loss += from_loss.item() * B
            train_to_loss   += to_loss.item()   * B
            train_from_acc  += (from_logits.argmax(1) == tgt_from).sum().item()
            train_to_acc    += (to_logits.argmax(1)   == tgt_to  ).sum().item()
            # Joint accuracy: both squares correct
            train_move_acc += (
                (from_logits.argmax(1) == tgt_from) &
                (to_logits.argmax(1)   == tgt_to)
            ).sum().item()
            if val_targets is not None:
                # FIX: Just add the item. We will divide by the number of batches at the end.
                # This keeps the scale human-readable (e.g., 2.5 pawns).
                train_value_loss += value_loss.item()

        train_loss      /= train_size
        train_from_loss /= train_size
        train_to_loss   /= train_size
        train_from_acc  /= train_size
        train_to_acc    /= train_size
        train_move_acc  /= train_size
        if value_labels is not None:
            train_value_loss /= train_size

        # ── validation ────────────────────────────────────────────────────────
        decoder.eval()
        val_loss = val_from_loss = val_to_loss = 0.0
        val_from_acc = val_to_acc = val_move_acc = 0.0
        val_value_loss = 0.0
        logit_stats = None

        with torch.no_grad():
            for batch in val_loader:
                if value_labels is not None:
                    batch_boards, batch_from, batch_to, batch_vals = batch
                    val_targets = batch_vals.to(device)
                else:
                    batch_boards, batch_from, batch_to = batch
                    val_targets = None

                b        = batch_boards.unsqueeze(1).to(device)
                tgt_from = batch_from.to(device)
                tgt_to   = batch_to.to(device)

                latents = encoder(b)
                from_logits, to_logits, pred_value = decoder(latents, from_sq=tgt_from)

                from_loss = criterion(from_logits, tgt_from)
                to_loss   = criterion(to_logits,   tgt_to)
                loss      = from_loss_weight * from_loss + to_loss
                if val_targets is not None:
                    value_loss = value_criterion(pred_value, val_targets)
                    loss = loss + value_loss

                B = batch_boards.size(0)
                val_loss      += loss.item()      * B
                val_from_loss += from_loss.item() * B
                val_to_loss   += to_loss.item()   * B
                val_from_acc  += (from_logits.argmax(1) == tgt_from).sum().item()
                val_to_acc    += (to_logits.argmax(1)   == tgt_to  ).sum().item()
                val_move_acc  += (
                    (from_logits.argmax(1) == tgt_from) &
                    (to_logits.argmax(1)   == tgt_to)
                ).sum().item()
                if val_targets is not None:
                    val_value_loss += value_loss.item() * B

                if logit_stats is None:
                    logit_stats = (from_logits.float(), to_logits.float())

        val_loss      /= val_size
        val_from_loss /= val_size
        val_to_loss   /= val_size
        val_from_acc  /= val_size
        val_to_acc    /= val_size
        val_move_acc  /= val_size
        if value_labels is not None:
            val_value_loss /= val_size
        elapsed        = time.time() - t0

        fl, tl = logit_stats
        if value_labels is not None:
            print(
                f"Epoch {epoch+1:2d}/{epochs} | "
                f"Loss {train_loss:.3f} (val:{train_value_loss:.3f})/{val_loss:.3f} (val:{val_value_loss:.3f}) | "
                f"From {train_from_loss:.3f}/{val_from_loss:.3f} ({train_from_acc:.3f}/{val_from_acc:.3f}) | "
                f"To {train_to_loss:.3f}/{val_to_loss:.3f} ({train_to_acc:.3f}/{val_to_acc:.3f}) | "
                f"Move {train_move_acc:.3f}/{val_move_acc:.3f} | "
                f"Logit std from={fl.std():.2f} to={tl.std():.2f} | "
                f"{elapsed:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch+1:2d}/{epochs} | "
                f"Loss {train_loss:.3f}/{val_loss:.3f} | "
                f"From {train_from_loss:.3f}/{val_from_loss:.3f} ({train_from_acc:.3f}/{val_from_acc:.3f}) | "
                f"To {train_to_loss:.3f}/{val_to_loss:.3f} ({train_to_acc:.3f}/{val_to_acc:.3f}) | "
                f"Move {train_move_acc:.3f}/{val_move_acc:.3f} | "
                f"Logit std from={fl.std():.2f} to={tl.std():.2f} | "
                f"{elapsed:.1f}s"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)
            ckpt = {
                "decoder":      decoder.state_dict(),
                "epoch":        epoch,
                "val_loss":     val_loss,
                "val_move_acc": val_move_acc,
            }
            if value_labels is not None:
                ckpt["val_value_loss"] = val_value_loss
            torch.save(ckpt, output_model_path)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Saved → {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",             default="checkpoints/checkpoint_epoch0060.pt")
    parser.add_argument("--dataset",          default="best_move/best_move_dataset.pt")
    parser.add_argument("--batch",            type=int,   default=64)
    parser.add_argument("--epochs",           type=int,   default=20)
    parser.add_argument("--lr",               type=float, default=5e-4)
    parser.add_argument("--label_smoothing",  type=float, default=0.1)
    parser.add_argument("--value_loss_weight", type=float, default=1.0,
                        help="Multiplier on the MSE value loss (default 1.0)")
    parser.add_argument("--grad_clip",        type=float, default=1.0)
    parser.add_argument("--from_loss_weight", type=float, default=2.0,
                        help="Weight on from-square loss relative to to-square loss (default 2.0)")
    parser.add_argument("--out",              default="best_move/factored_decoder_model.pt")
    args = parser.parse_args()

    train_decoder(
        args.ckpt, args.dataset, args.batch, args.epochs, args.lr,
        args.label_smoothing, args.value_loss_weight, args.grad_clip,
        args.from_loss_weight,
        output_model_path=args.out,
    )
