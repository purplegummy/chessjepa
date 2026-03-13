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
from model.acjepa import ActionConditionedChessJEPA
from util.config import JEPAConfig
from util.preprocess_pgn import board_to_tensor
from util.visualize_embeddings import tensor_to_board
from best_move.decoder import BestMoveDecoder


def create_legal_move_mask(board_tensors: torch.Tensor) -> torch.Tensor:
    """
    Create a mask for legal moves from board tensors.
    
    Args:
        board_tensors: (B, 17, 8, 8) tensor of board positions
        
    Returns:
        mask: (B, 4096) boolean tensor where True = legal move
    """
    batch_size = board_tensors.shape[0]
    mask = torch.zeros(batch_size, 4096, dtype=torch.bool)
    
    for b in range(batch_size):
        board_tensor = board_tensors[b]  # (17, 8, 8)
        board = tensor_to_board(board_tensor)
        
        for move in board.legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square
            move_idx = from_sq * 64 + to_sq
            mask[b, move_idx] = True
    
    return mask


def train_decoder(
    ckpt_path: str,
    dataset_path: str,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    label_smoothing: float = 0.2,
    value_loss_weight: float = 1.0,
    grad_clip: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    output_model_path: str = "best_move/decoder_model.pt",
):
    print(f"Loading JEPA checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: JEPAConfig = checkpoint["config"]

    print("Loading Context Encoder...")
    # Try AC model first; fall back to base JEPA
    try:
        jepa = ActionConditionedChessJEPA(
            encoder_kwargs=cfg.encoder_kwargs,
            predictor_kwargs=cfg.predictor_kwargs,
        ).to(device)
        jepa.load_state_dict(checkpoint["model"])
        print("  Using ActionConditionedChessJEPA")
    except Exception:
        jepa = ChessJEPA(
            encoder_kwargs=cfg.encoder_kwargs,
            predictor_kwargs=cfg.predictor_kwargs,
        ).to(device)
        jepa.load_state_dict(checkpoint["model"])
        print("  Using ChessJEPA (base)")

    encoder = jepa.context_encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    print(f"Loading Best Move Dataset: {dataset_path}")
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    boards = data["boards"]        # (N, 17, 8, 8)
    move_indices = data["move_indices"]  # (N,) long
    value_labels = data.get("evals", None)  # optional float tensor
    
    # Check for precomputed legal masks
    if "legal_masks" in data:
        legal_masks = data["legal_masks"]  # (N, 4096) bool tensor
        print(f"Using precomputed legal masks: {legal_masks.shape}")
        use_precomputed_masks = True
    else:
        legal_masks = None
        print("No precomputed legal masks found, will compute on-the-fly")
        use_precomputed_masks = False

    if value_labels is not None:
        # make sure we have float and a 1‑D vector
        value_labels = value_labels.float().view(-1)
        # if the magnitudes look like centipawns (>20) convert to pawns
        if value_labels.abs().max() > 20:
            print("Scaling value labels from centipawns to pawn units")
            value_labels /= 100.0
        if use_precomputed_masks:
            dataset = TensorDataset(boards, move_indices, value_labels, legal_masks)
        else:
            dataset = TensorDataset(boards, move_indices, value_labels)
    else:
        if use_precomputed_masks:
            dataset = TensorDataset(boards, move_indices, legal_masks)
        else:
            dataset = TensorDataset(boards, move_indices)

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    embed_dim   = cfg.encoder_kwargs.get("embed_dim", 256)
    num_patches = (cfg.board_size // cfg.patch_size) ** 2   # 16
    in_features = embed_dim * num_patches                    # 4096
    print(f"Initializing BestMoveDecoder on {device} (in_features={in_features} = {embed_dim}D × {num_patches} patches)...")
    decoder = BestMoveDecoder(in_features=in_features, hidden_features=512, num_layers=3, dropout=0.3).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    value_criterion = nn.MSELoss()  # we'll use this only if value labels provided
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
        train_value_loss = 0.0

        t0 = time.time()
        for batch in train_loader:
            # unpack according to dataset structure
            if value_labels is not None and use_precomputed_masks:
                batch_boards, batch_moves, batch_vals, batch_masks = batch
                batch_masks = batch_masks.to(device)
            elif value_labels is not None:
                batch_boards, batch_moves, batch_vals = batch
                batch_masks = None
            elif use_precomputed_masks:
                batch_boards, batch_moves, batch_masks = batch
                batch_masks = batch_masks.to(device)
            else:
                batch_boards, batch_moves = batch
                batch_masks = None
            
            val_targets = batch_vals.to(device) if value_labels is not None else None

            b = batch_boards.unsqueeze(1).to(device)   # (B, 1, 17, 8, 8)
            targets = batch_moves.to(device)            # (B,)

            optimizer.zero_grad()

            with torch.no_grad():
                latents = encoder(b)  # (B, 1, P, D) — decoder flattens patches internally

            logits, pred_value = decoder(latents)      # new dual-head
            
            # Apply legal move masking
            if use_precomputed_masks:
                legal_mask = batch_masks  # Already on device
            else:
                legal_mask = create_legal_move_mask(batch_boards).to(device)  # (B, 4096)
            masked_logits = logits.clone()
            masked_logits[~legal_mask] = -1e9  # Use large negative instead of -inf
            
            policy_loss = criterion(masked_logits, targets)
            if val_targets is not None:
                value_loss = value_criterion(pred_value, val_targets)
                loss = policy_loss + value_loss_weight * value_loss
            else:
                loss = policy_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            optimizer.step()

            B = batch_boards.size(0)
            train_loss += loss.item() * B
            train_correct += (masked_logits.argmax(dim=1) == targets).sum().item()
            if val_targets is not None:
                train_value_loss += value_loss.item() * B

        train_loss /= train_size
        train_acc = train_correct / train_size

        decoder.eval()
        val_loss = 0.0
        val_correct = 0
        val_value_loss = 0.0
        logit_stats_batch = None   # capture one batch for diagnostics
        with torch.no_grad():
            for batch in val_loader:
                # unpack according to dataset structure
                if value_labels is not None and use_precomputed_masks:
                    batch_boards, batch_moves, batch_vals, batch_masks = batch
                    batch_masks = batch_masks.to(device)
                elif value_labels is not None:
                    batch_boards, batch_moves, batch_vals = batch
                    batch_masks = None
                elif use_precomputed_masks:
                    batch_boards, batch_moves, batch_masks = batch
                    batch_masks = batch_masks.to(device)
                else:
                    batch_boards, batch_moves = batch
                    batch_masks = None
                
                val_targets = batch_vals.to(device) if value_labels is not None else None

                b = batch_boards.unsqueeze(1).to(device)
                targets = batch_moves.to(device)

                latents = encoder(b)  # (B, 1, P, D) — decoder flattens patches internally
                logits, pred_value = decoder(latents)
                
                # Apply legal move masking
                if use_precomputed_masks:
                    legal_mask = batch_masks  # Already on device
                else:
                    legal_mask = create_legal_move_mask(batch_boards).to(device)  # (B, 4096)
                masked_logits = logits.clone()
                masked_logits[~legal_mask] = -1e9  # Use large negative instead of -inf
                
                policy_loss = criterion(masked_logits, targets)
                if val_targets is not None:
                    value_loss = value_criterion(pred_value, val_targets)
                    loss = policy_loss + value_loss_weight * value_loss
                else:
                    loss = policy_loss

                # nan/inf guard
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf in validation loss, skipping batch")
                    continue

                val_loss += loss.item() * batch_boards.size(0)
                val_correct += (masked_logits.argmax(dim=1) == targets).sum().item()
                if val_targets is not None:
                    val_value_loss += value_loss.item() * batch_boards.size(0)

                if logit_stats_batch is None:
                    logit_stats_batch = masked_logits.float()

        val_loss /= val_size
        val_acc = val_correct / val_size
        if value_labels is not None:
            val_value_loss /= val_size
        elapsed = time.time() - t0

        # Logit diagnostics — high std signals overconfidence collapse
        lg = logit_stats_batch
        if value_labels is not None:
            print(
                f"Epoch {epoch+1:2d}/{epochs:2d} | "
                f"Train Loss: {train_loss:.4f} (val:{train_value_loss:.4f}) Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} (val:{val_value_loss:.4f}) Acc: {val_acc:.3f} | "
                f"Logits max={lg.max():.1f} min={lg.min():.1f} std={lg.std():.2f} | "
                f"Time: {elapsed:.2f}s"
            )
        else:
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
            ckpt = {
                "decoder": decoder.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            if value_labels is not None:
                ckpt["val_value_loss"] = val_value_loss
            torch.save(ckpt, output_model_path)

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
    parser.add_argument("--value_loss_weight", type=float, default=1.0,
                        help="Multiplier applied to the MSE value loss (default 1.0)")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--out", default="best_move/decoder_model.pt")
    args = parser.parse_args()

    train_decoder(
        args.ckpt, args.dataset, args.batch, args.epochs, args.lr,
        args.label_smoothing, args.value_loss_weight, args.grad_clip,
        output_model_path=args.out,
    )
