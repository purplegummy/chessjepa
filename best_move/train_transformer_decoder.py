"""
Train the Transformer Move Decoder on top of the frozen JEPA Context Encoder.
"""

import argparse
import os
import sys
import time
import chess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.acjepa import ActionConditionedChessJEPA
from util.config import JEPAConfig
from best_move.transformer_decoder import TransformerMoveDecoder

_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def tensor_to_board(t: torch.Tensor) -> chess.Board:
    """Convert a (17, 8, 8) board tensor back to a chess.Board (current player = white)."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    board = chess.Board(None)
    for i, piece in enumerate(_PIECES):
        for r, c in zip(*np.where(t[i] == 1)):
            board.set_piece_at(r * 8 + c, chess.Piece(piece, chess.WHITE))
        for r, c in zip(*np.where(t[i + 6] == 1)):
            board.set_piece_at(r * 8 + c, chess.Piece(piece, chess.BLACK))
    board.turn = chess.WHITE
    # Castling rights: ch12=current-KS, ch13=current-QS, ch14=opp-KS, ch15=opp-QS
    # current player = WHITE, opponent = BLACK in the reconstruction
    if t[12].any():
        board.castling_rights |= chess.BB_H1
    if t[13].any():
        board.castling_rights |= chess.BB_A1
    if t[14].any():
        board.castling_rights |= chess.BB_H8
    if t[15].any():
        board.castling_rights |= chess.BB_A8
    ep = np.where(t[16] == 1)
    if len(ep[0]) > 0:
        board.ep_square = int(ep[0][0]) * 8 + int(ep[1][0])
    return board


def create_legal_move_mask(board_tensors: torch.Tensor) -> torch.Tensor:
    """
    Create a mask for legal moves from board tensors.
    
    Args:
        board_tensors: (B, 18, 8, 8) tensor of board positions
        
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


def train_transformer_decoder(
    ckpt_path: str,
    dataset_path: str,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    label_smoothing: float = 0.2,
    grad_clip: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    output_model_path: str = "best_move/transformer_decoder_model.pt",
):
    print(f"Loading JEPA checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: JEPAConfig = checkpoint["config"]

    print("Loading Context Encoder...")
    jepa = ActionConditionedChessJEPA(
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
    boards = data["boards"]        # (N, 18, 8, 8)
    move_indices = data["move_indices"]  # (N,) long
    
    # Check for precomputed legal masks
    if "legal_masks" in data:
        legal_masks = data["legal_masks"]  # (N, 4096) bool tensor
        print(f"Using precomputed legal masks: {legal_masks.shape}")
        use_precomputed_masks = True
    else:
        legal_masks = None
        print("No precomputed legal masks found, will compute on-the-fly")
        use_precomputed_masks = False

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

    embed_dim = cfg.encoder_kwargs.get("embed_dim", 256)
    num_patches = (cfg.board_size // cfg.patch_size) ** 2   # 16
    print(f"Initializing TransformerMoveDecoder on {device} (embed_dim={embed_dim}, num_patches={num_patches})...")
    decoder = TransformerMoveDecoder(
        embed_dim=embed_dim,
        num_patches=num_patches,
        num_heads=8,
        ff_dim=512,
        num_layers=1,
        mlp_hidden=512,
        dropout=0.1
    ).to(device)

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
        for batch in train_loader:
            if use_precomputed_masks:
                batch_boards, batch_moves, batch_masks = batch
                batch_masks = batch_masks.to(device)  # (B, 4096)
            else:
                batch_boards, batch_moves = batch
                batch_masks = None

            b = batch_boards.unsqueeze(1).to(device, dtype=torch.float32)  # (B, 1, 17, 8, 8)
            targets = batch_moves.to(device)            # (B,)

            optimizer.zero_grad()

            with torch.no_grad():
                latents = encoder(b)  # (B, 1, P, D)

            logits = decoder(latents)  # (B, 4096)
            
            # Check for nan/inf in raw logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"WARNING: Raw logits contain nan/inf at epoch {epoch+1}")
                print(f"Logits stats: max={logits.max():.3f}, min={logits.min():.3f}")
                # Skip this batch
                continue
            
            # Apply legal move masking
            if use_precomputed_masks:
                legal_mask = batch_masks  # Already on device
            else:
                legal_mask = create_legal_move_mask(batch_boards).to(device)  # (B, 4096)
            masked_logits = logits.clone()
            masked_logits[~legal_mask] = -1e9  # Use large negative instead of -inf
            target_logits = masked_logits[torch.arange(targets.size(0)), targets]
            if (target_logits < -50).any():
                print(f"🚨 ERROR: Masking the ground truth move! Check board orientation.")
            loss = criterion(masked_logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            optimizer.step()

            B = batch_boards.size(0)
            train_loss += loss.item() * B
            train_correct += (masked_logits.argmax(dim=1) == targets).sum().item()

        train_loss /= train_size
        train_acc = train_correct / train_size

        decoder.eval()
        val_loss = 0.0
        val_correct = 0
        logit_stats_batch = None   # capture one batch for diagnostics

        with torch.no_grad():
            for batch in val_loader:
                # Use index access — batch may be a 3-tuple if precomputed masks exist
                batch_boards, batch_moves = batch[0], batch[1]

                b = batch_boards.unsqueeze(1).to(device, dtype=torch.float32)
                targets = batch_moves.to(device)

                latents = encoder(b)  # (B, 1, P, D)
                logits = decoder(latents)
                
                # Check for nan/inf in raw logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"WARNING: Val logits contain nan/inf at epoch {epoch+1}")
                    val_loss += float('inf') * batch_boards.size(0)
                    val_correct += 0
                    continue
                
                # Apply legal move masking
                legal_mask = create_legal_move_mask(batch_boards).to(device)  # (B, 4096)
                masked_logits = logits.clone()
                masked_logits[~legal_mask] = -1e9  # Use large negative instead of -inf
                
                loss = criterion(masked_logits, targets)

                val_loss += loss.item() * batch_boards.size(0)
                val_correct += (masked_logits.argmax(dim=1) == targets).sum().item()

                if logit_stats_batch is None:
                    logit_stats_batch = masked_logits.float()

        val_loss /= val_size
        val_acc = val_correct / val_size
        elapsed = time.time() - t0

        # Logit diagnostics — high std signals overconfidence collapse
        lg = logit_stats_batch
        # Filter out the masked values (-1e9) for statistics
        finite_mask = lg > -1e8  # Values that are not masked
        if finite_mask.any():
            finite_logits = lg[finite_mask]
            print(
                f"Epoch {epoch+1:2d}/{epochs:2d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
                f"Logits max={finite_logits.max():.1f} min={finite_logits.min():.1f} std={finite_logits.std():.2f} | "
                f"Time: {elapsed:.2f}s"
            )
        else:
            print(
                f"Epoch {epoch+1:2d}/{epochs:2d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
                f"Logits all masked | "
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
            torch.save(ckpt, output_model_path)

    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f}")
    print(f"Best decoder weights saved to {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints_ac/checkpoint_epoch0005.pt", help="Path to AC-JEPA checkpoint")
    parser.add_argument("--dataset", default="best_move/data/best_move_dataset.pt", help="Path to best-move dataset")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--out", default="best_move/transformer_decoder_model.pt")
    args = parser.parse_args()

    train_transformer_decoder(
        args.ckpt, args.dataset, args.batch, args.epochs, args.lr,
        args.label_smoothing, args.grad_clip,
        output_model_path=args.out,
    )