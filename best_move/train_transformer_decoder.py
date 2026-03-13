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
    value_loss_weight: float = 1.0,
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
    # Keep boards as uint8 to save VRAM (~676 MB vs 2.7 GB float32); cast in the loop
    boards = data["boards"].to(device)               # (N, 17, 8, 8) uint8
    move_indices = data["move_indices"].to(device)   # (N,) int64

    use_precomputed_masks = "legal_masks" in data
    if use_precomputed_masks:
        legal_masks = data["legal_masks"].to(device)  # (N, 4096) bool
        print(f"Using precomputed legal masks: {legal_masks.shape}")
    else:
        legal_masks = None
        print("No precomputed legal masks found, will compute on-the-fly")

    use_evals = "evals" in data
    if use_evals:
        evals = data["evals"].to(device)  # (N,) float32
        print(f"Using eval targets: {evals.shape}  mean={evals.mean():.3f}  std={evals.std():.3f}")
    else:
        evals = None
        print("No eval targets found — value head will not be trained")

    # Build dataset with all available tensors
    tensor_args = [boards, move_indices]
    if use_precomputed_masks:
        tensor_args.append(legal_masks)
    if use_evals:
        tensor_args.append(evals)
    dataset = TensorDataset(*tensor_args)

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    # num_workers=0: data is already on GPU — forking CUDA contexts would crash
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

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

    policy_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    value_criterion  = nn.MSELoss()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=0.05)

    print("-" * 60)
    print(f"Training on {train_size} samples. Validating on {val_size} samples.")
    print(f"Label smoothing: {label_smoothing}  |  Grad clip: {grad_clip}  |  Value weight: {value_loss_weight}")
    print("-" * 60)

    def unpack_batch(batch):
        """Unpack batch tuple into (boards, moves, masks, eval_targets)."""
        it = iter(batch)
        b_boards = next(it)
        b_moves  = next(it)
        b_masks  = next(it) if use_precomputed_masks else None
        b_evals  = next(it) if use_evals else None
        return b_boards, b_moves, b_masks, b_evals

    best_val_loss = float("inf")

    for epoch in range(epochs):
        decoder.train()
        train_loss = 0.0
        train_policy_loss = 0.0
        train_value_loss  = 0.0
        train_correct = 0

        t0 = time.time()
        for batch in train_loader:
            batch_boards, batch_moves, batch_masks, batch_evals = unpack_batch(batch)

            b = batch_boards.unsqueeze(1).float()  # (B, 1, 17, 8, 8)
            targets = batch_moves

            optimizer.zero_grad()

            with torch.no_grad():
                latents = encoder(b)  # (B, 1, P, D)

            logits, value = decoder(latents)  # (B, 4096), (B,)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"WARNING: Raw logits contain nan/inf at epoch {epoch+1}")
                continue

            # Apply legal move masking
            if use_precomputed_masks:
                legal_mask = batch_masks
            else:
                legal_mask = create_legal_move_mask(batch_boards).to(device)
            masked_logits = logits.clone()
            masked_logits[~legal_mask] = -1e9
            target_logits = masked_logits[torch.arange(targets.size(0)), targets]
            if (target_logits < -50).any():
                print(f"🚨 ERROR: Masking the ground truth move! Check board orientation.")

            p_loss = policy_criterion(masked_logits, targets)

            if use_evals:
                v_loss = value_criterion(value, batch_evals)
                loss = p_loss + value_loss_weight * v_loss
            else:
                v_loss = torch.tensor(0.0)
                loss = p_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            optimizer.step()

            B = batch_boards.size(0)
            train_loss        += loss.item() * B
            train_policy_loss += p_loss.item() * B
            train_value_loss  += v_loss.item() * B
            train_correct     += (masked_logits.argmax(dim=1) == targets).sum().item()

        train_loss        /= train_size
        train_policy_loss /= train_size
        train_value_loss  /= train_size
        train_acc = train_correct / train_size

        decoder.eval()
        val_loss        = 0.0
        val_policy_loss = 0.0
        val_value_loss  = 0.0
        val_correct = 0
        logit_stats_batch = None

        with torch.no_grad():
            for batch in val_loader:
                batch_boards, batch_moves, batch_masks, batch_evals = unpack_batch(batch)

                b = batch_boards.unsqueeze(1).float()
                targets = batch_moves

                latents = encoder(b)
                logits, value = decoder(latents)

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"WARNING: Val logits contain nan/inf at epoch {epoch+1}")
                    val_loss += float('inf') * batch_boards.size(0)
                    continue

                if use_precomputed_masks:
                    legal_mask = batch_masks
                else:
                    legal_mask = create_legal_move_mask(batch_boards).to(device)
                masked_logits = logits.clone()
                masked_logits[~legal_mask] = -1e9

                p_loss = policy_criterion(masked_logits, targets)
                if use_evals:
                    v_loss = value_criterion(value, batch_evals)
                    loss = p_loss + value_loss_weight * v_loss
                else:
                    v_loss = torch.tensor(0.0)
                    loss = p_loss

                B = batch_boards.size(0)
                val_loss        += loss.item() * B
                val_policy_loss += p_loss.item() * B
                val_value_loss  += v_loss.item() * B
                val_correct     += (masked_logits.argmax(dim=1) == targets).sum().item()

                if logit_stats_batch is None:
                    logit_stats_batch = masked_logits.float()

        val_loss        /= val_size
        val_policy_loss /= val_size
        val_value_loss  /= val_size
        val_acc = val_correct / val_size
        elapsed = time.time() - t0

        lg = logit_stats_batch
        finite_mask = lg > -1e8
        logit_info = "Logits all masked"
        if finite_mask.any():
            fl = lg[finite_mask]
            logit_info = f"Logits max={fl.max():.1f} min={fl.min():.1f} std={fl.std():.2f}"

        print(
            f"Epoch {epoch+1:2d}/{epochs:2d} | "
            f"Train Loss: {train_loss:.4f} (pol={train_policy_loss:.4f} val={train_value_loss:.4f}) Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} (pol={val_policy_loss:.4f} val={val_value_loss:.4f}) Acc: {val_acc:.3f} | "
            f"{logit_info} | Time: {elapsed:.2f}s"
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
    parser.add_argument("--ckpt", default="checkpoints_ac/checkpoint_epoch0005.pt", help="Path to AC-JEPA checkpoint")
    parser.add_argument("--dataset", default="best_move/data/best_move_dataset.pt", help="Path to best-move dataset")
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--value_weight", type=float, default=1.0, help="Weight for value loss relative to policy loss")
    parser.add_argument("--out", default="best_move/transformer_decoder_model.pt")
    args = parser.parse_args()

    train_transformer_decoder(
        args.ckpt, args.dataset, args.batch, args.epochs, args.lr,
        args.label_smoothing, args.grad_clip, args.value_weight,
        output_model_path=args.out,
    )