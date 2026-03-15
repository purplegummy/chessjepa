"""
Train the Transformer Move Decoder on top of the frozen JEPA Context Encoder.
"""

import argparse
import contextlib
import os
import sys
import time
import chess
import numpy as np
import torch
import torch.nn.functional as F
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
    batch_size = board_tensors.shape[0]
    mask = torch.zeros(batch_size, 4096, dtype=torch.bool)
    for b in range(batch_size):
        board = tensor_to_board(board_tensors[b])
        for move in board.legal_moves:
            mask[b, move.from_square * 64 + move.to_square] = True
    return mask


def legal_cross_entropy(logits: torch.Tensor, legal_mask: torch.Tensor,
                        targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    masked = logits.masked_fill(~legal_mask, float('-inf'))
    log_probs = F.log_softmax(masked, dim=-1)

    if label_smoothing == 0.0:
        return F.nll_loss(log_probs, targets)

    n_legal = legal_mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
    smooth = legal_mask.float() / n_legal * label_smoothing
    smooth.scatter_(1, targets.unsqueeze(1),
                    smooth.gather(1, targets.unsqueeze(1)) + (1.0 - label_smoothing))
    loss_terms = (smooth * log_probs).masked_fill(~legal_mask, 0.0)
    return -loss_terms.sum(dim=-1).mean()


def train_transformer_decoder(
    ckpt_path: str,
    dataset_path: str,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    label_smoothing: float = 0.2,
    grad_clip: float = 1.0,
    warmup_epochs: int = 5,
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

    print(f"Loading dataset: {dataset_path}")
    # Load from numpy memmap if available (instant load), else fall back to .pt
    _data_dir  = os.path.dirname(os.path.abspath(dataset_path))
    boards_npy = os.path.join(_data_dir, "elite_boards.npy")
    moves_npy  = os.path.join(_data_dir, "elite_moves.npy")
    if os.path.exists(boards_npy) and os.path.exists(moves_npy):
        print(f"Using numpy memmap: {boards_npy}")
        boards       = torch.from_numpy(np.load(boards_npy, mmap_mode="r")).to(device)
        move_indices = torch.from_numpy(np.load(moves_npy,  mmap_mode="r")).to(device)
    else:
        data         = torch.load(dataset_path, map_location="cpu", weights_only=False)
        boards       = data["boards"].to(device)
        move_indices = data["move_indices"].to(device)

    if boards.ndim == 4:                           # old single-frame format → add T=1 dim
        boards = boards.unsqueeze(1)

    # Check for sidecar masks file first, then fall back to embedded masks
    sidecar_path = dataset_path + ".masks"
    _npz_path = sidecar_path + ".npz"
    if os.path.exists(_npz_path):
        try:
            npz = np.load(_npz_path)
            N_masks, W = int(npz["shape"][0]), int(npz["shape"][1])
            legal_masks = torch.from_numpy(
                np.unpackbits(npz["masks"], axis=1)[:, :W].astype(bool)
            ).to(device)
            print(f"Using sidecar legal masks: {legal_masks.shape}")
            use_precomputed_masks = True
        except Exception as e:
            print(f"WARNING: sidecar masks file is corrupt ({e}), falling back to on-the-fly computation.")
            print(f"  Delete {_npz_path} and re-run precompute_masks.py when disk space is available.")
            legal_masks = None
            use_precomputed_masks = False
    elif os.path.exists(sidecar_path):
        try:
            legal_masks = torch.load(sidecar_path, map_location=device, weights_only=True)
            print(f"Using sidecar legal masks (legacy): {legal_masks.shape}")
            use_precomputed_masks = True
        except Exception as e:
            print(f"WARNING: sidecar masks file is corrupt ({e}), falling back to on-the-fly computation.")
            legal_masks = None
            use_precomputed_masks = False
    elif "legal_masks" in data:
        legal_masks = data["legal_masks"].to(device)
        print(f"Using embedded legal masks: {legal_masks.shape}")
        use_precomputed_masks = True
    else:
        legal_masks = None
        use_precomputed_masks = False
        print("No precomputed legal masks — will compute on-the-fly")

    if use_precomputed_masks:
        dataset = TensorDataset(boards, move_indices, legal_masks)
    else:
        dataset = TensorDataset(boards, move_indices)

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    embed_dim   = cfg.encoder_kwargs.get("embed_dim", 256)
    num_patches = (cfg.board_size // cfg.patch_size) ** 2
    print(f"Initializing TransformerMoveDecoder (embed_dim={embed_dim}, num_patches={num_patches})...")
    decoder = TransformerMoveDecoder(
        embed_dim=embed_dim,
        num_patches=num_patches,
        num_heads=8,
        ff_dim=512,
        num_layers=2,
        mlp_hidden=256,
        dropout=0.1,
        head_dropout=0.3,
        latent_dropout=0.1,
    ).to(device)

    # AMP: bfloat16 on CUDA (Ampere+, no scaler), float16 on older CUDA (needs scaler), off elsewhere
    if device == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp_ctx   = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_dtype = None
        amp_ctx   = contextlib.nullcontext
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    print(f"  AMP: {'enabled (' + str(amp_dtype) + ')' if amp_dtype else 'disabled'}")

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=0.01)
    warmup_scheduler  = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=lr / 10
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    print(f"-" * 60)
    print(f"Train: {train_size}  Val: {val_size}  |  label_smoothing={label_smoothing}  grad_clip={grad_clip}  warmup={warmup_epochs}")
    print("-" * 60)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        decoder.train()
        train_loss = train_correct = 0
        t0 = time.time()

        for batch in train_loader:
            if use_precomputed_masks:
                batch_boards, batch_moves, batch_masks = batch
            else:
                batch_boards, batch_moves = batch
                batch_masks = None

            b = batch_boards.float()
            targets = batch_moves

            optimizer.zero_grad()
            with torch.no_grad(), amp_ctx():
                latents = encoder(b)

            with amp_ctx():
                logits = decoder(latents)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"WARNING: logits contain nan/inf at epoch {epoch+1}")
                continue

            legal_mask = batch_masks if use_precomputed_masks else create_legal_move_mask(batch_boards).to(device)

            has_legal = legal_mask.any(dim=-1)
            if not has_legal.all():
                keep = has_legal
                logits, legal_mask, targets = logits[keep], legal_mask[keep], targets[keep]
                if targets.numel() == 0:
                    continue

            if not legal_mask[torch.arange(targets.size(0)), targets].all():
                print("WARNING: ground truth move is masked — check board orientation")

            with amp_ctx():
                loss = legal_cross_entropy(logits, legal_mask, targets, label_smoothing)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
                optimizer.step()

            masked_logits = logits.masked_fill(~legal_mask, float('-inf'))
            B = batch_boards.size(0)
            train_loss    += loss.item() * B
            train_correct += (masked_logits.argmax(dim=1) == targets).sum().item()

        train_loss /= train_size
        train_acc   = train_correct / train_size

        decoder.eval()
        val_loss = val_correct = 0
        logit_stats_batch = None

        with torch.no_grad():
            for batch in val_loader:
                if use_precomputed_masks:
                    batch_boards, batch_moves, batch_masks = batch
                else:
                    batch_boards, batch_moves = batch
                    batch_masks = None

                b = batch_boards.float()
                targets = batch_moves

                with amp_ctx():
                    latents = encoder(b)
                    logits  = decoder(latents)

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    val_loss += float('inf') * batch_boards.size(0)
                    continue

                legal_mask = batch_masks if use_precomputed_masks else create_legal_move_mask(batch_boards[:, -1] if batch_boards.ndim == 4 else batch_boards).to(device)

                has_legal = legal_mask.any(dim=-1)
                if not has_legal.all():
                    keep = has_legal
                    logits, legal_mask, targets = logits[keep], legal_mask[keep], targets[keep]
                    if targets.numel() == 0:
                        continue

                loss = legal_cross_entropy(logits, legal_mask, targets, label_smoothing)
                masked_logits = logits.masked_fill(~legal_mask, float('-inf'))

                B = batch_boards.size(0)
                val_loss    += loss.item() * B
                val_correct += (masked_logits.argmax(dim=1) == targets).sum().item()

                if logit_stats_batch is None:
                    logit_stats_batch = masked_logits.float()

        val_loss /= val_size
        val_acc   = val_correct / val_size
        elapsed   = time.time() - t0

        lg = logit_stats_batch
        if lg is not None and torch.isfinite(lg).any():
            fl = lg[torch.isfinite(lg)]
            logit_info = f"Logits max={fl.max():.1f} min={fl.min():.1f} std={fl.std():.2f}"
        else:
            logit_info = "Logits all masked"

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(
            f"Epoch {epoch+1:2d}/{epochs:2d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
            f"LR: {current_lr:.2e} | {logit_info} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            torch.save({
                "decoder": decoder.state_dict(),
                "epoch":    epoch,
                "val_loss": val_loss,
                "val_acc":  val_acc,
            }, output_model_path)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}  →  {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",            default="checkpoints_ac/checkpoint_epoch0010.pt")
    parser.add_argument("--dataset",         default="data/best_move_dataset_masks.pt")
    parser.add_argument("--batch",           type=int,   default=2048)
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--grad_clip",       type=float, default=1.0)
    parser.add_argument("--warmup_epochs",   type=int,   default=5)
    parser.add_argument("--out",             default="best_move/transformer_decoder_model.pt")
    args = parser.parse_args()

    train_transformer_decoder(
        args.ckpt, args.dataset, args.batch, args.epochs, args.lr,
        args.label_smoothing, args.grad_clip, args.warmup_epochs,
        output_model_path=args.out,
    )
