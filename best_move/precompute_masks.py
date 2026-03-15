"""Precompute legal-move masks for a best-move dataset.

Saves masks as a separate sidecar file (same path + '.masks') instead of
re-saving the entire dataset, avoiding the need for 2x disk space.

Usage:
    python best_move/precompute_masks.py \
        --input data/elite_dataset.pt

Output: data/elite_dataset.pt.masks  (just the BoolTensor, no boards copy)

The training script loads both files automatically when it finds the sidecar.
"""

import argparse
import os
import sys

import chess
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def tensor_to_board(t) -> chess.Board:
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


def precompute_masks(input_path: str):
    sidecar_path = input_path + ".masks"

    print(f"Loading {input_path}...")
    data = torch.load(input_path, map_location="cpu", weights_only=False)

    if "boards" not in data:
        raise KeyError("Expected dataset dict to contain 'boards' key")

    boards = data["boards"]  # (N, 17, 8, 8) or (N, T, 17, 8, 8)
    N = boards.shape[0]
    current_boards = boards[:, -1] if boards.ndim == 5 else boards  # (N, 17, 8, 8)
    del data  # free RAM — we only need the boards

    masks = torch.zeros((N, 4096), dtype=torch.bool)

    print(f"Precomputing {N:,} legal-move masks...")
    for i in tqdm(range(N)):
        board = tensor_to_board(current_boards[i])
        for move in board.legal_moves:
            masks[i, move.from_square * 64 + move.to_square] = True

    torch.save(masks, sidecar_path)
    print(f"Saved masks → {sidecar_path}  ({masks.shape}, {masks.nbytes / 1e6:.0f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to dataset .pt file")
    args = parser.parse_args()
    precompute_masks(args.input)


if __name__ == "__main__":
    main()
