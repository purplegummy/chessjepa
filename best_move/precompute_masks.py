"""Precompute legal-move masks for a best-move dataset.

This script reads a dataset saved by best_move/generate_dataset.py and adds
`legal_masks` containing a (N, 4096) boolean mask where each row marks legal
moves for the corresponding board.

Usage:
    python best_move/precompute_masks.py \
        --input best_move/data/best_move_dataset.pt \
        --output best_move/data/best_move_dataset_masked.pt

The output file is the same dictionary as the input plus a new key:
    "legal_masks": torch.BoolTensor(N, 4096)

This lets training scripts quickly apply legal-move masking without recomputing
legal move sets every batch.
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


def precompute_masks(input_path: str, output_path: str):
    print(f"Loading {input_path}...")
    data = torch.load(input_path, map_location="cpu", weights_only=False)

    if "boards" not in data:
        raise KeyError("Expected dataset dict to contain 'boards' key")

    boards = data["boards"]  # (N, 17, 8, 8) or (N, T, 17, 8, 8)
    N = boards.shape[0]
    # If sequence dataset, extract the last (current) frame for mask computation
    if boards.ndim == 4:
        current_boards = boards          # (N, 17, 8, 8)
    else:
        current_boards = boards[:, -1]   # (N, 17, 8, 8)

    masks = torch.zeros((N, 4096), dtype=torch.bool)

    print("Precomputing 4096-bit legal move masks...")
    for i in tqdm(range(N)):
        board = tensor_to_board(current_boards[i])
        for move in board.legal_moves:
            idx = move.from_square * 64 + move.to_square
            masks[i, idx] = True

    data["legal_masks"] = masks

    try:
        torch.save(data, output_path)
    except Exception:
        # Fall back to legacy zip serialization for very large files
        torch.save(data, output_path, _use_new_zipfile_serialization=False)

    print(f"Saved optimized dataset to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Precompute legal-move masks for a best-move dataset")
    parser.add_argument("--input", required=True, help="Path to input dataset (torch .pt)")
    parser.add_argument("--output", required=True, help="Path to output dataset (torch .pt)")
    args = parser.parse_args()
    precompute_masks(args.input, args.output)


if __name__ == "__main__":
    main()
