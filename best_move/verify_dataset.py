"""
Quick sanity-check: for N random samples in the dataset,
verify the stored move index is in the on-the-fly legal mask.
Prints details on any mismatches.

Usage:
    python best_move/verify_dataset.py --dataset best_move/data/best_move_dataset.pt --n 500
"""

import argparse
import os
import sys

import chess
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def tensor_to_board(t) -> chess.Board:
    """Reconstruct a board in tensor-coordinate space (always white-to-move)."""
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


def verify(dataset_path: str, n: int):
    print(f"Loading {dataset_path} ...")
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    boards = data["boards"]       # (N, 17, 8, 8)
    moves  = data["move_indices"] # (N,)
    N = len(moves)
    print(f"Dataset: {N:,} samples")

    import random
    indices = random.sample(range(N), min(n, N))

    errors = 0
    castling_errors = 0
    for i in indices:
        board_t = boards[i]          # (17, 8, 8)
        stored_idx = moves[i].item()  # int

        board = tensor_to_board(board_t)
        legal_indices = set(m.from_square * 64 + m.to_square for m in board.legal_moves)

        if stored_idx not in legal_indices:
            errors += 1
            from_sq = stored_idx // 64
            to_sq   = stored_idx % 64
            from_file = "abcdefgh"[from_sq % 8]
            from_rank = from_sq // 8 + 1
            to_file   = "abcdefgh"[to_sq % 8]
            to_rank   = to_sq // 8 + 1

            # Check if it's a castling issue
            castle_indices = set()
            king_sq = None
            for sq in range(64):
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.KING and piece.color == chess.WHITE:
                    king_sq = sq
            if king_sq is not None:
                # Typical castling move indices
                if from_sq == king_sq:
                    castling_errors += 1

            if errors <= 5:  # Print first 5 failures
                print(f"\n  Sample {i}: stored idx={stored_idx}")
                print(f"    from ({from_file}{from_rank}, tensor sq={from_sq}) → to ({to_file}{to_rank}, tensor sq={to_sq})")
                print(f"    Num legal moves: {len(legal_indices)}")
                print(f"    Legal from-squares: {sorted(set(idx // 64 for idx in legal_indices))}")
                if king_sq is not None:
                    print(f"    White king at sq={king_sq} (file={king_sq%8}, rank={king_sq//8})")

    print(f"\nResults: {errors}/{len(indices)} samples have invalid stored move ({errors/len(indices)*100:.1f}%)")
    if castling_errors:
        print(f"  Castling-related errors: {castling_errors}")
    if errors == 0:
        print("All good! Dataset move indices match legal masks.")
    elif errors == len(indices):
        print("ERROR: 100% failure — systematic coordinate mismatch. Regenerate the dataset.")
    else:
        print("Partial failures — likely edge cases (castling, en passant, unusual positions).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="best_move/data/best_move_dataset.pt")
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()
    verify(args.dataset, args.n)
