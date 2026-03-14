"""
Build a (board_tensor, move_index) dataset from Lichess Elite Database PGN files.

The Lichess Elite Database contains games played by players rated 2200+ on Lichess.
Download monthly PGN files from: https://database.nikonoel.fr/

Each position in every game becomes a training sample — the label is the move
the strong player chose to play from that position.

Usage:
    python best_move/generate_elite_dataset.py \\
        --pgn data/lichess_elite_2024-01.pgn \\
        --out data/elite_dataset.pt \\
        --max_games 200000

Multiple PGN files can be passed:
    python best_move/generate_elite_dataset.py \\
        --pgn data/elite_jan.pgn data/elite_feb.pgn \\
        --out data/elite_dataset.pt
"""

import argparse
import os
import sys
import random
import chess
import chess.pgn
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.preprocess_pgn import board_to_tensor


def uci_to_index(move: chess.Move, board: chess.Board) -> int:
    """
    Encode move as from_square*64 + to_square, with board-orientation flip
    applied when it is Black's turn (matches board_to_tensor's convention).
    Promotions are collapsed to queen.
    """
    from_sq = move.from_square
    to_sq   = move.to_square

    if board.turn == chess.BLACK:
        from_sq = (7 - from_sq // 8) * 8 + from_sq % 8
        to_sq   = (7 - to_sq   // 8) * 8 + to_sq   % 8

    return from_sq * 64 + to_sq


def process_pgn(
    pgn_paths: list[str],
    max_games: int,
    min_elo: int,
    skip_first_n: int,
    capture_ratio: float,
) -> tuple[list, list]:
    """
    Parse PGN files and collect (tensor, move_idx) pairs.
    Returns two lists: captures and non_captures.
    """
    captures     = []
    non_captures = []
    games_read   = 0
    positions    = 0
    skipped_elo  = 0

    for pgn_path in pgn_paths:
        print(f"Reading: {pgn_path}")
        with open(pgn_path, encoding="utf-8", errors="replace") as f:
            while games_read < max_games:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                # Filter by minimum ELO of both players
                try:
                    white_elo = int(game.headers.get("WhiteElo", "0") or "0")
                    black_elo = int(game.headers.get("BlackElo", "0") or "0")
                except ValueError:
                    skipped_elo += 1
                    continue

                if white_elo < min_elo or black_elo < min_elo:
                    skipped_elo += 1
                    continue

                games_read += 1
                board = game.board()
                move_num = 0

                for move in game.mainline_moves():
                    move_num += 1

                    # Skip opening moves — they're mostly theory and add noise
                    if move_num <= skip_first_n:
                        board.push(move)
                        continue

                    tensor = torch.from_numpy(board_to_tensor(board))
                    idx    = uci_to_index(move, board)

                    if board.is_capture(move):
                        captures.append((tensor, idx))
                    else:
                        non_captures.append((tensor, idx))

                    positions += 1
                    board.push(move)

        if games_read >= max_games:
            break

    print(f"Games parsed: {games_read:,}  |  skipped (ELO): {skipped_elo:,}  |  positions: {positions:,}")
    return captures, non_captures


def generate_elite_dataset(
    pgn_paths: list[str],
    output_path: str,
    max_games: int = 100_000,
    min_elo: int = 2200,
    skip_first_n: int = 8,
    capture_ratio: float = 0.35,
    max_samples: int | None = None,
):
    captures, non_captures = process_pgn(pgn_paths, max_games, min_elo, skip_first_n, capture_ratio)

    total_valid = len(captures) + len(non_captures)
    print(f"\nCaptures: {len(captures):,}  Non-captures: {len(non_captures):,}  Total: {total_valid:,}")

    if total_valid == 0:
        print("No valid samples found.")
        sys.exit(1)

    # Capture rebalancing: oversample captures up to capture_ratio
    raw_cap_frac = len(captures) / total_valid
    print(f"Raw capture fraction: {raw_cap_frac:.1%}  →  target: {capture_ratio:.1%}")

    n_cap = len(captures)
    n_non = len(non_captures)

    desired_cap = int(total_valid * capture_ratio)
    desired_non = total_valid - desired_cap

    if n_cap < desired_cap:
        desired_cap = n_cap
        desired_non = int(n_cap / capture_ratio) - n_cap
        desired_non = min(desired_non, n_non)

    random.shuffle(captures)
    random.shuffle(non_captures)

    all_samples = captures[:desired_cap] + non_captures[:desired_non]
    random.shuffle(all_samples)

    if max_samples is not None and len(all_samples) > max_samples:
        all_samples = all_samples[:max_samples]

    actual_frac = sum(1 for _, idx in all_samples
                      if len(all_samples) > 0) / len(all_samples)  # placeholder
    print(f"Final dataset: {len(all_samples):,} samples")

    boards_list, moves_list = zip(*all_samples)
    boards = torch.stack(boards_list)                      # (N, 17, 8, 8)
    moves  = torch.tensor(moves_list, dtype=torch.long)    # (N,)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save({"boards": boards, "move_indices": moves}, output_path)
    print(f"Saved → {output_path}  (boards: {boards.shape}, moves: {moves.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn_dir",        required=True,                    help="Folder containing Lichess Elite PGN files")
    parser.add_argument("--out",            default="data/elite_dataset.pt",  help="Output .pt path")
    parser.add_argument("--max_games",      type=int,   default=100_000,      help="Max games to read (default: 100k)")
    parser.add_argument("--min_elo",        type=int,   default=2200,         help="Minimum ELO for both players (default: 2200)")
    parser.add_argument("--skip_first_n",   type=int,   default=8,            help="Skip first N moves per game (opening theory)")
    parser.add_argument("--capture_ratio",  type=float, default=0.35,         help="Target fraction of capture moves (default: 0.35)")
    parser.add_argument("--max_samples",    type=int,   default=None,         help="Cap total dataset size (optional)")
    args = parser.parse_args()

    pgn_paths = sorted([
        os.path.join(args.pgn_dir, f)
        for f in os.listdir(args.pgn_dir)
        if f.endswith(".pgn")
    ])
    if not pgn_paths:
        print(f"No .pgn files found in {args.pgn_dir}")
        sys.exit(1)
    print(f"Found {len(pgn_paths)} PGN file(s) in {args.pgn_dir}")

    generate_elite_dataset(
        pgn_paths,
        args.out,
        args.max_games,
        args.min_elo,
        args.skip_first_n,
        args.capture_ratio,
        args.max_samples,
    )
