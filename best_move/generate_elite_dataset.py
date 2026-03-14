"""
Build a (board_sequence, move_index) dataset from Lichess Elite Database PGN files.

Each sample stores the last --seq_len board states leading up to the move, matching
the temporal context the JEPA encoder was trained on. Early positions are zero-padded.

The Lichess Elite Database contains games played by players rated 2200+ on Lichess.
Download monthly PGN files from: https://database.nikonoel.fr/

Streams PGN files one game at a time using reservoir sampling, so memory usage
is bounded to --max_samples regardless of how many files / games you pass in.

Usage:
    python best_move/generate_elite_dataset.py \\
        --pgn_dir data/elite_pgns/ \\
        --out data/elite_dataset.pt \\
        --max_samples 500000
"""

import argparse
import os
import sys
import random
from collections import deque
import chess
import chess.pgn
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.preprocess_pgn import board_to_tensor


def uci_to_index(move: chess.Move, board: chess.Board) -> int:
    """
    Encode move as from_square*64 + to_square with the same vertical flip that
    board_to_tensor applies when it is Black's turn.
    """
    from_sq = move.from_square
    to_sq   = move.to_square
    if board.turn == chess.BLACK:
        from_sq = (7 - from_sq // 8) * 8 + from_sq % 8
        to_sq   = (7 - to_sq   // 8) * 8 + to_sq   % 8
    return from_sq * 64 + to_sq


def generate_elite_dataset(
    pgn_paths: list[str],
    output_path: str,
    max_samples: int = 500_000,
    max_games: int | None = None,
    min_elo: int = 2200,
    capture_ratio: float = 0.35,
    seq_len: int = 16,
):
    # Fixed-size reservoirs for captures and non-captures
    cap_size     = int(max_samples * capture_ratio)
    non_cap_size = max_samples - cap_size
    cap_reservoir     = []   # each entry: (tensor, idx)
    non_cap_reservoir = []

    cap_seen     = 0   # total captures streamed so far
    non_cap_seen = 0

    games_read  = 0
    skipped_elo = 0
    positions   = 0

    pbar = tqdm(total=max_games, desc="Games", unit="game")

    for pgn_path in pgn_paths:
        with open(pgn_path, encoding="utf-8", errors="replace") as f:
            while True:
                if max_games is not None and games_read >= max_games:
                    break

                game = chess.pgn.read_game(f)
                if game is None:
                    break

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
                pbar.update(1)
                board = game.board()
                history: deque = deque(maxlen=seq_len)

                for move in game.mainline_moves():
                    history.append(board_to_tensor(board))

                    # Build (seq_len, 17, 8, 8) with zero-padding for early positions
                    frames = list(history)
                    if len(frames) < seq_len:
                        pad = [np.zeros((17, 8, 8), dtype=np.uint8)] * (seq_len - len(frames))
                        frames = pad + frames
                    seq_tensor = torch.from_numpy(np.stack(frames))  # (seq_len, 17, 8, 8)

                    idx    = uci_to_index(move, board)
                    sample = (seq_tensor, idx)
                    positions += 1


                    # Reservoir sampling (Algorithm R) per bucket
                    if board.is_capture(move):
                        cap_seen += 1
                        if len(cap_reservoir) < cap_size:
                            cap_reservoir.append(sample)
                        else:
                            j = random.randint(0, cap_seen - 1)
                            if j < cap_size:
                                cap_reservoir[j] = sample
                    else:
                        non_cap_seen += 1
                        if len(non_cap_reservoir) < non_cap_size:
                            non_cap_reservoir.append(sample)
                        else:
                            j = random.randint(0, non_cap_seen - 1)
                            if j < non_cap_size:
                                non_cap_reservoir[j] = sample

                    board.push(move)

        if max_games is not None and games_read >= max_games:
            break

    pbar.close()

    print(f"\nGames: {games_read:,}  |  skipped (ELO): {skipped_elo:,}  |  positions streamed: {positions:,}")
    print(f"Reservoir — captures: {len(cap_reservoir):,}  non-captures: {len(non_cap_reservoir):,}")

    all_samples = cap_reservoir + non_cap_reservoir
    if not all_samples:
        print("No samples collected.")
        sys.exit(1)

    random.shuffle(all_samples)

    boards_list, moves_list = zip(*all_samples)
    boards = torch.stack(boards_list)                   # (N, 17, 8, 8)
    moves  = torch.tensor(moves_list, dtype=torch.long) # (N,)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save({"boards": boards, "move_indices": moves}, output_path)
    print(f"Saved → {output_path}  (boards: {boards.shape}, moves: {moves.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn_dir",       required=True,                   help="Folder containing Lichess Elite PGN files")
    parser.add_argument("--out",           default="data/elite_dataset.pt", help="Output .pt path")
    parser.add_argument("--max_samples",   type=int,   default=500_000,     help="Max samples to keep in memory and save (default: 500k)")
    parser.add_argument("--max_games",     type=int,   default=None,        help="Stop after this many games (default: all)")
    parser.add_argument("--min_elo",       type=int,   default=2200,        help="Minimum ELO for both players (default: 2200)")
    parser.add_argument("--capture_ratio", type=float, default=0.35,        help="Target fraction of capture moves (default: 0.35)")
    parser.add_argument("--seq_len",       type=int,   default=16,          help="Number of board states per sample — must match JEPA seq_len (default: 16)")
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
        args.max_samples,
        args.max_games,
        args.min_elo,
        args.capture_ratio,
        args.seq_len,
    )
