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
import io
import os
import sys
import random
from collections import deque
from multiprocessing import Pool
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


def _flip_tensor(t_white: np.ndarray) -> np.ndarray:
    """Derive the black-perspective tensor from the white-perspective tensor."""
    t_black = np.empty_like(t_white)
    t_black[0:6]  = t_white[6:12, ::-1, :]
    t_black[6:12] = t_white[0:6,  ::-1, :]
    t_black[12]   = t_white[14]
    t_black[13]   = t_white[15]
    t_black[14]   = t_white[12]
    t_black[15]   = t_white[13]
    t_black[16]   = t_white[16, ::-1, :]
    return t_black


def _process_game(args) -> list[tuple[np.ndarray, int, bool]]:
    """
    Worker: parse one PGN string and return all (seq_array, move_idx, is_capture) tuples.
    Returns [] if the game is filtered out.
    """
    game_str, min_elo, seq_len = args
    game = chess.pgn.read_game(io.StringIO(game_str))
    if game is None:
        return []

    h = game.headers
    try:
        if int(h.get("WhiteElo", "0") or "0") < min_elo:
            return []
        if int(h.get("BlackElo", "0") or "0") < min_elo:
            return []
    except ValueError:
        return []

    board = game.board()
    history: deque = deque(maxlen=seq_len)
    PAD = np.zeros((17, 8, 8), dtype=np.uint8)
    results = []

    for move in game.mainline_moves():
        current_flip = board.turn == chess.BLACK
        t_white = board_to_tensor(board, force_flip=False)
        t_black = _flip_tensor(t_white)
        history.append((t_white, t_black))

        frames = [entry[int(current_flip)] for entry in history]
        if len(frames) < seq_len:
            frames = [PAD] * (seq_len - len(frames)) + frames

        seq = np.stack(frames)  # (seq_len, 17, 8, 8)
        idx = uci_to_index(move, board)
        is_capture = board.is_capture(move)
        results.append((seq, idx, is_capture))
        board.push(move)

    return results


def _read_game_strings(pgn_paths: list[str], max_games: int | None) -> list[str]:
    """Read all PGN files and split into individual game strings."""
    game_strings = []
    for path in pgn_paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = []
            for line in f:
                lines.append(line)
                if line.startswith("[Event ") and len(lines) > 1:
                    game_strings.append("".join(lines[:-1]))
                    lines = [line]
                    if max_games is not None and len(game_strings) >= max_games:
                        return game_strings
            if lines:
                game_strings.append("".join(lines))
            if max_games is not None and len(game_strings) >= max_games:
                return game_strings
    return game_strings


def generate_elite_dataset(
    pgn_paths: list[str],
    output_path: str,
    max_samples: int = 500_000,
    max_games: int | None = None,
    min_elo: int = 2200,
    capture_ratio: float = 0.35,
    seq_len: int = 16,
    num_workers: int = 32,
):
    cap_size     = int(max_samples * capture_ratio)
    non_cap_size = max_samples - cap_size
    cap_reservoir     = []
    non_cap_reservoir = []
    cap_seen     = 0
    non_cap_seen = 0
    positions    = 0

    print(f"Reading game strings from {len(pgn_paths)} PGN file(s)...")
    game_strings = _read_game_strings(pgn_paths, max_games)
    print(f"Found {len(game_strings):,} games — processing with {num_workers} workers...")

    worker_args = [(gs, min_elo, seq_len) for gs in game_strings]

    with Pool(num_workers) as pool:
        for game_samples in tqdm(
            pool.imap_unordered(_process_game, worker_args, chunksize=64),
            total=len(game_strings),
            desc="Games",
            unit="game",
        ):
            for seq, idx, is_capture in game_samples:
                positions += 1
                sample = (seq, idx)
                if is_capture:
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

    print(f"\nGames: {len(game_strings):,}  |  positions streamed: {positions:,}")
    print(f"Reservoir — captures: {len(cap_reservoir):,}  non-captures: {len(non_cap_reservoir):,}")

    all_samples = cap_reservoir + non_cap_reservoir
    if not all_samples:
        print("No samples collected.")
        sys.exit(1)

    random.shuffle(all_samples)

    boards_arr = np.stack([s[0] for s in all_samples])          # (N, seq_len, 17, 8, 8)
    moves_arr  = np.array([s[1] for s in all_samples], dtype=np.int64)  # (N,)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    boards = torch.from_numpy(boards_arr)
    moves  = torch.from_numpy(moves_arr)
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
    parser.add_argument("--workers",       type=int,   default=32,          help="Parallel worker processes (default: 32)")
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
        args.workers,
    )
