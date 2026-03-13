"""
Build a best-move dataset by combining two sources:

  1. Lichess puzzle database  (lichess_db_puzzle.csv)
     FEN = pre-puzzle position; Moves = "<setup_move> <solution> ..."
     → apply Moves[0] to reach the puzzle position, Moves[1] is the answer.
     Puzzles are tactically rich (many captures/forks/mates) so they
     naturally oversample forcing moves.

  2. Stockfish best-moves CSV  (stockfish_best_moves.csv)
     Columns: Game, Position (FEN), Best Move (UCI), Evaluation
     → direct FEN + best-move pairs from quiet / game positions.

Combining both gives coverage of:
  - Tactical forcing lines  (puzzles)
  - Positional / quiet moves (stockfish CSV)

Move index encoding: from_square * 64 + to_square  (0..4095)
Promotions are collapsed to queen.
"""

import argparse
import os
import sys
import random
import torch
import chess
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.preprocess_pgn import board_to_tensor

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"


# ── helpers ────────────────────────────────────────────────────────────────

def uci_to_index(uci: str, board: chess.Board) -> int | None:
    """Parse UCI string → flat index (from_sq*64 + to_sq), queen-promoting."""
    try:
        move = chess.Move.from_uci(uci.strip())
        if move.promotion is not None:
            move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
        if move not in board.legal_moves:
            # Try queen promotion variant in case the UCI omitted it
            move_q = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            if move_q not in board.legal_moves:
                return None
            move = move_q
        return move.from_square * 64 + move.to_square
    except Exception:
        return None


def load_puzzles(csv_path: str, max_samples: int) -> tuple[list, list, int]:
    """
    Load Lichess puzzle CSV.

    The FEN is the position BEFORE the opponent's last move.
    Moves[0] = opponent's forcing move (sets up the puzzle).
    Moves[1] = the correct best move for the solver.

    Returns (capture_samples, non_capture_samples, n_skipped).

    Puzzles don't come with numeric evaluations, but they are designed to
    illustrate winning tactics.  Instead of zero we assign a default value of
    +1.0 (a 'win') for every puzzle sample; this gives the value head a weak
    positive signal and avoids having to zero-weight those examples during
    training.  You can still tune `--value_loss_weight` when mixing with other
    datasets if desired.
    """
    print(f"Loading puzzles from: {csv_path}")
    df = pd.read_csv(csv_path, usecols=["FEN", "Moves"])
    print(f"  {len(df):,} puzzles available, sampling up to {max_samples:,}")

    if max_samples < len(df):
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    captures, non_captures, skipped = [], [], 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Puzzles"):
        try:
            moves = str(row["Moves"]).strip().split()
            if len(moves) < 2:
                skipped += 1
                continue

            board = chess.Board(str(row["FEN"]))

            # Apply the opponent's setup move
            setup = chess.Move.from_uci(moves[0])
            if setup not in board.legal_moves:
                skipped += 1
                continue
            board.push(setup)

            # The solution move is Moves[1]
            idx = uci_to_index(moves[1], board)
            if idx is None:
                skipped += 1
                continue

            tensor = torch.from_numpy(board_to_tensor(board))
            solution = chess.Move(idx // 64, idx % 64)
            is_cap = board.is_capture(solution)

            # puzzles represent winning tactics; label as +1.0
            (captures if is_cap else non_captures).append((tensor, idx, 3.0))

        except Exception:
            skipped += 1
            continue

    return captures, non_captures, skipped


def load_stockfish_csv(csv_path: str) -> tuple[list, list, int]:
    """
    Load stockfish_best_moves.csv.
    This is a header-less CSV with columns: Game, Position (FEN), Best Move (UCI), Evaluation
    """
    print(f"Loading Stockfish CSV: {csv_path}")
    
    # Updated names list to include 'Game' column (4 columns total)
    try:
        df = pd.read_csv(
            csv_path, 
            header=None, 
            names=["Game", "Position", "Best Move", "eval"],
            engine='python',
            on_bad_lines='skip'
        )
    except Exception as e:
        print(f"Critical error reading CSV: {e}")
        return [], [], 0

    # Drop duplicates on FEN to avoid identical positions with conflicting labels
    df = df.drop_duplicates(subset="Position").reset_index(drop=True)
    print(f"  {len(df):,} unique positions")

    captures, non_captures, skipped = [], [], 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Stockfish CSV"):
        try:
            # Ensure we're reading from 'Position' (the FEN column)
            board = chess.Board(str(row["Position"]).strip())
            idx = uci_to_index(str(row["Best Move"]), board)
            
            if idx is None:
                skipped += 1
                continue

            tensor = torch.from_numpy(board_to_tensor(board))
            move = chess.Move(idx // 64, idx % 64)
            is_cap = board.is_capture(move)
            
            # Parse evaluation
            eval_val = 0.0
            if "eval" in row and not pd.isna(row["eval"]):
                try:
                    eval_val = float(row["eval"])
                except ValueError:
                    eval_val = 0.0
                    
            # Clamp extreme values
            if abs(eval_val) > 1000:
                eval_val = 30.0 if eval_val > 0 else -30.0

            (captures if is_cap else non_captures).append((tensor, idx, eval_val))

        except Exception:
            skipped += 1
            continue

    return captures, non_captures, skipped


# ── rebalancing ────────────────────────────────────────────────────────────

def rebalance_and_save(
    captures: list,
    non_captures: list,
    capture_ratio: float,
    output_path: str,
):
    total = len(captures) + len(non_captures)
    raw_frac = len(captures) / max(total, 1)
    print(f"\nRaw  — total: {total:,}  captures: {len(captures):,} ({raw_frac:.1%})  non-captures: {len(non_captures):,}")
    print(f"Target capture ratio: {capture_ratio:.1%}")

    random.shuffle(captures)
    random.shuffle(non_captures)

    # Never discard captures; shrink non-captures to hit the ratio
    n_cap = len(captures)
    desired_non = int(n_cap / capture_ratio) - n_cap
    desired_non = min(desired_non, len(non_captures))

    selected = captures + non_captures[:desired_non]
    random.shuffle(selected)

    actual_frac = n_cap / len(selected)
    print(f"Final — total: {len(selected):,}  captures: {n_cap:,} ({actual_frac:.1%})\n")

    # determine if evals are included by inspecting tuple length
    if selected and len(selected[0]) == 3:
        boards_list, moves_list, eval_list = zip(*selected)
    else:
        boards_list, moves_list = zip(*selected)
        eval_list = None

    boards = torch.stack(boards_list)
    moves  = torch.tensor(moves_list, dtype=torch.long)

    out_dict = {"boards": boards, "move_indices": moves}
    if eval_list is not None:
        evals = torch.tensor(eval_list, dtype=torch.float32)
        out_dict["evals"] = evals

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(out_dict, output_path)
    shape_info = f"boards: {boards.shape}  moves: {moves.shape}"
    if eval_list is not None:
        shape_info += f"  evals: {evals.shape}"
    print(f"Saved → {output_path}")
    print(f"  {shape_info}")


# ── main ───────────────────────────────────────────────────────────────────

def main(
    puzzle_csv: str,
    stockfish_csv: str,
    output_path: str,
    max_puzzles: int,
    capture_ratio: float,
):
    all_captures, all_non_captures = [], []
    total_skipped = 0

    # ── source 1: puzzles ──────────────────────────────────────────────────
    if puzzle_csv and os.path.exists(puzzle_csv):
        cap, non, skip = load_puzzles(puzzle_csv, max_puzzles)
        print(f"  Puzzles   → captures: {len(cap):,}  non-captures: {len(non):,}  skipped: {skip:,}")
        all_captures    += cap
        all_non_captures += non
        total_skipped   += skip
    else:
        print(f"Puzzle CSV not found, skipping: {puzzle_csv}")

    # ── source 2: stockfish CSV ────────────────────────────────────────────
    if stockfish_csv and os.path.exists(stockfish_csv):
        cap, non, skip = load_stockfish_csv(stockfish_csv)
        print(f"  Stockfish → captures: {len(cap):,}  non-captures: {len(non):,}  skipped: {skip:,}")
        all_captures    += cap
        all_non_captures += non
        total_skipped   += skip
    else:
        print(f"Stockfish CSV not found, skipping: {stockfish_csv}")

    print(f"\nTotal skipped: {total_skipped:,}")

    if not all_captures and not all_non_captures:
        print("No valid samples collected. Check file paths.")
        sys.exit(1)

    rebalance_and_save(all_captures, all_non_captures, capture_ratio, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzles",
                        default="data/lichess_db_puzzle.csv",
                        help="Path to Lichess puzzle CSV")
    parser.add_argument("--stockfish",
                        default="data/stockfish_best_moves.csv",
                        help="Path to stockfish best-moves CSV")
    parser.add_argument("--out",
                        default="best_move/best_move_dataset.pt",
                        help="Output .pt path")
    parser.add_argument("--max_puzzles",
                        type=int, default=300_000,
                        help="Max puzzle rows to sample (default 300k)")
    parser.add_argument("--capture_ratio",
                        type=float, default=0.40,
                        help="Target capture fraction in final dataset (default 0.40)")
    args = parser.parse_args()

    main(args.puzzles, args.stockfish, args.out, args.max_puzzles, args.capture_ratio)
