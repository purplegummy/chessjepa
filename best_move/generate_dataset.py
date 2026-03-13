"""
Build a (board_tensor, best_move_index) dataset from the Kaggle
"Stockfish Best Moves Compilation" CSV.

Dataset: https://www.kaggle.com/datasets/yousefradwanlmao/stockfish-best-moves-compilation
Expected CSV columns (auto-detected): a FEN column and a UCI-move column.

Move index encoding: from_square * 64 + to_square  (0..4095)
Promotions are collapsed to queen so the index stays in the 4096-class space.

Capture rebalancing
--------------------
Best-move captures are a minority in raw data but are disproportionately
important tactically.  We oversample them up to --capture_ratio (default 35%)
of the final dataset so the decoder sees enough capture examples.
"""

import argparse
import os
import sys
import torch
import chess
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.preprocess_pgn import board_to_tensor


# ── column auto-detection ──────────────────────────────────────────────────

FEN_CANDIDATES  = ["fen", "FEN", "position", "board"]
MOVE_CANDIDATES = ["best_move", "move", "uci", "bestmove", "best move",
                   "stockfish_move", "Move", "Best Move"]
# evaluation column candidates (stockfish centipawn, game result, etc.)
EVAL_CANDIDATES = ["eval", "evaluation", "score", "value", "eval_cp"]

def _find_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    # fallback: first column containing 'fen' or 'move' (case-insensitive)
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in candidates):
            return c
    return None


# ── move helpers ───────────────────────────────────────────────────────────

def uci_to_index(uci: str) -> int | None:
    """
    Convert a UCI move string (e.g. 'e2e4', 'g7g8q') to a flat index
    from_square*64 + to_square.  Promotions are collapsed to queen.
    Returns None on parse error.
    """
    try:
        move = chess.Move.from_uci(uci.strip())
        if move.promotion is not None:
            move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
        return move.from_square * 64 + move.to_square
    except Exception:
        return None


# ── main ───────────────────────────────────────────────────────────────────

def generate_dataset(
    csv_path: str,
    output_path: str,
    capture_ratio: float = 0.35,
    fen_col: str | None = None,
    move_col: str | None = None,
    eval_col: str | None = None,
    no_header: bool = False,
):
    print(f"Reading CSV: {csv_path}")
    if no_header:
        # assume third column is evaluation if present
        df = pd.read_csv(csv_path, header=None, names=["fen", "move", "eval"])
    else:
        df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows, columns: {list(df.columns)}")

    # Auto-detect columns if not specified
    if fen_col is None:
        fen_col = _find_col(df.columns.tolist(), FEN_CANDIDATES)
    if move_col is None:
        move_col = _find_col(df.columns.tolist(), MOVE_CANDIDATES)
    if eval_col is None:
        eval_col = _find_col(df.columns.tolist(), EVAL_CANDIDATES)

    if fen_col is None or move_col is None:
        print(f"ERROR: could not detect FEN/move columns.")
        print(f"  Columns found: {list(df.columns)}")
        print(f"  Use --fen_col and --move_col to specify them manually.")
        sys.exit(1)

    print(f"  Using FEN column : '{fen_col}'")
    print(f"  Using move column: '{move_col}'")

    captures     = []   # (tensor, idx, eval)
    non_captures = []

    print("Processing rows…")
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fen_str  = str(row[fen_col]).strip()
        move_str = str(row[move_col]).strip()

        # Parse evaluation if available
        eval_val = None
        if eval_col is not None and eval_col in row:
            raw = row[eval_col]
            try:
                eval_val = float(raw)
            except Exception:
                s = str(raw).strip()
                if s in ("1", "1-0"):
                    eval_val = 1.0
                elif s in ("0", "0-1"):
                    eval_val = -1.0
                elif s in ("0.5", "1/2", "1/2-1/2", "½"):
                    eval_val = 0.0
                else:
                    # strip +/- and try again
                    try:
                        eval_val = float(s.replace("+", ""))
                    except Exception:
                        eval_val = None
            # clamp, convert mates or huge numbers to reasonable range
            if eval_val is not None:
                # if number is absurd (>1000) treat as mate and cap at ±30
                if abs(eval_val) > 1000:
                    eval_val = 30.0 if eval_val > 0 else -30.0
                # convert centipawns to pawns if needed later loader will handle

        # If we expected an eval column but couldn't parse it, drop the row
        if eval_col is not None and eval_val is None:
            skipped += 1
            continue

        # Parse FEN
        try:
            board = chess.Board(fen_str)
        except Exception:
            skipped += 1
            continue

        # Parse move
        idx = uci_to_index(move_str)
        if idx is None:
            skipped += 1
            continue

        # Validate the move is legal
        try:
            move = chess.Move(idx // 64, idx % 64)
            # check promotion variant too
            if move not in board.legal_moves:
                # try queen promotion
                move_q = chess.Move(idx // 64, idx % 64, promotion=chess.QUEEN)
                if move_q not in board.legal_moves:
                    skipped += 1
                    continue
        except Exception:
            skipped += 1
            continue

        tensor = torch.from_numpy(board_to_tensor(board))   # (17, 8, 8)

        is_capture = board.is_capture(chess.Move.from_uci(move_str.strip()))
        if is_capture:
            captures.append((tensor, idx, eval_val))
        else:
            non_captures.append((tensor, idx, eval_val))

    total_valid = len(captures) + len(non_captures)
    print(f"\nValid rows : {total_valid:,}  |  captures: {len(captures):,}  |  non-captures: {len(non_captures):,}")
    print(f"Skipped    : {skipped:,}")

    if total_valid == 0:
        print("No valid samples — check your CSV columns.")
        sys.exit(1)

    # ── capture rebalancing ────────────────────────────────────────────────
    raw_capture_frac = len(captures) / total_valid
    print(f"Raw capture fraction : {raw_capture_frac:.1%}  →  target: {capture_ratio:.1%}")

    # Work out how many of each bucket to keep / repeat
    # We never discard captures; we may discard excess non-captures.
    n_cap     = len(captures)
    n_non     = len(non_captures)
    target_total = total_valid  # keep dataset size the same as raw

    desired_cap = int(target_total * capture_ratio)
    desired_non = target_total - desired_cap

    # If we don't have enough captures, use all of them and shrink non-captures
    if n_cap < desired_cap:
        desired_cap = n_cap
        desired_non = int(n_cap / capture_ratio) - n_cap
        desired_non = min(desired_non, n_non)

    import random
    random.shuffle(captures)
    random.shuffle(non_captures)

    selected_cap = captures[:desired_cap]
    selected_non = non_captures[:desired_non]
    all_samples  = selected_cap + selected_non
    random.shuffle(all_samples)

    actual_frac = len(selected_cap) / len(all_samples)
    print(f"Final dataset: {len(all_samples):,} samples  |  captures: {len(selected_cap):,} ({actual_frac:.1%})")

    # unpack triples or pairs depending on presence of evals
    if eval_col is not None:
        boards_list, moves_list, eval_list = zip(*all_samples)
    else:
        boards_list, moves_list = zip(*all_samples)
        eval_list = None

    boards = torch.stack(boards_list)                          # (N, 17, 8, 8)
    moves  = torch.tensor(moves_list, dtype=torch.long)        # (N,)

    out_dict = {"boards": boards, "move_indices": moves}
    if eval_list is not None:
        evals = torch.tensor(eval_list, dtype=torch.float32)
        out_dict["evals"] = evals

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(out_dict, output_path)
    shape_info = f"boards: {boards.shape}, moves: {moves.shape}"
    if eval_list is not None:
        shape_info += f", evals: {evals.shape}"
    print(f"Saved → {output_path}  ({shape_info})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",           required=True,                          help="Path to the Kaggle CSV file")
    parser.add_argument("--out",           default="best_move/best_move_dataset.pt", help="Output .pt path")
    parser.add_argument("--capture_ratio", type=float, default=0.35,               help="Target fraction of capture-move samples (default 0.35)")
    parser.add_argument("--fen_col",       default=None,                           help="CSV column name for FEN strings (auto-detected if omitted)")
    parser.add_argument("--move_col",      default=None,                           help="CSV column name for UCI moves (auto-detected if omitted)")
    parser.add_argument("--eval_col",      default=None,                           help="CSV column name for evaluation labels (stockfish score or result)")
    parser.add_argument("--no_header",     action="store_true",                    help="CSV has no header row (columns assumed to be: fen, move, eval)")
    args = parser.parse_args()

    generate_dataset(
        args.csv,
        args.out,
        args.capture_ratio,
        args.fen_col,
        args.move_col,
        args.eval_col,
        args.no_header,
    )
