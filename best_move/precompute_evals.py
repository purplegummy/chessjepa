"""Precompute Stockfish evaluations for a best-move dataset.

Reads a dataset saved by best_move/generate_dataset.py (or precompute_masks.py),
evaluates every position with Stockfish, and saves the results as a new key:

    "eval_targets": torch.FloatTensor(N,)   — values in (-1, 1) via tanh(cp / 400)

Supports resuming: if a partial checkpoint exists at --output it will skip
already-evaluated positions and continue from where it left off.

Usage:
    python best_move/precompute_evals.py \\
        --input  best_move/data/best_move_dataset.pt \\
        --output best_move/data/best_move_dataset_evals.pt \\
        --depth 15 --threads 4

    # Resume an interrupted run (detects existing eval_targets automatically):
    python best_move/precompute_evals.py \\
        --input  best_move/data/best_move_dataset.pt \\
        --output best_move/data/best_move_dataset_evals.pt \\
        --resume
"""

import argparse
import math
import os
import sys

import chess
import chess.engine
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

SENTINEL = float("nan")   # marks un-evaluated slots when resuming


# ─────────────────────────────────────────────────────────────────────────────
# Board reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def tensor_to_board(t) -> chess.Board | None:
    """
    Convert a (17, 8, 8) board tensor back to a chess.Board.

    The encoding is color-invariant: channels 0-5 are always the current
    player's pieces, channels 6-11 the opponent's.  The board is already
    row-flipped when black was to move, so we always reconstruct as WHITE.

    Returns None if the position is illegal or the game is already over.
    """
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()

    board = chess.Board(fen=None)
    board.clear()

    for i, piece in enumerate(_PIECES):
        for r, c in zip(*np.where(t[i] > 0.5)):       # current player (white)
            board.set_piece_at(chess.square(int(c), int(r)), chess.Piece(piece, chess.WHITE))
        for r, c in zip(*np.where(t[i + 6] > 0.5)):   # opponent (black)
            board.set_piece_at(chess.square(int(c), int(r)), chess.Piece(piece, chess.BLACK))

    board.turn = chess.WHITE  # color-invariant encoding — always white to move

    # Castling rights: ch12=current-KS, ch13=current-QS, ch14=opp-KS, ch15=opp-QS
    board.castling_rights = chess.BB_EMPTY
    if t[12].any(): board.castling_rights |= chess.BB_H1
    if t[13].any(): board.castling_rights |= chess.BB_A1
    if t[14].any(): board.castling_rights |= chess.BB_H8
    if t[15].any(): board.castling_rights |= chess.BB_A8

    ep = np.argwhere(t[16] > 0.5)
    if len(ep):
        r, c = ep[0]
        board.ep_square = chess.square(int(c), int(r))

    try:
        if board.is_game_over() or not list(board.legal_moves):
            return None
    except Exception:
        return None

    return board


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation normalization
# ─────────────────────────────────────────────────────────────────────────────

def cp_to_value(cp: int, scale: float = 400.0) -> float:
    """Centipawns → (-1, 1) via tanh.  ±30 000 cp (mate) → ≈ ±1."""
    return math.tanh(cp / scale)


def evaluate(engine: chess.engine.SimpleEngine,
             board: chess.Board,
             depth: int,
             time_limit: float | None) -> float | None:
    """Return a normalized value in (-1, 1) from the current player's POV."""
    limit = chess.engine.Limit(depth=depth) if time_limit is None \
        else chess.engine.Limit(time=time_limit)
    try:
        info = engine.analyse(board, limit)
        score = info["score"].white()   # always from white's side (= current player)
        if score.is_mate():
            cp = 30_000 * (1 if score.mate() > 0 else -1)
        else:
            cp = score.score(mate_score=30_000)
        return cp_to_value(cp)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def precompute_evals(
    input_path: str,
    output_path: str,
    stockfish_bin: str = "stockfish",
    depth: int = 15,
    time_limit: float | None = None,
    threads: int = 1,
    hash_mb: int = 128,
    resume: bool = False,
    save_every: int = 1000,
):
    print(f"Loading {input_path} …")
    data = torch.load(input_path, map_location="cpu", weights_only=False)

    if "boards" not in data:
        raise KeyError("Dataset must contain a 'boards' key")

    boards_tensor = data["boards"]   # (N, 17, 8, 8)
    N = boards_tensor.shape[0]
    print(f"  {N:,} positions")

    # ── Resume support ────────────────────────────────────────────────────────
    start_idx = 0
    eval_targets = torch.full((N,), SENTINEL, dtype=torch.float32)

    if resume and os.path.exists(output_path):
        print(f"  Resuming from {output_path} …")
        existing = torch.load(output_path, map_location="cpu", weights_only=False)
        if "eval_targets" in existing:
            prev = existing["eval_targets"]
            n_done = int((~torch.isnan(prev)).sum())
            eval_targets[:len(prev)] = prev
            start_idx = n_done
            print(f"  {n_done:,} already evaluated — continuing from index {start_idx:,}")
    elif "eval_targets" in data:
        prev = data["eval_targets"]
        n_done = int((~torch.isnan(prev)).sum())
        eval_targets[:len(prev)] = prev
        start_idx = n_done
        print(f"  Found {n_done:,} existing eval_targets in input — continuing")

    remaining = N - start_idx
    if remaining == 0:
        print("All positions already evaluated.")
        return

    # ── Start Stockfish ───────────────────────────────────────────────────────
    print(f"\nStarting Stockfish ({stockfish_bin}) …")
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_bin)
        engine.configure({"Threads": threads, "Hash": hash_mb})
    except Exception as e:
        print(f"ERROR: Could not start Stockfish: {e}")
        print("Make sure stockfish is on PATH or pass --stockfish /path/to/sf")
        sys.exit(1)

    skipped = 0
    out_data = {**data, "eval_targets": eval_targets}

    def _save():
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(out_data, output_path)

    print(f"Evaluating {remaining:,} positions (depth={depth}, time={time_limit}s) …\n")

    try:
        with tqdm(total=remaining, unit=" pos") as pbar:
            for i in range(start_idx, N):
                board = tensor_to_board(boards_tensor[i])
                if board is None:
                    eval_targets[i] = 0.0   # treat invalid positions as drawn
                    skipped += 1
                else:
                    val = evaluate(engine, board, depth, time_limit)
                    eval_targets[i] = val if val is not None else 0.0
                    if val is None:
                        skipped += 1

                pbar.update(1)

                if (i + 1) % save_every == 0:
                    _save()

    finally:
        engine.quit()
        _save()   # always save on exit (including KeyboardInterrupt)

    n_nan = int(torch.isnan(eval_targets).sum())
    print(f"\nDone.  Evaluated: {N - skipped:,}  Skipped/defaulted: {skipped:,}  NaN remaining: {n_nan}")
    print(f"Saved → {output_path}")
    print(f"\nValue stats: mean={eval_targets.nanmean():.4f}  std={eval_targets[~torch.isnan(eval_targets)].std():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute Stockfish evaluations for a best-move dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",      required=True,         help="Input dataset (.pt)")
    parser.add_argument("--output",     required=True,         help="Output dataset (.pt)")
    parser.add_argument("--stockfish",  default="stockfish",   help="Stockfish binary path")
    parser.add_argument("--depth",      type=int,   default=15, help="Search depth per position")
    parser.add_argument("--time",       type=float, default=None, help="Time limit per position in seconds (overrides --depth)")
    parser.add_argument("--threads",    type=int,   default=1,  help="Stockfish threads")
    parser.add_argument("--hash",       type=int,   default=128, help="Stockfish hash table MB")
    parser.add_argument("--resume",     action="store_true",   help="Resume from existing --output file")
    parser.add_argument("--save_every", type=int,   default=1000, help="Checkpoint frequency (positions)")
    args = parser.parse_args()

    precompute_evals(
        input_path=args.input,
        output_path=args.output,
        stockfish_bin=args.stockfish,
        depth=args.depth,
        time_limit=args.time,
        threads=args.threads,
        hash_mb=args.hash,
        resume=args.resume,
        save_every=args.save_every,
    )
