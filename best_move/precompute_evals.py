"""Precompute Stockfish evaluations for a best-move dataset.

Reads a dataset saved by best_move/generate_dataset.py (or precompute_masks.py),
evaluates every position with Stockfish in parallel, and saves:

    "eval_targets": torch.FloatTensor(N,)   — values in (-1, 1) via tanh(cp / 400)

Each worker process owns its own Stockfish engine instance.
Supports resuming: pass --resume and point --output at the partial file.

Usage:
    python best_move/precompute_evals.py \\
        --input  best_move/data/best_move_dataset.pt \\
        --output best_move/data/best_move_dataset_evals.pt \\
        --depth 15 --workers 8

    # Resume an interrupted run:
    python best_move/precompute_evals.py \\
        --input  best_move/data/best_move_dataset.pt \\
        --output best_move/data/best_move_dataset_evals.pt \\
        --workers 8 --resume
"""

import argparse
import math
import multiprocessing as mp
import os
import sys

import chess
import chess.engine
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

SENTINEL = float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Board reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def tensor_to_board(t: np.ndarray) -> chess.Board | None:
    board = chess.Board(fen=None)
    board.clear()

    for i, piece in enumerate(_PIECES):
        for r, c in zip(*np.where(t[i] > 0.5)):
            board.set_piece_at(chess.square(int(c), int(r)), chess.Piece(piece, chess.WHITE))
        for r, c in zip(*np.where(t[i + 6] > 0.5)):
            board.set_piece_at(chess.square(int(c), int(r)), chess.Piece(piece, chess.BLACK))

    board.turn = chess.WHITE  # color-invariant encoding — always white to move

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
# Per-worker Stockfish engine (initialized once per process)
# ─────────────────────────────────────────────────────────────────────────────

_engine: chess.engine.SimpleEngine | None = None
_depth: int = 15
_time_limit: float | None = None


def _worker_init(stockfish_bin: str, depth: int, time_limit: float | None,
                 threads_per_worker: int, hash_mb: int):
    global _engine, _depth, _time_limit
    _depth = depth
    _time_limit = time_limit
    _engine = chess.engine.SimpleEngine.popen_uci(stockfish_bin)
    _engine.configure({"Threads": threads_per_worker, "Hash": hash_mb})


def _worker_eval(args: tuple[int, np.ndarray]) -> tuple[int, float]:
    """Evaluate a single board. Returns (original_index, value)."""
    idx, board_np = args
    board = tensor_to_board(board_np)
    if board is None:
        return idx, 0.0

    limit = chess.engine.Limit(depth=_depth) if _time_limit is None \
        else chess.engine.Limit(time=_time_limit)
    try:
        info = _engine.analyse(board, limit)
        score = info["score"].white()
        if score.is_mate():
            cp = 30_000 * (1 if score.mate() > 0 else -1)
        else:
            cp = score.score(mate_score=30_000)
        return idx, math.tanh(cp / 400.0)
    except Exception:
        return idx, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def precompute_evals(
    input_path: str,
    output_path: str,
    stockfish_bin: str = "stockfish",
    depth: int = 15,
    time_limit: float | None = None,
    num_workers: int = 4,
    threads_per_worker: int = 1,
    hash_mb: int = 128,
    resume: bool = False,
    save_every: int = 2000,
):
    print(f"Loading {input_path} …")
    data = torch.load(input_path, map_location="cpu", weights_only=False)

    if "boards" not in data:
        raise KeyError("Dataset must contain a 'boards' key")

    boards_tensor = data["boards"]   # (N, 17, 8, 8)
    N = boards_tensor.shape[0]
    print(f"  {N:,} positions")

    # ── Resume support ────────────────────────────────────────────────────────
    eval_targets = torch.full((N,), SENTINEL, dtype=torch.float32)
    todo_indices: list[int] = list(range(N))

    if resume and os.path.exists(output_path):
        print(f"  Resuming from {output_path} …")
        existing = torch.load(output_path, map_location="cpu", weights_only=False)
        if "eval_targets" in existing:
            prev = existing["eval_targets"]
            eval_targets[:len(prev)] = prev
            done = set(int(i) for i in torch.where(~torch.isnan(eval_targets))[0])
            todo_indices = [i for i in range(N) if i not in done]
            print(f"  {len(done):,} already evaluated — {len(todo_indices):,} remaining")
    elif "eval_targets" in data:
        prev = data["eval_targets"]
        eval_targets[:len(prev)] = prev
        done = set(int(i) for i in torch.where(~torch.isnan(eval_targets))[0])
        todo_indices = [i for i in range(N) if i not in done]
        print(f"  Found {len(done):,} existing eval_targets in input — {len(todo_indices):,} remaining")

    if not todo_indices:
        print("All positions already evaluated.")
        return

    # Pre-convert boards to numpy once (avoids repeated tensor→numpy in workers)
    boards_np = boards_tensor.numpy().astype(np.float32)

    out_data = {**data, "eval_targets": eval_targets}

    def _save():
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(out_data, output_path)

    total_sf_threads = num_workers * threads_per_worker
    print(f"\nStarting {num_workers} workers × {threads_per_worker} SF thread(s) = {total_sf_threads} total SF threads")
    print(f"Evaluating {len(todo_indices):,} positions (depth={depth}, time={time_limit}s) …\n")

    # Build lazy iterator of (index, board_numpy) pairs for todo positions only
    def job_iter():
        for i in todo_indices:
            yield i, boards_np[i]

    ctx = mp.get_context("spawn")   # spawn is safer with subprocess-backed engines
    completed = 0

    with ctx.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(stockfish_bin, depth, time_limit, threads_per_worker, hash_mb),
    ) as pool:
        try:
            with tqdm(total=len(todo_indices), unit=" pos") as pbar:
                for idx, val in pool.imap_unordered(_worker_eval, job_iter(), chunksize=8):
                    eval_targets[idx] = val
                    completed += 1
                    pbar.update(1)

                    if completed % save_every == 0:
                        _save()
        except KeyboardInterrupt:
            print("\nInterrupted — saving progress …")
            pool.terminate()
        finally:
            _save()

    n_nan = int(torch.isnan(eval_targets).sum())
    valid = eval_targets[~torch.isnan(eval_targets)]
    print(f"\nDone.  Evaluated: {completed:,}  NaN remaining: {n_nan}")
    print(f"Value stats: mean={valid.mean():.4f}  std={valid.std():.4f}")
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute Stockfish evaluations for a best-move dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",               required=True,          help="Input dataset (.pt)")
    parser.add_argument("--output",              required=True,          help="Output dataset (.pt)")
    parser.add_argument("--stockfish",           default="stockfish",    help="Stockfish binary path")
    parser.add_argument("--depth",               type=int,   default=15, help="Search depth per position")
    parser.add_argument("--time",                type=float, default=None, help="Time limit per position in seconds (overrides --depth)")
    parser.add_argument("--workers",             type=int,   default=4,  help="Number of parallel worker processes (each owns one SF engine)")
    parser.add_argument("--threads_per_worker",  type=int,   default=1,  help="Stockfish threads per worker")
    parser.add_argument("--hash",                type=int,   default=128, help="Stockfish hash table MB per worker")
    parser.add_argument("--resume",              action="store_true",    help="Resume from existing --output file")
    parser.add_argument("--save_every",          type=int,   default=2000, help="Checkpoint frequency (positions)")
    args = parser.parse_args()

    precompute_evals(
        input_path=args.input,
        output_path=args.output,
        stockfish_bin=args.stockfish,
        depth=args.depth,
        time_limit=args.time,
        num_workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        hash_mb=args.hash,
        resume=args.resume,
        save_every=args.save_every,
    )
