"""
Generate and add an 'actions' array to an existing chess_chunks.zarr store.

The existing zarr only has 'boards' (shape: N, 16, 17, 8, 8).
This script derives the chess move (from_sq, to_sq) that produced each
board state by diffing consecutive positions within each chunk.

Board encoding (from preprocess_pgn.py)
─────────────────────────────────────────
  channels 0-5  : current-side pieces (pawn, knight, bishop, rook, queen, king)
  channels 6-11 : opponent pieces (same order)
  channels 12-15: castling rights (unused for move recovery)
  channel 16    : en passant square

Move recovery logic (per time step t > 0)
───────────────────────────────────────────
  Because boards are color-invariant, consecutive boards have OPPOSITE
  perspectives (b0 = current player's view, b1 = opponent's view after
  the move).  Before diffing, b1 is re-flipped into b0's frame via
  _to_b0_perspective(): swap channels 0-5 ↔ 6-11 and flip all rows.

  After alignment, piece channels (0-11) differ between board[t-1] and
  board[t].  The "from" square is where a piece DISAPPEARED (had 1, now 0).
  The "to"   square is where a piece APPEARED   (had 0, now 1).

  Captures: a piece disappears from one colour AND another disappears from
  the target square at the same time.  We still want to_sq correctly.

  Promotions: a pawn disappears from rank 7/1 and a queen (or other piece)
  appears on rank 8/0.  We map these to from/to squares directly.

  t = 0: the first board in a chunk is the position BEFORE any move in the
  chunk was played; there is no prior board to diff against.  We write
  from_sq = to_sq = 64 (null / no-move sentinel) for this step.

  Recovered squares are in b0's coordinate system (current player's frame).

Output format
─────────────
  zarr key: 'actions'
  dtype:    int16    (values 0-64, well within int16 range)
  shape:    (N, 16, 2)   — same N as 'boards', 16 time steps, 2 = (from, to)
  chunks:   (128, 16, 2) — matches the boards chunk structure

Usage
─────
  python util/generate_actions.py --zarr data/chess_chunks.zarr
  python util/generate_actions.py --zarr data/chess_chunks.zarr --workers 8 --batch 4096

  # Test on first 1000 chunks only:
  python util/generate_actions.py --zarr data/chess_chunks.zarr --max_chunks 1000

Options
───────
  --zarr        Path to zarr store  (default: data/chess_chunks.zarr)
  --workers     Parallel worker processes for board diffing (default: 8)
  --batch       Chunks processed per write batch (default: 4096)
  --max_chunks  Stop after N chunks (useful for testing, default: all)
  --overwrite   Re-generate actions even if the key already exists
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import zarr
from tqdm import tqdm


NULL_SQ: int = 64   # sentinel for "no move" (t=0 of each chunk)

# Columns in the 8×8 board correspond to squares laid out as
# board[row, col] = square 8*row + col  (same as python-chess convention)


def _to_b0_perspective(b1: np.ndarray) -> np.ndarray:
    """
    Re-flip b1 into b0's coordinate frame.

    Because boards are color-invariant, b0 and b1 are always in opposite
    perspectives.  To diff them correctly we need to undo b1's flip:
      - swap current-side channels (0-5) ↔ opponent channels (6-11)
      - flip all rows (row r → row 7-r)

    Only piece channels 0-11 matter for move recovery; the result is used
    exclusively inside _recover_move_from_diff.
    """
    aligned = np.empty_like(b1)
    aligned[0:6]  = b1[6:12, ::-1, :]   # opponent → current, flip rows
    aligned[6:12] = b1[0:6,  ::-1, :]   # current → opponent, flip rows
    aligned[12:]  = b1[12:]              # castling / ep unused here
    return aligned


def _recover_move_from_diff(b0: np.ndarray, b1: np.ndarray) -> tuple[int, int]:
    """
    Recover (from_sq, to_sq) from two consecutive board tensors.

    b0, b1 : (17, 8, 8) uint8  (b1 will be re-aligned to b0's perspective)

    Strategy
    ---------
    1. Re-flip b1 into b0's coordinate frame (_to_b0_perspective).
    2. Cast to int16 BEFORE subtracting — uint8 wraps 0-1=255, not -1.
    3. Only diff channels 0-5 (moving player's own pieces).
       - Captures / en passant: opponent's disappearing piece is in ch6-11,
         ignored entirely — no special-case logic needed.
    4. King first (ch5): if the king moved, use its squares directly.
       Collapsing channels with sum() for castling would give both the king
       and rook a count of 1, and argmax would pick the lower coordinate
       (the rook for queenside castling) — wrong.  Checking ch5 explicitly
       bypasses this.
    5. For all other moves: collapse channels, pick the square where the
       most pieces changed as from_sq / to_sq.

    Returns NULL_SQ for both if the move cannot be confidently recovered.
    """
    b1_aligned = _to_b0_perspective(b1)

    # Cast to int16 first — uint8 arithmetic wraps (0 - 1 = 255, not -1)
    mover_0 = b0[:6].astype(np.int16)       # (6, 8, 8) — moving player's pieces
    mover_1 = b1_aligned[:6].astype(np.int16)

    diff = mover_1 - mover_0                # (6, 8, 8)

    # ── King move / castling (channel 5) — check first ───────────────────
    king_diff  = diff[5]                    # (8, 8)
    king_from  = np.argwhere(king_diff < 0)
    king_to    = np.argwhere(king_diff > 0)

    if len(king_from) > 0 and len(king_to) > 0:
        from_rc = king_from[0]
        to_rc   = king_to[0]
    else:
        # ── All other moves ───────────────────────────────────────────────
        vanished_per_sq = (diff < 0).sum(axis=0)  # (8, 8)
        appeared_per_sq = (diff > 0).sum(axis=0)

        from_candidates = np.argwhere(vanished_per_sq > 0)
        to_candidates   = np.argwhere(appeared_per_sq > 0)

        if len(from_candidates) == 0 or len(to_candidates) == 0:
            return NULL_SQ, NULL_SQ

        from_rc = from_candidates[np.argmax(vanished_per_sq[from_candidates[:, 0],
                                                             from_candidates[:, 1]])]
        to_rc   = to_candidates[np.argmax(appeared_per_sq[to_candidates[:, 0],
                                                           to_candidates[:, 1]])]

    from_sq = int(from_rc[0]) * 8 + int(from_rc[1])
    to_sq   = int(to_rc[0])   * 8 + int(to_rc[1])

    return from_sq, to_sq


def _process_chunk_batch(chunk_boards: np.ndarray) -> np.ndarray:
    """
    Recover actions for a batch of chunks.

    chunk_boards : (B, T, 17, 8, 8) uint8
    returns      : (B, T, 2)        int16    — (from_sq, to_sq) at each time step
    """
    B, T = chunk_boards.shape[:2]
    actions = np.full((B, T, 2), NULL_SQ, dtype=np.int16)

    for b in range(B):
        for t in range(1, T):
            from_sq, to_sq = _recover_move_from_diff(
                chunk_boards[b, t - 1], chunk_boards[b, t]
            )
            actions[b, t, 0] = from_sq
            actions[b, t, 1] = to_sq
        # t=0 stays as NULL_SQ (no prior board to diff against)

    return actions


def _process_chunk_batch_worker(args):
    """Multiprocessing-safe wrapper."""
    chunk_boards, = args
    return _process_chunk_batch(chunk_boards)


def generate_actions(
    zarr_path: str,
    batch_size: int = 4096,
    num_workers: int = 8,
    max_chunks: int | None = None,
    overwrite: bool = False,
):
    print("=" * 60)
    print("  Generating actions array for zarr store")
    print("=" * 60)
    print(f"  Zarr path  : {zarr_path}")
    print(f"  Workers    : {num_workers}")
    print(f"  Batch size : {batch_size:,}")

    abs_path = os.path.abspath(zarr_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"\n  Zarr store not found at: {abs_path}\n"
            f"  Make sure you ran preprocess_pgn.py first and pass the correct path.\n"
            f"  Example:  python util/generate_actions.py --zarr /full/path/to/chess_chunks.zarr"
        )

    store = zarr.open(abs_path, mode="a")   # append mode — keep existing data

    if "boards" not in store:
        raise RuntimeError(f"No 'boards' array in {zarr_path}")

    boards = store["boards"]   # (N, T, 17, 8, 8)
    N, T = boards.shape[0], boards.shape[1]

    if max_chunks is not None:
        N = min(N, max_chunks)
        print(f"  Max chunks : {N:,}  (limited by --max_chunks)")

    print(f"  Total chunks: {N:,}  (T={T})")

    if "actions" in store and not overwrite:
        existing = store["actions"]
        if existing.shape[0] >= N:
            print(f"  'actions' already exists with shape {existing.shape}. "
                  f"Use --overwrite to regenerate.")
            return
        else:
            print(f"  'actions' exists but has only {existing.shape[0]:,} rows; "
                  f"will regenerate from scratch.")
            del store["actions"]

    # Create empty actions array
    chunk_size = boards.chunks[0]
    actions = store.require_dataset(
        "actions",
        shape=(N, T, 2),
        chunks=(chunk_size, T, 2),
        dtype="int16",
        compressor=zarr.Blosc(cname="lz4", clevel=3),
        overwrite=True,
    )

    t0 = time.time()
    total_written = 0
    pbar = tqdm(total=N, desc="chunks", unit=" chunks")

    with Pool(processes=num_workers) as pool:
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            B = end - start

            # Load batch from zarr (decompresses LZ4 chunks)
            batch_boards = np.asarray(boards[start:end])   # (B, T, 17, 8, 8)

            # Split into per-worker sub-batches
            worker_batch = max(1, B // num_workers)
            sub_batches = [
                (batch_boards[i:i + worker_batch],)
                for i in range(0, B, worker_batch)
            ]

            # Process in parallel
            results = pool.map(_process_chunk_batch_worker, sub_batches)
            batch_actions = np.concatenate(results, axis=0)   # (B, T, 2)

            actions[start:end] = batch_actions
            total_written += B
            pbar.update(B)

    pbar.close()

    elapsed = time.time() - t0
    rate = total_written / elapsed
    print(f"\n  Done!  Wrote {total_written:,} chunks in {elapsed:.1f}s  ({rate:.0f} chunks/s)")
    print(f"  actions.shape : {actions.shape}")
    print(f"  actions.dtype : {actions.dtype}")

    # Quick sanity check on a random sample
    sample_idx = 42
    if sample_idx < total_written:
        sample = np.asarray(actions[sample_idx])
        print(f"\n  Sanity check (chunk {sample_idx}):")
        for t in range(min(5, T)):
            f, to = sample[t, 0], sample[t, 1]
            if f == NULL_SQ:
                print(f"    t={t}: null move (expected for t=0)")
            else:
                f_sq = chess_sq_name(int(f))
                t_sq = chess_sq_name(int(to))
                print(f"    t={t}: {f_sq} → {t_sq}  (indices {f}, {to})")


def chess_sq_name(sq: int) -> str:
    """Convert a square index (0-63) to algebraic notation e.g. 12 → 'e2'."""
    if sq == NULL_SQ:
        return "null"
    col = sq % 8
    row = sq // 8
    return "abcdefgh"[col] + str(row + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add an 'actions' array to chess_chunks.zarr"
    )
    parser.add_argument("--zarr",       default="data/chess_chunks.zarr",
                        help="Path to zarr store (default: data/chess_chunks.zarr)")
    parser.add_argument("--workers",    type=int, default=8,
                        help="Parallel worker processes (default: 8)")
    parser.add_argument("--batch",      type=int, default=4096,
                        help="Chunks per write batch (default: 4096)")
    parser.add_argument("--max_chunks", type=int, default=None,
                        help="Stop after N chunks (for testing)")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Overwrite existing actions array")
    args = parser.parse_args()

    generate_actions(
        zarr_path=args.zarr,
        batch_size=args.batch,
        num_workers=args.workers,
        max_chunks=args.max_chunks,
        overwrite=args.overwrite,
    )
