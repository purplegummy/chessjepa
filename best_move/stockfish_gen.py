"""
Generate Stockfish best-move labels for chess positions.

Positions are sourced from any combination of:
  1. A Lichess puzzle CSV  (--puzzles)
  2. A PGN file            (--pgn)
  3. Purely random legal positions synthesized on-the-fly (--random)

Output is a header-less CSV (FEN, best_move, eval_cp) compatible with
best_move/generate_dataset.py --no_header.

Usage examples
──────────────
  # 50k positions from the existing chess_chunks.zarr (no CSV needed!)
  python best_move/stockfish_gen.py \\
    --zarr data/chess_chunks.zarr \\
    --n 50000 --depth 15 \\
    --out best_move/data/stockfish_best_moves.csv

  # 50k positions from the Lichess puzzle CSV, depth 15
  python best_move/stockfish_gen.py \\
    --puzzles data/lichess_db_puzzle.csv \\
    --n 50000 --depth 15 \\
    --out best_move/data/stockfish_best_moves.csv

  # 100k positions from a PGN + append to existing CSV
  python best_move/stockfish_gen.py \\
    --pgn lichess_2024-01.pgn.zst \\
    --n 100000 --depth 12 \\
    --out best_move/data/stockfish_best_moves.csv --append

  # Mix: 20k from puzzles, 20k random, depth 18
  python best_move/stockfish_gen.py \\
    --puzzles data/lichess_db_puzzle.csv --random \\
    --n 40000 --depth 18 \\
    --out best_move/data/sf_labels.csv

  # Quick test (1000 positions)
  python best_move/stockfish_gen.py \\
    --puzzles data/lichess_db_puzzle.csv \\
    --n 1000 --depth 10

Requirements
────────────
  pip install chess stockfish tqdm
  (Stockfish binary must be on PATH or set with --stockfish)
"""

import argparse
import csv
import io
import os
import random
import sys

import chess
import chess.pgn
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Stockfish wrapper
# ─────────────────────────────────────────────────────────────────────────────

def make_engine(binary: str, threads: int = 1, hash_mb: int = 128):
    """Create a Stockfish subprocess via python-chess."""
    import chess.engine
    engine = chess.engine.SimpleEngine.popen_uci(binary)
    engine.configure({"Threads": threads, "Hash": hash_mb})
    return engine


def best_move(engine, board: chess.Board, depth: int, time_limit: float | None = None):
    """
    Query Stockfish for the best move.  Returns (uci_str, eval_cp) or (None, None).
    """
    import chess.engine
    limit = chess.engine.Limit(depth=depth) if time_limit is None else \
            chess.engine.Limit(time=time_limit)
    try:
        result = engine.analyse(board, limit)
        move   = result.get("pv", [None])[0]
        score  = result["score"].white()
        if score.is_mate():
            eval_cp = 30000 * (1 if score.mate() > 0 else -1)
        else:
            eval_cp = score.score(mate_score=30000)
        return move.uci() if move else None, eval_cp
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Board tensor → chess.Board
# ─────────────────────────────────────────────────────────────────────────────

# Piece order matches board_to_tensor in preprocess_pgn.py:
#   channels 0-5  : white (pawn, knight, bishop, rook, queen, king)
#   channels 6-11 : black (same order)
#   channel 12    : side to move (1=white, 0=black)
_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                chess.ROOK,  chess.QUEEN,  chess.KING]

def tensor_to_board(t: np.ndarray) -> chess.Board | None:
    """
    Convert a (17, 8, 8) float32 board tensor back to a chess.Board.
    Returns None if the board is invalid or game-over.
    """
    board = chess.Board(fen=None)   # empty board, no castling/en-passant
    board.clear()
    for ch, piece_type in enumerate(_PIECE_TYPES):
        mask = t[ch]   # white piece
        for row in range(8):
            for col in range(8):
                if mask[row, col] > 0.5:
                    sq = chess.square(col, row)
                    board.set_piece_at(sq, chess.Piece(piece_type, chess.WHITE))
    for ch, piece_type in enumerate(_PIECE_TYPES):
        mask = t[ch + 6]   # black piece
        for row in range(8):
            for col in range(8):
                if mask[row, col] > 0.5:
                    sq = chess.square(col, row)
                    board.set_piece_at(sq, chess.Piece(piece_type, chess.BLACK))
    board.turn = chess.WHITE  # encoding is color-invariant; board is always from current player's perspective
    try:
        board.status()   # raises ValueError on illegal position
        if board.is_game_over() or not list(board.legal_moves):
            return None
    except Exception:
        return None
    return board


# ─────────────────────────────────────────────────────────────────────────────
# Position sources
# ─────────────────────────────────────────────────────────────────────────────

def positions_from_zarr(zarr_path: str, n: int, seed: int = 42):
    """
    Stream up to n board positions directly from the chess_chunks.zarr store.
    Randomly samples chunks and time steps to avoid bias.
    """
    import zarr
    rng = np.random.default_rng(seed)

    store  = zarr.open(zarr_path, mode="r")
    if "boards" not in store:
        raise RuntimeError(f"No 'boards' array in {zarr_path}. Run preprocess_pgn.py first.")

    boards = store["boards"]   # (N, T, 17, 8, 8)
    N, T   = boards.shape[0], boards.shape[1]
    print(f"  Zarr: {N:,} chunks × {T} timesteps = {N*T:,} total boards")

    # Shuffle chunk order
    chunk_order = rng.permutation(N)
    count = 0

    for ci in chunk_order:
        if count >= n:
            break
        chunk = np.asarray(boards[ci])   # (T, 17, 8, 8)
        # Randomly shuffle timesteps within each chunk
        ts = rng.permutation(T)
        for t in ts:
            board = tensor_to_board(chunk[t])
            if board is not None:
                yield board
                count += 1
                if count >= n:
                    return

def positions_from_puzzles(csv_path: str, n: int, seed: int = 42):
    """
    Yield up to n board positions by replaying Lichess puzzle sequences.
    Each puzzle contributes up to (T-1) intermediate positions.
    """
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["FEN", "Moves"])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    count = 0
    for _, row in df.iterrows():
        board = chess.Board(row["FEN"])
        for uci in str(row["Moves"]).strip().split():
            try:
                board.push_uci(uci)
            except Exception:
                break
            if not board.is_game_over():
                yield board.copy()
                count += 1
                if count >= n:
                    return
        if count >= n:
            return


def positions_from_pgn(pgn_path: str, n: int, seed: int = 42, sample_rate: float = 0.1):
    """
    Stream a PGN (plain or .zst compressed) and randomly sample positions.
    sample_rate: probability of keeping each position (tune to get ~n quickly).
    """
    rng = random.Random(seed)
    count = 0

    def _open(path):
        if path.endswith(".zst"):
            import zstandard as zstd
            ctx = zstd.ZstdDecompressor()
            f = open(path, "rb")
            return io.TextIOWrapper(ctx.stream_reader(f), encoding="utf-8", errors="ignore")
        return open(path, encoding="utf-8", errors="ignore")

    with _open(pgn_path) as f:
        while count < n:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                if board.fullmove_number >= 5 and rng.random() < sample_rate:
                    if not board.is_game_over():
                        yield board.copy()
                        count += 1
                        if count >= n:
                            return


def random_positions(n: int, seed: int = 42, min_moves: int = 10, max_moves: int = 60):
    """
    Generate n random board positions by playing random legal moves from the start.
    """
    rng = random.Random(seed)
    count = 0
    while count < n:
        board = chess.Board()
        k = rng.randint(min_moves, max_moves)
        for _ in range(k):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                break
            board.push(rng.choice(legal))
        if not board.is_game_over() and len(list(board.legal_moves)) > 0:
            yield board.copy()
            count += 1


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Stockfish best-move labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Position sources (at least one required)
    src = parser.add_argument_group("Position sources (use one or more)")
    src.add_argument("--zarr",     default=None,
                     help="Path to chess_chunks.zarr (fastest — no CSV needed)")
    src.add_argument("--puzzles",   default=None,
                     help="Path to Lichess puzzle CSV (lichess_db_puzzle.csv)")
    src.add_argument("--pgn",       default=None,
                     help="Path to PGN or .pgn.zst file")
    src.add_argument("--random",    action="store_true",
                     help="Also generate random positions to reach --n")

    # Stockfish settings
    sf = parser.add_argument_group("Stockfish settings")
    sf.add_argument("--stockfish",  default="stockfish",
                    help="Path to Stockfish binary (or just 'stockfish' if on PATH)")
    sf.add_argument("--depth",      type=int, default=15,
                    help="Search depth per position")
    sf.add_argument("--time",       type=float, default=None,
                    help="Time limit per position in seconds (overrides --depth)")
    sf.add_argument("--threads",    type=int, default=1,
                    help="Stockfish threads")
    sf.add_argument("--hash",       type=int, default=128,
                    help="Stockfish hash table size in MB")

    # Volume / output
    parser.add_argument("--n",      type=int, required=True,
                        help="Number of positions to label")
    parser.add_argument("--out",    default="best_move/data/stockfish_best_moves.csv",
                        help="Output CSV path")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing CSV instead of overwriting")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--skip_mates", action="store_true",
                        help="Skip positions where Stockfish finds forced mate")

    args = parser.parse_args()

    if not args.puzzles and not args.pgn and not args.random and not args.zarr:
        parser.error("Specify at least one position source: --zarr, --puzzles, --pgn, or --random")

    print("=" * 55)
    print("  Stockfish Label Generator")
    print("=" * 55)
    print(f"  Target positions : {args.n:,}")
    print(f"  Stockfish depth  : {args.depth}  time={args.time}s")
    print(f"  Output           : {args.out}  (append={args.append})")
    print()

    # ── Build position stream ──────────────────────────────────────────────
    streams = []
    per_source = args.n

    if args.zarr:
        streams.append(positions_from_zarr(args.zarr, per_source, args.seed))
        print(f"  Source: Zarr chunks      ({args.zarr})")
    if args.puzzles:
        streams.append(positions_from_puzzles(args.puzzles, per_source, args.seed))
        print(f"  Source: Lichess puzzles  ({args.puzzles})")
    if args.pgn:
        streams.append(positions_from_pgn(args.pgn, per_source, args.seed))
        print(f"  Source: PGN              ({args.pgn})")
    if args.random:
        streams.append(random_positions(per_source, args.seed))
        print(f"  Source: Random positions")

    def interleave(*iters):
        """Round-robin across position sources."""
        iters = [iter(it) for it in iters]
        while iters:
            nxt = []
            for it in iters:
                try:
                    yield next(it)
                    nxt.append(it)
                except StopIteration:
                    pass
            iters = nxt

    position_stream = interleave(*streams)

    # ── Start Stockfish ────────────────────────────────────────────────────
    print(f"\n  Starting Stockfish ({args.stockfish}) …")
    try:
        engine = make_engine(args.stockfish, args.threads, args.hash)
    except Exception as e:
        print(f"  ERROR: Could not start Stockfish: {e}")
        print("  Install Stockfish and make sure it's on PATH, or use --stockfish /path/to/sf")
        sys.exit(1)

    # ── Output file ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mode = "a" if args.append else "w"
    written = 0
    skipped = 0

    try:
        with open(args.out, mode, newline="") as csvfile:
            writer = csv.writer(csvfile)
            pbar   = tqdm(total=args.n, unit=" pos", desc="labelling")

            for board in position_stream:
                if written >= args.n:
                    break

                uci, eval_cp = best_move(engine, board, args.depth, args.time)
                if uci is None:
                    skipped += 1
                    continue
                if args.skip_mates and abs(eval_cp) >= 30000:
                    skipped += 1
                    continue

                writer.writerow(["Generated", board.fen(), uci, eval_cp])
                written += 1
                pbar.update(1)

            pbar.close()
    finally:
        engine.quit()

    print(f"\n  Done!  Written: {written:,}  Skipped: {skipped:,}")
    print(f"  File  : {os.path.abspath(args.out)}")
    print()
    print("  Next step — build the decoder dataset:")
    print(f"    python best_move/generate_dataset.py --csv {args.out} --no_header")


if __name__ == "__main__":
    main()
