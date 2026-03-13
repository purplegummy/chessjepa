"""
Chess PGN → Zarr preprocessor for JEPA training.

Usage:
    python preprocess_pgn.py --input games.pgn.zst --output chess_chunks.zarr

Requirements:
    pip install chess zarr numpy zstandard tqdm
"""

import argparse
import io
import chess
import chess.pgn
import numpy as np
import zarr
import zstandard as zstd
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
CHUNK_SIZE   = 16        # moves per training chunk
MIN_ELO      = 1500      # minimum rating for both players
MIN_MOVES    = 20        # discard very short games
MIN_SECONDS  = 0       # 5-minute time control floor (0 = no filter)
BATCH_WRITE  = 10_000    # how many chunks to accumulate before writing to zarr
# ─────────────────────────────────────────────────────────────────────────────

PIECES = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN,  chess.KING,
]


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert a python-chess Board to a (18, 8, 8) float32 tensor."""
    # Flip board vertically if black to move for color invariance
    flip = board.turn == chess.BLACK
    t = np.zeros((18, 8, 8), dtype=np.float32)
    for i, piece in enumerate(PIECES):
        for sq in board.pieces(piece, chess.WHITE):
            r, c = sq // 8, sq % 8
            if flip:
                r = 7 - r
            t[i, r, c] = 1.0
        for sq in board.pieces(piece, chess.BLACK):
            r, c = sq // 8, sq % 8
            if flip:
                r = 7 - r
            t[i + 6, r, c] = 1.0
    # Turn is always white in the representation after flipping
    t[12] = 0.0
    # Castling rights: swap if flipped
    t[13] = float(board.has_kingside_castling_rights(chess.WHITE))
    t[14] = float(board.has_queenside_castling_rights(chess.WHITE))
    t[15] = float(board.has_kingside_castling_rights(chess.BLACK))
    t[16] = float(board.has_queenside_castling_rights(chess.BLACK))
    if flip:
        t[13], t[14] = t[14], t[13]  # swap kingside/queenside for white
        t[15], t[16] = t[16], t[15]  # swap for black
    # En passant
    if board.ep_square is not None:
        r, c = board.ep_square // 8, board.ep_square % 8
        if flip:
            r = 7 - r
        t[17, r, c] = 1.0
    return t


def game_to_chunks(game: chess.pgn.Game) -> list[np.ndarray]:
    """Convert a game to a list of (CHUNK_SIZE, 17, 8, 8) arrays."""
    board = game.board()
    positions = [board_to_tensor(board)]
    for move in game.mainline_moves():
        board.push(move)
        positions.append(board_to_tensor(board))

    chunks = []
    for i in range(0, len(positions) - CHUNK_SIZE, CHUNK_SIZE):
        chunks.append(np.stack(positions[i : i + CHUNK_SIZE]))  # (16,17,8,8)
    return chunks


def parse_elo(headers: chess.pgn.Headers, color: str) -> int:
    try:
        return int(headers.get(f"{color}Elo", 0))
    except ValueError:
        return 0


def parse_time_control(headers: chess.pgn.Headers) -> int:
    """Return base time in seconds, or 0 if unparseable."""
    tc = headers.get("TimeControl", "-")
    if tc in ("-", "?", ""):
        return 0
    try:
        return int(tc.split("+")[0])
    except ValueError:
        return 0


def should_keep(game: chess.pgn.Game) -> bool:
    h = game.headers
    if parse_elo(h, "White") < MIN_ELO or parse_elo(h, "Black") < MIN_ELO:
        return False
    if MIN_SECONDS and parse_time_control(h) < MIN_SECONDS:
        return False
    # count half-moves without replaying
    node, n = game, 0
    while node.variations:
        node = node.variations[0]
        n += 1
        if n >= MIN_MOVES * 2:
            break
    return n >= MIN_MOVES * 2


def open_pgn_stream(path: str):
    """Return a text-mode stream, handling .zst transparently."""
    if path.endswith(".zst"):
        fh  = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        raw  = dctx.stream_reader(fh)
        return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def main(input_path: str, output_path: str):
    print(f"Reading  : {input_path}")
    print(f"Writing  : {output_path}")
    print(f"Settings : elo≥{MIN_ELO}  moves≥{MIN_MOVES}  "
          f"tc≥{MIN_SECONDS}s  chunk={CHUNK_SIZE}")

    # zarr store — we'll resize as we go
    store  = zarr.open(output_path, mode="w")
    boards = store.require_dataset(
        "boards",
        shape=(0, CHUNK_SIZE, 18, 8, 8),
        chunks=(256, CHUNK_SIZE, 18, 8, 8),
        dtype="float32",
        compressor=zarr.Blosc(cname="lz4", clevel=3),
    )

    buffer       = []   # accumulate chunks before writing
    total_games  = 0
    kept_games   = 0
    total_chunks = 0

    with open_pgn_stream(input_path) as pgn:
        pbar = tqdm(desc="games", unit=" games")
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            total_games += 1
            pbar.update(1)

            if not should_keep(game):
                continue
            kept_games += 1

            for chunk in game_to_chunks(game):
                buffer.append(chunk)

            if len(buffer) >= BATCH_WRITE:
                arr = np.stack(buffer)                       # (B,16,17,8,8)
                boards.append(arr, axis=0)
                total_chunks += len(buffer)
                buffer = []
                pbar.set_postfix(kept=kept_games, chunks=total_chunks)

        # flush remainder
        if buffer:
            arr = np.stack(buffer)
            boards.append(arr, axis=0)
            total_chunks += len(buffer)

        pbar.close()

    print(f"\nDone.")
    print(f"  Total games parsed : {total_games:,}")
    print(f"  Games kept         : {kept_games:,}  "
          f"({100*kept_games/max(total_games,1):.1f}%)")
    print(f"  Chunks written     : {total_chunks:,}")
    print(f"  Final array shape  : {boards.shape}")
    size_gb = boards.nbytes / 1e9
    print(f"  Uncompressed size  : {size_gb:.1f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to .pgn or .pgn.zst")
    parser.add_argument("--output", default="chess_chunks.zarr",
                        help="Output zarr path (default: chess_chunks.zarr)")
    args = parser.parse_args()
    main(args.input, args.output)
