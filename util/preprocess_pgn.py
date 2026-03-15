import argparse
import io
import signal
import sys
import chess
import chess.pgn
import numpy as np
import zarr
import zstandard as zstd
from tqdm import tqdm
import multiprocessing as mp

# -- config --------------------------------------------------------------------
CHUNK_SIZE   = 16        
MIN_ELO      = 1500      
MIN_MOVES    = 20        
MIN_SECONDS  = 0       
BATCH_WRITE  = 5000    # Number of games to process before flushing to Zarr
NUM_WORKERS  = mp.cpu_count() - 1 
# -----------------------------------------------------------------------------

PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

def board_to_tensor(board: chess.Board, force_flip: bool | None = None) -> np.ndarray:
    """Convert board to (17, 8, 8) uint8 tensor with color invariance.

    force_flip: if provided, overrides the automatic flip-by-turn logic.
                Pass the current player's flip value to keep all frames in a
                history sequence oriented to the same perspective.
    """
    flip = board.turn == chess.BLACK if force_flip is None else force_flip
    t = np.zeros((17, 8, 8), dtype=np.uint8)

    us   = chess.BLACK if flip else chess.WHITE
    them = chess.WHITE if flip else chess.BLACK

    for i, piece in enumerate(PIECES):
        # Current side pieces (channels 0-5)
        sqs = list(board.pieces(piece, us))
        if sqs:
            sqs = np.array(sqs)
            rows = (7 - sqs // 8) if flip else (sqs // 8)
            cols = sqs % 8
            t[i, rows, cols] = 1

        # Opponent pieces (channels 6-11)
        sqs = list(board.pieces(piece, them))
        if sqs:
            sqs = np.array(sqs)
            rows = (7 - sqs // 8) if flip else (sqs // 8)
            cols = sqs % 8
            t[i + 6, rows, cols] = 1

    # Channels 12-15: Castling Rights (Current-KS, Current-QS, Opponent-KS, Opponent-QS)
    w_ks = board.has_kingside_castling_rights(chess.WHITE)
    w_qs = board.has_queenside_castling_rights(chess.WHITE)
    b_ks = board.has_kingside_castling_rights(chess.BLACK)
    b_qs = board.has_queenside_castling_rights(chess.BLACK)

    if flip:
        t[12], t[13], t[14], t[15] = int(b_ks), int(b_qs), int(w_ks), int(w_qs)
    else:
        t[12], t[13], t[14], t[15] = int(w_ks), int(w_qs), int(b_ks), int(b_qs)

    if board.ep_square is not None:
        sq = board.ep_square
        r = (7 - sq // 8) if flip else (sq // 8)
        t[16, r, sq % 8] = 1

    return t
def process_game_string(game_str: str) -> list[np.ndarray]:
    """Worker function: parses a single PGN string into a list of chunks."""
    game = chess.pgn.read_game(io.StringIO(game_str))
    if game is None: return []
    
    # Filter logic inside worker to save main process time
    h = game.headers
    try:
        white_elo = int(h.get("WhiteElo", 0))
        black_elo = int(h.get("BlackElo", 0))
        if white_elo < MIN_ELO or black_elo < MIN_ELO: return []
    except ValueError: return []

    board = game.board()
    boards = [board.copy()]
    for move in game.mainline_moves():
        board.push(move)
        boards.append(board.copy())

    if len(boards) < MIN_MOVES * 2: return []

    chunks = []
    for i in range(0, len(boards) - CHUNK_SIZE, CHUNK_SIZE):
        chunk_boards = boards[i : i + CHUNK_SIZE]
        # All frames encoded from the last board's perspective for spatial consistency
        flip = chunk_boards[-1].turn == chess.BLACK
        frames = [board_to_tensor(b, force_flip=flip) for b in chunk_boards]
        chunks.append(np.stack(frames))
    return chunks

def get_game_generator(pgn_handle):
    """Generator that yields individual game strings from the PGN file."""
    lines = []
    for line in pgn_handle:
        lines.append(line)
        if line.startswith("[Event ") and len(lines) > 1:
            yield "".join(lines[:-1])
            lines = [line]
    yield "".join(lines)

def main(input_path: str, output_path: str):
    store = zarr.open(output_path, mode="w")
    boards = store.require_dataset(
        "boards",
        shape=(0, CHUNK_SIZE, 17, 8, 8),
        chunks=(128, CHUNK_SIZE, 17, 8, 8),
        dtype="uint8",
        compressor=zarr.Blosc(cname="lz4", clevel=5),
    )

    print(f"Starting extraction with {NUM_WORKERS} workers...")

    games_seen = 0
    games_kept = 0
    chunks_saved = 0
    chunk_buffer = []

    def flush_and_exit(sig, frame):
        if chunk_buffer:
            print(f"\nInterrupted — flushing {len(chunk_buffer)} buffered chunks to disk...")
            boards.append(np.stack(chunk_buffer), axis=0)
            print(f"Saved. Total chunks on disk: {boards.shape[0]:,}")
        else:
            print(f"\nInterrupted — nothing buffered, {boards.shape[0]:,} chunks already on disk.")
        sys.exit(0)

    signal.signal(signal.SIGINT, flush_and_exit)
    signal.signal(signal.SIGTERM, flush_and_exit)

    with open_pgn_stream(input_path) as pgn:
        game_gen = get_game_generator(pgn)

        with mp.Pool(NUM_WORKERS) as pool:
            pbar = tqdm(
                pool.imap_unordered(process_game_string, game_gen, chunksize=50),
                desc="games",
                unit=" games",
                dynamic_ncols=True,
            )

            for game_chunks in pbar:
                games_seen += 1
                if game_chunks:
                    games_kept += 1
                    chunk_buffer.extend(game_chunks)

                if len(chunk_buffer) >= BATCH_WRITE:
                    boards.append(np.stack(chunk_buffer), axis=0)
                    chunks_saved += len(chunk_buffer)
                    chunk_buffer = []

                pbar.set_postfix(
                    kept=games_kept,
                    chunks=chunks_saved + len(chunk_buffer),
                    refresh=False,
                )

            if chunk_buffer:
                boards.append(np.stack(chunk_buffer), axis=0)
                chunks_saved += len(chunk_buffer)
                chunk_buffer = []

    print(f"Games seen  : {games_seen:,}")
    print(f"Games kept  : {games_kept:,}  ({100*games_kept/max(games_seen,1):.1f}%)")
    print(f"Chunks saved: {chunks_saved:,}")
    print(f"Final dataset shape: {boards.shape}")

def open_pgn_stream(path: str):
    if path.endswith(".zst"):
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        return io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="chess_chunks.zarr")
    args = parser.parse_args()
    main(args.input, args.output)