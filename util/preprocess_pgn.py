import argparse
import io
import chess
import chess.pgn
import numpy as np
import zarr
import zstandard as zstd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# -- config --------------------------------------------------------------------
CHUNK_SIZE   = 16        
MIN_ELO      = 1500      
MIN_MOVES    = 20        
MIN_SECONDS  = 0       
BATCH_WRITE  = 5000    # Number of games to process before flushing to Zarr
NUM_WORKERS  = mp.cpu_count() - 1 
# -----------------------------------------------------------------------------

PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert board to (18, 8, 8) uint8 tensor with color invariance."""
    flip = board.turn == chess.BLACK
    t = np.zeros((18, 8, 8), dtype=np.uint8)
    
    for i, piece in enumerate(PIECES):
        # White pieces (or 'Current Side' if flipped)
        for sq in board.pieces(piece, chess.WHITE):
            r, c = (7 - (sq // 8), sq % 8) if flip else (sq // 8, sq % 8)
            t[i, r, c] = 1
        # Black pieces (or 'Opponent' if flipped)
        for sq in board.pieces(piece, chess.BLACK):
            r, c = (7 - (sq // 8), sq % 8) if flip else (sq // 8, sq % 8)
            t[i + 6, r, c] = 1

    # Channels 13-16: Castling Rights (W-KS, W-QS, B-KS, B-QS)
    # If flipped, we swap White and Black rights entirely
    w_ks, w_qs = board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(chess.WHITE)
    b_ks, b_qs = board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(chess.BLACK)
    
    if flip:
        t[13], t[14], t[15], t[16] = int(b_ks), int(b_qs), int(w_ks), int(w_qs)
    else:
        t[13], t[14], t[15], t[16] = int(w_ks), int(w_qs), int(b_ks), int(b_qs)
        
    if board.ep_square is not None:
        sq = board.ep_square
        r, c = (7 - (sq // 8), sq % 8) if flip else (sq // 8, sq % 8)
        t[17, r, c] = 1
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
    positions = [board_to_tensor(board)]
    for move in game.mainline_moves():
        board.push(move)
        positions.append(board_to_tensor(board))

    if len(positions) < MIN_MOVES * 2: return []

    chunks = []
    for i in range(0, len(positions) - CHUNK_SIZE, CHUNK_SIZE):
        chunks.append(np.stack(positions[i : i + CHUNK_SIZE]))
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
        shape=(0, CHUNK_SIZE, 18, 8, 8),
        chunks=(128, CHUNK_SIZE, 18, 8, 8),
        dtype="uint8",
        compressor=zarr.Blosc(cname="lz4", clevel=5),
    )

    print(f"Starting extraction with {NUM_WORKERS} workers...")
    
    with open_pgn_stream(input_path) as pgn:
        game_gen = get_game_generator(pgn)
        
        with mp.Pool(NUM_WORKERS) as pool:
            # We process in large batches to keep the Zarr appends efficient
            batch_iterator = tqdm(pool.imap_unordered(process_game_string, game_gen, chunksize=50))
            
            chunk_buffer = []
            for game_chunks in batch_iterator:
                if game_chunks:
                    chunk_buffer.extend(game_chunks)
                
                if len(chunk_buffer) >= BATCH_WRITE:
                    boards.append(np.stack(chunk_buffer), axis=0)
                    chunk_buffer = []

            if chunk_buffer:
                boards.append(np.stack(chunk_buffer), axis=0)

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