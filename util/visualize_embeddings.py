"""
Chess V-JEPA — Latent Space Visualization

Loads a trained JEPA checkpoint, passes random board positions through the 
Context Encoder to get 256-dimensional embeddings, and visualizes them using 
PCA or UMAP in an interactive HTML plot.

Hovering over a point (which represents an 8x8 chess board state) will show 
metadata, and clicking a point will attempt to open a Lichess analysis board 
for that exact position's FEN.
"""

import argparse
import os
import random
import webbrowser

import chess
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import zarr
from tqdm import tqdm

from model.jepa import ChessJEPA
from util.config import JEPAConfig
from util.dataset import ChessChunkDataset

try:
    from sklearn.decomposition import PCA
except ImportError:
    print("Please install scikit-learn: pip install scikit-learn")
    exit(1)

try:
    import umap
except ImportError:
    print("Please install umap-learn: pip install umap-learn")
    exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Reverse mapping: Tensor (17, 8, 8) → chess.Board
# ─────────────────────────────────────────────────────────────────────────────
# We need this to reconstruct the FEN string from the raw tensor so we can 
# generate a Lichess link.
PIECES = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN,  chess.KING,
]

def tensor_to_board(t: torch.Tensor | np.ndarray) -> chess.Board:
    """Reconstruct a chess.Board from the (17, 8, 8) tensor."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
        
    board = chess.Board(None) # empty board
    
    # Planes 0-5: White pieces
    # Planes 6-11: Black pieces
    for i, piece in enumerate(PIECES):
        white_squares = np.where(t[i] == 1.0)
        for r, c in zip(*white_squares):
            sq = r * 8 + c
            board.set_piece_at(sq, chess.Piece(piece, chess.WHITE))
            
        black_squares = np.where(t[i+6] == 1.0)
        for r, c in zip(*black_squares):
            sq = r * 8 + c
            board.set_piece_at(sq, chess.Piece(piece, chess.BLACK))
            
    # Plane 12: Turn
    board.turn = bool(t[12, 0, 0] > 0.5)
    
    # (Castling rights could be reconstructed from planes 13-16, 
    # but for a simple visual inspection FEN, pieces + turn is usually enough)
    
    return board


def board_to_lichess_url(board: chess.Board) -> str:
    """Convert a board to a clickable Lichess analysis URL."""
    fen = board.fen().replace(" ", "_")
    return f"https://lichess.org/analysis/{fen}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Extract Embeddings
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    ckpt_path: str, 
    zarr_path: str, 
    num_samples: int = 2000, 
    device: str = "cpu"
):
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg: JEPAConfig = checkpoint["config"]
    
    # Initialize Context Encoder only
    model = ChessJEPA(**cfg.encoder_kwargs).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    encoder = model.context_encoder
    
    print(f"Loading dataset: {zarr_path}")
    dataset = ChessChunkDataset(zarr_path, split="val")
    
    print(f"Extracting {num_samples} random board embeddings...")
    embeddings = []
    metadata = []
    
    for _ in tqdm(range(num_samples)):
        # Pick a random chunk and a random position within that chunk
        chunk_idx = random.randint(0, len(dataset) - 1)
        chunk = dataset[chunk_idx] # (16, 17, 8, 8)
        pos_idx = random.randint(0, 15)
        
        board_tensor = chunk[pos_idx] # (17, 8, 8)
        
        # Add batch and time dimensions: (1, 1, 17, 8, 8)
        x = board_tensor.unsqueeze(0).unsqueeze(0).to(device)
        
        # Forward pass through encoder -> (1, 1, 256)
        latent = encoder(x)
        embeddings.append(latent.squeeze().cpu().numpy())
        
        # Reconstruct board for metadata
        board = tensor_to_board(board_tensor)
        
        # Calculate some simple metadata to color the plot by
        pieces_count = len(board.piece_map())
        phase = "Opening" if pieces_count > 28 else "Endgame" if pieces_count < 14 else "Middlegame"
        score_heuristic = sum(
            [9 if p.piece_type == chess.QUEEN else 5 if p.piece_type == chess.ROOK else 3 
             for p in board.piece_map().values() if p.color == chess.WHITE]
        ) - sum(
            [9 if p.piece_type == chess.QUEEN else 5 if p.piece_type == chess.ROOK else 3 
             for p in board.piece_map().values() if p.color == chess.BLACK]
        )
        
        metadata.append({
            "FEN": board.fen(),
            "Lichess_URL": board_to_lichess_url(board),
            "Turn": "White" if board.turn else "Black",
            "Piece_Count": pieces_count,
            "Phase": phase,
            "Material_Advantage": score_heuristic
        })

    return np.stack(embeddings), pd.DataFrame(metadata)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_interactive(embeddings: np.ndarray, df: pd.DataFrame, method: str = "umap", out_file: str = "embeddings.html"):
    print(f"Reducing dimensions from {embeddings.shape[1]} to 2 using {method.upper()}...")
    
    if method.lower() == "pca":
        reducer = PCA(n_components=2)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        
    reduced = reducer.fit_transform(embeddings)
    
    df["x"] = reduced[:, 0]
    df["y"] = reduced[:, 1]
    
    print("Generating interactive Plotly graph...")
    
    # We color by Game Phase, and size by Piece Count to reveal latent structure
    fig = px.scatter(
        df, 
        x="x", 
        y="y", 
        color="Phase",
        symbol="Turn",
        hover_data=["Material_Advantage", "Piece_Count", "FEN"],
        custom_data=["Lichess_URL"],
        title=f"Chess V-JEPA Latent Space ({method.upper()})",
        template="plotly_dark"
    )
    
    # Add JS to make the points clickable (opens Lichess URL)
    # Plotly's Python API doesn't have native click-to-URL, so we inject a little script
    fig.update_layout(clickmode='event+select')
    
    html_content = fig.to_html(include_plotlyjs="cdn", full_html=True)
    
    # Inject click handler
    js_snippet = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var plots = document.getElementsByClassName('plotly-graph-div');
        if (plots.length > 0) {
            plots[0].on('plotly_click', function(data) {
                var url = data.points[0].customdata[0];
                window.open(url, '_blank');
            });
        }
    });
    </script>
    """
    html_content = html_content.replace('</body>', f'{js_snippet}</body>')
    
    with open(out_file, "w") as f:
        f.write(html_content)
        
    print(f"Saved interactive plot to {out_file}")
    
    # Attempt to open it automatically
    full_path = os.path.abspath(out_file)
    print(f"Opening {full_path} in browser...")
    webbrowser.open(f"file://{full_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--zarr", default="data/chess_chunks.zarr", help="Path to zarr dataset")
    parser.add_argument("--samples", type=int, default=1500, help="Number of boards to sample")
    parser.add_argument("--method", choices=["pca", "umap"], default="umap", help="Dimensionality reduction method")
    parser.add_argument("--out", default="latent_space.html", help="Output HTML file name")
    args = parser.parse_args()
    
    # Fallback to CPU if MPS/CUDA not available, since we are just inferencing a few samples
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    embeddings, df = extract_embeddings(args.ckpt, args.zarr, args.samples, device)
    plot_interactive(embeddings, df, args.method, args.out)


if __name__ == "__main__":
    main()
