"""
Chess V-JEPA — Puzzle Similarity Explorer

Encodes Lichess puzzles through the frozen JEPA context encoder and visualizes
the embedding space. Similar tactical patterns should cluster together.

Usage:
  # Download the Lichess puzzle CSV first:
  #   curl -O https://database.lichess.org/lichess_db_puzzle.csv.zst
  #   unzstd lichess_db_puzzle.csv.zst

  python util/puzzle_similarity.py \\
    --ckpt checkpoints/checkpoint_epoch0050.pt \\
    --puzzles lichess_db_puzzle.csv \\
    --samples 3000

Color modes (switchable in-browser):
  - Theme   (fork, pin, mate, sacrifice, ...)
  - Rating  (puzzle difficulty)
  - Length  (number of moves)

Clicking a point highlights its 10 nearest neighbors in embedding space.
"""

import argparse
import json
import os
import sys
import subprocess
import webbrowser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from tqdm import tqdm

from model.jepa import ChessJEPA
from util.config import JEPAConfig
from util.preprocess_pgn import board_to_tensor

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_distances
except ImportError:
    print("Please install scikit-learn: pip install scikit-learn")
    exit(1)

try:
    import umap
except ImportError:
    print("Please install umap-learn: pip install umap-learn")
    exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Color palettes
# ─────────────────────────────────────────────────────────────────────────────

THEME_PALETTE = {
    "fork":             "#FF5722",
    "pin":              "#2196F3",
    "skewer":           "#9C27B0",
    "discoveredAttack": "#FF9800",
    "doubleCheck":      "#F44336",
    "backRank":         "#795548",
    "mateIn1":          "#E91E63",
    "mateIn2":          "#C2185B",
    "mateIn3":          "#880E4F",
    "mate":             "#AD1457",
    "sacrifice":        "#4CAF50",
    "quietMove":        "#607D8B",
    "xRayAttack":       "#00BCD4",
    "zugzwang":         "#FFEB3B",
    "deflection":       "#8BC34A",
    "attraction":       "#FF4081",
    "interference":     "#7C4DFF",
    "clearance":        "#00E5FF",
    "trappedPiece":     "#FFAB40",
    "endgame":          "#A1887F",
    "opening":          "#81C784",
    "middlegame":       "#64B5F6",
    "other":            "#888888",
}

RATING_PALETTE = {
    "Beginner (<1200)":     "#4CAF50",
    "Intermediate (1200-1600)": "#FFEB3B",
    "Advanced (1600-2000)": "#FF5722",
    "Expert (>2000)":       "#9C27B0",
}

LENGTH_PALETTE = {
    "1 move":   "#4CAF50",
    "2 moves":  "#2196F3",
    "3 moves":  "#FF9800",
    "4 moves":  "#F44336",
    "5+ moves": "#9C27B0",
}

# Theme display priority (most informative first)
THEME_PRIORITY = [
    "mateIn1", "mateIn2", "mateIn3", "mate",
    "fork", "pin", "skewer", "discoveredAttack", "doubleCheck",
    "sacrifice", "deflection", "attraction", "interference", "clearance",
    "xRayAttack", "trappedPiece", "zugzwang", "quietMove", "backRank",
    "endgame", "middlegame", "opening",
]


# ─────────────────────────────────────────────────────────────────────────────
# Puzzle parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_primary_theme(themes_str: str) -> str:
    themes = set(themes_str.strip().split())
    for t in THEME_PRIORITY:
        if t in themes:
            return t
    parts = themes_str.strip().split()
    return parts[0] if parts else "other"


def get_rating_bucket(rating: int) -> str:
    if rating < 1200:   return "Beginner (<1200)"
    elif rating < 1600: return "Intermediate (1200-1600)"
    elif rating < 2000: return "Advanced (1600-2000)"
    else:               return "Expert (>2000)"


def puzzle_to_board_tensors(fen: str, moves_str: str, max_len: int = 16) -> list[np.ndarray]:
    """
    Play out puzzle moves and return a list of (17,8,8) board tensors.
    In Lichess puzzles, the first move is the opponent's setup move;
    subsequent moves are the solution. We encode all positions.
    """
    board = chess.Board(fen)
    tensors = [board_to_tensor(board)]
    for move_uci in moves_str.strip().split():
        if len(tensors) >= max_len:
            break
        try:
            board.push_uci(move_uci)
            tensors.append(board_to_tensor(board))
        except Exception:
            break
    return tensors


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_puzzles(csv_path: str, num_samples: int, seed: int = 42) -> pd.DataFrame:
    print(f"Loading puzzles from {csv_path}...")
    df = pd.read_csv(
        csv_path,
        usecols=["PuzzleId", "FEN", "Moves", "Rating", "Themes", "GameUrl"],
    )
    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=seed).reset_index(drop=True)
    print(f"Sampled {len(df):,} puzzles")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_puzzle_embeddings(ckpt_path: str, puzzles_df: pd.DataFrame, device: str):
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: JEPAConfig = checkpoint["config"]

    model = ChessJEPA(
        encoder_kwargs=cfg.encoder_kwargs,
        predictor_kwargs=cfg.predictor_kwargs,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    encoder = model.context_encoder

    print("Encoding puzzle sequences...")
    embeddings = []
    metadata = []

    for row in tqdm(puzzles_df.itertuples(), total=len(puzzles_df)):
        try:
            board_arrays = puzzle_to_board_tensors(row.FEN, row.Moves, max_len=cfg.seq_len)
        except Exception:
            continue

        # Encode the full sequence at once: (1, T, 17, 8, 8)
        # The encoder uses spatiotemporal attention across all T positions,
        # which is much better than encoding each board independently.
        seq = torch.from_numpy(np.stack(board_arrays)).unsqueeze(0).to(device)
        latents = encoder(seq)          # (1, T, embed_dim)
        puzzle_emb = latents.squeeze(0).mean(dim=0).cpu().numpy()  # mean over T → (embed_dim,)
        embeddings.append(puzzle_emb)

        themes_str     = str(row.Themes)
        primary_theme  = get_primary_theme(themes_str)
        rating         = int(row.Rating) if pd.notna(row.Rating) else 1500
        moves_list     = str(row.Moves).strip().split()
        # Lichess: move[0] = opponent, moves[1:] = solution
        n_puzzle_moves = max(0, len(moves_list) - 1)
        if n_puzzle_moves <= 4:
            length_label = f"{n_puzzle_moves} move{'s' if n_puzzle_moves != 1 else ''}"
        else:
            length_label = "5+ moves"

        metadata.append({
            "PuzzleId":      row.PuzzleId,
            "FEN":           row.FEN,
            "Moves":         str(row.Moves),
            "Lichess_URL":   f"https://lichess.org/training/{row.PuzzleId}",
            "Themes":        themes_str,
            "Primary_Theme": primary_theme,
            "Rating":        rating,
            "Rating_Bucket": get_rating_bucket(rating),
            "N_Moves":       n_puzzle_moves,
            "Length_Label":  length_label,
        })

    return np.stack(embeddings), pd.DataFrame(metadata)


# ─────────────────────────────────────────────────────────────────────────────
# k-NN
# ─────────────────────────────────────────────────────────────────────────────

def compute_knn(embeddings: np.ndarray, k: int = 10) -> list[list[int]]:
    print(f"Computing {k}-NN for {len(embeddings)} puzzles...")
    dists = cosine_distances(embeddings)
    np.fill_diagonal(dists, np.inf)
    return np.argsort(dists, axis=1)[:, :k].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def categorical_colors(series: pd.Series, palette: dict) -> list[str]:
    return [palette.get(v, palette.get("other", "#888888")) for v in series]


def _open_in_browser(path: str):
    abs_path = os.path.abspath(path)
    print(f"Opening {abs_path} ...")
    if sys.platform == "darwin":
        subprocess.run(["open", abs_path])
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", abs_path])
    else:
        webbrowser.open(f"file://{abs_path}")


def plot_puzzles(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    method: str = "umap",
    out_file: str = "puzzle_space.html",
):
    print(f"Reducing dimensions ({method.upper()})...")
    reducer = (
        PCA(n_components=2) if method == "pca"
        else umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    )
    reduced = reducer.fit_transform(embeddings)
    df["x"], df["y"] = reduced[:, 0], reduced[:, 1]

    theme_colors  = categorical_colors(df["Primary_Theme"], THEME_PALETTE)
    rating_colors = categorical_colors(df["Rating_Bucket"], RATING_PALETTE)
    length_colors = categorical_colors(df["Length_Label"],  LENGTH_PALETTE)

    all_colors = {
        "Theme":  theme_colors,
        "Rating": rating_colors,
        "Length": length_colors,
    }
    legends = {
        "Theme":  THEME_PALETTE,
        "Rating": RATING_PALETTE,
        "Length": LENGTH_PALETTE,
    }

    knn = compute_knn(embeddings, k=10)

    hover = [
        f"<b>{row.Primary_Theme}</b> | Rating {row.Rating}<br>"
        f"Moves: {row.N_Moves} | Themes: {row.Themes}<br>"
        f"ID: {row.PuzzleId}"
        for row in df.itertuples()
    ]

    custom = list(zip(
        df["Lichess_URL"].tolist(),
        df["Primary_Theme"].tolist(),
        df["Rating"].tolist(),
        df["Themes"].tolist(),
        df["FEN"].tolist(),
        df["Moves"].tolist() if "Moves" in df.columns else [""] * len(df),
    ))

    fig = go.Figure(go.Scatter(
        x=df["x"].tolist(), y=df["y"].tolist(),
        mode="markers",
        marker=dict(color=theme_colors, size=5, opacity=0.80, line=dict(width=0)),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        customdata=custom,
    ))
    fig.update_layout(
        title=f"Chess V-JEPA Puzzle Space ({method.upper()})",
        template="plotly_dark", showlegend=False,
        margin=dict(t=50, b=20, l=20, r=260),
        xaxis=dict(showticklabels=False, title=""),
        yaxis=dict(showticklabels=False, title=""),
    )

    color_data_js = json.dumps(all_colors)
    legends_js    = json.dumps(legends)
    knn_js        = json.dumps(knn)

    # Build legend HTML for themes (only show themes that appear in the data)
    present_themes = df["Primary_Theme"].unique().tolist()

    options_html = "\n".join([
        '<option value="Theme">Puzzle Theme</option>',
        '<option value="Rating">Difficulty Rating</option>',
        '<option value="Length">Puzzle Length</option>',
    ])

    injection = """
<link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>

<style>
  #vis-panel {
    position: fixed; top: 16px; right: 16px; width: 240px;
    background: #1a1a2e; border: 1px solid #333; border-radius: 10px;
    padding: 14px; font-family: sans-serif; color: #ddd; z-index: 9999;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
  }
  #vis-panel h3 { margin: 0 0 10px; font-size: 13px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
  .mode-select {
    width: 100%; background: #2a2a42; color: #ddd;
    border: 1px solid #444; border-radius: 6px; padding: 6px 8px;
    font-size: 13px; cursor: pointer;
  }
  #legend-box { margin-top: 12px; font-size: 11px; max-height: 300px; overflow-y: auto; }
  .leg-item { display: flex; align-items: center; gap: 7px; margin: 4px 0; }
  .leg-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .action-btn {
    margin-top: 6px; width: 100%; background: #2a2a42; color: #aaa;
    border: 1px solid #444; border-radius: 6px; padding: 5px 8px;
    font-size: 12px; cursor: pointer; display: none;
  }
  .action-btn:hover { background: #3a3a52; color: #ddd; }

  /* Board modal */
  #board-overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.6); z-index: 10000;
  }
  #board-modal {
    position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
    background: #1a1a2e; border: 1px solid #444; border-radius: 12px;
    padding: 16px; z-index: 10001; width: 400px;
    font-family: sans-serif; color: #ddd;
    box-shadow: 0 8px 40px rgba(0,0,0,0.8);
  }
  #modal-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }
  #modal-title  { font-size: 14px; font-weight: bold; }
  #modal-themes { font-size: 11px; color: #888; margin-top: 3px; }
  #close-modal  { background: none; border: none; color: #888; font-size: 22px; cursor: pointer; line-height: 1; padding: 0; }
  #close-modal:hover { color: #ddd; }
  #board-wrap   { width: 368px; margin: 0 auto; }
  #move-nav     { display: flex; align-items: center; gap: 8px; margin-top: 10px; }
  .nav-btn {
    background: #2a2a42; border: 1px solid #444; color: #ddd; border-radius: 6px;
    padding: 5px 12px; cursor: pointer; font-size: 16px;
  }
  .nav-btn:hover { background: #3a3a52; }
  .nav-btn:disabled { opacity: 0.3; cursor: default; }
  #move-label   { flex: 1; text-align: center; font-size: 12px; color: #aaa; }
  #lichess-btn  {
    display: block; margin-top: 10px; text-align: center;
    background: #2a2a42; border: 1px solid #555; color: #64B5F6;
    border-radius: 6px; padding: 6px; font-size: 12px; text-decoration: none;
  }
  #lichess-btn:hover { background: #3a3a52; }
  #similar-label { font-size: 11px; color: #666; margin-top: 8px; text-align: center; }
</style>

<div id="vis-panel">
  <h3>Color by</h3>
  <select class="mode-select" id="color-mode-select">""" + options_html + """</select>
  <button class="action-btn" id="reset-btn" onclick="resetHighlight()">Clear selection</button>
  <div id="legend-box"></div>
</div>

<div id="board-overlay" onclick="closeBoard()"></div>
<div id="board-modal" style="display:none">
  <div id="modal-header">
    <div>
      <div id="modal-title"></div>
      <div id="modal-themes"></div>
    </div>
    <button id="close-modal" onclick="closeBoard()">×</button>
  </div>
  <div id="board-wrap"></div>
  <div id="move-nav">
    <button class="nav-btn" id="prev-btn" onclick="prevMove()">◀</button>
    <div id="move-label"></div>
    <button class="nav-btn" id="next-btn" onclick="nextMove()">▶</button>
  </div>
  <a id="lichess-btn" href="#" target="_blank">Open on Lichess →</a>
  <div id="similar-label"></div>
</div>

<script>
const colorData     = """ + color_data_js + """;
const legends       = """ + legends_js + """;
const knn           = """ + knn_js + """;
const presentThemes = """ + json.dumps(present_themes) + """;

let chessBoard = null;
let chessGame  = null;
let puzzleMoves = [];   // all UCI moves from CSV (move[0] = opponent setup, rest = solution)
let moveIdx = 0;        // index into puzzleMoves currently shown (0 = after opponent's move)
let fenHistory = [];    // FEN at each step for back navigation

function showBoard(fen, movesStr, url, theme, rating, themes, nNeighbors) {
  puzzleMoves = movesStr.trim().split(' ').filter(Boolean);
  fenHistory  = [];

  // Replay from initial FEN
  chessGame = new Chess(fen);
  fenHistory.push(chessGame.fen());

  // Apply each move to build history
  for (const m of puzzleMoves) {
    chessGame.move(m, {sloppy: true});
    fenHistory.push(chessGame.fen());
  }

  // Start display after opponent's first move (index 1)
  moveIdx = Math.min(1, fenHistory.length - 1);

  const orientation = new Chess(fenHistory[moveIdx]).turn() === 'w' ? 'white' : 'black';

  if (chessBoard) chessBoard.destroy();
  chessBoard = Chessboard('board-wrap', {
    position: fenHistory[moveIdx],
    orientation: orientation,
    pieceTheme: 'https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/img/chesspieces/wikipedia/{piece}.png',
  });

  document.getElementById('modal-title').textContent  = `${theme} | Rating ${rating}`;
  document.getElementById('modal-themes').textContent = themes;
  document.getElementById('lichess-btn').href         = url;
  document.getElementById('similar-label').textContent =
    nNeighbors > 0 ? `${nNeighbors} similar puzzles highlighted` : '';

  updateMoveLabel();
  document.getElementById('board-overlay').style.display = 'block';
  document.getElementById('board-modal').style.display   = 'block';
}

function updateMoveLabel() {
  // moveIdx 0 = before opponent's move, 1 = puzzle start, 2+ = solution moves
  const total   = puzzleMoves.length;   // includes opponent's move
  const solStep = moveIdx - 1;          // solution move index (0-based)
  const solTotal = total - 1;
  let label;
  if (moveIdx === 0)         label = 'Initial position';
  else if (moveIdx === 1)    label = `Puzzle start (${solTotal} moves)`;
  else                       label = `Move ${solStep} / ${solTotal}`;
  document.getElementById('move-label').textContent  = label;
  document.getElementById('prev-btn').disabled = moveIdx === 0;
  document.getElementById('next-btn').disabled = moveIdx === fenHistory.length - 1;
}

function nextMove() {
  if (moveIdx < fenHistory.length - 1) {
    moveIdx++;
    chessBoard.position(fenHistory[moveIdx]);
    updateMoveLabel();
  }
}

function prevMove() {
  if (moveIdx > 0) {
    moveIdx--;
    chessBoard.position(fenHistory[moveIdx]);
    updateMoveLabel();
  }
}

function closeBoard() {
  document.getElementById('board-overlay').style.display = 'none';
  document.getElementById('board-modal').style.display   = 'none';
}

// Close on Escape
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeBoard();
  if (e.key === 'ArrowRight') nextMove();
  if (e.key === 'ArrowLeft')  prevMove();
});

document.addEventListener('DOMContentLoaded', function() {
  var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
  var currentMode = 'Theme';
  var baseColors  = colorData['Theme'].slice();

  function applyMode(mode) {
    currentMode = mode;
    baseColors  = colorData[mode].slice();
    Plotly.restyle(plotDiv, {'marker.color': [baseColors], 'marker.opacity': [0.80], 'marker.size': [5]});
    updateLegend(mode);
    document.getElementById('reset-btn').style.display = 'none';
  }

  function updateLegend(mode) {
    const box = document.getElementById('legend-box');
    const leg = legends[mode];
    let entries = Object.entries(leg);
    if (mode === 'Theme') {
      entries = entries.filter(([lbl]) => presentThemes.includes(lbl) || lbl === 'other');
    }
    box.innerHTML = entries.map(([lbl, color]) =>
      `<div class="leg-item"><div class="leg-dot" style="background:${color}"></div>${lbl}</div>`
    ).join('');
  }

  document.getElementById('color-mode-select').addEventListener('change', function() {
    applyMode(this.value);
  });

  plotDiv.on('plotly_click', function(data) {
    const pt        = data.points[0];
    const idx       = pt.pointIndex;
    const neighbors = knn[idx];
    const url       = pt.customdata[0];
    const theme     = pt.customdata[1];
    const rating    = pt.customdata[2];
    const themes    = pt.customdata[3];
    const fen       = pt.customdata[4];
    const moves     = pt.customdata[5];

    // Highlight neighbors
    const highlight = baseColors.map((c, i) => {
      if (i === idx)              return '#FFD700';
      if (neighbors.includes(i)) return '#FF9800';
      return 'rgba(60,60,60,0.15)';
    });
    const sizes = baseColors.map((c, i) =>
      i === idx ? 10 : (neighbors.includes(i) ? 7 : 4)
    );
    Plotly.restyle(plotDiv, {
      'marker.color': [highlight],
      'marker.opacity': [1.0],
      'marker.size': [sizes],
    });
    document.getElementById('reset-btn').style.display = 'block';

    // Show board modal
    showBoard(fen, moves, url, theme, rating, themes, neighbors.length);
  });

  window.resetHighlight = function() { applyMode(currentMode); };
  applyMode('Theme');
});
</script>
"""

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    html = html.replace("</body>", injection + "\n</body>")
    with open(out_file, "w") as f:
        f.write(html)
    print(f"Saved to {out_file}")
    _open_in_browser(out_file)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    required=True,                   help="JEPA checkpoint .pt")
    parser.add_argument("--puzzles", default="data/lichess_db_puzzle.csv", help="Path to lichess_db_puzzle.csv")
    parser.add_argument("--samples", type=int, default=3000,          help="Number of puzzles to sample")
    parser.add_argument("--method",  choices=["pca", "umap"],         default="umap")
    parser.add_argument("--out",     default="outputs/puzzle_space.html")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--cache",   default=None,
                        help="Path to cache file (.pkl). Saves after encoding; loads on next run to skip encoding.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    if args.cache and os.path.exists(args.cache):
        print(f"Loading cached embeddings from {args.cache} ...")
        import pickle
        with open(args.cache, "rb") as f:
            cached = pickle.load(f)
        embeddings, meta_df = cached["embeddings"], cached["metadata"]
    else:
        puzzles_df = load_puzzles(args.puzzles, args.samples, args.seed)
        embeddings, meta_df = extract_puzzle_embeddings(args.ckpt, puzzles_df, device)
        if args.cache:
            import pickle
            print(f"Saving embeddings to cache: {args.cache}")
            with open(args.cache, "wb") as f:
                pickle.dump({"embeddings": embeddings, "metadata": meta_df}, f)

    plot_puzzles(embeddings, meta_df, args.method, args.out)


if __name__ == "__main__":
    main()
