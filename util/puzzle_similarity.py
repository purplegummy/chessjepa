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
    "< 1000":    "#4CAF50",
    "1000-1200": "#8BC34A",
    "1200-1400": "#FFEB3B",
    "1400-1600": "#FF9800",
    "1600-1800": "#FF5722",
    "1800-2000": "#F44336",
    "2000-2200": "#9C27B0",
    "> 2200":    "#1A237E",
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
    if rating < 1000:   return "< 1000"
    elif rating < 1200: return "1000-1200"
    elif rating < 1400: return "1200-1400"
    elif rating < 1600: return "1400-1600"
    elif rating < 1800: return "1600-1800"
    elif rating < 2000: return "1800-2000"
    elif rating < 2200: return "2000-2200"
    else:               return "> 2200"


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

        # Encode each position, then mean-pool → single puzzle embedding
        pos_embs = []
        for arr in board_arrays:
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
            latent = encoder(t)  # (1, 1, embed_dim)
            pos_embs.append(latent.squeeze().cpu().numpy())

        embeddings.append(np.mean(pos_embs, axis=0))

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
  #info-box {
    margin-top: 10px; padding: 8px; background: #2a2a42; border-radius: 6px;
    font-size: 11px; display: none; border: 1px solid #444;
  }
  #info-box a { color: #64B5F6; text-decoration: none; }
  #info-box a:hover { text-decoration: underline; }
  .action-btn {
    margin-top: 6px; width: 100%; background: #2a2a42; color: #aaa;
    border: 1px solid #444; border-radius: 6px; padding: 5px 8px;
    font-size: 12px; cursor: pointer; display: none;
  }
  .action-btn:hover { background: #3a3a52; color: #ddd; }
</style>

<div id="vis-panel">
  <h3>Color by</h3>
  <select class="mode-select" id="color-mode-select">""" + options_html + """</select>
  <button class="action-btn" id="reset-btn" onclick="resetHighlight()">Clear selection</button>
  <div id="info-box"></div>
  <div id="legend-box"></div>
</div>

<script>
const colorData     = """ + color_data_js + """;
const legends       = """ + legends_js + """;
const knn           = """ + knn_js + """;
const presentThemes = """ + json.dumps(present_themes) + """;

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
    document.getElementById('info-box').style.display  = 'none';
  }

  function updateLegend(mode) {
    const box = document.getElementById('legend-box');
    const leg = legends[mode];
    let entries = Object.entries(leg);
    // For Theme mode, only show themes present in the data
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
    const pt       = data.points[0];
    const idx      = pt.pointIndex;
    const neighbors = knn[idx];
    const url      = pt.customdata[0];
    const theme    = pt.customdata[1];
    const rating   = pt.customdata[2];
    const themes   = pt.customdata[3];

    // Highlight: gold for clicked, orange for neighbors, dim rest
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

    // Info panel
    const infoBox = document.getElementById('info-box');
    infoBox.style.display = 'block';
    infoBox.innerHTML = `
      <b>${theme}</b> | Rating ${rating}<br>
      <span style="color:#999;font-size:10px">${themes}</span><br>
      <div style="margin-top:6px">
        10 similar puzzles highlighted in orange<br>
        <a href="${url}" target="_blank">Open on Lichess →</a>
      </div>`;
    document.getElementById('reset-btn').style.display = 'block';
  });

  window.resetHighlight = function() { applyMode(currentMode); };
  applyMode('Theme');
});
</script>
"""

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
    parser.add_argument("--puzzles", required=True,                   help="Path to lichess_db_puzzle.csv")
    parser.add_argument("--samples", type=int, default=3000,          help="Number of puzzles to sample")
    parser.add_argument("--method",  choices=["pca", "umap"],         default="umap")
    parser.add_argument("--out",     default="puzzle_space.html")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    puzzles_df = load_puzzles(args.puzzles, args.samples, args.seed)
    embeddings, meta_df = extract_puzzle_embeddings(args.ckpt, puzzles_df, device)
    plot_puzzles(embeddings, meta_df, args.method, args.out)


if __name__ == "__main__":
    main()
