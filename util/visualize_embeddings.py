"""
Chess V-JEPA — Latent Space Visualization

Single mode:
  python util/visualize_embeddings.py --ckpt checkpoints/epoch15.pt --zarr data/chess_chunks.zarr

Comparison mode (side-by-side):
  python util/visualize_embeddings.py --ckpt checkpoints/epoch15.pt --ckpt2 checkpoints/epoch50.pt --zarr data/chess_chunks.zarr

Both checkpoints encode the exact same sampled positions so distances are meaningful.

Color modes (switchable in-browser):
  - Game Phase  (Opening / Middlegame / Endgame)
  - Turn        (White / Black to move)
  - Material    (White Winning / Equal / Black Winning)
  - Piece Count (few / mid / many)
  - Game        (same chunk = same game, click to highlight)
  - Distance    (comparison only — L2 distance between the two checkpoints' embeddings)
"""

import argparse
import json
import os
import random
import sys
import webbrowser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
# 1. Tensor → chess.Board
# ─────────────────────────────────────────────────────────────────────────────

PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def tensor_to_board(t: torch.Tensor | np.ndarray) -> chess.Board:
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    board = chess.Board(None)
    for i, piece in enumerate(PIECES):
        for r, c in zip(*np.where(t[i] == 1.0)):
            board.set_piece_at(r * 8 + c, chess.Piece(piece, chess.WHITE))
        for r, c in zip(*np.where(t[i + 6] == 1.0)):
            board.set_piece_at(r * 8 + c, chess.Piece(piece, chess.BLACK))
    board.turn = bool(t[12, 0, 0] > 0.5)
    return board


def board_to_lichess_url(board: chess.Board) -> str:
    return f"https://lichess.org/analysis/{board.fen().replace(' ', '_')}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Position sampling (shared between checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

def load_positions(zarr_path: str, num_samples: int) -> tuple[list[torch.Tensor], pd.DataFrame]:
    """Sample random board positions from the dataset and build metadata."""
    print(f"Loading dataset: {zarr_path}")
    dataset = ChessChunkDataset(zarr_path, split="val")

    print(f"Sampling {num_samples} random positions...")
    board_tensors = []
    metadata = []

    for _ in tqdm(range(num_samples)):
        chunk_idx = random.randint(0, len(dataset) - 1)
        chunk = dataset[chunk_idx]          # (16, 17, 8, 8)
        pos_idx = random.randint(0, 15)
        board_tensor = chunk[pos_idx]       # (17, 8, 8)
        board_tensors.append(board_tensor)

        board = tensor_to_board(board_tensor)
        pieces_count = len(board.piece_map())
        phase = "Opening" if pieces_count > 28 else "Endgame" if pieces_count < 14 else "Middlegame"

        white_mat = sum(
            9 if p.piece_type == chess.QUEEN else 5 if p.piece_type == chess.ROOK else 3
            for p in board.piece_map().values() if p.color == chess.WHITE
        )
        black_mat = sum(
            9 if p.piece_type == chess.QUEEN else 5 if p.piece_type == chess.ROOK else 3
            for p in board.piece_map().values() if p.color == chess.BLACK
        )
        adv = white_mat - black_mat
        material_label = "White Winning" if adv > 2 else "Black Winning" if adv < -2 else "Equal"
        piece_bucket = "Few (<14)" if pieces_count < 14 else "Many (>28)" if pieces_count > 28 else "Mid (14-28)"

        metadata.append({
            "FEN": board.fen(),
            "Lichess_URL": board_to_lichess_url(board),
            "Turn": "White" if board.turn else "Black",
            "Piece_Count": pieces_count,
            "Phase": phase,
            "Material_Label": material_label,
            "Piece_Bucket": piece_bucket,
            "Game_ID": chunk_idx,
            "Move_In_Game": pos_idx,
        })

    return board_tensors, pd.DataFrame(metadata)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Encode positions with a checkpoint
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_positions(ckpt_path: str, board_tensors: list[torch.Tensor], device: str) -> np.ndarray:
    """Run the context encoder from a checkpoint over a list of board tensors."""
    print(f"Encoding with: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: JEPAConfig = checkpoint["config"]

    model = ChessJEPA(
        encoder_kwargs=cfg.encoder_kwargs,
        predictor_kwargs=cfg.predictor_kwargs,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    encoder = model.context_encoder

    embeddings = []
    for board_tensor in board_tensors:
        x = board_tensor.unsqueeze(0).unsqueeze(0).to(device)
        latent = encoder(x)
        embeddings.append(latent.squeeze().cpu().numpy())

    return np.stack(embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Color helpers
# ─────────────────────────────────────────────────────────────────────────────

PHASE_PALETTE    = {"Opening": "#4CAF50", "Middlegame": "#2196F3", "Endgame": "#FF9800"}
TURN_PALETTE     = {"White": "#EEEEEE",   "Black": "#555555"}
MATERIAL_PALETTE = {"White Winning": "#64B5F6", "Equal": "#A5D6A7", "Black Winning": "#EF9A9A"}
PIECE_PALETTE    = {"Few (<14)": "#FF9800", "Mid (14-28)": "#2196F3", "Many (>28)": "#4CAF50"}


def categorical_colors(series: pd.Series, palette: dict) -> list[str]:
    return [palette.get(v, "#888888") for v in series]


def game_color_array(game_ids: pd.Series) -> list[str]:
    import plotly.express as px
    pool = (
        px.colors.qualitative.Plotly + px.colors.qualitative.D3
        + px.colors.qualitative.G10 + px.colors.qualitative.T10
        + px.colors.qualitative.Alphabet
    )
    unique = sorted(game_ids.unique())
    color_map = {gid: pool[i % len(pool)] for i, gid in enumerate(unique)}
    return [color_map[g] for g in game_ids]


def distance_color_array(emb1: np.ndarray, emb2: np.ndarray) -> list[str]:
    """Green = similar, Red = very different."""
    dists = np.linalg.norm(emb1 - emb2, axis=1)
    dists_norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)
    return [f"hsl({int(120 * (1 - d))}, 80%, 50%)" for d in dists_norm]


def build_color_modes(df: pd.DataFrame, game_ids: pd.Series) -> tuple[dict, dict]:
    """Return (color_arrays, legend_dicts) for all modes."""
    colors = {
        "Phase":    categorical_colors(df["Phase"],         PHASE_PALETTE),
        "Turn":     categorical_colors(df["Turn"],          TURN_PALETTE),
        "Material": categorical_colors(df["Material_Label"], MATERIAL_PALETTE),
        "Pieces":   categorical_colors(df["Piece_Bucket"],  PIECE_PALETTE),
        "Game":     game_color_array(game_ids),
    }
    legends = {
        "Phase":    PHASE_PALETTE,
        "Turn":     TURN_PALETTE,
        "Material": MATERIAL_PALETTE,
        "Pieces":   PIECE_PALETTE,
        "Game":     {},
    }
    return colors, legends


def hover_texts(df: pd.DataFrame) -> list[str]:
    return [
        f"<b>Phase:</b> {row.Phase} | <b>Turn:</b> {row.Turn}<br>"
        f"<b>Material:</b> {row.Material_Label} | <b>Pieces:</b> {row.Piece_Count}<br>"
        f"<b>Move in game:</b> {row.Move_In_Game}<br>"
        f"<b>FEN:</b> {row.FEN}"
        for row in df.itertuples()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 5a. Single-checkpoint plot
# ─────────────────────────────────────────────────────────────────────────────

PANEL_CSS = """
<style>
  #vis-panel {
    position: fixed; top: 16px; right: 16px; width: 220px;
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
  #legend-box { margin-top: 12px; font-size: 12px; }
  .leg-item { display: flex; align-items: center; gap: 7px; margin: 5px 0; }
  .leg-dot { width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; }
  .action-btn {
    margin-top: 8px; width: 100%; background: #2a2a42; color: #aaa;
    border: 1px solid #444; border-radius: 6px; padding: 5px 8px;
    font-size: 12px; cursor: pointer; display: none;
  }
  .action-btn:hover { background: #3a3a52; color: #ddd; }
  .hint { margin-top: 8px; font-size: 11px; color: #666; display: none; }
</style>
"""


def plot_single(embeddings: np.ndarray, df: pd.DataFrame, method: str, out_file: str, label: str = ""):
    print(f"Reducing dimensions ({method.upper()})...")
    reducer = PCA(n_components=2) if method == "pca" else umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    reduced = reducer.fit_transform(embeddings)
    df["x"], df["y"] = reduced[:, 0], reduced[:, 1]

    colors, legends = build_color_modes(df, df["Game_ID"])

    custom = list(zip(df["Lichess_URL"].tolist(), df["Game_ID"].tolist()))
    fig = go.Figure(go.Scatter(
        x=df["x"].tolist(), y=df["y"].tolist(),
        mode="markers",
        marker=dict(color=colors["Phase"], size=5, opacity=0.80, line=dict(width=0)),
        text=hover_texts(df),
        hovertemplate="%{text}<extra></extra>",
        customdata=custom,
    ))
    title = f"Chess V-JEPA Latent Space ({method.upper()})" + (f" — {label}" if label else "")
    fig.update_layout(
        title=title, template="plotly_dark", showlegend=False,
        margin=dict(t=50, b=20, l=20, r=260),
        xaxis=dict(showticklabels=False, title=""),
        yaxis=dict(showticklabels=False, title=""),
    )

    color_data_js = json.dumps(colors)
    legends_js    = json.dumps(legends)
    game_ids_js   = json.dumps(df["Game_ID"].tolist())

    options_html = "\n".join([
        '<option value="Phase">Game Phase</option>',
        '<option value="Turn">Turn to Move</option>',
        '<option value="Material">Material Balance</option>',
        '<option value="Pieces">Piece Count</option>',
        '<option value="Game">Game (same game = same color)</option>',
    ])

    injection = PANEL_CSS + f"""
<div id="vis-panel">
  <h3>Color by</h3>
  <select class="mode-select" id="color-mode-select">{options_html}</select>
  <div class="hint" id="game-hint">Click a point to highlight its game</div>
  <button class="action-btn" id="reset-btn" onclick="resetHighlight()">Reset highlight</button>
  <div id="legend-box"></div>
</div>
<script>
const colorData = {color_data_js};
const legends   = {legends_js};
const gameIds   = {game_ids_js};
document.addEventListener('DOMContentLoaded', function() {{
  var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
  var currentMode = 'Phase';
  function applyMode(mode) {{
    currentMode = mode;
    Plotly.restyle(plotDiv, {{'marker.color': [colorData[mode]], 'marker.opacity': [0.80]}});
    updateLegend(mode);
    document.getElementById('reset-btn').style.display = 'none';
    document.getElementById('game-hint').style.display = mode === 'Game' ? 'block' : 'none';
  }}
  function updateLegend(mode) {{
    const box = document.getElementById('legend-box');
    const leg = legends[mode];
    if (!leg || Object.keys(leg).length === 0) {{
      box.innerHTML = '<div style="color:#666;font-size:11px;margin-top:4px">Each color = one game</div>';
      return;
    }}
    box.innerHTML = Object.entries(leg).map(([lbl, color]) =>
      `<div class="leg-item"><div class="leg-dot" style="background:${{color}}"></div>${{lbl}}</div>`
    ).join('');
  }}
  document.getElementById('color-mode-select').addEventListener('change', function() {{ applyMode(this.value); }});
  plotDiv.on('plotly_click', function(data) {{
    const pt = data.points[0];
    if (currentMode === 'Game') {{
      const clickedGame = pt.customdata[1];
      const highlight = gameIds.map(gid => gid === clickedGame ? '#FFD700' : 'rgba(80,80,80,0.15)');
      Plotly.restyle(plotDiv, {{'marker.color': [highlight], 'marker.opacity': [1.0]}});
      document.getElementById('reset-btn').style.display = 'block';
    }}
    window.open(pt.customdata[0], '_blank');
  }});
  window.resetHighlight = function() {{ applyMode('Game'); }};
  applyMode('Phase');
}});
</script>
"""
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    html = html.replace("</body>", injection + "\n</body>")
    with open(out_file, "w") as f:
        f.write(html)
    print(f"Saved to {out_file}")
    webbrowser.open(f"file://{os.path.abspath(out_file)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5b. Two-checkpoint comparison plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(
    emb1: np.ndarray, df1: pd.DataFrame,
    emb2: np.ndarray, df2: pd.DataFrame,
    labels: tuple[str, str],
    method: str,
    out_file: str,
):
    print(f"Reducing combined embeddings ({method.upper()})...")
    combined = np.vstack([emb1, emb2])
    reducer = PCA(n_components=2) if method == "pca" else umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    reduced = reducer.fit_transform(combined)
    n = len(emb1)
    r1, r2 = reduced[:n], reduced[n:]

    df1["x"], df1["y"] = r1[:, 0], r1[:, 1]
    df2["x"], df2["y"] = r2[:, 0], r2[:, 1]

    colors1, legends = build_color_modes(df1, df1["Game_ID"])
    colors2, _       = build_color_modes(df2, df2["Game_ID"])

    dist_colors = distance_color_array(emb1, emb2)
    colors1["Distance"] = dist_colors
    colors2["Distance"] = dist_colors
    legends["Distance"] = {}

    custom1 = list(zip(df1["Lichess_URL"].tolist(), df1["Game_ID"].tolist()))
    custom2 = list(zip(df2["Lichess_URL"].tolist(), df2["Game_ID"].tolist()))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"{labels[0]}", f"{labels[1]}"],
        horizontal_spacing=0.04,
    )
    fig.add_trace(go.Scatter(
        x=df1["x"].tolist(), y=df1["y"].tolist(), mode="markers",
        marker=dict(color=colors1["Phase"], size=5, opacity=0.80, line=dict(width=0)),
        text=hover_texts(df1), hovertemplate="%{text}<extra></extra>",
        customdata=custom1, name=labels[0],
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df2["x"].tolist(), y=df2["y"].tolist(), mode="markers",
        marker=dict(color=colors2["Phase"], size=5, opacity=0.80, line=dict(width=0)),
        text=hover_texts(df2), hovertemplate="%{text}<extra></extra>",
        customdata=custom2, name=labels[1],
    ), row=1, col=2)

    fig.update_layout(
        title=f"Chess V-JEPA Latent Space Comparison ({method.upper()})",
        template="plotly_dark", showlegend=False,
        margin=dict(t=60, b=20, l=20, r=260),
    )
    for axis in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
        fig.update_layout(**{axis: dict(showticklabels=False, title="")})

    color_data_js = json.dumps({"ckpt1": colors1, "ckpt2": colors2})
    legends_js    = json.dumps(legends)
    game_ids_js   = json.dumps(df1["Game_ID"].tolist())   # same positions → same game IDs

    options_html = "\n".join([
        '<option value="Phase">Game Phase</option>',
        '<option value="Turn">Turn to Move</option>',
        '<option value="Material">Material Balance</option>',
        '<option value="Pieces">Piece Count</option>',
        '<option value="Game">Game (same game = same color)</option>',
        '<option value="Distance">Embedding Distance (green=similar, red=different)</option>',
    ])

    dist_legend_html = """
      <div style="margin-top:8px;font-size:12px">
        <div class="leg-item"><div class="leg-dot" style="background:hsl(120,80%,50%)"></div>Similar</div>
        <div class="leg-item"><div class="leg-dot" style="background:hsl(60,80%,50%)"></div>Moderate</div>
        <div class="leg-item"><div class="leg-dot" style="background:hsl(0,80%,50%)"></div>Very different</div>
      </div>"""

    injection = PANEL_CSS + f"""
<div id="vis-panel">
  <h3>Color by</h3>
  <select class="mode-select" id="color-mode-select">{options_html}</select>
  <div class="hint" id="game-hint">Click a point to highlight its game</div>
  <button class="action-btn" id="reset-btn" onclick="resetHighlight()">Reset highlight</button>
  <div id="legend-box"></div>
</div>
<script>
const colorData = {color_data_js};
const legends   = {legends_js};
const gameIds   = {game_ids_js};
const distLegendHtml = `{dist_legend_html}`;
document.addEventListener('DOMContentLoaded', function() {{
  var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
  var currentMode = 'Phase';
  function applyMode(mode) {{
    currentMode = mode;
    Plotly.restyle(plotDiv, {{
      'marker.color': [colorData.ckpt1[mode], colorData.ckpt2[mode]],
      'marker.opacity': [0.80, 0.80],
    }}, [0, 1]);
    updateLegend(mode);
    document.getElementById('reset-btn').style.display = 'none';
    document.getElementById('game-hint').style.display = mode === 'Game' ? 'block' : 'none';
  }}
  function updateLegend(mode) {{
    const box = document.getElementById('legend-box');
    if (mode === 'Distance') {{ box.innerHTML = distLegendHtml; return; }}
    const leg = legends[mode];
    if (!leg || Object.keys(leg).length === 0) {{
      box.innerHTML = '<div style="color:#666;font-size:11px;margin-top:4px">Each color = one game</div>';
      return;
    }}
    box.innerHTML = Object.entries(leg).map(([lbl, color]) =>
      `<div class="leg-item"><div class="leg-dot" style="background:${{color}}"></div>${{lbl}}</div>`
    ).join('');
  }}
  document.getElementById('color-mode-select').addEventListener('change', function() {{ applyMode(this.value); }});
  function handleClick(traceIdx, pt) {{
    if (currentMode === 'Game') {{
      const clickedGame = pt.customdata[1];
      const highlight = gameIds.map(gid => gid === clickedGame ? '#FFD700' : 'rgba(80,80,80,0.15)');
      Plotly.restyle(plotDiv, {{'marker.color': [highlight, highlight], 'marker.opacity': [1.0, 1.0]}}, [0, 1]);
      document.getElementById('reset-btn').style.display = 'block';
    }}
    window.open(pt.customdata[0], '_blank');
  }}
  plotDiv.on('plotly_click', function(data) {{
    const pt = data.points[0];
    handleClick(pt.curveNumber, pt);
  }});
  window.resetHighlight = function() {{ applyMode('Game'); }};
  applyMode('Phase');
}});
</script>
"""
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    html = html.replace("</body>", injection + "\n</body>")
    with open(out_file, "w") as f:
        f.write(html)
    print(f"Saved to {out_file}")
    webbrowser.open(f"file://{os.path.abspath(out_file)}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    required=True,                    help="Primary checkpoint .pt")
    parser.add_argument("--ckpt2",   default=None,                     help="Second checkpoint for comparison")
    parser.add_argument("--zarr",    default="data/chess_chunks.zarr", help="Path to zarr dataset")
    parser.add_argument("--samples", type=int, default=1500,           help="Number of boards to sample")
    parser.add_argument("--method",  choices=["pca", "umap"],          default="umap")
    parser.add_argument("--out",     default="latent_space.html")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    board_tensors, df = load_positions(args.zarr, args.samples)
    emb1 = encode_positions(args.ckpt, board_tensors, device)

    if args.ckpt2:
        emb2 = encode_positions(args.ckpt2, board_tensors, device)
        label1 = os.path.basename(args.ckpt)
        label2 = os.path.basename(args.ckpt2)
        plot_comparison(emb1, df.copy(), emb2, df.copy(), (label1, label2), args.method, args.out)
    else:
        plot_single(emb1, df, args.method, args.out, label=os.path.basename(args.ckpt))


if __name__ == "__main__":
    main()
