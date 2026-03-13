# Chess JEPA

A JEPA (Joint-Embedding Predictive Architecture) trained on chess games. The model learns to predict future board states from past ones in latent space, producing representations that capture chess strategy and tactics.

Two training modes are available:
- **Base JEPA** (`train.py`) — predicts future board states from context alone
- **AC-JEPA** (`train_ac.py`) — same, but also conditions on the chess moves (actions) that produced each state

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Additional packages for visualization:
```bash
pip install scikit-learn umap-learn plotly pandas
```

---

## Full Training Pipeline

### Step 1 — Get PGN data

Download a Lichess game database (compressed PGN):

```bash
# Example: April 2024 standard games
curl -O https://database.lichess.org/standard/lichess_db_standard_rated_2024-04.pgn.zst
```

The preprocessor accepts both `.pgn` and `.pgn.zst` files directly.

---

### Step 2 — Preprocess PGN → Zarr

Convert raw PGN games into a chunked zarr dataset of board tensors.

```bash
python util/preprocess_pgn.py \
    --input lichess_db_standard_rated_2024-04.pgn.zst \
    --output data/chess_chunks.zarr
```

**What this does:**
- Filters games below 1500 Elo or under 20 moves
- Converts each board position to an `(18, 8, 8)` uint8 tensor:
  - Channels 0–5: current side's pieces (pawn, knight, bishop, rook, queen, king)
  - Channels 6–11: opponent's pieces (same order)
  - Channel 12: unused (zero) — turn is encoded via board flip for color invariance
  - Channels 13–16: castling rights
  - Channel 17: en passant square
- Packs consecutive positions into chunks of 16 and writes to zarr

**Output:** `data/chess_chunks.zarr` with shape `(N, 16, 18, 8, 8)`

---

### Step 3 — Generate actions (AC-JEPA only)

If you plan to train the action-conditioned model, generate the `actions` array from the board data:

```bash
python util/generate_actions.py \
    --zarr data/chess_chunks.zarr \
    --workers 8
```

This adds an `actions` array of shape `(N, 16, 2)` to the same zarr store, where each entry is `(from_sq, to_sq)` in 0–63 board coordinates. The first timestep per chunk is always `(64, 64)` (null — no prior board to diff against).

To test on a small slice first:
```bash
python util/generate_actions.py --zarr data/chess_chunks.zarr --max_chunks 1000
```

---

### Step 4 — Train

#### Base JEPA

```bash
python train.py --zarr_path data/chess_chunks.zarr
```

Checkpoints are saved to `checkpoints/` every 5 epochs by default.

#### AC-JEPA (recommended)

```bash
python train_ac.py --zarr_path data/chess_chunks.zarr
```

Checkpoints are saved to `checkpoints_ac/`.

> If you skipped Step 3, `train_ac.py` still runs but the predictor receives null moves — training works, just without the action signal.

#### Common options (both scripts)

| Flag | Default | Description |
|------|---------|-------------|
| `--zarr_path` | `chess_chunks.zarr` | Path to zarr store |
| `--batch_size` | `512` | Samples per batch |
| `--learning_rate` | auto | LR (default: linear scale from 1.5e-4 @ batch 256) |
| `--max_epochs` | `100` | Number of epochs |
| `--max_steps` | — | Stop after N steps (useful for smoke tests) |
| `--device` | `cuda` | `cuda`, `mps`, or `cpu` |
| `--num_workers` | `14` | DataLoader workers |
| `--checkpoint_dir` | `checkpoints` / `checkpoints_ac` | Where to save checkpoints |
| `--resume_from` | — | Path to a checkpoint to resume from |

**Smoke test (verify the pipeline works end-to-end):**
```bash
python train.py --zarr_path data/chess_chunks.zarr --max_steps 100
```

#### Resuming training

```bash
python train.py --resume_from checkpoints/checkpoint_epoch0010.pt
python train_ac.py --resume_from checkpoints_ac/checkpoint_epoch0010.pt
```

---

## Key hyperparameters

All defaults live in [util/config.py](util/config.py) in `JEPAConfig`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seq_len` | 16 | Board positions per training chunk |
| `in_channels` | 18 | Input channels per board |
| `embed_dim` | 256 | Encoder hidden dimension |
| `encoder_depth` | 6 | Transformer blocks in encoder |
| `predictor_dim` | 128 | Predictor bottleneck dimension |
| `target_ratio` | 0.4 | ~40% of positions used as prediction targets |
| `mask_mode` | `causal` | `causal` (predict future) or `random` |
| `ema_momentum_start` | 0.996 | EMA momentum at step 0 |
| `ema_momentum_end` | 1.0 | EMA momentum at final step |
| `warmup_epochs` | 10 | Epochs for linear LR warmup |

---

## Visualization

### Latent space of board positions

```bash
python util/visualize_embeddings.py \
    --ckpt checkpoints/checkpoint_epoch0050.pt \
    --zarr data/chess_chunks.zarr
```

Compare two checkpoints side-by-side:
```bash
python util/visualize_embeddings.py \
    --ckpt checkpoints/checkpoint_epoch0010.pt \
    --ckpt2 checkpoints/checkpoint_epoch0050.pt \
    --zarr data/chess_chunks.zarr
```

### Puzzle similarity (AC-JEPA)

Download the Lichess puzzle database first:
```bash
curl -O https://database.lichess.org/lichess_db_puzzle.csv.zst
unzstd lichess_db_puzzle.csv.zst -o data/lichess_db_puzzle.csv
```

Then run:
```bash
python util/viz_puzzles.py \
    --ckpt checkpoints_ac/checkpoint_epoch0050.pt \
    --puzzles data/lichess_db_puzzle.csv \
    --samples 3000
```

Cache embeddings to skip re-encoding on subsequent runs:
```bash
python util/viz_puzzles.py \
    --ckpt checkpoints_ac/checkpoint_epoch0050.pt \
    --cache puzzle_cache.pkl \
    --samples 3000
```

---

## Project structure

```
chessjepa/
├── train.py                    # Base JEPA training loop
├── train_ac.py                 # Action-conditioned JEPA training loop
├── model/
│   ├── encoder.py              # Patch-based transformer encoder
│   ├── predictor.py            # Base JEPA predictor
│   ├── acpredictor.py          # Action-conditioned predictor
│   ├── jepa.py                 # ChessJEPA (context + EMA target encoder)
│   └── acjepa.py               # ActionConditionedChessJEPA
└── util/
    ├── config.py               # JEPAConfig dataclass (all hyperparameters)
    ├── preprocess_pgn.py       # PGN → zarr board tensors
    ├── generate_actions.py     # Add actions array to zarr store
    ├── dataset.py              # PyTorch Dataset wrappers for zarr
    ├── masking.py              # Temporal mask generation
    ├── visualize_embeddings.py # Interactive latent space plot
    └── viz_puzzles.py          # Puzzle similarity explorer
```
