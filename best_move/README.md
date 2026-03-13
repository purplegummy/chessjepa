# Best Move Decoder

Probes the frozen AC-JEPA context encoder to predict best chess moves.

A lightweight **TransformerMoveDecoder** sits on top of the encoder's patch
latents and outputs logits over all 4096 possible from→to square pairs.
Only the decoder is trained; the encoder weights are kept frozen throughout.

---

## Pipeline

```
Lichess puzzle CSV / Stockfish CSV
        ↓
  generate_puzzle_dataset.py   (or generate_dataset.py)
        ↓
  best_move/data/best_move_dataset.pt
        ↓
  train_transformer_decoder.py
        ↓
  best_move/transformer_decoder_model.pt
```

---

## Step 1 — Build the dataset

**From Lichess puzzles** (recommended — tactically rich):

```bash
python best_move/generate_puzzle_dataset.py \
  --puzzles data/lichess_db_puzzle.csv \
  --out best_move/data/best_move_dataset.pt \
  --max_puzzles 300000 \
  --capture_ratio 0.40
```

**From a Stockfish best-moves CSV** (quiet / positional positions):

```bash
python best_move/generate_dataset.py \
  --csv data/stockfish_best_moves.csv \
  --out best_move/data/best_move_dataset.pt \
  --capture_ratio 0.35
```

**Both combined** (best coverage):

```bash
python best_move/generate_puzzle_dataset.py \
  --puzzles  data/lichess_db_puzzle.csv \
  --stockfish data/stockfish_best_moves.csv \
  --out best_move/data/best_move_dataset.pt
```

The `.pt` file contains:
- `boards`       — `(N, 17, 8, 8)` uint8, color-invariant (current player always at bottom)
- `move_indices` — `(N,)` int64, encoded as `from_sq * 64 + to_sq` in tensor-space coordinates
- `evals`        — `(N,)` float32, stockfish centipawns or `3.0` for puzzle wins (optional)

---

## Step 2 — Train the decoder

```bash
python best_move/train_transformer_decoder.py \
  --ckpt    checkpoints_ac/checkpoint_epoch0005.pt \
  --dataset best_move/data/best_move_dataset.pt \
  --epochs  20 \
  --batch   64 \
  --lr      1e-4 \
  --out     best_move/transformer_decoder_model.pt
```

| Flag               | Default                                      | Description                          |
|--------------------|----------------------------------------------|--------------------------------------|
| `--ckpt`           | `checkpoints_ac/checkpoint_epoch0005.pt`     | Frozen AC-JEPA checkpoint            |
| `--dataset`        | `best_move/data/best_move_dataset.pt`        | Dataset built in Step 1              |
| `--epochs`         | `20`                                         | Training epochs                      |
| `--batch`          | `64`                                         | Batch size                           |
| `--lr`             | `1e-4`                                       | Learning rate (AdamW)                |
| `--label_smoothing`| `0.0`                                        | Cross-entropy label smoothing        |
| `--grad_clip`      | `1.0`                                        | Gradient norm clip                   |
| `--out`            | `best_move/transformer_decoder_model.pt`     | Output path for best decoder weights |

The best checkpoint (lowest val loss) is saved automatically during training.

---

## Architecture

```
board tensor (17, 8, 8)
      ↓ frozen AC-JEPA context encoder
patch latents (1, 16, 256)   ← single board, 16 spatial patches, embed_dim=256
      ↓ TransformerMoveDecoder
  + learned positional embedding
  → 2× TransformerBlock (pre-norm, MHA + FFN)
  → flatten → (16 × 256 = 4096,)
  → Linear(4096, 512) → GELU → LayerNorm
  → Linear(512, 4096)
      ↓
move logits (4096,)   →  legal-move mask  →  argmax = predicted move
```

**Move encoding**: moves are indexed as `from_square * 64 + to_square`.
All promotions are collapsed to queen (no under-promotions in training data).
Squares are in the **tensor coordinate system** (rows flipped for black-to-move
positions to match the color-invariant board encoding).

---

## Files

| File                       | Purpose                                              |
|----------------------------|------------------------------------------------------|
| `generate_puzzle_dataset.py` | Build dataset from Lichess puzzles + optional Stockfish CSV |
| `generate_dataset.py`      | Build dataset from a generic Stockfish best-moves CSV |
| `transformer_decoder.py`   | `TransformerMoveDecoder` model definition            |
| `train_transformer_decoder.py` | Training loop (frozen encoder + trainable decoder) |
| `stockfish_gen.py`         | Generate a Stockfish best-moves CSV from PGN files   |
| `precompute_masks.py`      | Pre-compute legal-move masks to speed up training    |
| `gui_server.py`            | Flask server for the interactive move demo GUI       |
| `gui/`                     | HTML/JS/CSS for the interactive board demo           |
