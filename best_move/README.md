# Best Move Decoder

Probes the frozen AC-JEPA context encoder to predict best chess moves.

A lightweight **TransformerMoveDecoder** sits on top of the encoder's patch
latents and outputs logits over all 4096 possible from→to square pairs.
Only the decoder is trained; the encoder weights are kept frozen throughout.

---

## Pipeline

```
Lichess Elite Database PGNs  (https://database.nikonoel.fr/)
        ↓
  generate_elite_dataset.py
        ↓
  data/elite_dataset.pt
        ↓
  precompute_masks.py
        ↓
  data/elite_dataset_masks.pt
        ↓
  train_transformer_decoder.py
        ↓
  best_move/transformer_decoder_model.pt
```

---

## Step 1 — Build the dataset

Download monthly PGN files from https://database.nikonoel.fr/ into a folder, then:

```bash
python best_move/generate_elite_dataset.py \
  --pgn_dir   data/elite_pgns/ \
  --out        data/elite_dataset.pt \
  --max_samples 1000000 \
  --max_games  30000 \
  --min_elo    2200
```

| Flag             | Default     | Description                                              |
|------------------|-------------|----------------------------------------------------------|
| `--pgn_dir`      | required    | Folder containing `.pgn` files                          |
| `--out`          | `data/elite_dataset.pt` | Output path                              |
| `--max_samples`  | `500000`    | Reservoir size — caps memory and dataset size            |
| `--max_games`    | all         | Stop after N games (e.g. `30000` ≈ 1M positions)        |
| `--min_elo`      | `2200`      | Minimum ELO for both players                             |
| `--capture_ratio`| `0.35`      | Target fraction of capture moves                         |
| `--seq_len`      | `16`        | Board history length — must match JEPA `seq_len`         |

The `.pt` file contains:
- `boards`       — `(N, 16, 17, 8, 8)` uint8 — last 16 board states per sample, zero-padded for early positions
- `move_indices` — `(N,)` int64 — encoded as `from_sq * 64 + to_sq` in tensor-space coordinates

---

## Step 2 — Precompute legal move masks

Speeds up training by pre-computing legal move masks (avoids recomputing every batch):

```bash
python best_move/precompute_masks.py \
  --input data/elite_dataset.pt
```

The output is written as a sidecar file alongside the input: `data/elite_dataset.pt.masks`.
Pass the original `.pt` path to the trainer — it will detect and load the sidecar automatically.

---

## Step 3 — Train the decoder

```bash
python best_move/train_transformer_decoder.py \
  --ckpt           checkpoints_ac/checkpoint_epoch0020.pt \
  --dataset        data/elite_dataset.pt \
  --batch          512 \
  --epochs         30 \
  --lr             3e-4 \
  --label_smoothing 0.0 \
  --grad_clip      1.0 \
  --out            best_move/transformer_decoder_modelv79.pt \
  
```

| Flag               | Default                                  | Description                          |
|--------------------|------------------------------------------|--------------------------------------|
| `--ckpt`           | `checkpoints_ac/checkpoint_epoch0010.pt` | Frozen AC-JEPA checkpoint            |
| `--dataset`        | `data/best_move_dataset_masks.pt`        | Dataset built in Steps 1–2           |
| `--batch`          | `2048`                                   | Batch size                           |
| `--epochs`         | `20`                                     | Training epochs                      |
| `--lr`             | `1e-4`                                   | Learning rate (AdamW + cosine decay) |
| `--label_smoothing`| `0.0`                                    | Cross-entropy label smoothing        |
| `--grad_clip`      | `1.0`                                    | Gradient norm clip                   |
| `--out`            | `best_move/transformer_decoder_model.pt` | Output path for best decoder weights |

The best checkpoint (lowest val loss) is saved automatically during training.

---

## Architecture

```
board sequence (16, 17, 8, 8)
      ↓ frozen AC-JEPA context encoder
patch latents (16, 16, 256)   ← 16 timesteps, 16 spatial patches, embed_dim=256
      ↓ take last timestep → (B, 16, 256)
      ↓ TransformerMoveDecoder
  + latent dropout (0.1)
  + learned positional embedding
  → prepend [CLS] token → (B, 17, 256)
  → 2× TransformerBlock (pre-norm, MHA + FFN, ff_dim=512)
  → LayerNorm
  → extract [CLS] → (B, 256)
  → Dropout(0.3) → Linear(256, 256) → GELU → LayerNorm → Dropout(0.3)
  → from_head: Linear(256, 64)  ‖  to_head: Linear(256, 64)
  → outer sum → (B, 64, 64) → flatten → (B, 4096)
      ↓
move logits (4096,)  →  legal-move mask  →  argmax = predicted move
```

**Move encoding**: moves are indexed as `from_square * 64 + to_square`.
All promotions are collapsed to queen.
Squares are in the **current player's coordinate system**: all 16 history frames
in a sample are encoded from the perspective of the player making the move
(board is flipped for black-to-move so their pieces are always on ranks 1–2).
Move indices are flipped consistently with `uci_to_index` in `generate_elite_dataset.py`.

---

## Files

| File                          | Purpose                                                       |
|-------------------------------|---------------------------------------------------------------|
| `generate_elite_dataset.py`   | Stream Lichess Elite PGNs into a sequence dataset             |
| `generate_dataset.py`         | Build dataset from a generic Stockfish best-moves CSV         |
| `generate_puzzle_dataset.py`  | Build dataset from Lichess puzzles CSV                        |
| `precompute_masks.py`         | Pre-compute legal-move masks to speed up training             |
| `transformer_decoder.py`      | `TransformerMoveDecoder` model definition                     |
| `train_transformer_decoder.py`| Training loop (frozen encoder + trainable decoder)            |
| `gui_server.py`               | FastAPI server for the interactive move demo GUI              |
| `gui/`                        | HTML/JS/CSS for the interactive board demo                    |
