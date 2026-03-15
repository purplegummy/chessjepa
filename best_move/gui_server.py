"""
FastAPI server for playing chess against the frozen JEPA Context Encoder
and trained BestMoveDecoder.
"""

import contextlib
import os
import sys
import numpy as np
import torch
import chess
import chess.engine
from collections import deque
from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.jepa import ChessJEPA
from model.acjepa import ActionConditionedChessJEPA
from util.config import JEPAConfig
from best_move.transformer_decoder import TransformerMoveDecoder
from util.preprocess_pgn import board_to_tensor

MAX_HISTORY = 16  # encoder was trained on chunks of this length


def _flip_sq(sq: int) -> int:
    """Mirror a square index vertically (rank flip) to match board_to_tensor's black-to-move encoding."""
    return (7 - sq // 8) * 8 + sq % 8


def create_legal_move_mask_from_board(board: chess.Board, flip: bool = False) -> torch.Tensor:
    """Create a mask for legal moves from a chess.Board.

    If flip=True, square indices are mirrored vertically to match the flipped
    tensor space produced by board_to_tensor when black is to move.
    """
    mask = torch.zeros(4096, dtype=torch.bool)
    for move in board.legal_moves:
        from_sq = move.from_square
        to_sq = move.to_square
        if flip:
            from_sq = _flip_sq(from_sq)
            to_sq   = _flip_sq(to_sq)
        mask[from_sq * 64 + to_sq] = True
    return mask


@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_models()
    yield

app = FastAPI(title="Chess JEPA Best Move", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
ENCODER = None
DECODER = None

# AMP context for inference — bfloat16 on Ampere+ CUDA, float16 on older CUDA, off elsewhere
if torch.cuda.is_available():
    _AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    _AMP_CTX   = lambda: torch.autocast(device_type="cuda", dtype=_AMP_DTYPE)
else:
    _AMP_CTX   = contextlib.nullcontext

# session_id → deque of board tensors (numpy, (17,8,8) uint8), capped at MAX_HISTORY
_SESSION_HISTORY: Dict[str, deque] = {}


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    default="checkpoints/checkpoint_epoch0060.pt")
    parser.add_argument("--decoder", default="best_move/models/transformer_decoder_model.pt")
    args, _ = parser.parse_known_args()
    return args

_ARGS = _parse_args()


async def load_models():
    global ENCODER, DECODER

    ckpt_path    = _ARGS.ckpt
    decoder_path = _ARGS.decoder

    if not os.path.exists(ckpt_path):
        print(f"ERROR: JEPA checkpoint not found: {ckpt_path}")
        return
    if not os.path.exists(decoder_path):
        print(f"ERROR: Decoder checkpoint not found: {decoder_path}")
        return

    print(f"Loading models onto {DEVICE}...")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg: JEPAConfig = checkpoint["config"]

    # Try AC model first; fall back to base JEPA
    try:
        jepa = ActionConditionedChessJEPA(
            encoder_kwargs=cfg.encoder_kwargs,
            predictor_kwargs=cfg.predictor_kwargs,
        ).to(DEVICE)
        jepa.load_state_dict(checkpoint["model"])
        print("  Encoder: ActionConditionedChessJEPA")
    except Exception:
        jepa = ChessJEPA(
            encoder_kwargs=cfg.encoder_kwargs,
            predictor_kwargs=cfg.predictor_kwargs,
        ).to(DEVICE)
        jepa.load_state_dict(checkpoint["model"])
        print("  Encoder: ChessJEPA (base)")

    ENCODER = jepa.context_encoder
    ENCODER.eval()
    for p in ENCODER.parameters():
        p.requires_grad = False

    embed_dim   = cfg.encoder_kwargs.get("embed_dim", 256)
    num_patches = (cfg.board_size // cfg.patch_size) ** 2   # 16

    decoder_ckpt = torch.load(decoder_path, map_location=DEVICE, weights_only=False)
    state = decoder_ckpt["decoder"] if "decoder" in decoder_ckpt else decoder_ckpt
    num_layers = sum(1 for k in state if "transformer_blocks." in k and ".attn.in_proj_weight" in k)
    # Infer mlp_hidden and ff_dim from saved weights so the model always matches the checkpoint
    mlp_hidden = state["mlp.1.weight"].shape[0]
    ff_dim     = state["transformer_blocks.0.ff.0.weight"].shape[0]
    DECODER = TransformerMoveDecoder(
        embed_dim=embed_dim,
        num_patches=num_patches,
        num_heads=8,
        ff_dim=ff_dim,
        num_layers=num_layers,
        mlp_hidden=mlp_hidden,
        dropout=0.1,
        head_dropout=0.0,  # dropout irrelevant at inference; no saved weights to mismatch
    ).to(DEVICE)
    print(f"  Decoder: TransformerMoveDecoder (embed_dim={embed_dim}, num_patches={num_patches}, num_layers={num_layers}, mlp_hidden={mlp_hidden}, ff_dim={ff_dim})")
    DECODER.load_state_dict(state)
    DECODER.eval()
    print("Models loaded successfully.")


class BestMoveRequest(BaseModel):
    fen: str
    session_id: str = ""   # provide a stable ID per game to enable history tracking
    top_n: int = 5


class NewGameRequest(BaseModel):
    session_id: str


@app.post("/api/new_game")
async def new_game(req: NewGameRequest):
    """Clear the board history for a session (call at the start of each game)."""
    _SESSION_HISTORY.pop(req.session_id, None)
    return {"status": "ok"}


def _build_sequence(history: list, extra_tensor: np.ndarray | None = None) -> torch.Tensor:
    """
    Stack history (list of (17,8,8) uint8 arrays) into a (1, T, 17, 8, 8) float tensor.
    If extra_tensor is given, append it first (used to build next-position sequences).
    """
    frames = list(history)
    if extra_tensor is not None:
        frames = (frames + [extra_tensor])[-MAX_HISTORY:]
    seq = np.stack(frames, axis=0)
    return torch.from_numpy(seq).unsqueeze(0).float().to(DEVICE)


@app.post("/api/best_move")
async def get_best_move(req: BestMoveRequest):
    if ENCODER is None or DECODER is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    try:
        board = chess.Board(req.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {e}")

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise HTTPException(status_code=400, detail="No legal moves in this position.")

    # Build board tensor for current position and update session history
    current_tensor = board_to_tensor(board)  # (17, 8, 8) uint8

    if req.session_id:
        if req.session_id not in _SESSION_HISTORY:
            _SESSION_HISTORY[req.session_id] = deque(maxlen=MAX_HISTORY)
        _SESSION_HISTORY[req.session_id].append(current_tensor)
        history = list(_SESSION_HISTORY[req.session_id])
    else:
        history = [current_tensor]

    flip = board.turn == chess.BLACK

    def _tensor_idx(move: chess.Move) -> int:
        f = _flip_sq(move.from_square) if flip else move.from_square
        t = _flip_sq(move.to_square)   if flip else move.to_square
        return f * 64 + t

    with torch.no_grad(), _AMP_CTX():
        tensor = _build_sequence(history)           # (1, T, 17, 8, 8)
        latents = ENCODER(tensor)                   # (1, T, P, D)
        logits = DECODER(latents).squeeze(0).float()  # (4096,) — cast back to float32 for masking

        legal_mask    = create_legal_move_mask_from_board(board, flip=flip).to(DEVICE)
        masked_logits = logits.masked_fill(~legal_mask, float('-inf'))
        probs_all     = torch.softmax(masked_logits, dim=0)  # (4096,)

        policy_scores = [
            (move, probs_all[_tensor_idx(move)].item())
            for move in legal_moves
        ]
        policy_scores.sort(key=lambda x: x[1], reverse=True)

    top_n = min(req.top_n, len(policy_scores))
    top_moves = []
    for move, prob in policy_scores[:top_n]:
        temp = board.copy()
        san = temp.san(move)
        top_moves.append({
            "uci":  move.uci(),
            "san":  san,
            "prob": round(prob, 4),
        })

    best_move = policy_scores[0][0]
    return {
        "move":        best_move.uci(),
        "san":         top_moves[0]["san"],
        "top_moves":   top_moves,
        "history_len": len(history),
    }


# Serve static GUI files
GUI_DIR = os.path.join(os.path.dirname(__file__), "gui")
os.makedirs(GUI_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=GUI_DIR), name="static")


@app.get("/")
async def root():
    index_path = os.path.join(GUI_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "GUI not found at best_move/gui/index.html"}


if __name__ == "__main__":
    import uvicorn
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    uvicorn.run("best_move.gui_server:app", host="127.0.0.1", port=8001, reload=True)
