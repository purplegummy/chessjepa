"""
FastAPI server for playing chess against the frozen JEPA Context Encoder
and trained BestMoveDecoder.
"""

import os
import sys
import torch
import chess
import chess.engine
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.jepa import ChessJEPA
from model.acjepa import ActionConditionedChessJEPA
from util.config import JEPAConfig
from util.preprocess_pgn import board_to_tensor
from best_move.decoder import BestMoveDecoder
from best_move.factored_decoder import FactoredMoveDecoder

app = FastAPI(title="Chess JEPA Best Move")

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


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    default="checkpoints/checkpoint_epoch0060.pt")
    parser.add_argument("--decoder", default="best_move/factored_decoder_model2.pt")
    args, _ = parser.parse_known_args()
    return args

_ARGS = _parse_args()


@app.on_event("startup")
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
    in_features = embed_dim * num_patches                    # 4096

    decoder_ckpt = torch.load(decoder_path, map_location=DEVICE, weights_only=False)
    state = decoder_ckpt["decoder"] if "decoder" in decoder_ckpt else decoder_ckpt
    is_factored = any("from_sq_embed" in k for k in state.keys())

    if is_factored:
        DECODER = FactoredMoveDecoder(in_features=in_features, hidden=512, num_hidden=2).to(DEVICE)
    else:
        # Auto-detect in_features from the first linear layer's weight shape
        ckpt_in_features = state["net.0.weight"].shape[1]
        # Detect if dropout was used: with dropout, 3-layer net has 9 param groups;
        # without dropout, 7. Check by presence of net.8 (final Linear with dropout).
        ckpt_has_dropout = "net.8.weight" in state
        ckpt_dropout = 0.3 if ckpt_has_dropout else 0.0
        DECODER = BestMoveDecoder(
            in_features=ckpt_in_features,
            hidden_features=512,
            num_layers=3,
            dropout=ckpt_dropout,
        ).to(DEVICE)
        print(f"  Decoder: BestMoveDecoder (in_features={ckpt_in_features}, dropout={ckpt_dropout})")

    DECODER.load_state_dict(state)
    DECODER.eval()
    print("Models loaded successfully.")


class BestMoveRequest(BaseModel):
    fen: str
    top_n: int = 5


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

    tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        latents = ENCODER(tensor)

        if isinstance(DECODER, FactoredMoveDecoder):
            # score_all returns (1, 64, 64) plus value
            score_matrix, pred_value = DECODER.score_all(latents)
            score_matrix = score_matrix.squeeze(0)  # (64, 64)
            move_scores: list[tuple[chess.Move, float]] = [
                (move, score_matrix[move.from_square, move.to_square].item())
                for move in legal_moves
            ]
            value_out = pred_value.item()
        else:
            logits, pred_value = DECODER(latents)
            logits = logits.squeeze(0)  # (4096,)
            move_scores = [
                (move, logits[move.from_square * 64 + move.to_square].item())
                for move in legal_moves
            ]
            value_out = pred_value.item()
    raw = torch.tensor([s for _, s in move_scores])
    probs = torch.softmax(raw, dim=0).tolist()

    top_n = min(req.top_n, len(move_scores))
    top_moves = []
    for i in range(top_n):
        move = move_scores[i][0]
        # Get SAN notation using a temporary board copy
        temp = board.copy()
        san  = temp.san(move)
        top_moves.append({
            "uci":  move.uci(),
            "san":  san,
            "prob": round(probs[i], 4),
        })

    best_move = move_scores[0][0]
    resp = {
        "move":       best_move.uci(),
        "san":        top_moves[0]["san"],
        "confidence": round(probs[0], 4),
        "top_moves":  top_moves,
    }
    # value prediction (in pawn units) from the decoder
    if "value_out" in locals():
        resp["value"] = round(value_out, 3)
    return resp


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
