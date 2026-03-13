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
from best_move.transformer_decoder import TransformerMoveDecoder
from util.preprocess_pgn import board_to_tensor

# Old BestMoveDecoder for backward compatibility with checkpoints without value head
class OldBestMoveDecoder(torch.nn.Module):
    def __init__(self, in_features, hidden_features=512, num_layers=3, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(torch.nn.Linear(in_features if i == 0 else hidden_features, hidden_features))
            layers.append(torch.nn.GELU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.LayerNorm(hidden_features))
        layers.append(torch.nn.Linear(hidden_features, 4096))  # NUM_MOVES
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
    parser.add_argument("--decoder", default="best_move/models/transformer_decoder_model.pt")
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
    is_transformer = any("transformer_blocks" in k for k in state.keys())

    if is_transformer:
        embed_dim = cfg.encoder_kwargs.get("embed_dim", 256)
        num_patches = (cfg.board_size // cfg.patch_size) ** 2   # 16
        num_layers = sum(1 for k in state if "transformer_blocks." in k and ".attn.in_proj_weight" in k)
        DECODER = TransformerMoveDecoder(
            embed_dim=embed_dim,
            num_patches=num_patches,
            num_heads=8,
            ff_dim=512,
            num_layers=num_layers,
            mlp_hidden=512,
            dropout=0.1
        ).to(DEVICE)
        print(f"  Decoder: TransformerMoveDecoder (embed_dim={embed_dim}, num_patches={num_patches}, num_layers={num_layers})")
    else:
       
        # 1. Detect in_features from the first layer
        if "initial_layer.0.weight" in state:
            ckpt_in_features = state["initial_layer.0.weight"].shape[1]
        elif "trunk.0.weight" in state:
            ckpt_in_features = state["trunk.0.weight"].shape[1]
        elif "net.0.weight" in state:
            ckpt_in_features = state["net.0.weight"].shape[1]
        else:
            raise ValueError("Cannot detect in_features from decoder checkpoint")
                
        # 2. Identify the architecture by checking keys
        has_value_head = any("value_trunk" in k for k in state)
        # 3. Detect Dropout/Structure
        # In your class: block is [Linear, GELU, (Dropout), LayerNorm]
        # If trunk.2 is a LayerNorm (shape [512]), then Dropout was 0.0.
        # If trunk.2 is a Dropout (no weights), then trunk.3 is the LayerNorm.
        
        # We check the shape of trunk.2 to see if it's a LayerNorm weight
        ckpt_dropout = 0.0
        if "trunk.2.weight" in state:
            # If trunk.2 exists and has shape [512], it's LayerNorm -> Dropout was 0.0
            if state["trunk.2.weight"].shape == torch.Size([512]):
                ckpt_dropout = 0.0
            else:
                ckpt_dropout = 0.3 # Or whatever your default was
        elif "trunk.3.weight" in state:
             # If trunk.3 is the first LayerNorm, trunk.2 was likely Dropout
             ckpt_dropout = 0.3

        if has_value_head:
            # Use your new BestMoveDecoder class
            DECODER = BestMoveDecoder(
                in_features=ckpt_in_features,
                hidden_features=512,
                num_layers=3,
                dropout=ckpt_dropout,
            ).to(DEVICE)
            print(f"  Decoder: BestMoveDecoder (new, in_features={ckpt_in_features}, dropout={ckpt_dropout})")
        else:
            # Fallback for old models using 'net'
            DECODER = OldBestMoveDecoder(
                in_features=ckpt_in_features,
                hidden_features=512,
                num_layers=3,
                dropout=ckpt_dropout,
            ).to(DEVICE)
            print(f"  Decoder: OldBestMoveDecoder (in_features={ckpt_in_features}, dropout={ckpt_dropout})")
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

    flip = board.turn == chess.BLACK
    tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        latents = ENCODER(tensor)

        if isinstance(DECODER, TransformerMoveDecoder):
            # Transformer decoder outputs logits in tensor (possibly flipped) space.
            # board_to_tensor flips the board when black is to move, so move indices
            # must be mirrored vertically to match that flipped coordinate space.
            logits = DECODER(latents).squeeze(0)  # (4096,)

            legal_mask = create_legal_move_mask_from_board(board, flip=flip).to(DEVICE)
            masked_logits = logits.clone()
            masked_logits[~legal_mask] = -1e9

            def _tensor_idx(move: chess.Move) -> int:
                f = _flip_sq(move.from_square) if flip else move.from_square
                t = _flip_sq(move.to_square)   if flip else move.to_square
                return f * 64 + t

            move_scores = [
                (move, masked_logits[_tensor_idx(move)].item())
                for move in legal_moves
            ]
            value_out = None  # No value prediction
        else:
            # Non-factored decoders
            if isinstance(DECODER, OldBestMoveDecoder):
                logits = DECODER(latents)
                pred_value = torch.tensor(0.0)
            else:
                logits, pred_value = DECODER(latents)
            logits = logits.squeeze(0)  # (4096,)

            # Apply legal move masking
            legal_mask = create_legal_move_mask_from_board(board).to(DEVICE)
            masked_logits = logits.clone()
            masked_logits[~legal_mask] = -1e9

            move_scores = [
                (move, masked_logits[move.from_square * 64 + move.to_square].item())
                for move in legal_moves
            ]
            value_out = pred_value.item()
    # 1. Sort the moves so the highest score is first
    move_scores.sort(key=lambda x: x[1], reverse=True)

    # 2. Now calculate probabilities
    raw = torch.tensor([s for _, s in move_scores])
    probs = torch.softmax(raw, dim=0).tolist()

    # 3. Populate the response
    top_n = min(req.top_n, len(move_scores))
    top_moves = []
    for i in range(top_n):
        move, score = move_scores[i]
        temp = board.copy()
        san = temp.san(move)
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
    if value_out is not None:
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
