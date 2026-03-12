"""
FastAPI Server to evaluate FEN positions using the frozen JEPA Context Encoder
and the trained WinProbabilityDecoder.
"""

import os
import sys
import torch
import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.jepa import ChessJEPA
from util.config import JEPAConfig
from util.preprocess_pgn import board_to_tensor
from win_evaluation.decoder import WinProbabilityDecoder

import chess.engine
import numpy as np

# Convert CP to WDL (same as generation script)
def cp_to_wdl(cp: int) -> float:
    win_prob_percent = 50 + 50 * (2 / (1 + np.exp(-0.00368208 * cp)) - 1)
    return win_prob_percent / 100.0

app = FastAPI(title="Chess JEPA Win Evaluator")

# Allow CORS for local GUI development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
ENCODER = None
DECODER = None
STOCKFISH_ENGINE = None

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    default="checkpoints/checkpoint_epoch0010.pt", help="JEPA checkpoint")
    parser.add_argument("--decoder", default="win_evaluation/decoder_model.pt",     help="Decoder checkpoint")
    # uvicorn adds its own args; ignore unknowns
    args, _ = parser.parse_known_args()
    return args

_ARGS = _parse_args()

@app.on_event("startup")
async def load_models():
    global ENCODER, DECODER, STOCKFISH_ENGINE
    print(f"Loading models onto {DEVICE}...")

    ckpt_path = _ARGS.ckpt
    decoder_path = _ARGS.decoder
    
    if not os.path.exists(ckpt_path) or not os.path.exists(decoder_path):
        print(f"Error: Could not find checkpoints at {ckpt_path} or {decoder_path}")
        return
        
    # Load Encoder
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg: JEPAConfig = checkpoint["config"]
    
    jepa = ChessJEPA(
        encoder_kwargs=cfg.encoder_kwargs,
        predictor_kwargs=cfg.predictor_kwargs,
    ).to(DEVICE)
    jepa.load_state_dict(checkpoint["model"])
    
    ENCODER = jepa.context_encoder
    ENCODER.eval()
    
    # Load Decoder
    decoder_ckpt = torch.load(decoder_path, map_location=DEVICE, weights_only=False)
    DECODER = WinProbabilityDecoder(in_features=cfg.encoder_kwargs["embed_dim"]).to(DEVICE)
    DECODER.load_state_dict(decoder_ckpt["decoder"])
    DECODER.eval()
    print("JEPA Models loaded successfully.")
    
    # Load Stockfish
    stockfish_path = "/opt/homebrew/bin/stockfish"
    if os.path.exists(stockfish_path):
        try:
            _, STOCKFISH_ENGINE = await chess.engine.popen_uci(stockfish_path)
            print("Stockfish engine loaded successfully.")
        except Exception as e:
            print(f"Failed to load Stockfish engine: {e}")
    else:
        print(f"Stockfish engine not found at {stockfish_path}")

@app.on_event("shutdown")
async def shutdown_models():
    if STOCKFISH_ENGINE:
        await STOCKFISH_ENGINE.quit()

class EvaluateRequest(BaseModel):
    fen: str

@app.post("/api/evaluate")
async def evaluate_position(req: EvaluateRequest):
    if ENCODER is None or DECODER is None:
        raise HTTPException(status_code=503, detail="JEPA Models are not loaded.")
        
    try:
        board = chess.Board(req.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN string: {e}")
        
    # 1. JEPA Decoder Evaluation
    tensor = board_to_tensor(board)
    tensor = torch.from_numpy(tensor).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        latents = ENCODER(tensor)
        jepa_win_prob = DECODER(latents).item()
        
    # 2. Stockfish Evaluation
    sf_win_prob = None
    sf_cp = None
    sf_mate = None
    
    stockfish_path = "/opt/homebrew/bin/stockfish"
    if os.path.exists(stockfish_path):
        try:
            # Spawn per-request to avoid Uvicorn asyncio loop entanglement and Signal crashes
            with chess.engine.SimpleEngine.popen_uci(stockfish_path) as sf:
                info = sf.analyse(board, chess.engine.Limit(time=0.05))
                score = info["score"].white()
                
                if score.is_mate():
                    sf_mate = score.mate()
                    cp = 10000 if sf_mate > 0 else -10000
                else:
                    cp = score.score()
                    sf_cp = cp / 100.0 if cp is not None else None
                    
                sf_win_prob = cp_to_wdl(cp)
        except Exception as e:
            print(f"Stockfish error: {e}")

    return {
        "fen": req.fen,
        "jepa_win_probability": jepa_win_prob,
        "stockfish_win_probability": sf_win_prob,
        "stockfish_cp": sf_cp,
        "stockfish_mate": sf_mate
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
    return {"message": "GUI index.html not found in win_evaluation/gui/"}

if __name__ == "__main__":
    import uvicorn
    # Make sure we're in the project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    uvicorn.run("win_evaluation.gui_server:app", host="127.0.0.1", port=8000, reload=True)
