"""
Script to manually verify Win Probability predictions from the trained decoder
on clear advantage/disadvantage positions using the frozen encoder context.
"""

import os
import sys
import torch
import chess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.jepa import ChessJEPA
from util.config import JEPAConfig
from util.preprocess_pgn import board_to_tensor
from win_evaluation.decoder import WinProbabilityDecoder

def evaluate_fen(fen: str, encoder, decoder, device):
    board = chess.Board(fen)
    tensor = board_to_tensor(board) # (17, 8, 8)
    
    # Needs to be (B, 1, 17, 8, 8)
    tensor = torch.from_numpy(tensor).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        latents = encoder(tensor)
        win_prob = decoder(latents).item()
        
    return win_prob

def verify(ckpt_path: str, decoder_path: str, device: str = "cpu"):
    print(device)
    # 1. Load context encoder
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: JEPAConfig = checkpoint["config"]
    
    jepa = ChessJEPA(
        encoder_kwargs=cfg.encoder_kwargs,
        predictor_kwargs=cfg.predictor_kwargs,
    ).to(device)
    jepa.load_state_dict(checkpoint["model"])
    
    encoder = jepa.context_encoder
    encoder.eval()
    
    # 2. Load decoder
    decoder_ckpt = torch.load(decoder_path, map_location=device, weights_only=False)
    decoder = WinProbabilityDecoder(in_features=cfg.encoder_kwargs["embed_dim"]).to(device)
    decoder.load_state_dict(decoder_ckpt["decoder"])
    decoder.eval()
    
    # 3. Test well known positions
    test_positions = [
        ("Start Position", chess.STARTING_FEN),
        ("White +Queen (K vs K+Q)", "8/8/8/8/8/8/3Q4/K1k5 w - - 0 1"),
        ("Black +Queen (K+Q vs K)", "8/8/8/8/8/8/3q4/K1k5 w - - 0 1"),
        ("Mate in 1 for White", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1"),
    ]
    
    for name, fen in test_positions:
        prob = evaluate_fen(fen, encoder, decoder, device)
        print(f"[{name}] Win Prob: {prob:.3f} | FEN: {fen}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/checkpoint_epoch0010.pt")
    parser.add_argument("--decoder", default="win_evaluation/decoder_model.pt")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    verify(args.ckpt, args.decoder, device)
