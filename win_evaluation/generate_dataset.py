"""
Script to generate a subset of the dataset with win probability labels using Stockfish.
"""

import argparse
import os
import sys
import random
import torch
import numpy as np
import chess
import chess.engine
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.dataset import ChessChunkDataset
from util.visualize_embeddings import tensor_to_board

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

def cp_to_wdl(cp: int) -> float:
    """
    Convert Centipawn evaluation to a Win Probability (WDL) in [0.0, 1.0].
    Using the Lichess formula: 
    Win Prob = 50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1) / 100
    """
    win_prob_percent = 50 + 50 * (2 / (1 + np.exp(-0.00368208 * cp)) - 1)
    return win_prob_percent / 100.0

def generate_dataset(zarr_path: str, output_path: str, num_samples: int, time_limit: float = 0.05):
    print(f"Loading dataset from: {zarr_path}")
    dataset = ChessChunkDataset(zarr_path, split="train") # Can use train or val
    
    # Initialize engine
    if not os.path.exists(STOCKFISH_PATH):
        print(f"Error: Stockfish engine not found at {STOCKFISH_PATH}")
        sys.exit(1)
        
    print(f"Initializing Stockfish from {STOCKFISH_PATH}")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    
    board_tensors = []
    win_probs = []
    
    print(f"Generating labels for {num_samples} samples...")
    for _ in tqdm(range(num_samples)):
        chunk_idx = random.randint(0, len(dataset) - 1)
        chunk = dataset[chunk_idx] # (16, 17, 8, 8)
        pos_idx = random.randint(0, 15)
        
        board_tensor = chunk[pos_idx] # (17, 8, 8)
        board = tensor_to_board(board_tensor)
        
        try:
            # We evaluate from White's perspective
            info = engine.analyse(board, chess.engine.Limit(time=time_limit))
            score = info["score"].white()
            
            if score.is_mate():
                # Mate in N: assign 1.0 for mate-in-X, 0.0 for mated-in-X
                cp = 10000 if score.mate() > 0 else -10000
            else:
                cp = score.score()
                
            wdl = cp_to_wdl(cp)
            
            board_tensors.append(board_tensor.clone())
            win_probs.append(wdl)
            
        except Exception as e:
            print(f"Error evaluating position: {e}")
            continue
            
    engine.quit()
    
    if board_tensors:
         boards = torch.stack(board_tensors)
         probs = torch.tensor(win_probs, dtype=torch.float32).unsqueeze(1) # (N, 1)
         
         print(f"Saving dataset to {output_path}...")
         print(f"Boards shape: {boards.shape}, Probs shape: {probs.shape}")
         os.makedirs(os.path.dirname(output_path), exist_ok=True)
         torch.save({"boards": boards, "win_probs": probs}, output_path)
         print("Done!")
    else:
         print("Failed to generate any samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default="data/chess_chunks.zarr", help="Path to zarr dataset")
    parser.add_argument("--out", default="win_evaluation/win_eval_dataset.pt", help="Output path for the .pt dataset")
    parser.add_argument("--samples", type=int, default=10000, help="Number of positions to evaluate")
    parser.add_argument("--time", type=float, default=0.01, help="Time limit per evaluation in seconds")
    args = parser.parse_args()
    
    generate_dataset(args.zarr, args.out, args.samples, args.time)
