"""
Train the Win Probability Decoder on top of the Frozen JEPA Context Encoder.
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.jepa import ChessJEPA
from util.config import JEPAConfig
from win_evaluation.decoder import WinProbabilityDecoder

def train_decoder(
    ckpt_path: str,
    dataset_path: str,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    output_model_path: str = "win_evaluation/decoder_model.pt"
):
    print(f"Loading JEPA checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: JEPAConfig = checkpoint["config"]
    
    # Init context encoder
    print("Loading Context Encoder...")
    jepa = ChessJEPA(
        encoder_kwargs=cfg.encoder_kwargs,
        predictor_kwargs=cfg.predictor_kwargs,
    ).to(device)
    jepa.load_state_dict(checkpoint["model"])
    
    # We only need the context encoder, and it will be completely frozen
    encoder = jepa.context_encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
        
    print(f"Loading Win Evaluation Dataset: {dataset_path}")
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    boards = data["boards"] # (N, 17, 8, 8)
    win_probs = data["win_probs"] # (N, 1)
    
    # We add sequence and chunk dimensions to the boards (N, 1, 1, 17, 8, 8) 
    # to match the JEPA encoder's expected (B, T, V, 17, 8, 8) for spatiotemporal
    # Or simplified to just (B, 1, 17, 8, 8) or whatever the encoder expects.
    # The normal training loop passes (B, 16, 17, 8, 8)
    dataset = TensorDataset(boards, win_probs)
    
    # 80/20 train/val split
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    print(f"Initializing WinProbabilityDecoder on {device}...")
    decoder = WinProbabilityDecoder(in_features=cfg.encoder_kwargs["embed_dim"], hidden_features=512, num_layers=3).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)
    
    print("-" * 50)
    print(f"Training on {train_size} samples. Validating on {val_size} samples.")
    print("-" * 50)
    
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        decoder.train()
        train_loss = 0.0
        
        t0 = time.time()
        for batch_boards, batch_probs in train_loader:
            # Add temporal dimensions since context encoder expects at least (B, SeqLen, ...)
            # We are extracting 1 board snapshot at a time, so SeqLen=1
            b = batch_boards.unsqueeze(1).to(device) # (B, 1, 17, 8, 8)
            t = batch_probs.to(device) # (B, 1)
            
            optimizer.zero_grad()
            
            # Extract latents (no grad tracking since encoder is frozen)
            with torch.no_grad():
                # encoder returns (B*SpatialTokens, EmbedDim) if flat, or 
                # depending on context_encoder return shape
                latents = encoder(b) # -> (B, L, EmbedDim)
                
            pred = decoder(latents) # -> (B, 1)
            
            loss = criterion(pred, t)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_boards.size(0)
            
        train_loss /= train_size
        
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_boards, batch_probs in val_loader:
                b = batch_boards.unsqueeze(1).to(device)
                t = batch_probs.to(device)
                
                latents = encoder(b)
                pred = decoder(latents)
                
                loss = criterion(pred, t)
                val_loss += loss.item() * batch_boards.size(0)
                
        val_loss /= val_size
        time_elapsed = time.time() - t0
        
        print(f"Epoch {epoch+1:2d}/{epochs:2d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Time: {time_elapsed:.2f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            torch.save({
                "decoder": decoder.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, output_model_path)
            
    print(f"Training complete. Best Validation MSE: {best_val_loss:.4f}")
    print(f"Best decoder weights saved to {output_model_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/checkpoint_epoch0010.pt", help="Path to frozen JEPA checkpoint")
    parser.add_argument("--dataset", default="win_evaluation/win_eval_dataset.pt", help="Path to generated target labels dataset")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()
    
    train_decoder(args.ckpt, args.dataset, args.batch, args.epochs, args.lr)
