"""Precompute legal-move masks for a best-move dataset.

This script reads a dataset saved by best_move/generate_dataset.py and adds
`legal_masks` containing a (N, 4096) boolean mask where each row marks legal
moves for the corresponding board.

Usage:
    python best_move/precompute_masks.py \
        --input best_move/data/best_move_dataset.pt \
        --output best_move/data/best_move_dataset_masked.pt

The output file is the same dictionary as the input plus a new key:
    "legal_masks": torch.BoolTensor(N, 4096)

This lets training scripts quickly apply legal-move masking without recomputing
legal move sets every batch.
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm

# Ensure util/ is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.visualize_embeddings import tensor_to_board


def precompute_masks(input_path: str, output_path: str):
    print(f"Loading {input_path}...")
    data = torch.load(input_path, map_location="cpu", weights_only=False)

    if "boards" not in data:
        raise KeyError("Expected dataset dict to contain 'boards' key")

    boards = data["boards"]  # (N, 17, 8, 8)
    N = boards.shape[0]

    masks = torch.zeros((N, 4096), dtype=torch.bool)

    print("Precomputing 4096-bit legal move masks...")
    for i in tqdm(range(N)):
        board = tensor_to_board(boards[i])
        for move in board.legal_moves:
            idx = move.from_square * 64 + move.to_square
            masks[i, idx] = True

    data["legal_masks"] = masks

    try:
        torch.save(data, output_path)
    except Exception:
        # Fall back to legacy zip serialization for very large files
        torch.save(data, output_path, _use_new_zipfile_serialization=False)

    print(f"Saved optimized dataset to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Precompute legal-move masks for a best-move dataset")
    parser.add_argument("--input", required=True, help="Path to input dataset (torch .pt)")
    parser.add_argument("--output", required=True, help="Path to output dataset (torch .pt)")
    args = parser.parse_args()
    precompute_masks(args.input, args.output)


if __name__ == "__main__":
    main()
