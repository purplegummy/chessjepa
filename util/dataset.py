"""
Chess V-JEPA — Dataset

Wraps the preprocessed zarr store as a PyTorch Dataset.

Your zarr store (chess_chunks.zarr) has this structure:
    └── boards : uint8 array of shape (N, 16, 18, 8, 8)
                 N chunks, each with 16 consecutive board positions
                 18 channels per board (12 piece planes + 4 castling + 1 en passant planes,
                 color-invariant via board flip so current player is always "at the bottom")

Each __getitem__ call returns one chunk: (16, 18, 8, 8) — a sequence of
16 board states ready to be split into context/target by the masking strategy.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import zarr


class ChessChunkDataset(Dataset):
    """
    PyTorch Dataset backed by a zarr store.

    Why zarr instead of loading everything into RAM?
    ─────────────────────────────────────────────────
    •  zarr arrays are chunked and compressed on disk (using LZ4).
    •  We memory-map them lazily — only the chunks actually accessed
       get decompressed and loaded.  This lets us handle datasets
       much larger than available RAM.
    •  Random access by index is O(1) since zarr stores chunk offsets.

    Args:
        zarr_path : path to the zarr store (e.g. "chess_chunks.zarr")
        split     : "train" or "val" — uses last 5% as validation
    """

    def __init__(self, zarr_path: str, split: str = "train", val_fraction: float = 0.05):
        super().__init__()
        # Use ThreadSynchronizer for thread-safe access if using multiple workers
        # (though PyTorch workers are separate processes, it's good practice)
        store = zarr.open(zarr_path, mode="r")
        self.boards = store["boards"]  # shape: (N, 16, 18, 8, 8)
        
        # Zarr chunks are (128, 16, 18, 8, 8). If we ask for idx=0, it decompresses
        # the entire chunk. If we don't cache it, idx=1 decompresses it again!
        self.chunk_size = self.boards.chunks[0]  # Should be 256
        self.cached_chunk_idx = -1
        self.cached_chunk_data = None

        total = self.boards.shape[0]
        val_start = int(total * (1.0 - val_fraction))

        if split == "train":
            self.start = 0
            self.end = val_start
        elif split == "val":
            self.start = val_start
            self.end = total
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            boards : (16, 18, 8, 8) uint8 tensor — one game chunk
        """
        global_idx = self.start + idx
        
        # Which zarr chunk does this index belong to?
        zarr_chunk_idx = global_idx // self.chunk_size
        
        # If we don't have this chunk in memory, load the ENTIRE chunk (256 items)
        # This is a massive speedup vs reading 1 item (which secretly loads 256 anyway)
        if zarr_chunk_idx != self.cached_chunk_idx:
            start_idx = zarr_chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.boards.shape[0])
            # Load into RAM as numpy array
            self.cached_chunk_data = np.asarray(self.boards[start_idx:end_idx])
            self.cached_chunk_idx = zarr_chunk_idx
            
        # Get the item from our RAM cache
        local_idx = global_idx % self.chunk_size
        raw = self.cached_chunk_data[local_idx]
        
        return torch.from_numpy(raw)  # float32 tensor


def build_dataloaders(
    zarr_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    val_fraction: float = 0.05,
) -> tuple[DataLoader, DataLoader]:
    """
    Convenience function to create train and val DataLoaders.

    Args:
        zarr_path    : path to the zarr store
        batch_size   : samples per batch (default 64)
        num_workers  : parallel data loading workers (default 4)
        val_fraction : fraction of data for validation (default 0.05 = 5%)

    Returns:
        (train_loader, val_loader)
    """
    train_ds = ChessChunkDataset(zarr_path, split="train", val_fraction=val_fraction)
    val_ds = ChessChunkDataset(zarr_path, split="val", val_fraction=val_fraction)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,          # MUST BE FALSE for sequential zarr caching
        num_workers=num_workers,
        pin_memory=True,        # faster GPU transfer on NVIDIA
        drop_last=True,         # drop incomplete final batch for stable training
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Action-Conditioned Dataset
# ─────────────────────────────────────────────────────────────────────────────


class ActionChessChunkDataset(Dataset):
    """
    PyTorch Dataset that returns both board states AND chess move actions.

    Expected zarr store structure
    ─────────────────────────────
        boards  : uint8    (N, T, 18, 8, 8)   — board state sequences
        actions : int16    (N, T, 2)           — move sequences
                  actions[i, t] = (from_sq, to_sq) of the move that produced
                  boards[i, t].  Use 64 for null / unknown (e.g. first board).

    If the zarr store does not contain an 'actions' array (e.g. the original
    boards-only store), __getitem__ will return a null-filled actions tensor
    (all 64s) of shape (T, 2).  This lets you run the AC model in a degraded
    "no-action" mode while the data pipeline is being upgraded.

    Args:
        zarr_path     : path to the zarr store
        split         : "train" or "val"
        val_fraction  : fraction held out for validation (default 0.05)
        null_sq_idx   : square index to use for null / padding moves (default 64)
    """

    NULL_SQ: int = 64   # the "no move" sentinel (outside valid 0-63 board squares)

    def __init__(
        self,
        zarr_path: str,
        split: str = "train",
        val_fraction: float = 0.05,
        null_sq_idx: int = 64,
    ):
        super().__init__()
        self.null_sq_idx = null_sq_idx

        store = zarr.open(zarr_path, mode="r")

        self.boards = store["boards"]       # (N, T, 18, 8, 8)
        self.has_actions = "actions" in store
        self.actions = store["actions"] if self.has_actions else None

        # Chunk-level cache (same strategy as ChessChunkDataset)
        self.chunk_size = self.boards.chunks[0]
        self._board_chunk_idx = -1
        self._board_chunk_data = None
        self._action_chunk_idx = -1
        self._action_chunk_data = None

        total = self.boards.shape[0]
        val_start = int(total * (1.0 - val_fraction))

        if split == "train":
            self.start, self.end = 0, val_start
        elif split == "val":
            self.start, self.end = val_start, total
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        # Infer T from the boards array shape
        self._seq_len = self.boards.shape[1]

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            boards  : (T, 18, 8, 8) uint8  tensor
            actions : (T, 2)        int64  tensor  — (from_sq, to_sq) per step
                      null moves are filled with self.null_sq_idx (64)
        """
        global_idx = self.start + idx
        zarr_chunk_idx = global_idx // self.chunk_size

        # ── Load boards chunk if needed ───────────────────────────────────
        if zarr_chunk_idx != self._board_chunk_idx:
            s = zarr_chunk_idx * self.chunk_size
            e = min(s + self.chunk_size, self.boards.shape[0])
            self._board_chunk_data = np.asarray(self.boards[s:e])
            self._board_chunk_idx = zarr_chunk_idx

        local_idx = global_idx % self.chunk_size
        board_np = self._board_chunk_data[local_idx]              # (T, 18, 8, 8)
        boards_t = torch.from_numpy(board_np)

        # ── Load actions chunk if needed ──────────────────────────────────
        if self.has_actions:
            if zarr_chunk_idx != self._action_chunk_idx:
                s = zarr_chunk_idx * self.chunk_size
                e = min(s + self.chunk_size, self.actions.shape[0])
                self._action_chunk_data = np.asarray(self.actions[s:e])
                self._action_chunk_idx = zarr_chunk_idx

            action_np = self._action_chunk_data[local_idx]        # (T, 2)
            actions_t = torch.from_numpy(action_np).long()
        else:
            # Fallback: all null moves — AC predictor still runs but gets no info
            actions_t = torch.full((self._seq_len, 2), self.null_sq_idx, dtype=torch.long)

        return boards_t, actions_t


def build_ac_dataloaders(
    zarr_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    val_fraction: float = 0.05,
) -> tuple[DataLoader, DataLoader]:
    """
    Convenience function to create action-conditioned train and val DataLoaders.

    Each batch yields:
        boards  : (B, T, 18, 8, 8) uint8
        actions : (B, T, 2)        int64

    Args:
        zarr_path    : path to the zarr store
        batch_size   : samples per batch (default 64)
        num_workers  : parallel data loading workers (default 4)
        val_fraction : fraction of data for validation (default 0.05 = 5%)

    Returns:
        (train_loader, val_loader)
    """
    train_ds = ActionChessChunkDataset(zarr_path, split="train", val_fraction=val_fraction)
    val_ds   = ActionChessChunkDataset(zarr_path, split="val",   val_fraction=val_fraction)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,       # Keep False for sequential zarr chunk caching
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
