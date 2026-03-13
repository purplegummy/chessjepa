"""
Chess V-JEPA — Dataset

Wraps the preprocessed zarr store as a PyTorch Dataset.
Includes lazy initialization for safe multiprocessing and a custom
chunk sampler to shuffle data without breaking Zarr's LZ4 cache.
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import zarr


class ZarrChunkSampler(Sampler):
    """Shuffles the order of Zarr chunks, but reads sequentially within them."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.chunk_size = dataset.chunk_size
        self.num_items = len(dataset)
        self.num_chunks = math.ceil(self.num_items / self.chunk_size)

    def __iter__(self):
        # 1. Shuffle the order of the Zarr chunks globally
        chunk_order = torch.randperm(self.num_chunks).tolist()

        # 2. Build the list of individual item indices based on the shuffled chunks
        indices = []
        for chunk_idx in chunk_order:
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.num_items)
            indices.extend(range(start_idx, end_idx))

        return iter(indices)

    def __len__(self):
        return self.num_items


class ActionChessChunkDataset(Dataset):
    """
    PyTorch Dataset that returns both board states AND chess move actions.
    Uses lazy initialization so multiple PyTorch workers don't crash the OS
    by sharing the same Zarr file handle.
    """
    NULL_SQ: int = 64

    def __init__(
        self,
        zarr_path: str,
        split: str = "train",
        val_fraction: float = 0.05,
        null_sq_idx: int = 64,
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.null_sq_idx = null_sq_idx

        # 1. Open temporarily JUST to grab metadata in the main process
        temp_store = zarr.open(zarr_path, mode="r")
        temp_boards = temp_store["boards"]
        total = temp_boards.shape[0]
        self._seq_len = temp_boards.shape[1]
        self.chunk_size = temp_boards.chunks[0]
        self.has_actions = "actions" in temp_store
        
        # Do NOT store temp_store in self. Let it garbage collect.
        self.store = None
        self.boards = None
        self.actions = None

        # 2. Setup caching variables
        self._board_chunk_idx = -1
        self._board_chunk_data = None
        self._action_chunk_idx = -1
        self._action_chunk_data = None

        val_start = int(total * (1.0 - val_fraction))

        if split == "train":
            self.start, self.end = 0, val_start
        elif split == "val":
            self.start, self.end = val_start, total
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

    def _lazy_init(self):
        """Called by the worker processes to open their own Zarr connections."""
        if self.store is None:
            self.store = zarr.open(self.zarr_path, mode="r")
            self.boards = self.store["boards"]
            if self.has_actions:
                self.actions = self.store["actions"]

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure the worker has an open connection
        self._lazy_init()

        global_idx = self.start + idx
        zarr_chunk_idx = global_idx // self.chunk_size

        # ── Load boards chunk if needed ───────────────────────────────────
        if zarr_chunk_idx != self._board_chunk_idx:
            s = zarr_chunk_idx * self.chunk_size
            e = min(s + self.chunk_size, self.boards.shape[0])
            self._board_chunk_data = np.asarray(self.boards[s:e])
            self._board_chunk_idx = zarr_chunk_idx

        local_idx = global_idx % self.chunk_size
        board_np = self._board_chunk_data[local_idx]              
        boards_t = torch.from_numpy(board_np)

        # ── Load actions chunk if needed ──────────────────────────────────
        if self.has_actions:
            if zarr_chunk_idx != self._action_chunk_idx:
                s = zarr_chunk_idx * self.chunk_size
                e = min(s + self.chunk_size, self.actions.shape[0])
                self._action_chunk_data = np.asarray(self.actions[s:e])
                self._action_chunk_idx = zarr_chunk_idx

            action_np = self._action_chunk_data[local_idx]        
            actions_t = torch.from_numpy(action_np).long()
        else:
            actions_t = torch.full((self._seq_len, 2), self.null_sq_idx, dtype=torch.long)

        return boards_t, actions_t


def build_ac_dataloaders(
    zarr_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    val_fraction: float = 0.05,
) -> tuple[DataLoader, DataLoader]:
    
    train_ds = ActionChessChunkDataset(zarr_path, split="train", val_fraction=val_fraction)
    val_ds   = ActionChessChunkDataset(zarr_path, split="val",   val_fraction=val_fraction)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=ZarrChunkSampler(train_ds), # Custom chunk-level shuffle
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False, # Validation does not need shuffling
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader