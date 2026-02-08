"""
Pytorch Datasets for global latent vectors.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def get_w_by_index(idx: int, root_dir: str) -> torch.Tensor:
    """
    Retrieve W latent vector by index from the dataset directory.

    Args:
        idx (int): Index of the sample.
        root_dir (str): Root directory of the dataset.

    Returns:
        torch.Tensor: Loaded W latent vector.

    Raises:
        FileNotFoundError: If the file cannot be found.
    """
    # Assuming directory structure: root_dir / dir_idx / idx.npy
    # where dir_idx = idx // 1000 (typical convention)
    dir_idx = idx // 1000

    # Try hierarchical path first
    w_path = os.path.join(root_dir, str(dir_idx), f"{idx}.npy")

    if not os.path.exists(w_path):
        # Fallback to flat path
        w_path = os.path.join(root_dir, f"{idx}.npy")

    if os.path.exists(w_path):
        w = np.load(w_path)
        return torch.tensor(w)
    else:
        raise FileNotFoundError(f"Latent w not found at {w_path}")


class WDataSet(Dataset):
    """
    Dataset class for loading W latent vectors.
    """

    def __init__(self, root_dir: str, length: int = 6999) -> None:
        """
        Initialize WDataSet.

        Args:
            root_dir (str): Path to dataset root.
            length (int): Total number of samples.
        """
        self.root_dir = root_dir
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        return get_w_by_index(idx, self.root_dir)


class ConcatDataset(Dataset):
    """
    Dataset wrapper to zip multiple datasets together.
    """

    def __init__(self, datasets: list[Dataset]) -> None:
        """
        Initialize ConcatDataset.

        Args:
            datasets (list[Dataset]): List of datasets to zip.
        """
        self.datasets = datasets

    def __getitem__(self, i: int) -> tuple[Any, ...]:
        return tuple(d[i] for d in self.datasets)

    def __len__(self) -> int:
        return min(len(d) for d in self.datasets)
