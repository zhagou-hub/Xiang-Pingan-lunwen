"""Patch extraction utilities for ULA-Net."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def pad_cube(cube: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Reflect-pad HSI cube (H, W, L) so every pixel has a full KxK neighborhood.
    """
    if patch_size % 2 == 0:
        raise ValueError(f"patch_size must be odd, got {patch_size}")
    pad = patch_size // 2
    return np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")


def extract_patches(cube: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Extract a patch for every spatial location.

    Parameters
    ----------
    cube : (H, W, L)
    patch_size : K (odd)

    Returns
    -------
    patches : (H*W, L, K, K)
    """
    h, w, l_bands = cube.shape
    padded = pad_cube(cube, patch_size)
    k = patch_size
    patches = np.empty((h * w, l_bands, k, k), dtype=np.float32)
    idx = 0
    for i in range(h):
        for j in range(w):
            patch = padded[i : i + k, j : j + k, :]  # (K, K, L)
            patches[idx] = np.transpose(patch, (2, 0, 1))
            idx += 1
    return patches


def center_spectra_from_cube(cube: np.ndarray) -> np.ndarray:
    """Return center-pixel spectra as (H*W, L)."""
    h, w, l_bands = cube.shape
    return cube.reshape(h * w, l_bands).astype(np.float32)


class HyperspectralPatchDataset(Dataset):
    """Dataset yielding (patch, center_spectrum) pairs."""

    def __init__(self, cube: np.ndarray, patch_size: int) -> None:
        self.cube = np.asarray(cube, dtype=np.float32)
        self.patch_size = patch_size
        self.patches = extract_patches(self.cube, patch_size)
        self.centers = center_spectra_from_cube(self.cube)
        self.height, self.width, self.n_bands = self.cube.shape

    def __len__(self) -> int:
        return self.patches.shape[0]

    def __getitem__(self, index: int):
        patch = torch.from_numpy(self.patches[index])
        center = torch.from_numpy(self.centers[index])
        return patch, center
