"""Loaders for Samson / Jasper Ridge .mat files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


# Jasper Ridge water-vapor / noisy bands to remove (1-based indices from the paper).
JASPER_REMOVE_BANDS_1BASED = (
    list(range(1, 4))
    + list(range(108, 113))
    + list(range(154, 167))
    + list(range(220, 225))
)


def _first_existing(keys: list[str], mapping: dict) -> np.ndarray | None:
    lower_map = {str(k).lower(): k for k in mapping.keys() if not str(k).startswith("__")}
    for key in keys:
        real = lower_map.get(key.lower())
        if real is not None:
            return np.asarray(mapping[real])
    return None


def _scalar(arr: np.ndarray | None) -> int | None:
    if arr is None:
        return None
    return int(np.asarray(arr).reshape(-1)[0])


def _to_hwl(
    cube: np.ndarray,
    n_row: int | None = None,
    n_col: int | None = None,
) -> np.ndarray:
    """
    Normalize cube to (H, W, L).

    Supports:
      - (H, W, L) / (L, H, W)
      - (L, N) with n_row * n_col == N  (Zhu Feiyun style: Samson V, Jasper Y)
    """
    cube = np.asarray(cube)
    if cube.ndim == 2:
        if n_row is None or n_col is None:
            raise ValueError(
                f"2-D cube shape {cube.shape} needs nRow/nCol metadata to reshape."
            )
        l_bands, n_pixels = cube.shape
        if n_row * n_col != n_pixels:
            # try (N, L)
            if cube.shape[0] == n_row * n_col:
                cube = cube.T
                l_bands, n_pixels = cube.shape
            else:
                raise ValueError(
                    f"Cannot reshape {cube.shape} with nRow={n_row}, nCol={n_col}"
                )
        # MATLAB/.mat stores spatial pixels in column-major (Fortran) order.
        # Using NumPy default (C) order rotates/mirrors the scene vs the paper figures.
        cube = cube.T.reshape(n_row, n_col, l_bands, order="F")
    elif cube.ndim == 3:
        if cube.shape[0] < 32 and cube.shape[1] >= 32 and cube.shape[2] >= 32:
            cube = np.transpose(cube, (1, 2, 0))  # (L,H,W) -> (H,W,L)
        elif cube.shape[1] == cube.shape[2] and cube.shape[0] != cube.shape[1]:
            cube = np.transpose(cube, (1, 2, 0))  # (L,H,W)
    else:
        raise ValueError(f"Expected 2-D or 3-D cube, got shape {cube.shape}")
    return cube.astype(np.float32)


def _to_abundance_hwP(a: np.ndarray, height: int, width: int, n_endmembers: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:
        if a.shape == (height, width, n_endmembers):
            return a
        if a.shape == (n_endmembers, height, width):
            return np.transpose(a, (1, 2, 0))
        if a.shape[0] == height and a.shape[1] == width:
            return a
    if a.ndim == 2:
        if a.shape == (height * width, n_endmembers):
            return a.reshape(height, width, n_endmembers, order="F")
        if a.shape == (n_endmembers, height * width):
            return a.T.reshape(height, width, n_endmembers, order="F")
    raise ValueError(f"Cannot interpret abundance shape {a.shape} for H={height}, W={width}, P={n_endmembers}")


def _to_endmember_LP(m: np.ndarray, n_bands: int, n_endmembers: int) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.shape == (n_bands, n_endmembers):
        return m
    if m.shape == (n_endmembers, n_bands):
        return m.T
    # fallback: bands on longer axis
    if m.ndim == 2 and max(m.shape) == n_bands:
        return m if m.shape[0] == n_bands else m.T
    raise ValueError(f"Cannot interpret endmember shape {m.shape} for L={n_bands}, P={n_endmembers}")


def remove_bands(cube: np.ndarray, remove_1based: list[int]) -> np.ndarray:
    keep = [i for i in range(cube.shape[2]) if (i + 1) not in set(remove_1based)]
    return cube[:, :, keep]


def load_mat_dataset(
    data_dir: str | Path,
    dataset_name: str,
    n_endmembers: int,
    drop_jasper_bands: bool = True,
) -> dict:
    """
    Load a real hyperspectral unmixing .mat dataset.

    Expected directory layout:
      data/raw/samson/*.mat
      data/raw/jasper_ridge/*.mat

    Flexible key names are supported (Y/X/data/cube, A/A_true, M/M_true/EDmembers, ...).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            f"See data/README.md for download and placement instructions."
        )

    mat_files = sorted(data_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found in {data_dir}\n"
            f"Place the {dataset_name} dataset here. See data/README.md."
        )

    # Prefer a single main cube file; if multiple, merge keys.
    arrays: dict[str, np.ndarray] = {}
    for path in mat_files:
        raw = loadmat(path)
        for k, v in raw.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray):
                arrays[k] = v

    cube = _first_existing(
        ["Y", "y", "X", "x", "data", "cube", "V", "img", "hsi", "samson", "jasperRidge2_R198", "jasper"],
        arrays,
    )
    if cube is None:
        candidates = [
            v for v in arrays.values() if isinstance(v, np.ndarray) and v.ndim in (2, 3) and v.size > 1000
        ]
        if not candidates:
            raise KeyError(
                f"Could not find HSI cube in {data_dir}. Keys={list(arrays.keys())}. "
                f"Rename/provide a variable such as Y / V / data / cube."
            )
        cube = max(candidates, key=lambda a: a.size)

    n_row = _scalar(_first_existing(["nRow", "nrow", "H", "height"], arrays))
    n_col = _scalar(_first_existing(["nCol", "ncol", "W", "width"], arrays))
    # Common defaults for the paper's ROIs when metadata is missing.
    if n_row is None or n_col is None:
        name = dataset_name.lower()
        if name == "samson" and cube.ndim == 2 and max(cube.shape) == 9025:
            n_row, n_col = 95, 95
        elif name.startswith("jasper") and cube.ndim == 2 and max(cube.shape) == 10000:
            n_row, n_col = 100, 100

    cube = _to_hwl(cube, n_row=n_row, n_col=n_col)

    # Zhu-style Jasper stores uint16 DN; scale by maxValue when present.
    max_value = _first_existing(["maxValue", "maxvalue"], arrays)
    if max_value is not None:
        mv = float(np.asarray(max_value).reshape(-1)[0])
        if mv > 1.0:
            cube = cube / mv

    if dataset_name.lower() in {"jasper_ridge", "jasper"} and drop_jasper_bands:
        if cube.shape[2] >= 220:
            cube = remove_bands(cube, JASPER_REMOVE_BANDS_1BASED)

    h, w, l_bands = cube.shape
    result = {
        "Y": cube,
        "dataset": dataset_name,
        "n_bands": l_bands,
        "height": h,
        "width": w,
    }

    a = _first_existing(["A", "A_true", "abundance", "abundances", "GT", "label"], arrays)
    # Do not treat spectral scaling 'S' as endmembers.
    m = _first_existing(["M", "M_true", "EDmembers", "endmembers", "E"], arrays)

    if a is not None:
        try:
            result["A"] = _to_abundance_hwP(a, h, w, n_endmembers)
        except ValueError:
            # abundance may correspond to different P; still try reshape with a.shape[-1]
            p = a.shape[0] if a.ndim == 3 and a.shape[0] <= 16 else a.shape[-1]
            result["A"] = _to_abundance_hwP(a, h, w, p)

    if m is not None:
        p = result["A"].shape[-1] if "A" in result else n_endmembers
        # if jasper bands were removed, endmembers must match remaining bands
        m_arr = np.asarray(m, dtype=np.float32)
        if m_arr.ndim == 2 and max(m_arr.shape) != l_bands and max(m_arr.shape) >= 220 and drop_jasper_bands:
            m_hwl_like = m_arr if m_arr.shape[0] > m_arr.shape[1] else m_arr.T  # (L,P)
            m_cube = m_hwl_like.T  # noop keep (L,P)
            # treat as (L,P) then remove bands
            if m_hwl_like.shape[0] >= 220:
                keep = [i for i in range(m_hwl_like.shape[0]) if (i + 1) not in set(JASPER_REMOVE_BANDS_1BASED)]
                m_arr = m_hwl_like[keep, :]
        result["M"] = _to_endmember_LP(m_arr, l_bands, p)

    return result
