"""Vertex Component Analysis (VCA) — Nascimento & Bioucas-Dias style."""

from __future__ import annotations

import numpy as np


def vca(y: np.ndarray, n_endmembers: int, seed: int = 0) -> np.ndarray:
    """
    Extract endmembers with VCA.

    Parameters
    ----------
    y : (L, N) bands x pixels
    n_endmembers : P

    Returns
    -------
    (L, P) endmember matrix
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"y must be (L, N), got {y.shape}")

    l_bands, n_pixels = y.shape
    p = int(n_endmembers)
    if p < 1 or p > min(l_bands, n_pixels):
        raise ValueError(f"invalid n_endmembers={p} for shape {y.shape}")

    # SNR estimate
    y_mean = y.mean(axis=1, keepdims=True)
    yd = y - y_mean
    cov = (yd @ yd.T) / n_pixels
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(np.sort(eigvals)[::-1], 1e-12, None)
    snr_est = 10 * np.log10(eigvals[:p].sum() / (eigvals[p:].sum() + 1e-12))
    snr_th = 15 + 10 * np.log10(p)

    if snr_est > snr_th:
        u, _, _ = np.linalg.svd(y @ y.T / n_pixels, full_matrices=False)
        u = u[:, :p]
        y_p = u.T @ y
        y_p = y_p / (np.sum(np.abs(y_p), axis=0, keepdims=True) + 1e-12)
        proj = u
    else:
        u, _, _ = np.linalg.svd(yd @ yd.T / n_pixels, full_matrices=False)
        u = u[:, : p - 1]
        c = np.sqrt(np.mean(y**2))
        y_p = np.vstack([u.T @ y, c * np.ones((1, n_pixels))])
        y_p = y_p / (np.sum(np.abs(y_p), axis=0, keepdims=True) + 1e-12)
        proj = np.hstack([u, np.zeros((l_bands, 1))])
        # reconstruct helper unused; select from original y via indices

    indices = np.zeros(p, dtype=int)
    a = np.zeros((p, p), dtype=np.float64)
    a[0, 0] = 1.0
    for i in range(p):
        w = rng.standard_normal(p)
        f = w - a @ np.linalg.pinv(a) @ w
        nrm = np.linalg.norm(f)
        f = f / (nrm + 1e-12)
        scores = np.abs(f @ y_p)
        idx = int(np.argmax(scores))
        indices[i] = idx
        a[:, i] = y_p[:, idx]

    return y[:, indices].astype(np.float32)
