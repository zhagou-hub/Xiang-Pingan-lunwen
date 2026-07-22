"""Evaluation metrics: aRMSE and eSAD."""

from __future__ import annotations

import numpy as np


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def abundance_rmse(a_true: np.ndarray, a_est: np.ndarray) -> float:
    """
    Abundance RMSE.

    Accepts shapes (N, P) or (H, W, P).
    """
    a_true = _as_numpy(a_true).reshape(-1, a_true.shape[-1])
    a_est = _as_numpy(a_est).reshape(-1, a_est.shape[-1])
    if a_true.shape != a_est.shape:
        raise ValueError(f"abundance shape mismatch: {a_true.shape} vs {a_est.shape}")
    return float(np.sqrt(np.mean((a_true - a_est) ** 2)))


def endmember_sad(m_true: np.ndarray, m_est: np.ndarray, eps: float = 1e-12) -> float:
    """
    Mean spectral angle distance (radians) over endmembers.

    Accepts (P, L) or (L, P). Arrays are aligned to (P, L).
    """
    m_true = _as_numpy(m_true)
    m_est = _as_numpy(m_est)
    if m_true.shape != m_est.shape:
        if m_true.T.shape == m_est.shape:
            m_true = m_true.T
        else:
            raise ValueError(f"endmember shape mismatch: {m_true.shape} vs {m_est.shape}")
    # Prefer (P, L): more bands than endmembers typically.
    if m_true.shape[0] > m_true.shape[1]:
        m_true = m_true.T
        m_est = m_est.T

    sads = []
    for i in range(m_true.shape[0]):
        t = m_true[i]
        e = m_est[i]
        cos = np.dot(t, e) / (np.linalg.norm(t) * np.linalg.norm(e) + eps)
        cos = np.clip(cos, -1.0, 1.0)
        sads.append(float(np.arccos(cos)))
    return float(np.mean(sads))


def match_endmembers_by_sad(
    m_true: np.ndarray,
    m_est: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Greedy match estimated endmembers to ground-truth by SAD.

    Returns reordered m_est (P,L), permutation indices, and leaves m_true as (P,L).
    """
    m_true = _as_numpy(m_true)
    m_est = _as_numpy(m_est)
    if m_true.shape[0] > m_true.shape[1]:
        m_true = m_true.T
    if m_est.shape[0] > m_est.shape[1]:
        m_est = m_est.T

    p = m_true.shape[0]
    used = set()
    perm: list[int] = []
    for i in range(p):
        best_j, best_sad = -1, 1e9
        for j in range(p):
            if j in used:
                continue
            cos = np.dot(m_true[i], m_est[j]) / (
                np.linalg.norm(m_true[i]) * np.linalg.norm(m_est[j]) + 1e-12
            )
            sad = float(np.arccos(np.clip(cos, -1.0, 1.0)))
            if sad < best_sad:
                best_sad, best_j = sad, j
        used.add(best_j)
        perm.append(best_j)
    return m_true, m_est[perm], perm
