"""Synthetic HSI generation following the paper's ELMM-style mixing model."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _gaussian_random_field(height: int, width: int, length_scale: float, rng: np.random.Generator) -> np.ndarray:
    """Approximate smooth field via low-frequency filtered noise."""
    noise = rng.normal(size=(height, width))
    # FFT low-pass smoothing for a spatially correlated abundance field
    noise_f = np.fft.fft2(noise)
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    scale = np.exp(-2 * (np.pi**2) * (length_scale**2) * (fx**2 + fy**2))
    field = np.real(np.fft.ifft2(noise_f * scale))
    field = (field - field.min()) / (field.max() - field.min() + 1e-12)
    return field


def generate_endmembers(n_bands: int, n_endmembers: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate smooth synthetic endmember signatures, shape (L, P).
    """
    wavelengths = np.linspace(0, 1, n_bands)
    endmembers = np.zeros((n_bands, n_endmembers), dtype=np.float64)
    for p in range(n_endmembers):
        spectrum = np.zeros(n_bands)
        n_peaks = rng.integers(2, 5)
        for _ in range(n_peaks):
            center = rng.uniform(0.05, 0.95)
            width = rng.uniform(0.02, 0.12)
            amp = rng.uniform(0.2, 1.0)
            spectrum += amp * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
        spectrum += rng.uniform(0.05, 0.25)
        spectrum = np.clip(spectrum, 1e-3, None)
        spectrum = spectrum / spectrum.max()
        endmembers[:, p] = spectrum
    return endmembers


def generate_abundances(
    height: int,
    width: int,
    n_endmembers: int,
    length_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate spatially smooth abundances with ANC + ASC, shape (H, W, P).
    """
    fields = np.stack(
        [_gaussian_random_field(height, width, length_scale, rng) for _ in range(n_endmembers)],
        axis=-1,
    )
    # sharpen a bit for sparsity
    fields = fields ** 1.5
    abundances = fields / (fields.sum(axis=-1, keepdims=True) + 1e-12)
    return abundances.astype(np.float64)


def apply_elmm_mixing(
    endmembers: np.ndarray,
    abundances: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    scale_min: float = 0.75,
    scale_max: float = 1.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Y = M A S + N  (pixel-wise diagonal scaling), returns cube (H,W,L) and scales (H,W,P).
    endmembers: (L, P), abundances: (H, W, P)

    Paper: scaling factors drawn in [0.75, 1.25] via a 2-D Gaussian field, then clipped.
    """
    h, w, p = abundances.shape
    l_bands = endmembers.shape[0]
    # Spatially smooth scales in [scale_min, scale_max]
    scales = np.zeros((h, w, p), dtype=np.float64)
    for k in range(p):
        field = _gaussian_random_field(h, w, length_scale=max(h, w) / 8.0, rng=rng)
        scales[:, :, k] = scale_min + (scale_max - scale_min) * field

    cube = np.zeros((h, w, l_bands), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            m_scaled = endmembers * scales[i, j][None, :]
            cube[i, j] = m_scaled @ abundances[i, j]

    power = np.mean(cube**2)
    noise_power = power / (10 ** (snr_db / 10.0) + 1e-12)
    noise = rng.normal(0.0, np.sqrt(noise_power), size=cube.shape)
    cube = np.clip(cube + noise, 0.0, None)
    return cube.astype(np.float32), scales.astype(np.float32)


def generate_synthetic_scene(cfg: dict) -> dict[str, np.ndarray]:
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    height = int(cfg["height"])
    width = int(cfg["width"])
    n_bands = int(cfg["n_bands"])
    n_endmembers = int(cfg["n_endmembers"])
    length_scale = float(cfg.get("length_scale", 8.0))
    snr_db = float(cfg.get("snr_db", 30.0))
    scale_min = float(cfg.get("scale_min", 0.75))
    scale_max = float(cfg.get("scale_max", 1.25))

    endmembers = generate_endmembers(n_bands, n_endmembers, rng)
    abundances = generate_abundances(height, width, n_endmembers, length_scale, rng)
    cube, scales = apply_elmm_mixing(
        endmembers,
        abundances,
        snr_db,
        rng,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    return {
        "Y": cube,  # (H, W, L)
        "A": abundances.astype(np.float32),
        "M": endmembers.astype(np.float32),  # (L, P)
        "S": scales,
    }


def save_synthetic_scene(scene: dict[str, np.ndarray], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "synthetic_scene.npz"
    np.savez_compressed(path, **scene)
    return path


def load_synthetic_scene(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}
