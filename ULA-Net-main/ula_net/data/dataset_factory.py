"""Build experiment cubes from config (synthetic or real)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .real_datasets import load_mat_dataset
from .synthetic import generate_synthetic_scene, load_synthetic_scene, save_synthetic_scene
from .vca import vca


def build_cube(cfg: dict) -> dict:
    """
    Returns a dict with at least:
      Y: (H, W, L)
      M_init: (L, P) from VCA
    and optionally A, M ground truth.
    """
    dataset = str(cfg["dataset"]).lower()
    n_endmembers = int(cfg["n_endmembers"])
    seed = int(cfg.get("seed", 42))

    if dataset == "synthetic":
        synth_cfg = cfg.get("synthetic", {})
        out_dir = Path(cfg.get("synthetic_dir", "data/synthetic"))
        cache = out_dir / "synthetic_scene.npz"
        regenerate = bool(cfg.get("regenerate_synthetic", False))
        if cache.exists() and not regenerate:
            scene = load_synthetic_scene(cache)
        else:
            merged = {
                "seed": seed,
                "height": int(synth_cfg.get("height", cfg.get("height", 70))),
                "width": int(synth_cfg.get("width", cfg.get("width", 70))),
                "n_bands": int(synth_cfg.get("n_bands", cfg.get("n_bands", 200))),
                "n_endmembers": n_endmembers,
                "length_scale": float(synth_cfg.get("length_scale", 8.0)),
                "scale_min": float(synth_cfg.get("scale_min", 0.75)),
                "scale_max": float(synth_cfg.get("scale_max", 1.25)),
                "snr_db": float(synth_cfg.get("snr_db", 30.0)),
            }
            scene = generate_synthetic_scene(merged)
            save_synthetic_scene(scene, out_dir)
        cube = scene["Y"]
        data = {"Y": cube, "A": scene["A"], "M": scene["M"], "dataset": "synthetic"}
    elif dataset in {"samson", "jasper_ridge", "jasper"}:
        name = "jasper_ridge" if dataset.startswith("jasper") else "samson"
        data_dir = cfg.get("data_dir") or (
            "data/raw/jasper_ridge" if name == "jasper_ridge" else "data/raw/samson"
        )
        data = load_mat_dataset(data_dir, name, n_endmembers=n_endmembers)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    y = data["Y"]
    h, w, l_bands = y.shape
    y_matrix = y.reshape(-1, l_bands).T  # (L, N)
    # Paper: "ten experiments and averaged the results" = 10 full train runs,
    # NOT averaging 10 VCA matrices (that blurs vertices and hurts unmixing).
    n_vca = int(cfg.get("n_vca_runs", 1))
    if n_vca <= 1:
        m_init = vca(y_matrix, n_endmembers=n_endmembers, seed=seed)
    else:
        # Optional: pick the VCA run whose simplex volume proxy is largest
        candidates = [vca(y_matrix, n_endmembers=n_endmembers, seed=seed + i) for i in range(n_vca)]
        scores = []
        for m in candidates:
            # volume proxy: product of singular values of centered endmembers
            mc = m - m.mean(axis=1, keepdims=True)
            s = np.linalg.svd(mc, compute_uv=False)
            scores.append(float(np.prod(s[:n_endmembers]) + 1e-12))
        m_init = candidates[int(np.argmax(scores))]
    data["M_init"] = m_init.astype(np.float32)
    data["n_bands"] = l_bands
    data["height"] = h
    data["width"] = w
    return data


def load_experiment_data(cfg: dict) -> dict:
    return build_cube(cfg)
