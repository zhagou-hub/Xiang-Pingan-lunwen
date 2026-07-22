#!/usr/bin/env python
"""Generate and cache the synthetic hyperspectral scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ula_net.data.synthetic import generate_synthetic_scene, save_synthetic_scene
from ula_net.utils import load_yaml


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic HSI for ULA-Net")
    p.add_argument("--config", type=str, default="configs/synthetic.yaml")
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_yaml(cfg_path)
    synth = cfg.get("synthetic", cfg)
    scene_cfg = {
        "seed": int(cfg.get("seed", 42)),
        "height": int(synth.get("height", 64)),
        "width": int(synth.get("width", 64)),
        "n_bands": int(synth.get("n_bands", 200)),
        "n_endmembers": int(cfg.get("n_endmembers", synth.get("n_endmembers", 4))),
        "length_scale": float(synth.get("length_scale", 8.0)),
        "scale_min": float(synth.get("scale_min", 0.75)),
        "scale_max": float(synth.get("scale_max", 1.25)),
        "snr_db": float(synth.get("snr_db", 30.0)),
    }
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / cfg.get("synthetic_dir", "data/synthetic")
    scene = generate_synthetic_scene(scene_cfg)
    path = save_synthetic_scene(scene, out_dir)
    print(f"Saved synthetic scene to {path}")
    print(f"Y={scene['Y'].shape}, A={scene['A'].shape}, M={scene['M'].shape}")


if __name__ == "__main__":
    main()
