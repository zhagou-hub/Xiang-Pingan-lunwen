#!/usr/bin/env python
"""Train ULA-Net from a YAML config (single run)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ula_net.train import run_training
from ula_net.utils import load_yaml


def parse_args():
    p = argparse.ArgumentParser(description="Train ULA-Net")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs")
    p.add_argument("--seed", type=int, default=None, help="Override seed")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_yaml(cfg_path)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg["file_prefix"] = f"seed{int(cfg.get('seed', 42))}_"
    result = run_training(cfg, project_root=ROOT)
    print(f"Training finished. Outputs: {result['run_dir']} (prefix={result['file_prefix']})")


if __name__ == "__main__":
    main()
