#!/usr/bin/env python
"""Evaluate a trained ULA-Net checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ula_net.evaluate import run_evaluation
from ula_net.utils import load_yaml


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ULA-Net")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    cfg = load_yaml(cfg_path)
    result = run_evaluation(cfg, ckpt, project_root=ROOT)
    print(f"Evaluation finished. Metrics saved under {result['out_dir']}")


if __name__ == "__main__":
    main()
