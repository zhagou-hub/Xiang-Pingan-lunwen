#!/usr/bin/env python
"""Visualize abundance maps / endmembers / loss curves from a run directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ula_net.visualize import run_visualization


def parse_args():
    p = argparse.ArgumentParser(description="Visualize ULA-Net outputs")
    p.add_argument("--run-dir", type=str, required=True, help="Directory containing predictions.npz")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_visualization(run_dir)


if __name__ == "__main__":
    main()
