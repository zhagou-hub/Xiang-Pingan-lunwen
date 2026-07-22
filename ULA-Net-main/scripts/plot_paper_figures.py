#!/usr/bin/env python
"""Generate paper-style GT vs Ours abundance maps and endmember curves."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ula_net.visualize import run_paper_style_visualization


DEFAULT_DIRS = [
    "outputs/synthetic",
    "outputs/samson",
    "outputs/jasper_ridge",
]


def parse_args():
    p = argparse.ArgumentParser(description="Paper-style GT vs Ours figures")
    p.add_argument(
        "--run-dir",
        type=str,
        action="append",
        default=None,
        help="Dataset output dir, e.g. outputs/samson (repeatable).",
    )
    p.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Specific predictions file name, e.g. seed42_predictions.npz",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="",
        help="File prefix, e.g. seed42_",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_dirs = args.run_dir or DEFAULT_DIRS
    for rd in run_dirs:
        path = Path(rd)
        if not path.is_absolute():
            path = ROOT / path
        run_paper_style_visualization(
            path,
            predictions_name=args.predictions,
            prefix=args.prefix,
        )
    print("Done.")


if __name__ == "__main__":
    main()
