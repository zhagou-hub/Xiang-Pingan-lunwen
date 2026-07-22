#!/usr/bin/env python
"""
Paper protocol: run the full experiment N times (default 10) with different seeds,
then average aRMSE / eSAD.

All files are written flat under outputs/<dataset>/  (no nested run folders).
Example: outputs/samson/seed42_best.pt, outputs/samson/metrics_mean.json

Usage:
  python scripts/train_multirun.py --config configs/samson.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ula_net.train import run_training
from ula_net.utils import ensure_dir, load_yaml, save_json


def parse_args():
    p = argparse.ArgumentParser(description="ULA-Net multi-run training (paper: 10 runs)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--n-runs", type=int, default=None, help="Override n_experiments (paper=10)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--base-seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_yaml(cfg_path)

    n_runs = int(args.n_runs if args.n_runs is not None else cfg.get("n_experiments", 10))
    base_seed = int(args.base_seed if args.base_seed is not None else cfg.get("seed", 42))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    out_dir = ensure_dir(
        ROOT / cfg.get("output_root", "outputs") / str(cfg["dataset"])
    )
    all_metrics: list[dict] = []

    print(f"Paper-style multi-run: n_runs={n_runs}, epochs={cfg.get('epochs', 100)}, base_seed={base_seed}")
    print(f"All outputs -> {out_dir}")

    for i in range(n_runs):
        seed = base_seed + i
        run_cfg = dict(cfg)
        run_cfg["seed"] = seed
        run_cfg["file_prefix"] = f"seed{seed}_"
        print("\n" + "=" * 60)
        print(f"Run {i + 1}/{n_runs}  seed={seed}")
        print("=" * 60)
        result = run_training(run_cfg, project_root=ROOT)
        metrics = result.get("metrics") or {}
        metrics["seed"] = seed
        all_metrics.append(metrics)
        save_json(out_dir / f"seed{seed}_metrics.json", metrics)
        print(f"Run {i + 1} metrics: {metrics}")

    keys = [k for k in all_metrics[0].keys() if k != "seed"]
    mean = {k: float(np.mean([m[k] for m in all_metrics if k in m])) for k in keys}
    std = {k: float(np.std([m[k] for m in all_metrics if k in m])) for k in keys}
    payload = {
        "n_runs": n_runs,
        "base_seed": base_seed,
        "epochs": int(cfg.get("epochs", 100)),
        "per_run": all_metrics,
        "mean": mean,
        "std": std,
    }
    save_json(out_dir / "metrics_mean.json", payload)
    print("\n" + "=" * 60)
    print("MULTI-RUN MEAN (± std)")
    for k in keys:
        print(f"  {k}: {mean[k]:.6f} ± {std[k]:.6f}")
    print(f"Saved: {out_dir / 'metrics_mean.json'}")


if __name__ == "__main__":
    main()
