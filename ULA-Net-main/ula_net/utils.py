"""Common helpers: device, seeding, config I/O, run directories."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_output_dir(output_root: str | Path, dataset_name: str) -> Path:
    """Flat layout: outputs/<dataset>/  (no nested run folders)."""
    return ensure_dir(Path(output_root) / dataset_name)


def make_run_dir(output_root: str | Path, dataset_name: str, run_name: str | None = None) -> Path:
    """
    Always return outputs/<dataset>/ only.
    run_name is ignored for directory creation (kept for API compatibility).
    Use file prefixes (e.g. seed_42_) to distinguish multi-runs.
    """
    return dataset_output_dir(output_root, dataset_name)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_config_copy(cfg: dict[str, Any], run_dir: Path, filename: str = "config.yaml") -> None:
    ensure_dir(run_dir)
    with open(run_dir / filename, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
