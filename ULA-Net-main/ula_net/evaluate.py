"""Evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ula_net.data.dataset_factory import build_cube
from ula_net.metrics import abundance_rmse, endmember_sad, match_endmembers_by_sad
from ula_net.train import create_model, infer_full_image
from ula_net.utils import get_device, save_json, set_seed


def evaluate_predictions(
    preds: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if "A" in data and data["A"] is not None and np.size(data["A"]) > 0:
        a_est = preds["A"]
        a_true = data["A"]
        # align endmember order via SAD matching if M available
        if "M" in data and np.size(data["M"]) > 0:
            m_true, m_est, perm = match_endmembers_by_sad(data["M"], preds["M"])
            a_est = a_est[..., perm]
            metrics["eSAD"] = endmember_sad(m_true, m_est)
            metrics["aRMSE"] = abundance_rmse(a_true, a_est)
        else:
            metrics["aRMSE"] = abundance_rmse(a_true, a_est)
    elif "M" in data and np.size(data.get("M", [])) > 0:
        m_true, m_est, _ = match_endmembers_by_sad(data["M"], preds["M"])
        metrics["eSAD"] = endmember_sad(m_true, m_est)
    return metrics


def run_evaluation(cfg: dict, checkpoint: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    project_root = Path(project_root) if project_root else Path.cwd()
    checkpoint = Path(checkpoint)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(prefer_cuda=bool(cfg.get("prefer_cuda", True)))

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = dict(cfg)
    cfg.update({k: v for k, v in ckpt.get("cfg", {}).items() if k not in cfg or cfg[k] is None})

    data = build_cube(cfg)
    cfg["n_bands"] = int(data["n_bands"])
    model = create_model(cfg, ckpt.get("m_init", data["M_init"]), device)
    model.load_state_dict(ckpt["model_state"])

    preds = infer_full_image(
        model,
        data["Y"],
        patch_size=int(cfg["patch_size"]),
        device=device,
        batch_size=int(cfg.get("eval_batch_size", 256)),
    )
    metrics = evaluate_predictions(preds, data)
    out_dir = checkpoint.parent
    save_json(out_dir / "metrics.json", metrics)
    np.savez_compressed(
        out_dir / "predictions_eval.npz",
        A=preds["A"],
        Y_hat=preds["Y_hat"],
        S=preds["S"],
        M=preds["M"],
    )
    print("Metrics:", metrics)
    return {"metrics": metrics, "predictions": preds, "data": data, "out_dir": out_dir}
