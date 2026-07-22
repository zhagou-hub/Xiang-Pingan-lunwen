"""Training loop for ULA-Net."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ula_net.data.dataset_factory import build_cube
from ula_net.data.patches import HyperspectralPatchDataset
from ula_net.losses import ULANetLoss
from ula_net.models.ula_net import RuaAE
from ula_net.utils import ensure_dir, get_device, make_run_dir, save_config_copy, save_json, set_seed


def _file_prefix(cfg: dict) -> str:
    """Prefix for flat files in outputs/<dataset>/, e.g. seed42_."""
    if cfg.get("file_prefix"):
        return str(cfg["file_prefix"])
    seed = cfg.get("seed")
    if seed is not None:
        return f"seed{int(seed)}_"
    return ""


def create_model(cfg: dict, m_init: np.ndarray, device: torch.device) -> RuaAE:
    endmembers = torch.from_numpy(np.asarray(m_init, dtype=np.float32))  # (L, P)
    model = RuaAE(
        n_bands=int(cfg.get("n_bands", m_init.shape[0])),
        n_endmembers=int(cfg["n_endmembers"]),
        patch_size=int(cfg["patch_size"]),
        endmembers=endmembers,
        model_dim=int(cfg.get("model_dim", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        scale_range=float(cfg.get("epsilon", 0.2)),
    )
    return model.to(device)


def train_one_epoch(
    model: RuaAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ULANetLoss,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "loss_re": 0.0,
        "loss_mse": 0.0,
        "loss_sparsity": 0.0,
        "loss_gate": 0.0,
    }
    n_batches = 0
    for patches, centers in loader:
        patches = patches.to(device)
        centers = centers.to(device)
        optimizer.zero_grad(set_to_none=True)
        recon, abd, _scale = model(patches, return_abd=True)
        loss, stats = criterion(centers, recon, abd, model.gate_values)
        loss.backward()
        optimizer.step()
        model.project_parameters_()
        for k in totals:
            totals[k] += stats.get(k, 0.0)
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def infer_full_image(
    model: RuaAE,
    cube: np.ndarray,
    patch_size: int,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    model.eval()
    dataset = HyperspectralPatchDataset(cube, patch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    abds = []
    recons = []
    scales = []
    for patches, _centers in loader:
        patches = patches.to(device)
        recon, abd, scale = model(patches, return_abd=True)
        abds.append(abd.cpu().numpy())
        recons.append(recon.cpu().numpy())
        scales.append(scale.cpu().numpy())
    h, w, l_bands = cube.shape
    p = abds[0].shape[1]
    abundance = np.concatenate(abds, axis=0).reshape(h, w, p)
    reconstruction = np.concatenate(recons, axis=0).reshape(h, w, l_bands)
    edm_scale = np.concatenate(scales, axis=0).reshape(h, w, p)
    endmembers = model.endmember.detach().cpu().numpy()  # (P, L)
    return {
        "A": abundance,
        "Y_hat": reconstruction,
        "S": edm_scale,
        "M": endmembers.T,  # (L, P)
    }


def run_training(cfg: dict, project_root: str | Path | None = None) -> dict[str, Any]:
    project_root = Path(project_root) if project_root else Path.cwd()
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(prefer_cuda=bool(cfg.get("prefer_cuda", True)))

    data = build_cube(cfg)
    cube = data["Y"]
    cfg = dict(cfg)
    cfg["n_bands"] = int(data["n_bands"])

    dataset = HyperspectralPatchDataset(cube, int(cfg["patch_size"]))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 100)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        drop_last=False,
    )

    model = create_model(cfg, data["M_init"], device)
    criterion = ULANetLoss(
        alpha=float(cfg.get("alpha", 0.1)),
        beta=float(cfg.get("beta", 0.01)),
        mse_weight=float(cfg.get("mse_weight", 1.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 6e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg.get("lr_step", 20)),
        gamma=float(cfg.get("lr_gamma", 0.99)),
    )

    run_dir = make_run_dir(
        project_root / cfg.get("output_root", "outputs"),
        str(cfg["dataset"]),
        cfg.get("run_name"),
    )
    ensure_dir(run_dir)
    prefix = _file_prefix(cfg)
    save_config_copy(cfg, run_dir, filename=f"{prefix}config.yaml")
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    epochs = int(cfg.get("epochs", 100))

    print(f"Device: {device}")
    print(f"Cube: {cube.shape}, patches: {len(dataset)}, out_dir: {run_dir}, prefix: {prefix!r}")

    for epoch in range(1, epochs + 1):
        stats = train_one_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        stats["epoch"] = float(epoch)
        stats["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append(stats)
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss={stats['loss']:.6f} re={stats['loss_re']:.6f} "
            f"mse={stats.get('loss_mse', 0.0):.6f} "
            f"sp={stats['loss_sparsity']:.6f} gate={stats['loss_gate']:.6f} "
            f"lr={stats['lr']:.6g}"
        )
        if stats["loss"] < best_loss:
            best_loss = stats["loss"]
            best_path = run_dir / f"{prefix}best.pt"
            ensure_dir(best_path.parent)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "m_init": data["M_init"],
                },
                best_path,
            )

    last_path = run_dir / f"{prefix}last.pt"
    ensure_dir(last_path.parent)
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": cfg,
            "epoch": epochs,
            "m_init": data["M_init"],
        },
        last_path,
    )
    save_json(run_dir / f"{prefix}history.json", {"history": history})

    preds = infer_full_image(
        model,
        cube,
        patch_size=int(cfg["patch_size"]),
        device=device,
        batch_size=int(cfg.get("eval_batch_size", 256)),
    )
    np.savez_compressed(
        run_dir / f"{prefix}predictions.npz",
        A=preds["A"],
        Y_hat=preds["Y_hat"],
        S=preds["S"],
        M=preds["M"],
        Y=cube,
        A_true=data["A"] if "A" in data else np.array([]),
        M_true=data["M"] if "M" in data else np.array([]),
    )

    from ula_net.evaluate import evaluate_predictions

    metrics = evaluate_predictions(preds, data)
    save_json(run_dir / f"{prefix}metrics.json", metrics)
    print("Metrics:", metrics)
    return {"run_dir": run_dir, "history": history, "data": data, "predictions": preds, "metrics": metrics, "file_prefix": prefix}
