"""Paper-style visualization: abundance heatmaps + endmember curves (GT vs Ours)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ula_net.metrics import match_endmembers_by_sad

# Dataset-specific endmember names (paper order).
ENDMEMBER_NAMES = {
    "samson": ["Soil", "Tree", "Water"],
    "jasper_ridge": ["Tree", "Water", "Dirt", "Road"],
    "jasper": ["Tree", "Water", "Dirt", "Road"],
    "synthetic": None,  # filled as Endmember-1..P
}


def _to_LP(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"endmember must be 2-D, got {m.shape}")
    return m if m.shape[0] >= m.shape[1] else m.T  # (L, P)


def _infer_dataset_name(run_dir: Path) -> str:
    # Flat layout: outputs/<dataset>/
    if run_dir.name in ENDMEMBER_NAMES:
        return run_dir.name
    name = str(run_dir).lower()
    if "samson" in name:
        return "samson"
    if "jasper" in name:
        return "jasper_ridge"
    return "synthetic"


def _endmember_titles(dataset: str, n_endmembers: int) -> list[str]:
    names = ENDMEMBER_NAMES.get(dataset)
    if names and len(names) == n_endmembers:
        return names
    return [f"Endmember-{i + 1}" for i in range(n_endmembers)]


def plot_abundance_gt_vs_ours(
    a_true: np.ndarray,
    a_est: np.ndarray,
    save_path: str | Path,
    titles: list[str],
) -> None:
    """
    Paper-style abundance maps: one row per endmember, columns = GT | Ours.
    a_*: (H, W, P), already matched in endmember order.
    """
    h, w, p = a_true.shape
    fig, axes = plt.subplots(p, 2, figsize=(6.2, 2.6 * p))
    if p == 1:
        axes = np.asarray([axes])

    for i in range(p):
        for j, (img, col_title) in enumerate(
            [(a_true[:, :, i], "GT"), (a_est[:, :, i], "Ours")]
        ):
            ax = axes[i, j]
            im = ax.imshow(img, cmap="jet", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_title, fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(titles[i], fontsize=11, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle("Abundance maps (per endmember)", fontsize=13, y=1.01)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_endmember_curves_gt_vs_ours(
    m_true: np.ndarray,
    m_est: np.ndarray,
    save_path: str | Path,
    titles: list[str],
) -> None:
    """
    Paper-style endmember curves: red dashed = GT, blue solid = Ours.
    m_*: (L, P), already matched.
    """
    m_true = _to_LP(m_true)
    m_est = _to_LP(m_est)
    p = m_true.shape[1]
    fig, axes = plt.subplots(1, p, figsize=(3.4 * p, 3.0), sharey=True)
    if p == 1:
        axes = [axes]

    for i in range(p):
        ax = axes[i]
        ax.plot(m_true[:, i], color="red", linestyle="--", linewidth=1.8, label="GT")
        ax.plot(m_est[:, i], color="tab:blue", linestyle="-", linewidth=1.6, label="Ours")
        ax.set_title(titles[i], fontsize=11)
        ax.set_xlabel("Bands")
        if i == 0:
            ax.set_ylabel("Reflectance")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Endmember curves (GT vs Ours)", fontsize=13, y=1.02)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_paper_style_visualization(
    run_dir: str | Path,
    predictions_name: str | None = None,
    prefix: str = "",
) -> dict[str, Path]:
    """
    Load predictions and write GT vs Ours figures into the same folder.

    predictions_name: explicit file name, or None to auto-pick:
      - <prefix>predictions.npz if given
      - predictions.npz
      - first *predictions.npz found
    """
    run_dir = Path(run_dir)
    if predictions_name:
        pred_path = run_dir / predictions_name
    elif prefix:
        pred_path = run_dir / f"{prefix}predictions.npz"
    else:
        pred_path = run_dir / "predictions.npz"
        if not pred_path.exists():
            cands = sorted(run_dir.glob("*predictions.npz"))
            if not cands:
                raise FileNotFoundError(f"No *predictions.npz in {run_dir}")
            pred_path = cands[0]

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {pred_path}")

    stem = pred_path.name.replace("predictions.npz", "").rstrip("_")
    tag = f"{stem}_" if stem else ""

    data = np.load(pred_path, allow_pickle=False)
    a_est = np.asarray(data["A"], dtype=np.float64)
    m_est = _to_LP(data["M"])

    if "A_true" not in data.files or data["A_true"].size == 0:
        raise ValueError(f"No abundance GT in {pred_path}")
    if "M_true" not in data.files or data["M_true"].size == 0:
        raise ValueError(f"No endmember GT in {pred_path}")

    a_true = np.asarray(data["A_true"], dtype=np.float64)
    m_true = _to_LP(data["M_true"])

    m_true_pl, m_est_pl, perm = match_endmembers_by_sad(m_true, m_est)
    a_est_aligned = a_est[..., perm]
    m_true_lp = m_true_pl.T
    m_est_lp = m_est_pl.T

    dataset = _infer_dataset_name(run_dir)
    titles = _endmember_titles(dataset, a_true.shape[-1])

    abund_path = run_dir / f"{tag}abundance_gt_vs_ours.png"
    curve_path = run_dir / f"{tag}endmember_curves_gt_vs_ours.png"
    plot_abundance_gt_vs_ours(a_true, a_est_aligned, abund_path, titles)
    plot_endmember_curves_gt_vs_ours(m_true_lp, m_est_lp, curve_path, titles)

    print(f"[{dataset}] from {pred_path.name}")
    print(f"[{dataset}] abundance -> {abund_path}")
    print(f"[{dataset}] curves    -> {curve_path}")
    return {"abundance": abund_path, "curves": curve_path}


# Keep thin wrappers used by scripts/visualize.py
def plot_abundance_maps(abundance: np.ndarray, save_path: str | Path, titles: list[str] | None = None) -> None:
    titles = titles or [f"Endmember {i + 1}" for i in range(abundance.shape[-1])]
    # single-panel fallback: only Ours
    dummy = np.zeros_like(abundance)
    # If caller only has Ours, show Ours in both columns with note — better just show Ours row.
    p = abundance.shape[-1]
    fig, axes = plt.subplots(1, p, figsize=(3.0 * p, 2.8))
    if p == 1:
        axes = [axes]
    for i in range(p):
        im = axes[i].imshow(abundance[:, :, i], cmap="jet", vmin=0.0, vmax=1.0)
        axes[i].set_title(titles[i])
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], fraction=0.046)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_endmembers(
    m_est: np.ndarray,
    save_path: str | Path,
    m_true: np.ndarray | None = None,
) -> None:
    m_est = _to_LP(m_est)
    p = m_est.shape[1]
    titles = [f"Endmember-{i + 1}" for i in range(p)]
    if m_true is not None and np.asarray(m_true).size > 0:
        m_true = _to_LP(m_true)
        _, m_est_m, perm = match_endmembers_by_sad(m_true, m_est)
        plot_endmember_curves_gt_vs_ours(m_true, m_est_m.T, save_path, titles)
    else:
        fig, axes = plt.subplots(1, p, figsize=(3.4 * p, 3.0))
        if p == 1:
            axes = [axes]
        for i in range(p):
            axes[i].plot(m_est[:, i], color="tab:blue")
            axes[i].set_title(titles[i])
            axes[i].set_xlabel("Bands")
            axes[i].set_ylabel("Reflectance")
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_loss_curve(history: list[dict], save_path: str | Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(epochs, [h["loss"] for h in history], label="total")
    ax.plot(epochs, [h["loss_re"] for h in history], label="SSD")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_visualization(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    run_paper_style_visualization(run_dir)
    hist_path = run_dir / "history.json"
    if hist_path.exists():
        import json

        with open(hist_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        plot_loss_curve(payload["history"], run_dir / "loss_curve.png")
