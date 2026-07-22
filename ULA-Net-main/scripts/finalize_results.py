"""Finalize figures from best seeds, write comparison table, delete intermediate files."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ula_net.visualize import run_paper_style_visualization

OUTPUTS = ROOT / "outputs"

# Author Proposed (paper tables, averages)
AUTHOR = {
    "synthetic": {"aRMSE": 0.0306, "eSAD": 0.0165},
    "samson": {"aRMSE": 0.0837, "eSAD": 0.0567},
    "jasper_ridge": {"aRMSE": 0.1360, "eSAD": 0.1863},
}


def pick_best_seed(metrics_mean: dict) -> int:
    """Pick seed with lowest aRMSE for visualization."""
    runs = metrics_mean["per_run"]
    best = min(runs, key=lambda r: r["aRMSE"])
    return int(best["seed"])


def main() -> None:
    rows = []
    for dataset in ["synthetic", "samson", "jasper_ridge"]:
        ddir = OUTPUTS / dataset
        mean_path = ddir / "metrics_mean.json"
        if not mean_path.exists():
            raise FileNotFoundError(mean_path)
        mean_payload = json.loads(mean_path.read_text(encoding="utf-8"))
        seed = pick_best_seed(mean_payload)
        prefix = f"seed{seed}_"
        print(f"[{dataset}] plotting from {prefix}predictions.npz")
        run_paper_style_visualization(ddir, prefix=prefix)

        src_a = ddir / f"seed{seed}_abundance_gt_vs_ours.png"
        src_c = ddir / f"seed{seed}_endmember_curves_gt_vs_ours.png"
        dst_a = ddir / "abundance_gt_vs_ours.png"
        dst_c = ddir / "endmember_curves_gt_vs_ours.png"
        shutil.copy2(src_a, dst_a)
        shutil.copy2(src_c, dst_c)
        print(f"[{dataset}] -> {dst_a.name}, {dst_c.name}")

        ours_a = mean_payload["mean"]["aRMSE"]
        ours_e = mean_payload["mean"]["eSAD"]
        std_a = mean_payload["std"]["aRMSE"]
        std_e = mean_payload["std"]["eSAD"]
        auth = AUTHOR[dataset]
        rows.append(
            {
                "dataset": dataset,
                "author_aRMSE": auth["aRMSE"],
                "ours_aRMSE": ours_a,
                "ours_aRMSE_std": std_a,
                "author_eSAD": auth["eSAD"],
                "ours_eSAD": ours_e,
                "ours_eSAD_std": std_e,
                "n_runs": mean_payload["n_runs"],
                "vis_seed": seed,
            }
        )

    # Write comparison table
    md_path = OUTPUTS / "comparison_author_vs_ours.md"
    csv_path = OUTPUTS / "comparison_author_vs_ours.csv"
    lines = [
        "# ULA-Net: Author vs Reproduction",
        "",
        "Ours = **mean ± std over 10 runs** (paper protocol). Lower is better.",
        "",
        "| Dataset | Metric | Author (Proposed) | Ours (10-run mean ± std) | Gap |",
        "|---------|--------|------------------:|-------------------------:|----:|",
    ]
    csv_lines = ["dataset,metric,author,ours_mean,ours_std,gap"]
    for r in rows:
        for metric, ak, ok, sk in [
            ("aRMSE", "author_aRMSE", "ours_aRMSE", "ours_aRMSE_std"),
            ("eSAD", "author_eSAD", "ours_eSAD", "ours_eSAD_std"),
        ]:
            author = r[ak]
            ours = r[ok]
            std = r[sk]
            gap = ours - author
            name = {"synthetic": "Synthetic", "samson": "Samson", "jasper_ridge": "Jasper Ridge"}[r["dataset"]]
            lines.append(
                f"| {name} | {metric} | {author:.4f} | {ours:.4f} ± {std:.4f} | {gap:+.4f} |"
            )
            csv_lines.append(
                f"{r['dataset']},{metric},{author:.6f},{ours:.6f},{std:.6f},{gap:.6f}"
            )
    lines += [
        "",
        "## Figure files",
        "",
        "| Dataset | Abundance | Endmember curves | Vis seed (best aRMSE) |",
        "|---------|-----------|------------------|----------------------:|",
    ]
    for r in rows:
        name = r["dataset"]
        lines.append(
            f"| {name} | `outputs/{name}/abundance_gt_vs_ours.png` | "
            f"`outputs/{name}/endmember_curves_gt_vs_ours.png` | {r['vis_seed']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Author numbers: Proposed column in paper Tables II / III–IV / V–VI.",
        "- Synthetic scene is self-generated (not the authors' USGS file), so synthetic gap is expected.",
        "- Abundance/endmember maps use Fortran-order reshape (MATLAB .mat convention).",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")

    # Delete intermediate recording files; keep final figures + metrics_mean + comparison table
    keep_names = {
        "metrics_mean.json",
        "abundance_gt_vs_ours.png",
        "endmember_curves_gt_vs_ours.png",
        ".gitkeep",
    }
    keep_global = {
        "comparison_author_vs_ours.md",
        "comparison_author_vs_ours.csv",
        ".gitkeep",
    }
    deleted = 0
    for dataset in ["synthetic", "samson", "jasper_ridge"]:
        ddir = OUTPUTS / dataset
        for f in ddir.iterdir():
            if f.is_file() and f.name not in keep_names:
                f.unlink()
                deleted += 1
                print(f"deleted {f.relative_to(ROOT)}")
    for f in OUTPUTS.iterdir():
        if f.is_file() and f.name not in keep_global and f.suffix in {".md", ".csv"}:
            # keep only comparison table we just wrote; remove old comparison if any duplicate handled
            if f.name.startswith("comparison_author_vs_ours"):
                continue
        if f.is_file() and f.name not in keep_global and f.name.endswith((".png", ".pt", ".npz", ".json", ".yaml")):
            # stray files at outputs root
            if f.name not in keep_global:
                f.unlink()
                deleted += 1
    print(f"Deleted {deleted} intermediate files.")
    print("Kept: metrics_mean.json, abundance/endmember PNGs, comparison table.")


if __name__ == "__main__":
    main()
