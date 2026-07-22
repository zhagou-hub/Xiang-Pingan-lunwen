# Full paper protocol: 10 runs x 100 epochs for 3 datasets, then figures.
# Outputs go flat under outputs/<dataset>/
# Uses script location so Chinese absolute paths are not needed.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "Working directory: $Root"
Write-Host "========================================"
Write-Host "ULA-Net multi-run training"
Write-Host "========================================"

Write-Host ""
Write-Host "[1/3] Samson ..."
python scripts/train_multirun.py --config configs/samson.yaml
if ($LASTEXITCODE -ne 0) { throw "Samson failed with exit $LASTEXITCODE" }

Write-Host ""
Write-Host "[2/3] Jasper Ridge ..."
python scripts/train_multirun.py --config configs/jasper_ridge.yaml
if ($LASTEXITCODE -ne 0) { throw "Jasper failed with exit $LASTEXITCODE" }

Write-Host ""
Write-Host "[3/3] Synthetic ..."
python scripts/train_multirun.py --config configs/synthetic.yaml
if ($LASTEXITCODE -ne 0) { throw "Synthetic failed with exit $LASTEXITCODE" }

Write-Host ""
Write-Host "[Plot] GT vs Ours figures (seed42) ..."
python scripts/plot_paper_figures.py --run-dir outputs/samson --prefix seed42_
python scripts/plot_paper_figures.py --run-dir outputs/jasper_ridge --prefix seed42_
python scripts/plot_paper_figures.py --run-dir outputs/synthetic --prefix seed42_

Write-Host ""
Write-Host "ALL DONE."
Write-Host "Metrics: outputs/*/metrics_mean.json"
Write-Host "Figures: outputs/*/*abundance_gt_vs_ours.png"
