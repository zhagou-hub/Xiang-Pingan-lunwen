# ULA-Net: Author vs Reproduction

Ours = **mean ± std over 10 runs** (paper protocol). Lower is better.

| Dataset | Metric | Author (Proposed) | Ours (10-run mean ± std) | Gap |
|---------|--------|------------------:|-------------------------:|----:|
| Synthetic | aRMSE | 0.0306 | 0.2457 ± 0.0690 | +0.2151 |
| Synthetic | eSAD | 0.0165 | 0.2756 ± 0.0940 | +0.2591 |
| Samson | aRMSE | 0.0837 | 0.2258 ± 0.0754 | +0.1421 |
| Samson | eSAD | 0.0567 | 0.0964 ± 0.0973 | +0.0397 |
| Jasper Ridge | aRMSE | 0.1360 | 0.2377 ± 0.0079 | +0.1017 |
| Jasper Ridge | eSAD | 0.1863 | 0.2432 ± 0.0398 | +0.0569 |

## Figure files

| Dataset | Abundance | Endmember curves | Vis seed (best aRMSE) |
|---------|-----------|------------------|----------------------:|
| synthetic | `outputs/synthetic/abundance_gt_vs_ours.png` | `outputs/synthetic/endmember_curves_gt_vs_ours.png` | 45 |
| samson | `outputs/samson/abundance_gt_vs_ours.png` | `outputs/samson/endmember_curves_gt_vs_ours.png` | 50 |
| jasper_ridge | `outputs/jasper_ridge/abundance_gt_vs_ours.png` | `outputs/jasper_ridge/endmember_curves_gt_vs_ours.png` | 51 |

## Notes

- Author numbers: Proposed column in paper Tables II / III–IV / V–VI.
- Synthetic scene is self-generated (not the authors' USGS file), so synthetic gap is expected.
- Abundance/endmember maps use Fortran-order reshape (MATLAB .mat convention).
