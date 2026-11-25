# Aggregated results

`final_grid_20260830/` contains the aggregate output of the final six-channel experiment
grid:

- `comparison_summary.csv` — policy-level means and intervals over the grid.
- `seed_metrics.csv` — per-seed metrics, the source rows for the summary.
- `summary.md` — human-readable rendering of the comparison table.

Per-run logs, checkpoints and raw rollout files are not part of this repository; the
seed-level evidence blocks that back the reported comparisons live in `../reproducibility/`.
