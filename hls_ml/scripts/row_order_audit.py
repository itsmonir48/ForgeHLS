"""
row_order_audit.py

Checks whether the row order of the ForgeHLS release carries information about
which variants of a kernel are Pareto-optimal. If it does, any procedure that
breaks ties by row index (stable sorts, "first of each group") leaks the label.

Pareto front per kernel: non-dominated on (Worst-caseLatency, ResourceMetric),
as in pareto_dse.py. Kernels with >= 30 variants.

Output: hls_ml/results/row_order_audit.txt
"""

import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR
from pareto_dse import MIN_ROWS, pareto_mask, resource_metric


def main():
    data = np.load(DATA_DIR / "dataset_arrays.npz")
    Y = data["y_targets"].astype(float)
    groups = data["group_ids"]
    X = data["X_features"]
    unroll = X[:, list(data["feature_names"]).index("pragma_max_unroll_factor")]
    latency = Y[:, 0]
    res = resource_metric(Y[:, 1], Y[:, 3], Y[:, 2], Y[:, 4])

    positions, first_share_pareto, first_share_all, n = [], [], [], 0
    for kernel in np.unique(groups):
        idx = np.flatnonzero(groups == kernel)
        if len(idx) < MIN_ROWS:
            continue
        n += 1
        mask = pareto_mask(latency[idx], res[idx])
        rel = np.arange(len(idx)) / (len(idx) - 1)
        positions.append(rel[mask].mean())
        first = ~pd.Series(unroll[idx]).duplicated().to_numpy()
        first_share_pareto.append(first[mask].mean())
        first_share_all.append(first.mean())

    lines = [
        "Row-order audit of the ForgeHLS release",
        "=" * 60,
        f"Kernels with >= {MIN_ROWS} variants: {n}",
        f"Mean relative row position of Pareto variants within their kernel: "
        f"{np.mean(positions):.3f} (0.5 expected if order were random)",
        f"Share of Pareto variants that are the first row with their max unroll factor: "
        f"{np.mean(first_share_pareto):.3f} (all variants: {np.mean(first_share_all):.3f})",
    ]
    text = "\n".join(lines)
    print(text)
    (RESULTS_DIR / "row_order_audit.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
