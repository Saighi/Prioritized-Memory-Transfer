"""Visualization for the continual-learning loop (`src.continual`).

  - dashboard(hist):   static 4-panel matplotlib overview (retention heatmaps + growth + fractions).
  - retention(hist):   interactive plotly retention heatmap (Storage vs buffer-only baseline).

Matplotlib / plotly are imported lazily so the core has no plotting dependency on import.
"""
from __future__ import annotations

import numpy as np


def dashboard(hist, cfg=None):
    """Static 4-panel overview: the money plot is the retention heatmap (Storage keeps every
    memory; the buffer-only baseline keeps only the diagonal = the latest)."""
    import matplotlib.pyplot as plt

    H = hist.to_numpy()
    R_store = H["retention_storage"]      # (n_memories, n_cycles)
    R_base = H["retention_baseline"]
    n_cycles = R_store.shape[1]
    cyc = np.arange(1, n_cycles + 1)
    vmax = float(np.nanmax([np.nanmax(R_store), np.nanmax(R_base)])) if n_cycles else 1.0

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    im0 = ax[0, 0].imshow(R_store, aspect="auto", cmap="viridis_r", vmin=0, vmax=vmax,
                          origin="lower", extent=[0.5, n_cycles + 0.5, -0.5, R_store.shape[0] - 0.5])
    ax[0, 0].set_title("Storage retention  ‖M_Z mᵢ‖  (low = retained)")
    ax[0, 0].set_xlabel("cycle (memory added)"); ax[0, 0].set_ylabel("memory index i")
    fig.colorbar(im0, ax=ax[0, 0], fraction=0.046)

    im1 = ax[0, 1].imshow(R_base, aspect="auto", cmap="viridis_r", vmin=0, vmax=vmax,
                          origin="lower", extent=[0.5, n_cycles + 0.5, -0.5, R_base.shape[0] - 0.5])
    ax[0, 1].set_title("Buffer-only baseline retention (catastrophic forgetting)")
    ax[0, 1].set_xlabel("cycle"); ax[0, 1].set_ylabel("memory index i")
    fig.colorbar(im1, ax=ax[0, 1], fraction=0.046)

    ax[1, 0].plot(cyc, H["n_retained"], "-o", label="memories retained in Storage")
    ax[1, 0].plot(cyc, cyc, "--", color="gray", label="# memories seen")
    ax[1, 0].set_title("Storage accumulates the whole stream")
    ax[1, 0].set_xlabel("cycle"); ax[1, 0].set_ylabel(f"# retained (‖M_Z mᵢ‖ < {hist.tol})")
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(cyc, H["retained_frac"], "-o", label="continual (storage)")
    ax[1, 1].plot(cyc, H["baseline_retained_frac"], "-s", label="buffer-only baseline")
    ax[1, 1].plot(cyc, H["consolidation_deficit"], "-^", color="C3", label="consolidation deficit")
    ax[1, 1].set_title(f"Fraction of memories retained (tol={hist.tol})")
    ax[1, 1].set_xlabel("cycle"); ax[1, 1].set_ylim(-0.05, 1.15); ax[1, 1].legend(fontsize=8)

    fig.suptitle("Continual learning: buffer → synthesis → storage consolidation", fontsize=12)
    fig.tight_layout()
    return fig


def retention(hist):
    """Interactive plotly retention heatmap for Storage (rows = memories, cols = cycles)."""
    import plotly.graph_objects as go
    R = hist.retention_matrix("storage")
    fig = go.Figure(go.Heatmap(z=R, colorscale="Viridis_r", colorbar=dict(title="‖M_Z mᵢ‖")))
    fig.update_layout(
        title="Storage retention over the stream (low = memory still nulled = retained)",
        xaxis_title="cycle", yaxis_title="memory index",
        template="plotly_white", height=460,
    )
    return fig
