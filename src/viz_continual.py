"""Visualization for the continual-learning loop (`src.continual`).

  - dashboard(hist):   static 4-panel matplotlib overview (retention heatmaps + growth + fractions).
  - retention(hist):   interactive plotly retention heatmap (Storage vs buffer-only baseline).

Matplotlib / plotly are imported lazily so the core has no plotting dependency on import.
"""
from __future__ import annotations

import numpy as np

from .viz_style import PLOTLY_LAYOUT, despine_all, mpl_style


def dashboard(hist, cfg=None):
    """Static 4-panel overview: the money plot is the retention heatmap (Storage keeps every
    memory; the buffer-only baseline keeps only the diagonal = the latest)."""
    import matplotlib.pyplot as plt
    mpl_style()

    H = hist.to_numpy()
    R_store = H["retention_storage"]      # (n_memories, n_cycles)
    R_base = H["retention_baseline"]
    n_cycles = R_store.shape[1]
    cyc = np.arange(1, n_cycles + 1)
    vmax = float(np.nanmax([np.nanmax(R_store), np.nanmax(R_base)])) if n_cycles else 1.0

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    for a, R, title in ((ax[0, 0], R_store, "Storage retention"),
                        (ax[0, 1], R_base, "Buffer-only baseline")):
        im = a.imshow(R, aspect="auto", cmap="viridis_r", vmin=0, vmax=vmax,
                      origin="lower", extent=[0.5, n_cycles + 0.5, -0.5, R.shape[0] - 0.5])
        a.set_title(title)
        a.set_xlabel("cycle")
        a.set_ylabel("memory index $i$")
        a.grid(False)
        despine_all(a)
        cb = fig.colorbar(im, ax=a, fraction=0.046)
        cb.set_label(r"$\|M_Z\, m_i\|$  (low = retained)")

    ax[1, 0].plot(cyc, H["n_retained"], "-o", lw=2, label="retained in Storage")
    ax[1, 0].plot(cyc, cyc, "--", color="gray", label="seen")
    ax[1, 0].set_title("Accumulation")
    ax[1, 0].set_xlabel("cycle")
    ax[1, 0].set_ylabel("# memories")
    ax[1, 0].legend()

    ax[1, 1].plot(cyc, H["retained_frac"], "-o", lw=2, label="Storage")
    ax[1, 1].plot(cyc, H["baseline_retained_frac"], "-s", lw=2, label="buffer-only")
    ax[1, 1].plot(cyc, H["consolidation_deficit"], "-^", lw=2, color="C3",
                  label="consolidation deficit")
    ax[1, 1].set_title(f"Retention fraction (tol = {hist.tol})")
    ax[1, 1].set_xlabel("cycle")
    ax[1, 1].set_ylabel("fraction")
    ax[1, 1].set_ylim(-0.05, 1.15)
    ax[1, 1].legend()

    fig.suptitle("Continual learning: buffer → synthesis → storage")
    fig.tight_layout()
    return fig


def retention(hist):
    """Interactive plotly retention heatmap for Storage (rows = memories, cols = cycles)."""
    import plotly.graph_objects as go
    R = hist.retention_matrix("storage")
    fig = go.Figure(go.Heatmap(z=R, colorscale="Viridis_r",
                               colorbar=dict(title="‖M_Z mᵢ‖")))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title_text="Storage retention (low = retained)",
        xaxis_title="cycle", yaxis_title="memory index",
        height=460,
    )
    return fig
