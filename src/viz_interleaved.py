"""Visualization for the interleaved merge (`src.interleaved`).

  - crosstalk(hist):   THE plot — each teacher's residual over bouts, shaded by which teacher was
                       rehearsed, showing the decaying sawtooth.
  - dashboard(hist):   crosstalk + the union deficit falling to zero.
  - crosstalk_plotly(hist): interactive version.

Matplotlib / plotly are imported lazily.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .artifacts import InterleavedBuildInfo
from .viz_style import PLOTLY_LAYOUT, mpl_style


def _shade_bouts(ax, active, labels):
    """Shade the background of each bout by which teacher was rehearsed."""
    colors = ["#cfe8ff", "#ffe0cf"]
    for b, a in enumerate(active):
        ax.axvspan(b - 0.5, b + 0.5, color=colors[a % 2], alpha=0.5, lw=0)
    # legend proxies
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=colors[i % 2], alpha=0.5, label=f"rehearsing {labels[i]}")
               for i in range(len(labels))]
    return handles


def crosstalk(hist, ax=None):
    """Residual of each teacher's memory over bouts — the crosstalk-cancellation sawtooth."""
    import matplotlib.pyplot as plt
    mpl_style()
    H = hist.to_numpy()
    b = H["bout"]
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.5))
    shade = _shade_bouts(ax, H["active"], hist.labels)
    l1, = ax.plot(b, H["resid1"], "-o", color="#1f6fb2", ms=5, lw=2,
                  label=f"{hist.labels[0]} memory")
    l2, = ax.plot(b, H["resid2"], "-o", color="#c1440e", ms=5, lw=2,
                  label=f"{hist.labels[1]} memory")
    ax.set_title("Crosstalk cancellation")
    ax.set_xlabel("replay bout")
    ax.set_ylabel(r"residual  $\|M_S U_k\|$")
    ax.legend(handles=[l1, l2, *shade], loc="upper right")
    return ax


def dashboard(hist, info: Optional[InterleavedBuildInfo] = None):
    """Two panels: the crosstalk sawtooth and the union deficit (subspace being constructed)."""
    import matplotlib.pyplot as plt
    mpl_style()
    H = hist.to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    crosstalk(hist, ax=ax[0])
    ax[1].plot(H["bout"], H["union_deficit"], "-o", color="#3a7d44", ms=5, lw=2)
    ax[1].set_title("Union deficit")
    ax[1].set_xlabel("replay bout")
    ax[1].set_ylabel(r"$\|M_S U_\Sigma\|_F^2$")
    ax[1].set_yscale("log")
    if info is not None:
        fig.suptitle(
            f"Interleaved merge: {info.rank1}+{info.rank2} memories, "
            f"overlap {info.overlap}, "
            rf"$r_\Sigma$ = {info.r_Sigma}")
    fig.tight_layout()
    return fig


def crosstalk_plotly(hist):
    """Interactive crosstalk sawtooth."""
    import plotly.graph_objects as go
    H = hist.to_numpy()
    b = H["bout"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=b, y=H["resid1"], mode="lines+markers",
                             name=f"{hist.labels[0]} memory"))
    fig.add_trace(go.Scatter(x=b, y=H["resid2"], mode="lines+markers",
                             name=f"{hist.labels[1]} memory"))
    fig.add_trace(go.Scatter(x=b, y=H["union_deficit"], mode="lines", name="union deficit",
                             line=dict(dash="dash", color="green")))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title_text="Crosstalk cancellation over replay bouts",
        xaxis_title="replay bout", yaxis_title="residual",
        height=430,
    )
    return fig
