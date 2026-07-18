"""Shared figure styling (paper-oriented): larger fonts, concise titles, no top/right spines.

`mpl_style()` updates matplotlib rcParams in place; `PLOTLY_LAYOUT` is a base layout dict to
splat into `fig.update_layout(**PLOTLY_LAYOUT, ...)`.
"""
from __future__ import annotations

MPL_RC = {
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 17,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "axes.titlepad": 10,
    "legend.frameon": False,
}


def mpl_style() -> None:
    """Apply the shared matplotlib style (call once at the top of every figure builder)."""
    import matplotlib as mpl
    mpl.rcParams.update(MPL_RC)


def despine_all(ax) -> None:
    """Remove every spine (for image/heatmap panels where the frame is just noise)."""
    for s in ax.spines.values():
        s.set_visible(False)


PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(size=15),
    title=dict(font=dict(size=18)),
    margin=dict(t=60, r=30),
)
