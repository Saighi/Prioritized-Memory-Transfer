"""Interactive plotly figures (render inline in the VS Code interactive window; needs
`nbformat`). Each returns a plotly Figure — call `.show()` in a notebook cell.

  - staircase(hist):    novelty-spectrum eigenvalues over time (zoom/hover).
  - raster(hist):       |cos(x_T, m_p)| heatmap over time.
  - trajectory_3d(...):  x_T projected onto 3 memory directions, animated with a time slider
                         (the "scrub the replay" centerpiece).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .history import History
from .model import TwoPopModel
from .viz_style import PLOTLY_LAYOUT


def staircase(hist: History):
    import plotly.graph_objects as go
    H = hist.to_numpy()
    t, spec = H["t"], H["novelty_spec"]
    fig = go.Figure()
    for j in range(spec.shape[1]):
        fig.add_trace(go.Scatter(x=t, y=spec[:, j], mode="lines", name=f"dir {j}"))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title_text="Novelty staircase",
        xaxis_title="time", yaxis_title="eig(Uᵀ N_S U)",
        height=420,
    )
    return fig


def raster(hist: History):
    import plotly.graph_objects as go
    H = hist.to_numpy()
    t = H["t"]
    z = np.abs(H["align"]).T   # (P, T)
    fig = go.Figure(go.Heatmap(
        x=t, y=list(range(z.shape[0])), z=z, colorscale="Magma", zmin=0, zmax=1,
        colorbar=dict(title="|cos(x_T, m_p)|"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title_text="Replay raster",
        xaxis_title="time", yaxis_title="memory index p",
        height=420,
    )
    return fig


def trajectory_3d(hist: History, model: TwoPopModel, info: Optional[Dict] = None,
                  basis: str = "patterns", max_frames: int = 120):
    """Animate x_T's path projected onto 3 directions, with a time slider.

    basis="patterns": project onto the first 3 stored memories (axes are interpretable as
    'how much of m0/m1/m2'); basis="manifold": project onto the first 3 columns of U_T.
    """
    import plotly.graph_objects as go
    H = hist.to_numpy()
    t = H["t"]
    xT = torch.as_tensor(H["x_T"])                 # (T, d)

    if basis == "manifold" and info is not None:
        B3 = info["U_T"][:, :3].cpu()
        labels = ["u0", "u1", "u2"]
    else:
        B3 = model.patterns[:, :3].cpu()
        labels = ["m0", "m1", "m2"]
    coords = (xT @ B3).numpy()                     # (T, 3)

    # base path, colored by time
    base = go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="lines",
        line=dict(width=4, color=t, colorscale="Viridis"),
        opacity=0.55, name="x_T path",
        hovertemplate="t=%{customdata:.1f}<extra></extra>", customdata=t,
    )
    head = go.Scatter3d(
        x=[coords[0, 0]], y=[coords[0, 1]], z=[coords[0, 2]], mode="markers",
        marker=dict(size=6, color="red"), name="x_T(t)",
    )

    idx = np.unique(np.linspace(0, len(t) - 1, min(max_frames, len(t))).astype(int))
    frames = [
        go.Frame(
            data=[go.Scatter3d(x=[coords[i, 0]], y=[coords[i, 1]], z=[coords[i, 2]],
                               mode="markers", marker=dict(size=6, color="red"))],
            traces=[1], name=f"{t[i]:.0f}",
        )
        for i in idx
    ]

    fig = go.Figure(data=[base, head], frames=frames)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title_text="x_T trajectory in memory coordinates",
        height=620,
        scene=dict(xaxis_title=labels[0], yaxis_title=labels[1], zaxis_title=labels[2]),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.05, y=0.05,
            buttons=[
                dict(label="▶ play", method="animate",
                     args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True)]),
                dict(label="⏸ pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.1, len=0.85, y=0.0,
            currentvalue=dict(prefix="t = "),
            steps=[dict(method="animate", label=f"{t[i]:.0f}",
                        args=[[f"{t[i]:.0f}"], dict(mode="immediate",
                              frame=dict(duration=0, redraw=True))]) for i in idx],
        )],
    )
    return fig
