"""Pure SMACOF embed-and-drape toolkit (model-agnostic).

Borrowed from a Hopfield energy-landscape project: take a cloud of high-dimensional states,
embed them into a *reconstructed* (non-isometric, intuitive) 2-D layout by metric MDS / SMACOF,
then drape a scalar height over that layout and render it as a plotly Surface. The reconstruction
is deliberately NOT the true geometry — it just lays out states so that near-states sit near each
other, which is enough to read plateaus, walls, and valleys off the draped scalar.

No `torch`, no `src` — only numpy with scipy / scikit-learn / plotly lazy-imported inside the
functions that need them, so importing this module is cheap and dependency-light. Every figure
helper returns a bare plotly object; the caller decides `.show()` / `.write_html()`.

Design rule for animations (enforced by callers): compute the layout ONCE and reuse it for every
frame, so only the draped height changes and the embedding never jitters.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# A dark-background-friendly perceptual ramp (plasma-ish), low = deep/cool, high = bright.
INFERNO = [
    [0.00, "#0d0887"], [0.15, "#3b049a"], [0.30, "#7201a8"], [0.45, "#a52c60"],
    [0.60, "#d44842"], [0.75, "#ed7953"], [0.90, "#fbb61a"], [1.00, "#fcffa4"],
]
# A diverging ramp for signed scalars (e.g. Phi): negative = blue valley, 0 = pale, positive = red wall.
DIVERGING = "RdBu_r"

_BG = "#0a0a1a"


# ----------------------------------------------------------------- distances & embedding
def pairwise_dist(X: np.ndarray, metric: Optional[np.ndarray] = None) -> np.ndarray:
    """(n,n) pairwise distances among the rows of X (n,d).

    metric=None  -> plain Euclidean ‖x−y‖ (the stable default; use this for fixed layouts).
    metric=A     -> Mahalanobis-like ‖A(x−y)‖ for a (d,d) matrix A: distance in the model's own
                    metric (e.g. A = N_S gives "how distinguishable to the student"). Implemented
                    as Euclidean distance after the linear map x ↦ A xᵀ.
    """
    from scipy.spatial.distance import cdist

    X = np.asarray(X, dtype=float)
    Y = X if metric is None else X @ np.asarray(metric, dtype=float).T
    return cdist(Y, Y)


def smacof_layout(D: np.ndarray, seed: int = 0, max_iter: int = 400, n_init: int = 4) -> np.ndarray:
    """Metric MDS / SMACOF on a precomputed dissimilarity matrix D -> (n,2), standardized
    (zero mean, unit std per axis) so downstream grid spans are scale-stable."""
    from sklearn.manifold import MDS

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed,
              max_iter=max_iter, n_init=n_init, normalized_stress="auto")
    Z = mds.fit_transform(np.asarray(D, dtype=float))
    Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-12)
    return Z


# ----------------------------------------------------------------- drape a scalar -> surface grid
def drape_surface(layout: np.ndarray, heights: np.ndarray, grid_n: int = 160,
                  sigma: float = 4.0, margin: float = 0.35, dilate: int = 4):
    """Interpolate scattered (layout, heights) onto a regular grid and mask outside the data hull.

    Mirrors the reference pipeline: cubic griddata (nearest-fill the NaNs it leaves) -> gaussian
    smoothing -> Delaunay convex-hull mask (dilated a few pixels) so the surface stops at the data
    rather than ballooning to the grid corners. Returns (Xi, Yi, Zi) as 2-D meshgrids; Zi is NaN
    outside the hull (plotly renders that as a clean hole).
    """
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter, binary_dilation
    from scipy.spatial import Delaunay

    layout = np.asarray(layout, dtype=float)
    heights = np.asarray(heights, dtype=float)
    xmin, xmax = layout[:, 0].min() - margin, layout[:, 0].max() + margin
    ymin, ymax = layout[:, 1].min() - margin, layout[:, 1].max() + margin
    xi = np.linspace(xmin, xmax, grid_n)
    yi = np.linspace(ymin, ymax, grid_n)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata(layout, heights, (Xi, Yi), method="cubic")
    Znear = griddata(layout, heights, (Xi, Yi), method="nearest")
    Zi[np.isnan(Zi)] = Znear[np.isnan(Zi)]
    Zi = gaussian_filter(Zi, sigma=sigma)

    hull = Delaunay(layout)
    inside = hull.find_simplex(np.c_[Xi.ravel(), Yi.ravel()]) >= 0
    inside = binary_dilation(inside.reshape(Xi.shape), iterations=dilate)
    Zi = Zi.copy()
    Zi[~inside] = np.nan
    return Xi, Yi, Zi


# ----------------------------------------------------------------- plotly building blocks
def surface_trace(Xi, Yi, Zi, cmin=None, cmax=None, colorscale=INFERNO,
                  showscale=True, name="", colorbar_title="height"):
    """A NaN-masked go.Surface with soft lighting on a dark background."""
    import plotly.graph_objects as go

    Zi = np.asarray(Zi)
    if cmin is None:
        cmin = float(np.nanmin(Zi))
    if cmax is None:
        cmax = float(np.nanmax(Zi))
    return go.Surface(
        x=Xi, y=Yi, z=Zi, cmin=cmin, cmax=cmax, colorscale=colorscale,
        showscale=showscale, name=name, opacity=0.95,
        colorbar=dict(title=dict(text=colorbar_title, font=dict(color="white")),
                      tickfont=dict(color="white"), len=0.6),
        contours=dict(z=dict(show=True, usecolormap=True, project_z=False,
                             highlightcolor="rgba(255,255,255,0.15)")),
        lighting=dict(ambient=0.45, diffuse=0.65, specular=0.25, roughness=0.55, fresnel=0.2),
        lightposition=dict(x=0, y=0, z=100000),
        hovertemplate="height = %{z:.3f}<extra></extra>",
    )


def dark_scene(zlabel: str, zrange: Optional[Sequence[float]] = None) -> dict:
    """Scene dict: x/y hidden (SMACOF coords are meaningless), z labelled, dark background."""
    z = dict(title=dict(text=zlabel, font=dict(color="white")), showbackground=False,
             gridcolor="rgba(255,255,255,0.12)", zeroline=False, tickfont=dict(color="white"))
    if zrange is not None:
        z["range"] = list(zrange)
    return dict(
        xaxis=dict(visible=False, showbackground=False),
        yaxis=dict(visible=False, showbackground=False),
        zaxis=z, bgcolor=_BG,
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.85)),
        aspectratio=dict(x=1.2, y=1.2, z=0.6),
    )


def dark_layout(title: str, height: int = 640) -> dict:
    """update_layout kwargs for a single dark figure."""
    return dict(
        title=dict(text=title, x=0.5, font=dict(color="white", size=16)),
        paper_bgcolor=_BG, font=dict(color="white"),
        margin=dict(l=0, r=0, t=55, b=0), height=height,
    )


def anim_controls(labels: Sequence[str], duration: int = 140, slider_prefix: str = "t = ") -> dict:
    """Play/pause buttons + a time slider, one step per frame (frames are named '0','1',...)."""
    return dict(
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.05, y=0.06, xanchor="left",
            bgcolor="rgba(255,255,255,0.10)", font=dict(color="white"),
            buttons=[
                dict(label="▶ play", method="animate",
                     args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True)]),
                dict(label="⏸ pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.1, len=0.85, y=0.0, font=dict(color="white"),
            currentvalue=dict(prefix=slider_prefix, font=dict(color="white")),
            steps=[dict(method="animate", label=str(lab),
                        args=[[f"{i}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                   for i, lab in enumerate(labels)],
        )],
        uirevision="keep",
    )
