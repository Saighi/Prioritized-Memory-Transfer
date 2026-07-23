"""Eigenspace-geometry figures: how the student's operators S_S and N_S look, and morph, in a
tiny d=3 network where the memory manifold is a 2-D plane and the off-manifold direction is a line.

Three independent visualizations, each with a static triptych (S empty / mid / transferred)
and a time animation. They are deliberately kept separate (no overcrowding):

  1. novelty_sphere_*    : the unit state-sphere colored by novelty(u) = uᵀ N_S(t) u, with the
                           memory plane's great circle and the x_T comet on it. The memory
                           circle cools toward 0 as it is learned; off-manifold heats toward 1.
  2. stretch_ellipsoid_* : the image of the unit sphere under S_S(t) (normalized per frame by
                           its top eigenvalue, so it shows shape): a sphere flattening toward a
                           line as the two memory eigenvalues fall to 0.
  3. energy_valley_*     : the surface z = ½ xᵀ S_S(t) x over the slice [memory dir, off-manifold
                           normal]: a round bowl develops a flat valley along the memory direction.

Conventions match viz_interactive.py: plotly is lazy-imported inside each builder, and every
public builder returns a bare plotly Figure (the caller decides `.show()`). Operators are
recomputed from each weight snapshot via pure helpers, so the live model is never mutated.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .diagnostics import novelty_operator
from .viz_style import PLOTLY_LAYOUT
from .history import History
from .model import TwoPopModel


# --------------------------------------------------------------- pure operator helpers
# (operate on a snapshot W_S, never touch the live model)
def s_s_of(W_S: torch.Tensor) -> torch.Tensor:
    """S_S = M_Sᵀ M_S with M_S = I - W_S."""
    I = torch.eye(W_S.shape[-1], dtype=W_S.dtype, device=W_S.device)
    M = I - W_S
    return M.transpose(-2, -1) @ M


def n_s_of(W_S: torch.Tensor, pi_TS: float, pi_S: float) -> torch.Tensor:
    """N_S = pi_S S_S (pi_TS I + pi_S S_S)^-1 (the novelty operator)."""
    return novelty_operator(s_s_of(W_S), pi_TS, pi_S)


# --------------------------------------------------------------- per-frame data extractor
def eigenframes(hist: History, model: TwoPopModel, info: Optional[object] = None,
                n_frames: Optional[int] = None) -> List[Dict]:
    """One dict per weight snapshot: t, S_S, N_S (numpy), eigvals/eigvecs of S_S, and the
    x_T path/head recorded up to that snapshot's time. Sub-sample to n_frames if given."""
    snaps = hist.weight_snaps
    if not snaps:
        raise ValueError("no weight snapshots; set SimConfig.n_weight_snapshots > 0")
    if n_frames is not None and n_frames < len(snaps):
        keep = np.unique(np.linspace(0, len(snaps) - 1, n_frames).astype(int))
        snaps = [snaps[i] for i in keep]

    t_rec = np.asarray(hist.t)
    xT = torch.stack(hist.x_T).numpy() if hist.x_T else np.zeros((1, model.d))

    frames: List[Dict] = []
    for s in snaps:
        W = s["W_S"]
        S = s_s_of(W)
        N = n_s_of(W, model.pi_TS, model.pi_S)
        evals, evecs = torch.linalg.eigh(S)        # ascending
        mask = t_rec <= s["t"] + 1e-9
        path = xT[mask] if mask.any() else xT[:1]
        frames.append(dict(
            t=float(s["t"]),
            S_S=S.numpy(), N_S=N.numpy(),
            evals=evals.numpy(), evecs=evecs.numpy(),
            x_T_path=path, x_T_head=path[-1],
        ))
    return frames


# --------------------------------------------------------------- geometry helpers
def _unit_sphere(n_theta: int = 40, n_phi: int = 22):
    theta = np.linspace(0, 2 * np.pi, n_theta)     # azimuth
    phi = np.linspace(0, np.pi, n_phi)             # polar
    T, Ph = np.meshgrid(theta, phi)
    return np.sin(Ph) * np.cos(T), np.sin(Ph) * np.sin(T), np.cos(Ph)


def _great_circle(U2: np.ndarray, n: int = 200) -> np.ndarray:
    """The memory plane's intersection with the unit sphere: cos s·u0 + sin s·u1."""
    s = np.linspace(0, 2 * np.pi, n)
    return np.cos(s)[:, None] * U2[:, 0][None, :] + np.sin(s)[:, None] * U2[:, 1][None, :]


def _off_axis(info: object) -> np.ndarray:
    """The off-manifold normal: eigenvector of the LARGEST eigenvalue of S_T."""
    evals, evecs = torch.linalg.eigh(info.S_T)
    return evecs[:, -1].cpu().numpy()


def _pick3(frames: List[Dict]) -> List[Dict]:
    n = len(frames)
    if n == 1:
        return [frames[0]] * 3
    return [frames[0], frames[n // 2], frames[-1]]


def _novelty_on_sphere(N_S: np.ndarray, X, Y, Z) -> np.ndarray:
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)        # (Nv, 3)
    return np.einsum("ni,ij,nj->n", V, N_S, V).reshape(X.shape)


# the play/pause + slider block, shared by every animation (mirrors viz_interactive)
def _anim_controls(frames: List[Dict], duration: int = 120) -> dict:
    return dict(
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.05, y=0.05,
            buttons=[
                dict(label="▶ play", method="animate",
                     args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True)]),
                dict(label="⏸ pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.1, len=0.85, y=0.0, currentvalue=dict(prefix="t = "),
            steps=[dict(method="animate", label=f"{fr['t']:.0f}",
                        args=[[f"{i}"], dict(mode="immediate",
                              frame=dict(duration=0, redraw=True))])
                   for i, fr in enumerate(frames)],
        )],
        uirevision="keep",
    )


# =============================================================== 1. NOVELTY SPHERE
def _sphere_surface(N_S, X, Y, Z, cmax, show):
    import plotly.graph_objects as go
    return go.Surface(x=X, y=Y, z=Z, surfacecolor=_novelty_on_sphere(N_S, X, Y, Z),
                      colorscale="Viridis", cmin=0.0, cmax=cmax, showscale=show,
                      colorbar=dict(title="novelty"), opacity=1.0)


def _comet(fr):
    import plotly.graph_objects as go
    p, h = fr["x_T_path"] * 1.02, fr["x_T_head"] * 1.02     # lift off the surface a touch
    path = go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
                        line=dict(width=3, color="#444"), opacity=0.55, name="x_T path")
    head = go.Scatter3d(x=[h[0]], y=[h[1]], z=[h[2]], mode="markers",
                        marker=dict(size=5, color="red"), name="x_T(t)")
    return path, head


_SCENE_UNIT = dict(xaxis=dict(range=[-1.05, 1.05], title=""),
                   yaxis=dict(range=[-1.05, 1.05], title=""),
                   zaxis=dict(range=[-1.05, 1.05], title=""), aspectmode="cube")


def novelty_sphere_animated(hist, model, info, n_frames=None):
    import plotly.graph_objects as go
    frames = eigenframes(hist, model, info, n_frames)
    X, Y, Z = _unit_sphere()
    circ = _great_circle(info.U_T[:, :2].cpu().numpy()) * 1.02
    cmax = max(0.34, max(float(_novelty_on_sphere(fr["N_S"], X, Y, Z).max()) for fr in frames))

    circ_tr = go.Scatter3d(x=circ[:, 0], y=circ[:, 1], z=circ[:, 2], mode="lines",
                           line=dict(width=5, color="#1f9e89"), name="memory plane")
    p0, h0 = _comet(frames[0])
    data0 = [_sphere_surface(frames[0]["N_S"], X, Y, Z, cmax, True), circ_tr, p0, h0]

    go_frames = []
    for i, fr in enumerate(frames):
        p, h = _comet(fr)
        go_frames.append(go.Frame(
            data=[_sphere_surface(fr["N_S"], X, Y, Z, cmax, True), p, h],
            traces=[0, 2, 3], name=f"{i}"))      # trace 1 (circle) is static

    fig = go.Figure(data=data0, frames=go_frames)
    fig.update_layout(**PLOTLY_LAYOUT, title_text="Novelty on the state sphere",
                      height=620, scene=_SCENE_UNIT,
                      **_anim_controls(frames))
    return fig


def novelty_sphere_triptych(hist, model, info, n_frames=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    frames = eigenframes(hist, model, info, n_frames)
    X, Y, Z = _unit_sphere()
    circ = _great_circle(info.U_T[:, :2].cpu().numpy()) * 1.02
    cmax = max(0.34, max(float(_novelty_on_sphere(fr["N_S"], X, Y, Z).max()) for fr in frames))

    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
                        subplot_titles=("S empty", "mid-transfer", "transferred"))
    for c, fr in enumerate(_pick3(frames), start=1):
        fig.add_trace(_sphere_surface(fr["N_S"], X, Y, Z, cmax, c == 3), row=1, col=c)
        fig.add_trace(go.Scatter3d(x=circ[:, 0], y=circ[:, 1], z=circ[:, 2], mode="lines",
                                   line=dict(width=4, color="#1f9e89"), showlegend=False), row=1, col=c)
        p, h = fr["x_T_path"] * 1.02, fr["x_T_head"] * 1.02
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
                                   line=dict(width=3, color="#444"), opacity=0.55, showlegend=False), row=1, col=c)
        fig.add_trace(go.Scatter3d(x=[h[0]], y=[h[1]], z=[h[2]], mode="markers",
                                   marker=dict(size=4, color="red"), showlegend=False), row=1, col=c)
    fig.update_scenes(**_SCENE_UNIT)
    fig.update_layout(**PLOTLY_LAYOUT, title_text="Novelty on the state sphere", height=430)
    return fig


# =============================================================== 2. STRETCH ELLIPSOID
def _ellipsoid_surface(S_S, X, Y, Z, lam_max, show):
    import plotly.graph_objects as go
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    Vd = V @ (S_S / lam_max).T
    rad = np.linalg.norm(Vd, axis=1).reshape(X.shape)
    return go.Surface(x=Vd[:, 0].reshape(X.shape), y=Vd[:, 1].reshape(X.shape),
                      z=Vd[:, 2].reshape(X.shape), surfacecolor=rad, colorscale="Magma",
                      cmin=0.0, cmax=1.0, showscale=show, colorbar=dict(title="‖S_S u‖"), opacity=0.95)


def _ellipsoid_axes(fr, lam_max, legend):
    import plotly.graph_objects as go
    out = []
    for i in range(3):                                  # eigh ascending: 0,1 = memory, 2 = off-manifold
        v = fr["evecs"][:, i] * (fr["evals"][i] / lam_max)
        name = "off-manifold axis" if i == 2 else "memory axis (→0)"
        out.append(go.Scatter3d(x=[0, v[0]], y=[0, v[1]], z=[0, v[2]], mode="lines",
                                line=dict(width=7, color="#d62728" if i == 2 else "#1f77b4"),
                                name=name, showlegend=(legend and i in (0, 2))))
    return out


def stretch_ellipsoid_animated(hist, model, info, n_frames=None):
    import plotly.graph_objects as go
    frames = eigenframes(hist, model, info, n_frames)
    X, Y, Z = _unit_sphere()

    fr0 = frames[0]
    lam0 = float(fr0["evals"].max())
    data0 = [_ellipsoid_surface(fr0["S_S"], X, Y, Z, lam0, True)] + _ellipsoid_axes(fr0, lam0, True)

    go_frames = []
    for i, fr in enumerate(frames):
        lam = float(fr["evals"].max())
        go_frames.append(go.Frame(
            data=[_ellipsoid_surface(fr["S_S"], X, Y, Z, lam, True)] + _ellipsoid_axes(fr, lam, True),
            traces=[0, 1, 2, 3], name=f"{i}"))

    fig = go.Figure(data=data0, frames=go_frames)
    fig.update_layout(**PLOTLY_LAYOUT, title_text="Stretch ellipsoid of S_S",
                      height=620, scene=_SCENE_UNIT,
                      **_anim_controls(frames))
    return fig


def stretch_ellipsoid_triptych(hist, model, info, n_frames=None):
    from plotly.subplots import make_subplots
    frames = eigenframes(hist, model, info, n_frames)
    X, Y, Z = _unit_sphere()
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
                        subplot_titles=("S empty", "mid-transfer", "transferred"))
    for c, fr in enumerate(_pick3(frames), start=1):
        lam = float(fr["evals"].max())
        fig.add_trace(_ellipsoid_surface(fr["S_S"], X, Y, Z, lam, c == 3), row=1, col=c)
        for tr in _ellipsoid_axes(fr, lam, legend=(c == 1)):
            fig.add_trace(tr, row=1, col=c)
    fig.update_scenes(**_SCENE_UNIT)
    fig.update_layout(**PLOTLY_LAYOUT, title_text="Stretch ellipsoid of S_S", height=430)
    return fig


# =============================================================== 3. ENERGY VALLEY
def _valley_grid(info, ng: int = 50, span: float = 1.5):
    u_mem = info.U_T[:, 0].cpu().numpy()
    n_off = _off_axis(info)
    a = np.linspace(-span, span, ng)
    A, Bm = np.meshgrid(a, a)
    P = A[..., None] * u_mem[None, None, :] + Bm[..., None] * n_off[None, None, :]   # (ng,ng,3)
    return A, Bm, P


def _valley_Z(S_S, P):
    return 0.5 * np.einsum("ijk,kl,ijl->ij", P, S_S, P)


def _valley_surface(A, Bm, Z, zmax, show):
    import plotly.graph_objects as go
    return go.Surface(x=A, y=Bm, z=Z, surfacecolor=Z, colorscale="Viridis",
                      cmin=0.0, cmax=zmax, showscale=show, colorbar=dict(title="energy"))


def _SCENE_VALLEY(zmax):
    return dict(xaxis=dict(title="memory dir"), yaxis=dict(title="off-manifold"),
                zaxis=dict(title="½xᵀS_S x", range=[0, zmax]), aspectmode="cube")


def energy_valley_animated(hist, model, info, n_frames=None):
    import plotly.graph_objects as go
    frames = eigenframes(hist, model, info, n_frames)
    A, Bm, P = _valley_grid(info)
    Zs = [_valley_Z(fr["S_S"], P) for fr in frames]
    zmax = float(max(Z.max() for Z in Zs))

    fig = go.Figure(data=[_valley_surface(A, Bm, Zs[0], zmax, True)],
                    frames=[go.Frame(data=[_valley_surface(A, Bm, Zs[i], zmax, True)],
                                     traces=[0], name=f"{i}") for i in range(len(frames))])
    fig.update_layout(**PLOTLY_LAYOUT, title_text="Energy valley  ½xᵀS_S x",
                      height=620, scene=_SCENE_VALLEY(zmax),
                      **_anim_controls(frames))
    return fig


def energy_valley_triptych(hist, model, info, n_frames=None):
    from plotly.subplots import make_subplots
    frames = eigenframes(hist, model, info, n_frames)
    A, Bm, P = _valley_grid(info)
    picks = _pick3(frames)
    Zs = [_valley_Z(fr["S_S"], P) for fr in frames]
    zmax = float(max(Z.max() for Z in Zs))
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
                        subplot_titles=("S empty", "mid-transfer", "transferred"))
    for c, fr in enumerate(picks, start=1):
        fig.add_trace(_valley_surface(A, Bm, _valley_Z(fr["S_S"], P), zmax, c == 3), row=1, col=c)
    fig.update_scenes(**_SCENE_VALLEY(zmax))
    fig.update_layout(**PLOTLY_LAYOUT, title_text="Energy valley  ½xᵀS_S x", height=430)
    return fig
