"""The six SMACOF-landscape figures for the two-population predictive-coding model.

This is where the model meets the embedding: it calls the live `pmt` network (via `model` / `info`
/ `hist` objects the notebook produces) for the operators S_T, N_S and the W_S weight snapshots,
then hands scalar heights to the pure `embedding` toolkit to lay out and drape.

Everything reduces, in the fast-S limit, to quadratic forms in the teacher state x_T on the sphere
‖x_T‖=r0 (see two_population_memory_transfer_model.md §4, §8, §9):

    F_T(x)   = (pi_T /2) xᵀ S_T x          zero on the memory manifold ker(M_T); rises off it
    F_S*(x)  = (pi_TS/2) xᵀ N_S x          the novelty surface; 0 on learned dirs, >0 on unlearned
    Phi(x)   = F_T - F_S*                   the saddle: flat memory floor, valleys at novel dirs,
                                            off-manifold walls

HONEST CAVEAT (carried in the captions): the linear model's VFE is a smooth quadratic form with NO
local basins on the manifold — the manifold is exactly flat. A faithful render is ONE flat plateau,
not Hopfield-style per-memory valleys. The drama is the surface morphing over training (novel-
direction valleys filling in) and the sleep-vs-wake sign flip, not static basins.

Operators are recomputed from each W_S snapshot by pure local helpers (s_s_of / n_s_of) so the live
model is never mutated. Heights depend on S_T, N_S, pi_T, pi_TS — NOT on the sign of pi_ST; only the
*simulated W_S trajectory* differs between sleep and wake, which is exactly what figure 5 contrasts.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import embedding as emb


# ----------------------------------------------------------------- operator helpers (numpy, pure)
# Mirror model.S_S() / model.novelty_operator(); take a snapshot W_S, never touch the model.
def s_s_of(W_S: np.ndarray) -> np.ndarray:
    """S_S = M_Sᵀ M_S with M_S = I - W_S."""
    W = np.asarray(W_S, dtype=float)
    M = np.eye(W.shape[-1]) - W
    return M.T @ M


def n_s_of(W_S: np.ndarray, pi_TS: float, pi_S: float) -> np.ndarray:
    """N_S = pi_S S_S (pi_TS I + pi_S S_S)^-1 (the novelty operator)."""
    S = s_s_of(W_S)
    I = np.eye(S.shape[-1])
    return pi_S * S @ np.linalg.inv(pi_TS * I + pi_S * S)


def _np(x) -> np.ndarray:
    """torch tensor or array -> float numpy (cpu)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


# ----------------------------------------------------------------- height (scalar) fields
def _quad(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Row-wise quadratic form xᵀ A x for each row of X (n,d) -> (n,)."""
    return np.einsum("ni,ij,nj->n", X, A, X)


def heights_FT(X, S_T, pi_T):                 # teacher self-energy (flat plateau)
    return 0.5 * pi_T * _quad(X, _np(S_T))


def heights_FS_self(X, W_S, pi_S):            # student's own pattern-completion energy
    return 0.5 * pi_S * _quad(X, s_s_of(_np(W_S)))


def heights_novelty(X, N_S, pi_TS):           # F_S* = novelty surface
    return 0.5 * pi_TS * _quad(X, _np(N_S))


def heights_Phi(X, S_T, N_S, pi_T, pi_TS):    # the saddle Phi = F_T - F_S*
    return heights_FT(X, S_T, pi_T) - heights_novelty(X, N_S, pi_TS)


# ----------------------------------------------------------------- state-cloud sampling
def sample_states(model, info, n: int = 1200, p_manifold: float = 0.55, p_off: float = 0.25,
                  noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """A cloud of teacher states x_T on the sphere ‖x‖=r0, to be SMACOF-embedded.

    Mixture (mirrors the Hopfield sampler): on-manifold mixtures (random combinations of the memory
    basis U_T), off-manifold points (a manifold mixture plus a component along the steepest off-
    manifold normal), and isotropic random states. The P stored patterns are prepended as the first
    P rows (anchors). Every row is jittered by `noise` and renormalized to r0.
    """
    rng = np.random.default_rng(seed)
    U_T = _np(info["U_T"])                      # (d, k) memory basis
    d, k = U_T.shape
    r0 = float(model.r0)
    patterns = _np(model.patterns).T            # (P, d)

    # steepest off-manifold normal = top eigenvector of S_T
    evals, evecs = np.linalg.eigh(_np(info["S_T"]))
    n_off = evecs[:, -1]

    n_man = int(p_manifold * n)
    n_offc = int(p_off * n)
    n_rand = max(0, n - n_man - n_offc - len(patterns))

    def _manifold_mix(m):
        c = rng.standard_normal((m, k))
        return c @ U_T.T                        # (m, d), lies in ker(M_T)

    man = _manifold_mix(n_man)
    off = _manifold_mix(n_offc) + rng.uniform(0.4, 1.6, (n_offc, 1)) * n_off[None, :]
    rnd = rng.standard_normal((n_rand, d))

    X = np.concatenate([patterns, man, off, rnd], axis=0)
    X = X + noise * rng.standard_normal(X.shape)
    X *= r0 / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X


def make_layout(model, info, X=None, n: int = 1200, seed: int = 0,
                metric: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Sample (if needed) and SMACOF-embed once. Returns (X, layout). Reuse the SAME pair across
    figures so the embedding is stable (the fixed-layout rule)."""
    if X is None:
        X = sample_states(model, info, n=n, seed=seed)
    layout = emb.smacof_layout(emb.pairwise_dist(X, metric=metric), seed=seed)
    return X, layout


# ----------------------------------------------------------------- shared helpers
def _ensure(model, info, X, layout, n, seed):
    if X is None or layout is None:
        X, layout = make_layout(model, info, X=X, n=n, seed=seed)
    return X, layout


def _anchor_markers(layout, heights, n_anchor, label="memory pattern", lift=0.02):
    """The first n_anchor sampled states (the stored patterns) as cyan diamonds on the surface."""
    import plotly.graph_objects as go

    span = float(np.nanmax(heights) - np.nanmin(heights)) or 1.0
    z = heights[:n_anchor] + lift * span
    return go.Scatter3d(
        x=layout[:n_anchor, 0], y=layout[:n_anchor, 1], z=z, mode="markers",
        marker=dict(size=4, color="cyan", symbol="diamond", line=dict(width=1, color="black")),
        name=label, hovertemplate=f"{label}<extra></extra>",
    )


def _snaps(hist, n_frames):
    """Sub-sample weight snapshots to at most n_frames; returns list of {t, W_S(numpy)}."""
    snaps = hist.weight_snaps
    if not snaps:
        raise ValueError("no weight snapshots — set SimConfig.n_weight_snapshots > 0")
    if n_frames is not None and n_frames < len(snaps):
        keep = np.unique(np.linspace(0, len(snaps) - 1, n_frames).astype(int))
        snaps = [snaps[i] for i in keep]
    return [(float(s["t"]), _np(s["W_S"])) for s in snaps]


# =============================================================== 1. TEACHER FLAT PLATEAU
def teacher_plateau(model, info, layout=None, X=None, n: int = 1200, seed: int = 0):
    """Static: F_T draped over the embedding. The memory manifold is a flat valley floor; off-
    manifold rises into walls. Visualizes 'T is flat' — deliberately featureless on the manifold."""
    import plotly.graph_objects as go

    X, layout = _ensure(model, info, X, layout, n, seed)
    h = heights_FT(X, info["S_T"], model.pi_T)
    Xi, Yi, Zi = emb.drape_surface(layout, h)

    P = _np(model.patterns).shape[1]
    fig = go.Figure([
        emb.surface_trace(Xi, Yi, Zi, colorscale=emb.INFERNO, colorbar_title="F_T"),
        _anchor_markers(layout, h, P),
    ])
    fig.update_layout(scene=emb.dark_scene("F_T  =  ½ π_T xᵀ S_T x"),
                      **emb.dark_layout("Teacher plateau — F_T over the SMACOF embedding<br>"
                                        "<sub>flat on the memory manifold, walls off it (the memories live on the flat floor)</sub>"))
    return fig


# =============================================================== 2. STUDENT CARVING ITS MANIFOLD
def student_carving_animated(hist, model, info, layout=None, X=None, n: int = 1200,
                             seed: int = 0, n_frames: Optional[int] = 30, grid_n: int = 100):
    """Animated over training: F_S^self = ½ π_S xᵀ S_S x on a FIXED layout. On the sphere ‖x‖=r0 it
    starts UNIFORM (S empty → S_S=I, isotropic), then carves valleys along the directions S learns
    (those eigenvalues fall to 0) while off-manifold rises into walls (those eigenvalues grow)."""
    import plotly.graph_objects as go

    X, layout = _ensure(model, info, X, layout, n, seed)
    snaps = _snaps(hist, n_frames)
    Hs = [heights_FS_self(X, W, model.pi_S) for _, W in snaps]
    drapes = [emb.drape_surface(layout, h, grid_n=grid_n) for h in Hs]
    cmax = max(float(np.nanmax(d[2])) for d in drapes)
    Xi, Yi, _ = drapes[0]

    def surf(i):
        return emb.surface_trace(Xi, Yi, drapes[i][2], cmin=0.0, cmax=cmax,
                                 colorscale=emb.INFERNO, colorbar_title="½xᵀS_S x")

    fig = go.Figure(
        data=[surf(0)],
        frames=[go.Frame(data=[surf(i)], traces=[0], name=f"{i}") for i in range(len(snaps))],
    )
    fig.update_layout(scene=emb.dark_scene("F_Sˢᵉˡᶠ", zrange=[0, cmax]),
                      **emb.dark_layout("Student carving its manifold — ½ π_S xᵀ S_S x over training<br>"
                                        "<sub>uniform on the sphere at first → valleys carve along learned directions, walls rise off-manifold</sub>"),
                      **emb.anim_controls([f"{t:.0f}" for t, _ in snaps]))
    return fig


# =============================================================== 3. SADDLE CROSS-SECTIONS
def saddle_cross_sections(hist, model, info, span: float = 1.0, n_t: int = 121):
    """Static pair (1×2): Phi along a 1-D line t·u through x_T-space.

      • bowl  — an OFF-manifold direction: F_T's restoring term dominates → convex (T damps it).
      • hill  — a NOVEL memory direction at t=0 (S empty): concave → T slides off (the replay drive).
                Overlaid dashed: the SAME direction at the final snapshot, flattened as S learns it
                ('learning flips the hill into a flat floor', releasing the drive).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    S_T = _np(info["S_T"])
    U_T = _np(info["U_T"])
    pi_T, pi_TS = model.pi_T, model.pi_TS
    snaps = _snaps(hist, None)
    N0 = n_s_of(snaps[0][1], pi_TS, model.pi_S)        # S empty → high isotropic novelty
    Nf = n_s_of(snaps[-1][1], pi_TS, model.pi_S)       # final → manifold learned

    # off-manifold normal (bowl direction) and the most-novel manifold direction at t=0 (hill)
    ev, evec = np.linalg.eigh(S_T)
    u_off = evec[:, -1]
    Rn = U_T.T @ N0 @ U_T
    w, V = np.linalg.eigh(Rn)
    u_novel = U_T @ V[:, -1]
    u_off /= np.linalg.norm(u_off); u_novel /= np.linalg.norm(u_novel)

    t = np.linspace(-span, span, n_t)

    def phi_along(u, N):
        a = 0.5 * pi_T * float(u @ S_T @ u) - 0.5 * pi_TS * float(u @ N @ u)
        return a * t ** 2

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("bowl — off-manifold direction (T damps; stable)",
                                        "hill — novel memory direction (T slides off → replay)"))
    fig.add_trace(go.Scatter(x=t, y=phi_along(u_off, Nf), mode="lines",
                             line=dict(color="#4fc3f7", width=3), name="off-manifold"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=phi_along(u_novel, N0), mode="lines",
                             line=dict(color="#ef5350", width=3), name="novel (S empty)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=phi_along(u_novel, Nf), mode="lines",
                             line=dict(color="#bbbbbb", width=2, dash="dash"),
                             name="same dir, after learning"), row=1, col=2)
    for c in (1, 2):
        fig.update_xaxes(title_text="displacement t along direction", zeroline=True,
                         zerolinecolor="rgba(255,255,255,0.3)", row=1, col=c)
        fig.update_yaxes(title_text="Φ = F_T − F_S*", zeroline=True,
                         zerolinecolor="rgba(255,255,255,0.3)", row=1, col=c)
    fig.update_layout(template="plotly_dark",
                      **emb.dark_layout("Saddle cross-sections of Φ — why the sleep flow is a saddle<br>"
                                        "<sub>bowl along predictable directions, hill along novel ones; learning flattens the hill</sub>",
                                        height=460))
    return fig


# =============================================================== 4. DEFLATING-NOVELTY (Phi) SURFACE
def deflating_novelty_animated(hist, model, info, layout=None, X=None, n: int = 1200,
                               seed: int = 0, n_frames: Optional[int] = 30, grid_n: int = 100):
    """Animated over training: Φ = F_T − F_S* on a FIXED layout. The memory manifold is the flat
    floor; novel directions are VALLEYS (T is driven there during replay); off-manifold are walls.
    As S learns, the valleys fill up to the floor — the novelty-spectrum staircase as a landscape."""
    import plotly.graph_objects as go

    X, layout = _ensure(model, info, X, layout, n, seed)
    snaps = _snaps(hist, n_frames)
    S_T = info["S_T"]
    Hs = [heights_Phi(X, S_T, n_s_of(W, model.pi_TS, model.pi_S), model.pi_T, model.pi_TS)
          for _, W in snaps]
    drapes = [emb.drape_surface(layout, h, grid_n=grid_n) for h in Hs]
    # symmetric color range centered on the flat memory floor (0): valleys blue, walls red.
    # Robust to a few very tall off-manifold wall points (percentile, not max).
    M = max(float(np.nanpercentile(np.abs(d[2]), 96)) for d in drapes)
    Xi, Yi, _ = drapes[0]

    def surf(i):
        return emb.surface_trace(Xi, Yi, drapes[i][2], cmin=-M, cmax=M,
                                 colorscale=emb.DIVERGING, colorbar_title="Φ")

    fig = go.Figure(
        data=[surf(0)],
        frames=[go.Frame(data=[surf(i)], traces=[0], name=f"{i}") for i in range(len(snaps))],
    )
    fig.update_layout(scene=emb.dark_scene("Φ = F_T − F_S*"),
                      **emb.dark_layout("Deflating novelty — Φ over training (the primary result, as a landscape)<br>"
                                        "<sub>flat memory floor · novel-direction valleys filling in as S learns · off-manifold walls</sub>"),
                      **emb.anim_controls([f"{t:.0f}" for t, _ in snaps]))
    return fig


# =============================================================== 5. SLEEP vs WAKE
def sleep_vs_wake(hist_sleep, hist_wake, model, info, layout=None, X=None,
                  n: int = 1200, seed: int = 0):
    """Static side-by-side (1×2 scenes): the FINAL Φ surface after a sleep run (π_ST<0, transfer)
    vs a wake run (π_ST>0, no transfer drive). Same teacher, same layout. Sleep fills the novel-
    direction valleys (transfer complete → flat floor); wake leaves them deep (the recall control).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    X, layout = _ensure(model, info, X, layout, n, seed)
    S_T = info["S_T"]

    def final_phi(hist):
        _, W = _snaps(hist, None)[-1]
        h = heights_Phi(X, S_T, n_s_of(W, model.pi_TS, model.pi_S), model.pi_T, model.pi_TS)
        return emb.drape_surface(layout, h)

    dS, dW = final_phi(hist_sleep), final_phi(hist_wake)
    M = max(float(np.nanpercentile(np.abs(dS[2]), 96)), float(np.nanpercentile(np.abs(dW[2]), 96)))

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("sleep  (π_ST < 0)  — valleys filled: transfer",
                                        "wake  (π_ST > 0)  — valleys remain: no transfer"))
    fig.add_trace(emb.surface_trace(dS[0], dS[1], dS[2], cmin=-M, cmax=M,
                                    colorscale=emb.DIVERGING, showscale=False), row=1, col=1)
    fig.add_trace(emb.surface_trace(dW[0], dW[1], dW[2], cmin=-M, cmax=M,
                                    colorscale=emb.DIVERGING, colorbar_title="Φ"), row=1, col=2)
    sc = emb.dark_scene("Φ")
    fig.update_layout(scene=sc, scene2=sc,
                      **emb.dark_layout("Sleep vs wake — the reversed-precision sign flip<br>"
                                        "<sub>identical teacher &amp; embedding; only sign(π_ST) differs. Sleep transfers, wake does not.</sub>"))
    return fig


# =============================================================== 6. FLOW FIELD (descent on Phi)
def flow_field(model, info, layout=None, X=None, n: int = 1200, seed: int = 0,
               W_S=None, grid_n: int = 80, n_arrows: int = 24):
    """Static 2-D: a quiver of steepest descent on the draped Φ landscape — i.e. the teacher's sleep
    flow (T descends Φ ⇒ it is driven into the novel-direction valleys). Φ uses the supplied W_S
    (default: S empty → maximal drive). A filled Φ contour underlays the arrows.
    """
    import plotly.graph_objects as go
    import plotly.figure_factory as ff

    X, layout = _ensure(model, info, X, layout, n, seed)
    if W_S is None:
        W_S = np.zeros((model.d, model.d))      # S empty: every memory direction still novel
    N_S = n_s_of(_np(W_S), model.pi_TS, model.pi_S)
    h = heights_Phi(X, info["S_T"], N_S, model.pi_T, model.pi_TS)
    Xi, Yi, Zi = emb.drape_surface(layout, h, grid_n=grid_n)

    # steepest descent on Phi: (u,v) = -∇Zi (NaN outside the hull)
    xi, yi = Xi[0, :], Yi[:, 0]
    valid = ~np.isnan(Zi)
    gy, gx = np.gradient(np.where(valid, Zi, np.nanmean(Zi)), yi, xi)
    u = np.where(valid, -gx, np.nan)
    v = np.where(valid, -gy, np.nan)

    # subsample to a readable quiver of valid in-hull arrows (1-D), normalized to show direction
    s = max(1, grid_n // n_arrows)
    XS, YS = np.meshgrid(xi[::s], yi[::s])
    US, VS = u[::s, ::s], v[::s, ::s]
    m = np.isfinite(US) & np.isfinite(VS)
    xs, ys, us, vs = XS[m], YS[m], US[m], VS[m]
    spmax = float(np.hypot(us, vs).max()) or 1.0
    us, vs = us / spmax, vs / spmax
    dx = float(xi[1] - xi[0])
    quiv = ff.create_quiver(xs, ys, us, vs, scale=dx * s * 0.9, arrow_scale=0.35,
                            line=dict(color="rgba(255,255,255,0.85)", width=1.2), name="descent flow")

    fig = go.Figure([
        go.Contour(x=xi, y=yi, z=Zi, colorscale=emb.DIVERGING, contours=dict(coloring="heatmap"),
                   colorbar=dict(title=dict(text="Φ", font=dict(color="white")),
                                 tickfont=dict(color="white")), opacity=0.9, name="Φ"),
        quiv.data[0],
    ])
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False, scaleanchor="x")
    fig.update_layout(**emb.dark_layout("Teacher's sleep flow — steepest descent on Φ<br>"
                                        "<sub>streamlines run downhill into the novel-direction valleys: the replay drive (S empty)</sub>",
                                        height=620))
    return fig
