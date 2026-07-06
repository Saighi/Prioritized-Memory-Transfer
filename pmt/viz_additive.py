"""Visualization for the additive three-network synthesis model.

  - dashboard(hist, info):     static 6-panel matplotlib overview of a run.
  - weight_snapshots(hist, info): W_S heatmaps over the run next to the two teacher weights.
  - staircase(hist):           interactive plotly novelty staircase on U_Sigma.
  - source_balance(hist):      interactive plotly source-novelty rebalancing (J1, J2, J12).

Matplotlib / plotly are imported lazily so the core has no plotting dependency on import.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def dashboard(hist, info: Optional[Dict] = None):
    """Static 6-panel overview of an additive-synthesis run (spec section 19 observables)."""
    import matplotlib.pyplot as plt

    H = hist.to_numpy()
    t = H["t"]
    fig, ax = plt.subplots(3, 2, figsize=(13, 11))

    # (0,0) transfer deficits E_Sigma, E1, E2 -> floor
    ax[0, 0].plot(t, H["E_Sigma"], label=r"$E_\Sigma=\|M_S U_\Sigma\|_F^2$", lw=2)
    ax[0, 0].plot(t, H["E1"], label=r"$E_1$", lw=1.3)
    ax[0, 0].plot(t, H["E2"], label=r"$E_2$", lw=1.3)
    ax[0, 0].set_title("Combined-subspace transfer deficit")
    ax[0, 0].set_xlabel("time"); ax[0, 0].set_ylabel("deficit"); ax[0, 0].legend(fontsize=8)

    # (0,1) restricted novelty staircase on U_Sigma
    spec = H["novelty_spec"]
    for j in range(spec.shape[1]):
        ax[0, 1].plot(t, spec[:, j], lw=1.1)
    ax[0, 1].set_title(r"Restricted novelty spectrum  eig$(U_\Sigma^\top N_S U_\Sigma)$")
    ax[0, 1].set_xlabel("time"); ax[0, 1].set_ylabel("novelty eigenvalue")

    # (1,0) source balancing J1, J2, J12
    ax[1, 0].plot(t, H["J1"], label=r"$J_1=\|\alpha_1 N_S x_1\|^2$", lw=1.5)
    ax[1, 0].plot(t, H["J2"], label=r"$J_2=\|\alpha_2 N_S x_2\|^2$", lw=1.5)
    ax[1, 0].plot(t, H["J12"], label=r"$J_{12}$ (cross)", lw=1.0, ls="--")
    ax[1, 0].set_title("Source novelty contributions (automatic rebalancing)")
    ax[1, 0].set_xlabel("time"); ax[1, 0].set_ylabel("novelty power"); ax[1, 0].legend(fontsize=8)

    # (1,1) termination signals
    ax[1, 1].plot(t, H["eps_Sigma_norm"] ** 2, label=r"$\|\varepsilon_\Sigma\|^2$", lw=1.5)
    ax[1, 1].plot(t, H["eps_S_norm"] ** 2, label=r"$\|\varepsilon_S\|^2$", lw=1.5)
    ax[1, 1].plot(t, H["dWS_norm"], label=r"$\|dW_S/dt\|_F$", lw=1.5)
    ax[1, 1].set_title("Common-error self-termination")
    ax[1, 1].set_xlabel("time"); ax[1, 1].set_yscale("log"); ax[1, 1].legend(fontsize=8)

    # (2,0) teacher manifold leakage / occupancy
    ax[2, 0].plot(t, H["L1"], label=r"$L_1=\|Q_1 x_1\|^2$", lw=1.5)
    ax[2, 0].plot(t, H["L2"], label=r"$L_2=\|Q_2 x_2\|^2$", lw=1.5)
    ax[2, 0].set_title("Teacher off-manifold leakage (structure guard)")
    ax[2, 0].set_xlabel("time"); ax[2, 0].set_ylabel("leakage"); ax[2, 0].legend(fontsize=8)

    # (2,1) mixture persistent-excitation margin
    ax[2, 1].plot(t, H["mix_min_eig"], color="C3", lw=1.5)
    ax[2, 1].set_title(r"Mixture exploration  $\lambda_{\min}(U_\Sigma^\top \Sigma_y U_\Sigma)$")
    ax[2, 1].set_xlabel("time"); ax[2, 1].set_ylabel(r"$\lambda_{\min}$")

    if info is not None:
        fig.suptitle(
            f"additive synthesis — geometry={info['cfg'].geometry}, "
            f"r1={info['rank1']}, r2={info['rank2']}, overlap={info['overlap']}, "
            f"r_Σ={info['r_Sigma']} (capacity_ok={info['capacity_ok']}), "
            f"ρ={info['rho']:.3f}",
            fontsize=11,
        )
    fig.tight_layout()
    return fig


def weight_snapshots(hist, info: Dict):
    """W_S heatmaps over the run, next to the two frozen teacher weight matrices."""
    import matplotlib.pyplot as plt

    snaps = hist.weight_snaps
    n = len(snaps)
    fig, ax = plt.subplots(1, n + 2, figsize=(2.4 * (n + 2), 2.6))
    vmax = max(float(s["W_S"].abs().max()) for s in snaps) if snaps else 1.0
    for i, s in enumerate(snaps):
        ax[i].imshow(s["W_S"].numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax[i].set_title(f"W_S  t={s['t']:.0f}", fontsize=8); ax[i].axis("off")
    for j, key in enumerate(("W1", "W2")):
        W = info[key].detach().cpu().numpy()
        ax[n + j].imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax[n + j].set_title(f"{key} (frozen)", fontsize=8); ax[n + j].axis("off")
    fig.suptitle("Synthesis weights W_S over training vs the two frozen teachers", fontsize=10)
    fig.tight_layout()
    return fig


def staircase(hist):
    """Interactive plotly novelty staircase on U_Sigma."""
    import plotly.graph_objects as go
    H = hist.to_numpy()
    t, spec = H["t"], H["novelty_spec"]
    fig = go.Figure()
    for j in range(spec.shape[1]):
        fig.add_trace(go.Scatter(x=t, y=spec[:, j], mode="lines", name=f"dir {j}"))
    fig.update_layout(
        title="Restricted novelty spectrum eig(U_Σᵀ N_S U_Σ) — the combined-subspace staircase",
        xaxis_title="time", yaxis_title="novelty eigenvalue",
        template="plotly_white", height=420,
    )
    return fig


def source_balance(hist):
    """Interactive plotly view of automatic source rebalancing (J1, J2, and total novelty)."""
    import plotly.graph_objects as go
    H = hist.to_numpy()
    t = H["t"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=H["J1"], mode="lines", name="J₁ (teacher 1)"))
    fig.add_trace(go.Scatter(x=t, y=H["J2"], mode="lines", name="J₂ (teacher 2)"))
    fig.add_trace(go.Scatter(x=t, y=H["J1"] + H["J2"] + H["J12"], mode="lines",
                             name="total novelty", line=dict(dash="dash", color="black")))
    fig.update_layout(
        title="Source-novelty rebalancing — a learned source loses its drive",
        xaxis_title="time", yaxis_title="novelty power",
        template="plotly_white", height=420,
    )
    return fig
