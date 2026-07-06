"""Static matplotlib/seaborn dashboard for a finished simulation.

Two entry points:
  - dashboard(hist, model, info): the 6-panel overview of the transfer dynamics.
  - weight_snapshots(hist, model): a row of W_S heatmaps over the run + the W_T target.

Both return a matplotlib Figure (displays inline in the VS Code interactive window).
"""
from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .history import History
from .model import TwoPopModel


def _style() -> None:
    sns.set_theme(context="notebook", style="whitegrid")


def dashboard(hist: History, model: TwoPopModel, info: Optional[Dict] = None):
    """6-panel overview: novelty-spectrum staircase, replay raster, energies, per-direction
    residual, W_S learning + manifold occupancy, and the S_T spectrum with the pi_ST guard."""
    _style()
    H = hist.to_numpy()
    t = H["t"]
    k = H["novelty_spec"].shape[1]
    P = H["align"].shape[1]

    fig, ax = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Prioritized memory transfer — network dynamics", fontsize=15, y=0.995)

    # (0,0) novelty-spectrum staircase (primary observable)
    a = ax[0, 0]
    for j in range(k):
        a.plot(t, H["novelty_spec"][:, j], lw=1.6)
    a.set(title="Restricted novelty spectrum  eig(Uᵀ N_S U)  — the staircase",
          xlabel="time", ylabel="novelty eigenvalue")
    a.axhline(model.pi_S / (model.pi_TS + model.pi_S), ls="--", c="k", alpha=0.4,
              label="n(1)=π_S/(π_TS+π_S) (S empty)")
    a.legend(loc="upper right", fontsize=8)

    # (0,1) replay raster: |cos(x_T, m_p)| over time
    a = ax[0, 1]
    im = a.imshow(np.abs(H["align"]).T, aspect="auto", origin="lower",
                  extent=[t[0], t[-1], -0.5, P - 0.5], cmap="magma", vmin=0, vmax=1)
    a.set(title="Replay raster  |cos(x_T, m_p)|", xlabel="time", ylabel="pattern p")
    a.set_yticks(range(P))
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    # (1,0) energies
    a = ax[1, 0]
    a.plot(t, H["F_T"], label="F_T", lw=1.6)
    a.plot(t, H["F_S"], label="F_S", lw=1.6)
    a.plot(t, H["Phi"], label="Φ = F_T − F_S", lw=1.6)
    a.set(title="Energies", xlabel="time", ylabel="energy")
    a.legend(fontsize=9)

    # (1,1) per-direction residual ||M_S m_p|| (spike then decay)
    a = ax[1, 1]
    for p in range(P):
        a.plot(t, H["residual"][:, p], lw=1.4, label=f"m{p}")
    a.set(title="S's residual per memory  ||M_S m_p||", xlabel="time", ylabel="residual")
    if P <= 8:
        a.legend(fontsize=8, ncol=2)

    # (2,0) W_S learning + manifold occupancy
    a = ax[2, 0]
    a.plot(t, H["WS_dist"], c="C3", lw=1.8, label="||W_S − W_T||_F")
    a.set(title="Learning progress & T's occupancy", xlabel="time", ylabel="||W_S − W_T||_F")
    a.legend(loc="upper right", fontsize=8)
    a2 = a.twinx()
    a2.plot(t, H["manifold_occ"], c="C0", lw=1.4, alpha=0.8)
    a2.set_ylabel("manifold occupancy of x_T", color="C0")
    a2.set_ylim(0, 1.02)
    a2.grid(False)

    # (2,1) S_T spectrum with pi_ST guard
    a = ax[2, 1]
    if info is not None and "S_T" in info:
        import torch
        evals = torch.linalg.eigvalsh(info["S_T"]).cpu().numpy()
        a.bar(range(len(evals)), np.sort(evals), color="C7")
        a.axhline(info["sigma2_min"], ls="--", c="C0", label=f"σ²_min={info['sigma2_min']:.3f}")
        a.axhline(abs(info["pi_ST"]), ls="-", c="C3", label=f"|π_ST|={abs(info['pi_ST']):.3f}")
        a.axhline(info["guard_scalar"], ls=":", c="C2",
                  label=f"scalar guard={info['guard_scalar']:.3f}")
        a.set(title="S_T spectrum & stability guard", xlabel="eigenvalue index",
              ylabel="eigenvalue")
        a.legend(fontsize=8)
    else:
        a.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def weight_snapshots(hist: History, model: TwoPopModel):
    """Row of W_S heatmaps over the run, with the frozen W_T target at the end."""
    _style()
    snaps = hist.weight_snaps
    n = len(snaps) + 1
    vmax = max(float(s["W_S"].abs().max()) for s in snaps) if snaps else 1.0
    vmax = max(vmax, float(model.W_T.abs().max()))
    fig, ax = plt.subplots(1, n, figsize=(2.7 * n, 3.0))
    if n == 1:
        ax = [ax]
    for i, s in enumerate(snaps):
        ax[i].imshow(s["W_S"].numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax[i].set_title(f"W_S  t={s['t']:.0f}", fontsize=9)
        ax[i].set_xticks([]); ax[i].set_yticks([])
    im = ax[-1].imshow(model.W_T.cpu().numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax[-1].set_title("W_T (target)", fontsize=9)
    ax[-1].set_xticks([]); ax[-1].set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.suptitle("W_S convergence toward W_T (note zero diagonal throughout)", y=1.04)
    return fig
