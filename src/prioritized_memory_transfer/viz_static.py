"""Static matplotlib dashboard for a finished simulation.

Two entry points:
  - dashboard(hist, model, info): the 6-panel overview of the transfer dynamics.
  - weight_snapshots(hist, model): a row of W_S heatmaps over the run + the W_T target.

Both return a matplotlib Figure (displays inline in the VS Code interactive window).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .history import History
from .model import TwoPopModel
from .viz_style import despine_all, mpl_style


def dashboard(
    hist: History,
    model: TwoPopModel,
    info=None,
):
    """6-panel overview: novelty staircase, replay raster, energies, per-memory residual,
    weight convergence + manifold occupancy, and the teacher spectrum with the pi_ST guard."""
    mpl_style()
    H = hist.to_numpy()
    t = H["t"]
    k = H["novelty_spec"].shape[1]
    P = H["align"].shape[1]

    fig, ax = plt.subplots(3, 2, figsize=(15, 12))

    # (0,0) novelty-spectrum staircase (primary observable)
    a = ax[0, 0]
    for j in range(k):
        a.plot(t, H["novelty_spec"][:, j], lw=1.8)
    a.set(title="Novelty staircase", xlabel="time",
          ylabel=r"eig$(U^\top N_S\, U)$")
    a.axhline(model.pi_S / (model.pi_TS + model.pi_S), ls="--", c="k", alpha=0.4,
              label=r"$n(1)$ (S empty)")
    a.legend(loc="upper right")

    # (0,1) replay raster: |cos(x_T, m_p)| over time
    a = ax[0, 1]
    im = a.imshow(np.abs(H["align"]).T, aspect="auto", origin="lower",
                  extent=[t[0], t[-1], -0.5, P - 0.5], cmap="magma", vmin=0, vmax=1)
    a.set(title="Replay raster", xlabel="time", ylabel="memory index $p$")
    a.set_yticks(range(P))
    a.grid(False)
    despine_all(a)
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    cb.set_label(r"$|\cos(x_T, m_p)|$")

    # (1,0) energies
    a = ax[1, 0]
    a.plot(t, H["F_T"], label="$F_T$", lw=1.8)
    a.plot(t, H["F_S"], label="$F_S$", lw=1.8)
    a.plot(t, H["Phi"], label=r"$\Phi = F_T - F_S$", lw=1.8)
    a.set(title="Free energies", xlabel="time", ylabel="energy")
    a.legend()

    # (1,1) per-memory residual ||M_S m_p|| (spike then decay)
    a = ax[1, 1]
    for p in range(P):
        a.plot(t, H["residual"][:, p], lw=1.6, label=f"$m_{p}$")
    a.set(title="Per-memory residual", xlabel="time", ylabel=r"$\|M_S\, m_p\|$")
    if P <= 8:
        a.legend(ncol=2)

    # (2,0) W_S learning + manifold occupancy
    a = ax[2, 0]
    a.plot(t, H["WS_dist"], c="C3", lw=2.0)
    a.set(title="Weight convergence", xlabel="time", ylabel=r"$\|W_S - W_T\|_F$")
    a2 = a.twinx()
    a2.plot(t, H["manifold_occ"], c="C0", lw=1.6, alpha=0.8)
    a2.set_ylabel(r"manifold occupancy of $x_T$", color="C0")
    a2.set_ylim(0, 1.02)
    a2.grid(False)
    a2.spines["right"].set_visible(True)

    # (2,1) S_T spectrum with pi_ST guard
    a = ax[2, 1]
    if info is not None:
        import torch
        evals = torch.linalg.eigvalsh(info.S_T).cpu().numpy()
        a.bar(range(len(evals)), np.sort(evals), color="C7")
        a.axhline(info.sigma2_min, ls="--", c="C0",
                  label=rf"$\sigma^2_{{\min}}$ = {info.sigma2_min:.2f}")
        a.axhline(abs(info.pi_ST), ls="-", c="C3",
                  label=rf"$|\pi_{{ST}}|$ = {abs(info.pi_ST):.2f}")
        a.axhline(info.guard_scalar, ls=":", c="C2",
                  label=f"guard = {info.guard_scalar:.2f}")
        a.set(title="Teacher spectrum & guard", xlabel="eigenvalue index",
              ylabel=r"eig$(S_T)$")
        a.legend()
    else:
        a.axis("off")

    fig.tight_layout()
    return fig


def weight_snapshots(hist: History, model: TwoPopModel):
    """Row of W_S heatmaps over the run, with the frozen W_T target at the end."""
    mpl_style()
    snaps = hist.weight_snaps
    n = len(snaps) + 1
    vmax = max(float(s["W_S"].abs().max()) for s in snaps) if snaps else 1.0
    vmax = max(vmax, float(model.W_T.abs().max()))
    fig, ax = plt.subplots(1, n, figsize=(2.7 * n, 3.0))
    if n == 1:
        ax = [ax]
    for i, s in enumerate(snaps):
        ax[i].imshow(s["W_S"].numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax[i].set_title(f"$W_S$   t = {s['t']:.0f}", fontsize=12)
        ax[i].set_xticks([]); ax[i].set_yticks([]); ax[i].grid(False)
        despine_all(ax[i])
    im = ax[-1].imshow(model.W_T.cpu().numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax[-1].set_title("$W_T$ (target)", fontsize=12)
    ax[-1].set_xticks([]); ax[-1].set_yticks([]); ax[-1].grid(False)
    despine_all(ax[-1])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.suptitle(r"$W_S \to W_T$ (zero diagonal throughout)", y=1.06)
    return fig
