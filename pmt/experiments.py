"""Shared sweep helpers that turn each analytic claim into arrays a notebook can plot
against theory:
  - sweep_circulation:        circulation(pi_ST) vs |pi_ST + pi_TS| * sqrt(d)  (signed pi_ST)
  - offmanifold_growth:       max growth eigenvalue of (-pi_T S_T + |pi_ST| N_S) vs |pi_ST|
    terminal_occupancy:       dynamical terminal manifold-occupancy vs |pi_ST|
  - timescale_discrimination: known/novel S-residual vs tau_S/tau_T (clamped perception)
  - effective_rank:           numerical rank of the pattern set

Saddle exactness is a *sign* phenomenon (exact at pi_ST = -pi_TS), so sweep the signed
precision. Stability is a *magnitude* phenomenon (noise-chasing when the reversed drive is too
strong), so sweep |pi_ST|; those helpers build replay models with pi_ST = -|pi_ST|.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

import numpy as np
import torch

from . import diagnostics as dg
from .config import ModelConfig, SimConfig
from .dynamics import simulate
from .model import bwd, build_system, fwd


# ----------------------------------------------------------------- saddle exactness
def sweep_circulation(base: ModelConfig, pi_STs: Sequence[float]):
    """Empirical circulation vs the theoretical |pi_ST + pi_TS|*sqrt(d) (signed pi_ST; the
    saddle is exact at pi_ST = -pi_TS)."""
    emp, theo = [], []
    for k in pi_STs:
        model, _ = build_system(replace(base, pi_ST=float(k), exact_saddle=False))
        emp.append(dg.circulation(model))
        theo.append(abs(float(k) + base.pi_TS) * math.sqrt(base.d))
    return np.asarray(pi_STs, float), np.asarray(emp), np.asarray(theo)


# ----------------------------------------------------------------- stability
def offmanifold_growth(base: ModelConfig, pi_ST_mags: Sequence[float]):
    """Max eigenvalue of the reduced growth operator G = -pi_T S_T + |pi_ST| N_S, with S having
    fully learned the manifold (W_S = W_T). |pi_ST| is the magnitude of the reversed precision;
    G lifts off 0 exactly at the stability bound."""
    model, info = build_system(replace(base, pi_ST=0.0))
    model.W_S = model.W_T.clone()              # S has consolidated the whole manifold
    S_T = info["S_T"]
    gmax = []
    for m in pi_ST_mags:
        G = -model.pi_T * S_T + float(m) * model.novelty_operator()
        gmax.append(float(torch.linalg.eigvalsh(G).max()))
    return np.asarray(pi_ST_mags, float), np.asarray(gmax), info


def terminal_occupancy(base: ModelConfig, pi_ST_mags: Sequence[float],
                       n_steps: int = 30000, dt: float = 0.5):
    """Dynamical test: run a full replay transfer at each |pi_ST| (so pi_ST = -|pi_ST|) and
    report the terminal fraction of x_T inside the teacher's memory manifold. ~1 below the guard
    (quiescent on-manifold), drops above it (T chases noise after transfer)."""
    occ, final_nov = [], []
    for m in pi_ST_mags:
        model, info = build_system(replace(base, pi_ST=-float(m)))
        sim = SimConfig(n_steps=n_steps, dt=dt, mode="adiabatic",
                        record_every=max(1, n_steps // 60), progress=False)
        H = simulate(model, sim, info).to_numpy()
        occ.append(float(np.mean(H["manifold_occ"][-5:])))
        final_nov.append(float(H["novelty_spec"][-5:].max()))
    return np.asarray(pi_ST_mags, float), np.asarray(occ), np.asarray(final_nov)


# ----------------------------------------------------------------- timescales
def _relax_S(model, x_T, tau_S, dt, T):
    """Clamp T at x_T, relax S (perception only) for wall-time T; return ||eps_TS||."""
    model.x_T = x_T.clone()
    model.x_S = torch.zeros_like(x_T)
    for _ in range(max(1, int(T / dt))):
        eps_TS = model.x_S - model.x_T
        eps_S = fwd(model.M_S, model.x_S)
        model.x_S = model.x_S + dt * (-model.pi_TS * eps_TS - model.pi_S * bwd(model.M_S, eps_S)) / tau_S
    return float((model.x_S - model.x_T).norm())


def timescale_discrimination(base: ModelConfig, ratios: Sequence[float],
                             known_idx, novel_idx, tau_T: float = 10.0):
    """For each tau_S/tau_T, pretrain S on `known_idx`, then clamp T to a known vs a novel
    pattern and relax S for one T-timescale. Returns the residual ||eps_TS|| in each case.
    A known pattern is only 'explained away' (small residual) when S is fast enough."""
    res_known, res_novel = [], []
    for r in ratios:
        tau_S = float(r) * tau_T
        model, _ = build_system(replace(base, tau_T=tau_T, tau_S=tau_S))
        model.pretrain(known_idx)
        dt = 0.1 * tau_S                      # keep S-Euler stable across the whole sweep
        mk = model.patterns[:, known_idx[0]]
        mn = model.patterns[:, novel_idx[0]]
        res_known.append(_relax_S(model, mk, tau_S, dt, T=tau_T))
        res_novel.append(_relax_S(model, mn, tau_S, dt, T=tau_T))
    return np.asarray(ratios, float), np.asarray(res_known), np.asarray(res_novel)


# ----------------------------------------------------------------- subspace rank
def effective_rank(M: torch.Tensor, tol: float = 1e-6) -> int:
    """Number of significant singular values of the pattern matrix (its subspace dim)."""
    s = torch.linalg.svdvals(M)
    return int((s > tol * s.max()).sum())
