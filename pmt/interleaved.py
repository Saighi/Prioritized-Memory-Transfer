"""Interleaved merge: build the combined memory subspace by rehearsing ONE teacher at a time.

The additive model sums the teachers (`y = alpha1 x1 + alpha2 x2`) and learns from the sum. That
needs each teacher to *roam* (>= 2 memories); a single-memory teacher is pinned by the reversed-
precision feedback, the sum collapses onto the rank-1 blend `(alpha1 m1 + alpha2 m2)` (the cross-term
`alpha1 alpha2 E[x1 x2^T]` corrupts the covariance), and the merge fails.

Interleaving never forms the sum. The plastic synthesis is coupled to exactly ONE teacher per replay
bout (its interface is `active`, the other is not), alternating bouts. The time-averaged covariance
it learns from is therefore `Sigma = p1 Sigma1 + p2 Sigma2` — **no cross-term, by construction** — so
it is full rank on `U_Sigma = U1 + U2` even for single-memory, correlated teachers. The back-and-forth
is the persistent co-excitation that lets the covariance inversion cancel the crosstalk: rehearse T1
-> nulls `m1`, nudges `m2`; rehearse T2 -> nulls `m2`, nudges `m1`; repeat -> both nulled. This is
interleaved replay/rehearsal, the standard cure for catastrophic forgetting.

Each bout is a single-teacher reversed-precision transfer (the same physics as the two-population
model), so the synthesis only ever learns from a network's *replayed activity*, never from clamped
input. `interleave_merge` is the reusable bout driver (also used by `pmt.continual`).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch

from .config import AdditiveSynthesisConfig, SimConfig
from .history import InterleavedHistory
from .macro import AdditiveInterface, MacroNetwork, Population
from .memory import build_W_T, make_teacher_subspaces


def build_interleaved_synthesis(cfg: AdditiveSynthesisConfig):
    """Two frozen teachers `T1, T2` + a plastic synthesis `S`, wired with TWO single-source
    interfaces `S<-T1`, `S<-T2` (both `active=False`; the driver toggles one on per bout). Returns
    `(macro, info)` with the memory bases `U1, U2, U_Sigma`, the target rank `r_Sigma`, per-teacher
    reversed precisions, and `kind="interleaved"`."""
    from .diagnostics import manifold_basis, spectral_gap, subspace_sum_basis

    I = torch.eye(cfg.d, dtype=cfg.dtype, device=cfg.device)
    M1, M2 = make_teacher_subspaces(cfg)
    W1, W2 = build_W_T(M1, cfg), build_W_T(M2, cfg)
    M_T1, M_T2 = I - W1, I - W2
    U1, U2 = manifold_basis(M_T1), manifold_basis(M_T2)
    U_Sigma, overlap, r_Sigma = subspace_sum_basis(U1, U2)

    sigma1_min = spectral_gap(M_T1.transpose(-2, -1) @ M_T1)
    sigma2_min = spectral_gap(M_T2.transpose(-2, -1) @ M_T2)

    def _rho(pi_teacher, sigma_min):
        if cfg.exact_saddle:
            return -float(cfg.pi_I)
        if isinstance(cfg.rho, str) and cfg.rho == "auto":
            return -cfg.rho_safety * pi_teacher * sigma_min      # single source: ||C||^2 = 1
        return float(cfg.rho)

    T1 = Population("T1", W1, cfg.pi_T1, cfg.tau_T1, plastic=False, sigma_xi=cfg.sigma_xi1, r=cfg.r1)
    T2 = Population("T2", W2, cfg.pi_T2, cfg.tau_T2, plastic=False, sigma_xi=cfg.sigma_xi2, r=cfg.r2)
    S = Population("S", torch.zeros_like(W1), cfg.pi_S, cfg.tau_S, plastic=True, eta=cfg.eta)

    interfaces = [
        AdditiveInterface(target="S", sources=["T1"], alpha=[1.0], pi_I=cfg.pi_I,
                          rho=_rho(cfg.pi_T1, sigma1_min), active=False),
        AdditiveInterface(target="S", sources=["T2"], alpha=[1.0], pi_I=cfg.pi_I,
                          rho=_rho(cfg.pi_T2, sigma2_min), active=False),
    ]
    macro = MacroNetwork([T1, T2, S], interfaces)

    info = {
        "kind": "interleaved",
        "cfg": cfg,
        "M1": M1, "M2": M2, "W1": W1, "W2": W2,
        "U1": U1, "U2": U2, "U_Sigma": U_Sigma,
        "rank1": U1.shape[1], "rank2": U2.shape[1], "overlap": overlap, "r_Sigma": r_Sigma,
        "capacity_ok": r_Sigma <= cfg.d - 1,
        "sigma1_min": sigma1_min, "sigma2_min": sigma2_min,
        "target": "S", "order": ["T1", "T2"],
        "bases": {"T1": U1, "T2": U2},
        "seed": cfg.seed,
    }
    return macro, info


def interleave_merge(
    macro: MacroNetwork,
    order: List[str],
    bases: Dict[str, torch.Tensor],
    target: str,
    sim: SimConfig,
    gen: torch.Generator,
    n_bouts: int,
    record: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    """Rehearse the sources in `order` one at a time, `sim.bout_steps` steps each, for `n_bouts` bouts.

    Each bout: activate only the (single-source) interface feeding from `order[bout % len(order)]`,
    re-seed that teacher's state on its own memory span (`bases[name]`), then run the reversed-
    precision transfer so the plastic `target` perceives that teacher and takes slow learning steps.
    `record(bout, active_index)` is called after each bout. Returns the learned `target` weight.
    """
    d = macro.d
    itf_of: Dict[str, AdditiveInterface] = {}
    for itf in macro.interfaces:
        for s in itf.sources:
            itf_of[s] = itf

    def seed(name: str) -> None:
        p = macro.populations[name]
        U = bases.get(name)
        x = torch.randn(d, generator=gen, dtype=macro.dtype, device=macro.device)
        if U is not None and U.shape[1] > 0:
            x = U @ (U.transpose(-2, -1) @ x)
        r = p.r if p.r is not None else 1.0
        p.x = x * (r / x.norm().clamp_min(1e-12))

    macro.populations[target].x = torch.zeros(d, dtype=macro.dtype, device=macro.device)
    for bout in range(n_bouts):
        name = order[bout % len(order)]
        for itf in macro.interfaces:
            itf.active = itf is itf_of[name]
        seed(name)
        for _ in range(sim.bout_steps):
            macro.step(sim.dt, gen, mode=sim.mode, substeps=sim.s_substeps)
        if record is not None:
            record(bout, order.index(name))
    return macro.populations[target].W


def simulate_interleaved(macro: MacroNetwork, sim: SimConfig, info: Dict) -> InterleavedHistory:
    """Run the interleaved merge of `T1, T2 -> S` and record the crosstalk-cancellation trace."""
    cfg: AdditiveSynthesisConfig = info["cfg"]
    gen = torch.Generator(device=macro.device).manual_seed(cfg.seed + 12345)
    hist = InterleavedHistory(labels=tuple(info["order"]))
    U1, U2, U_Sigma = info["U1"], info["U2"], info["U_Sigma"]
    n_bouts = max(1, sim.n_steps // sim.bout_steps)

    def record(bout: int, active: int) -> None:
        hist.record(bout, active, macro.populations["S"].M, U1, U2, U_Sigma)

    interleave_merge(macro, info["order"], info["bases"], "S", sim, gen, n_bouts, record)
    return hist
