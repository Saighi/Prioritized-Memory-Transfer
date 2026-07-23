"""Interleaved subspace addition: merge two teachers' memory subspaces into one plastic
network by rehearsing ONE teacher at a time.

The plastic synthesis `S` is coupled to exactly one teacher per replay bout (its interface is
`active`, the other is not), alternating bouts. The time-averaged covariance it learns from is
`Sigma = p1 Sigma1 + p2 Sigma2` — no teacher cross-term, by construction — so it is full rank
on `U_Sigma = U1 + U2` even for single-memory, correlated teachers. The back-and-forth is the
co-excitation that lets the covariance inversion cancel crosstalk: rehearse T1 -> nulls `m1`,
nudges `m2`; rehearse T2 -> nulls `m2`, nudges `m1`; repeat -> both nulled. This is interleaved
replay/rehearsal, the standard cure for catastrophic forgetting.

Each bout is a single-teacher reversed-precision transfer (the same physics as the
two-population model in `prioritized_memory_transfer.model`), so the synthesis only ever learns from a network's
*replayed activity*, never from clamped input. `interleave_merge` is the reusable bout driver
(also used by `prioritized_memory_transfer.continual`).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, Dict, Optional, Sequence

import torch

from .config import InterleavedConfig, SimConfig
from .history import InterleavedHistory
from .macro import CouplingInterface, MacroNetwork, Population, resolve_signed_precision, seed_state
from .memory import build_memory, make_teacher_subspaces


def build_interleaved_synthesis(
    cfg: InterleavedConfig,
) -> tuple[MacroNetwork, SimpleNamespace]:
    """Two frozen teachers `T1, T2` + a plastic synthesis `S`, wired with TWO single-source
    interfaces `S<-T1`, `S<-T2` (both `active=False`; the driver toggles one on per bout). Returns
    `(macro, info)` with the memory bases `U1, U2, U_Sigma`, the target rank `r_Sigma`, per-teacher
    reversed precisions, and `kind="interleaved"`."""
    from .diagnostics import manifold_basis, spectral_gap, subspace_sum_basis

    M1, M2 = make_teacher_subspaces(cfg)
    memory1, memory2 = build_memory(M1), build_memory(M2)
    W1, W2 = memory1.W, memory2.W
    M_T1, M_T2 = memory1.M_op, memory2.M_op
    U1, U2 = manifold_basis(M_T1), manifold_basis(M_T2)
    U_Sigma, overlap, r_Sigma = subspace_sum_basis(U1, U2)

    sigma1_min = spectral_gap(M_T1.transpose(-2, -1) @ M_T1)
    sigma2_min = spectral_gap(M_T2.transpose(-2, -1) @ M_T2)

    def _rho(pi_teacher, sigma_min):
        # single source: ||C||^2 = 1, so the guard is just pi_teacher * sigma_min
        return resolve_signed_precision(cfg.rho, guard=pi_teacher * sigma_min,
                                        safety=cfg.rho_safety,
                                        exact_saddle=cfg.exact_saddle, pi_pos=cfg.pi_I)

    T1 = Population("T1", W1, cfg.pi_T1, cfg.tau_T1, plastic=False, sigma_xi=cfg.sigma_xi1, r=cfg.r1)
    T2 = Population("T2", W2, cfg.pi_T2, cfg.tau_T2, plastic=False, sigma_xi=cfg.sigma_xi2, r=cfg.r2)
    S = Population("S", torch.zeros_like(W1), cfg.pi_S, cfg.tau_S, plastic=True, eta=cfg.eta)

    rho1 = _rho(cfg.pi_T1, sigma1_min)
    rho2 = _rho(cfg.pi_T2, sigma2_min)
    interfaces = [
        CouplingInterface(target="S", sources=["T1"], alpha=[1.0], pi_I=cfg.pi_I,
                          rho=rho1, active=False),
        CouplingInterface(target="S", sources=["T2"], alpha=[1.0], pi_I=cfg.pi_I,
                          rho=rho2, active=False),
    ]
    macro = MacroNetwork([T1, T2, S], interfaces)

    info = SimpleNamespace(
        cfg=cfg,
        U1=U1,
        U2=U2,
        U_Sigma=U_Sigma,
        overlap=overlap,
        r_Sigma=r_Sigma,
        target="S",
        order=("T1", "T2"),
        bases={"T1": U1, "T2": U2},
        rank1=U1.shape[1],
        rank2=U2.shape[1],
    )
    return macro, info


def interleave_merge(
    macro: MacroNetwork,
    order: Sequence[str],
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
    if target not in macro.populations:
        raise ValueError(f"unknown interleaved target population {target!r}.")
    if not macro.populations[target].plastic:
        raise ValueError(f"interleaved target {target!r} must be plastic.")
    order = list(order)
    if not order:
        raise ValueError("interleaved order must contain at least one source.")
    if len(set(order)) != len(order):
        raise ValueError(f"interleaved order contains duplicate sources: {order!r}.")
    if not isinstance(n_bouts, int) or n_bouts < 1:
        raise ValueError(f"n_bouts must be an integer >= 1 (got {n_bouts!r}).")

    itf_of: Dict[str, CouplingInterface] = {}
    for name in order:
        if name not in macro.populations:
            raise ValueError(f"unknown interleaved source population {name!r}.")
        matches = [
            itf for itf in macro.interfaces
            if itf.target == target and itf.sources == [name]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"source {name!r} must have exactly one single-source interface into "
                f"{target!r} (found {len(matches)})."
            )
        basis = bases.get(name)
        source = macro.populations[name]
        if basis is None or basis.ndim != 2 or basis.shape[0] != source.d:
            shape = None if basis is None else tuple(basis.shape)
            raise ValueError(
                f"basis for source {name!r} must have shape ({source.d}, k) (got {shape})."
            )
        itf_of[name] = matches[0]

    def seed(name: str) -> None:
        seed_state(macro.populations[name], bases[name], gen)

    target_pop = macro.populations[target]
    target_pop.x = torch.zeros(
        target_pop.d, dtype=target_pop.dtype, device=target_pop.device
    )
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


def simulate_interleaved(
    macro: MacroNetwork,
    sim: SimConfig,
    info: SimpleNamespace,
) -> InterleavedHistory:
    """Run the interleaved merge of `T1, T2 -> S` and record the crosstalk-cancellation trace."""
    cfg = info.cfg
    gen = torch.Generator(device=macro.device).manual_seed(cfg.seed + 12345)
    hist = InterleavedHistory(labels=info.order)
    U1, U2, U_Sigma = info.U1, info.U2, info.U_Sigma
    if sim.n_steps % sim.bout_steps:
        raise ValueError(
            "interleaved n_steps must be an exact multiple of bout_steps "
            f"(got n_steps={sim.n_steps}, bout_steps={sim.bout_steps})."
        )
    n_bouts = sim.n_steps // sim.bout_steps

    def record(bout: int, active: int) -> None:
        hist.record(bout, active, macro.populations["S"].M, U1, U2, U_Sigma)

    interleave_merge(macro, info.order, info.bases, info.target, sim, gen, n_bouts, record)
    return hist
