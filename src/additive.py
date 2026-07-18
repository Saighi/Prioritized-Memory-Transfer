"""The additive three-network memory-synthesis model, as an instance of the `src.macro` engine.

Two frozen teachers `T1`, `T2` are summed into one combined prediction `y = alpha1 x1 + alpha2 x2`;
a single common error `eps_Sigma = x_S - y` drives a plastic synthesis network `S`. In sleep the
teachers' signed interface precision `rho < 0` amplifies whatever part of the combined teacher
state `S` cannot yet predict, so `S` learns the subspace sum `U_Sigma = U1 + U2` with
novelty-homeostatic source rebalancing and clean self-termination. Full specification and analysis:
`three_network_additive_memory_synthesis.md`.

`build_additive_synthesis(cfg)` wires three `Population`s and one `AdditiveInterface` (or two
single-source interfaces for the separate-error control, spec section 3 / ablation A) into a
`MacroNetwork`; `simulate_additive` runs and records it (called via `src.simulate` when
`info["kind"] == "additive"`).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from .config import AdditiveSynthesisConfig, SimConfig
from .history import AdditiveHistory
from .macro import AdditiveInterface, MacroNetwork, Population, resolve_signed_precision, seed_state
from .memory import build_W_T, make_teacher_subspaces


def build_additive_synthesis(cfg: AdditiveSynthesisConfig):
    """Build the two frozen teachers and the plastic synthesis network, resolve the signed
    teacher-side precision `rho` against the structure guard (spec section 17), and return
    `(macro, info)`. `info` carries the memory bases `U1, U2, U_Sigma`, the target rank `r_Sigma`,
    the intersection dimension, per-teacher spectral gaps and guards, and `kind="additive"`."""
    from .diagnostics import manifold_basis, spectral_gap, subspace_sum_basis

    I = torch.eye(cfg.d, dtype=cfg.dtype, device=cfg.device)

    # frozen teacher memories with prescribed geometry
    M1, M2 = make_teacher_subspaces(cfg)
    W1 = build_W_T(M1, cfg)
    W2 = build_W_T(M2, cfg)
    M_T1, M_T2 = I - W1, I - W2
    U1 = manifold_basis(M_T1)                       # (d, rank1) orthonormal basis of ker M_T1
    U2 = manifold_basis(M_T2)
    U_Sigma, overlap, r_Sigma = subspace_sum_basis(U1, U2)

    sigma1_min = spectral_gap(M_T1.transpose(-2, -1) @ M_T1)
    sigma2_min = spectral_gap(M_T2.transpose(-2, -1) @ M_T2)

    # structure guard (spec section 17): beta ||C||^2 < min(pi_k sigma_k,min^2),
    # with C = [alpha1 I, alpha2 I] so ||C||^2 = alpha1^2 + alpha2^2 and beta = |rho|.
    Cnorm2 = cfg.alpha1 ** 2 + cfg.alpha2 ** 2
    guard = min(cfg.pi_T1 * sigma1_min, cfg.pi_T2 * sigma2_min) / Cnorm2
    rho = resolve_signed_precision(cfg.rho, guard=guard, safety=cfg.rho_safety,
                                   exact_saddle=cfg.exact_saddle, pi_pos=cfg.pi_I)

    r_leash1 = cfg.r1 if cfg.norm_constraint else None
    r_leash2 = cfg.r2 if cfg.norm_constraint else None
    T1 = Population("T1", W1, cfg.pi_T1, cfg.tau_T1, plastic=False, sigma_xi=cfg.sigma_xi1, r=r_leash1)
    T2 = Population("T2", W2, cfg.pi_T2, cfg.tau_T2, plastic=False, sigma_xi=cfg.sigma_xi2, r=r_leash2)
    S = Population("S", torch.zeros_like(W1), cfg.pi_S, cfg.tau_S, plastic=True, eta=cfg.eta)

    if cfg.separate_errors:
        interfaces = [
            AdditiveInterface(target="S", sources=["T1"], alpha=[cfg.alpha1], pi_I=cfg.pi_I, rho=rho),
            AdditiveInterface(target="S", sources=["T2"], alpha=[cfg.alpha2], pi_I=cfg.pi_I, rho=rho),
        ]
    else:
        interfaces = [
            AdditiveInterface(target="S", sources=["T1", "T2"],
                              alpha=[cfg.alpha1, cfg.alpha2], pi_I=cfg.pi_I, rho=rho),
        ]

    macro = MacroNetwork([T1, T2, S], interfaces)

    info = {
        "kind": "additive",
        "cfg": cfg,
        "M1": M1, "M2": M2,
        "W1": W1, "W2": W2,
        "U1": U1, "U2": U2, "U_Sigma": U_Sigma,
        "rank1": U1.shape[1], "rank2": U2.shape[1],
        "overlap": overlap, "r_Sigma": r_Sigma,
        "capacity_ok": r_Sigma <= cfg.d - 1,
        "sigma1_min": sigma1_min, "sigma2_min": sigma2_min,
        "guard": guard, "rho": rho, "rho_ok": abs(rho) < guard,
        "pi_I": cfg.pi_I, "pi_S": cfg.pi_S,
        "alpha1": cfg.alpha1, "alpha2": cfg.alpha2,
        "seed": cfg.seed,
        "precision_ok": cfg.pi_I > cfg.pi_S,
    }
    return macro, info


def _reset_additive(macro: MacroNetwork, cfg: AdditiveSynthesisConfig, info: Dict,
                    gen: torch.Generator) -> None:
    """Synthesis state -> 0; each teacher -> a random state, optionally projected into its own
    memory subspace (spec section 20 initial conditions), then renormalized to its leash."""
    macro.populations["S"].x = torch.zeros(cfg.d, dtype=cfg.dtype, device=cfg.device)
    for name, U, r in (("T1", info["U1"], cfg.r1), ("T2", info["U2"], cfg.r2)):
        seed_state(macro.populations[name], U if cfg.init_on_manifold else None, gen, r=r)


def simulate_additive(macro: MacroNetwork, sim: SimConfig, info: Dict) -> AdditiveHistory:
    """Run the additive-synthesis transfer and record the spec's success observables."""
    cfg: AdditiveSynthesisConfig = info["cfg"]
    gen = torch.Generator(device=macro.device).manual_seed(cfg.seed + 12345)
    _reset_additive(macro, cfg, info, gen)

    hist = AdditiveHistory()
    snap_steps = set(
        int(round(s)) for s in torch.linspace(0, sim.n_steps - 1, sim.n_weight_snapshots).tolist()
    )
    dt = sim.dt

    iterator = range(sim.n_steps)
    if sim.progress:
        regime = "replay" if info["rho"] < 0 else "recall"
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"additive[{sim.mode}/{regime}]")
        except Exception:
            pass

    for step in iterator:
        noise = None
        if cfg.correlated_noise:                     # teachers share one draw (ablation C)
            shared = torch.randn(cfg.d, generator=gen, dtype=cfg.dtype, device=cfg.device)
            noise = {"T1": shared, "T2": shared}
        macro.step(dt, gen, mode=sim.mode, substeps=sim.s_substeps, noise=noise)

        if step % sim.record_every == 0:
            hist.record(step * dt, macro, info)
        if step in snap_steps:
            hist.snapshot_weights(step * dt, step, macro)

    return hist
