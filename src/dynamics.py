"""Integrators and the simulation loop.

The heavy lifting is the engine's `MacroNetwork.step` (see `src.macro`); `simulate` just resets
state, drives the step, and records observables. Two integration modes (both handled by the
engine):
  - "full":      explicit Euler on the fast perception nodes (you can watch them relax); noise on
                 the environment nodes uses Euler-Maruyama sqrt(dt) scaling.
  - "adiabatic": replace each perception node by its exact steady state each step (infinite
                 timescale separation). Faster and matches the novelty-operator theory cleanly.

The sleep/wake regime is read off the sign of the interface's signed precision (`pi_ST` in the
two-population model, `rho` in the interleaved/continual models): < 0 = sleep/replay = transfer.

Invariants enforced every step by the engine: `||x|| = r` on leashed nodes and `diag(W) = 0` on
plastic nodes.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from .config import SimConfig
from .history import History
from .model import TwoPopModel


def simulate(model: TwoPopModel, sim: SimConfig, info: Optional[Dict] = None) -> History:
    """Run a two-population transfer simulation and record observables into a `History`."""
    cfg = model.cfg
    gen = torch.Generator(device=model.device).manual_seed(cfg.seed + 12345)

    model.reset_state(gen)
    if sim.pretrain_subset is not None:
        model.pretrain(sim.pretrain_subset)

    U_T = info["U_T"] if info is not None else None
    if U_T is None:
        from .diagnostics import manifold_basis
        U_T = manifold_basis(model.M_T)
    M = model.patterns
    macro = model.macro

    hist = History()
    snap_steps = set(
        int(round(s)) for s in torch.linspace(0, sim.n_steps - 1, sim.n_weight_snapshots).tolist()
    )

    dt = sim.dt
    iterator = range(sim.n_steps)
    if sim.progress:
        regime = "replay" if model.pi_ST < 0 else "recall"
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"simulate[{sim.mode}/{regime}]")
        except Exception:
            pass

    for step in iterator:
        macro.step(dt, gen, mode=sim.mode, substeps=sim.s_substeps)
        if step % sim.record_every == 0:
            hist.record(step * dt, model, U_T, M)
        if step in snap_steps:
            hist.snapshot_weights(step * dt, step, model)

    return hist
