"""pmt.macro — a small "network of networks" engine (the LEGO layer).

Compose predictive-coding **populations** (nodes) and **additive interfaces** (hyper-edges)
into a `MacroNetwork`. Both shipped models are instances of this one engine:

  - the two-population transfer model  = 2 populations (T, S) + 1 interface  (see `pmt.model`);
  - the three-network additive model   = 3 populations (T1, T2, S) + 1 interface (`pmt.additive`).

Nothing here is new mathematics — it is exactly the equations of
`two_population_memory_transfer_model.md` / `three_network_additive_memory_synthesis.md`,
factored so the *wiring* is data rather than code. That makes future architectures (chains,
trees, several students, general coordinate maps `C_k`) drop in without touching the integrator.

Conventions (inherited from `pmt.model`):
  - Tied weights: each population owns ONE recurrent matrix `W`; the top-down path uses
    `M = I - W`, the bottom-up path uses `M^T` (the same matrix transposed).
  - Row-vector application so an optional leading batch dim broadcasts cleanly:
        fwd(W, x) = x @ W.T   == W x     (top-down / prediction)
        bwd(W, x) = x @ W     == W^T x   (bottom-up / error feedback)
  - Zero diagonal on every `W` at all times (no autapses); the `I` in `M = I - W` is the
    structural self/leak term, not a synapse.

Reversed precision (the sleep/wake knob) lives on the interface as the *signed* teacher-side
precision `rho`: `rho < 0` is sleep/replay (teachers driven to disagree — the transfer drive),
`rho > 0` is wake/recall (teachers chase the prediction). The target side always uses an
ordinary positive precision `pi_I` (the student always descends its free energy).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .memory import zero_diag


# ----------------------------------------------------------------- tied tensor operators
def fwd(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Top-down / forward: returns W x."""
    return x @ W.transpose(-2, -1)


def bwd(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Bottom-up / transpose: returns W^T x (tied to the same W)."""
    return x @ W


def outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Outer product a b^T, shape (..., d, d)."""
    return a.unsqueeze(-1) * b.unsqueeze(-2)


# ------------------------------------------------------------------------------ node
class Population:
    """One predictive-coding network node.

    Owns a recurrent weight matrix `W` (zero diagonal), a self-precision `pi`, a time constant
    `tau`, and a live state `x`. Frozen teachers set `plastic=False`; a plastic synthesis/student
    sets `plastic=True` with a learning rate `eta`. `sigma_xi > 0` adds exploration noise and a
    finite `r` renormalizes `||x|| = r` each step (the amplitude leash); leave `r=None` for a
    population whose amplitude is set by its inputs (a synthesis network).
    """

    def __init__(
        self,
        name: str,
        W: torch.Tensor,
        pi: float,
        tau: float,
        *,
        plastic: bool = False,
        eta: float = 0.0,
        sigma_xi: float = 0.0,
        r: Optional[float] = None,
    ) -> None:
        self.name = name
        self.W = W                      # recurrent weights, zero diagonal
        self.pi = float(pi)
        self.tau = float(tau)
        self.plastic = bool(plastic)
        self.eta = float(eta)
        self.sigma_xi = float(sigma_xi)
        self.r = None if r is None else float(r)

        self.d = W.shape[0]
        self.dtype = W.dtype
        self.device = W.device
        self.I = torch.eye(self.d, dtype=self.dtype, device=self.device)
        self.x: Optional[torch.Tensor] = None

    # ----- tied operators -----
    @property
    def M(self) -> torch.Tensor:
        """Mismatch operator M = I - W."""
        return self.I - self.W

    def S_op(self) -> torch.Tensor:
        """Self-surprise operator S = M^T M."""
        M = self.M
        return M.transpose(-2, -1) @ M

    # ----- self error / energy / rates (external clamp v = 0) -----
    def eps_self(self) -> torch.Tensor:
        """Self prediction error M x (zero on this population's own memory manifold)."""
        return fwd(self.M, self.x)

    def F_self(self) -> torch.Tensor:
        """Self free energy 0.5 pi ||M x||^2."""
        return 0.5 * self.pi * (self.eps_self() ** 2).sum(-1)

    def rate_self(self) -> torch.Tensor:
        """The self term of the state rate: -pi M^T eps_self = -pi S x (before dividing by tau)."""
        return -self.pi * bwd(self.M, self.eps_self())

    def rate_W(self) -> torch.Tensor:
        """dW/dt = eta pi eps_self x^T (Hebbian on the self-error and activity). The diagonal is
        re-zeroed by the network after the update, so it is not removed here."""
        return self.eta * self.pi * outer(self.eps_self(), self.x)

    # ----- state maintenance -----
    def renorm(self) -> None:
        if self.r is None:
            return
        n = self.x.norm(dim=-1, keepdim=True)
        self.x = self.x * (self.r / n.clamp_min(1e-12))

    def zero_diag_W(self) -> None:
        self.W = zero_diag(self.W)


# ------------------------------------------------------------------------- hyper-edge
@dataclass
class AdditiveInterface:
    """One additive prediction/error hyper-edge.

    A set of `sources` is combined into a single prediction `y = sum_k alpha_k C_k x_k` and
    compared with the `target` state, giving the common error `eps = x_target - y`. Optional
    coordinate maps `C_k` (C_k : R^{d_k} -> R^{d_target}) let sources live in different neural
    coordinates; `None` means identity.

    Rate contributions:
      - to the target:   -pi_I * eps                     (ordinary positive precision)
      - to source k:     +rho * alpha_k * C_k^T eps      (signed teacher-side precision; <0 sleep)
    """

    target: str
    sources: List[str]
    alpha: List[float]
    pi_I: float
    rho: float
    C: Optional[List[Optional[torch.Tensor]]] = None   # coordinate maps; entry None => identity
    active: bool = True                                # inactive edges contribute nothing (interleaving)

    def _map_in(self, k: int, x: torch.Tensor) -> torch.Tensor:
        """Apply C_k to a source state: C_k x = x @ C_k^T (row-vector convention)."""
        Ck = None if self.C is None else self.C[k]
        return x if Ck is None else x @ Ck.transpose(-2, -1)

    def _map_back(self, k: int, e: torch.Tensor) -> torch.Tensor:
        """Apply C_k^T to a target-space error: C_k^T e = e @ C_k (row-vector convention)."""
        Ck = None if self.C is None else self.C[k]
        return e if Ck is None else e @ Ck

    def y(self, pops: Dict[str, Population]) -> torch.Tensor:
        """Combined additive prediction y = sum_k alpha_k C_k x_k."""
        acc = None
        for k, s in enumerate(self.sources):
            term = self.alpha[k] * self._map_in(k, pops[s].x)
            acc = term if acc is None else acc + term
        return acc

    def eps(self, pops: Dict[str, Population]) -> torch.Tensor:
        """Common interface error eps = x_target - y."""
        return pops[self.target].x - self.y(pops)

    def source_feedback(self, k: int, e: torch.Tensor) -> torch.Tensor:
        """Source-side drive +rho alpha_k C_k^T eps for source index k given the error e."""
        return self.rho * self.alpha[k] * self._map_back(k, e)


# --------------------------------------------------------------------------- the graph
class MacroNetwork:
    """A graph of `Population` nodes wired by `AdditiveInterface` hyper-edges.

    Assembles each population's state rate from its own self term plus every interface it takes
    part in, and steps the whole system with one integrator (`step`). A population that is the
    target of some interface is a *perception* node (it can be eliminated adiabatically); a
    population that only feeds interfaces is an *environment* node (it gets exploration noise and
    the amplitude leash). This mirrors `pmt.dynamics` exactly for the two-population case.
    """

    def __init__(self, populations: List[Population], interfaces: List[AdditiveInterface]) -> None:
        self.populations: Dict[str, Population] = {p.name: p for p in populations}
        self.interfaces: List[AdditiveInterface] = list(interfaces)
        ref = populations[0]
        self.d = ref.d
        self.dtype = ref.dtype
        self.device = ref.device

    # ----- roles (only ACTIVE interfaces count; inactive edges are transparent) -----
    def targets(self) -> set:
        return {itf.target for itf in self.interfaces if itf.active}

    def sources(self) -> set:
        ss: set = set()
        for itf in self.interfaces:
            if itf.active:
                ss.update(itf.sources)
        return ss

    # ----- assembled rates -----
    def rate_x(self, name: str) -> torch.Tensor:
        """Full dx/dt for population `name`: self term + all active interface contributions, over tau."""
        p = self.populations[name]
        drift = p.rate_self()
        for itf in self.interfaces:
            if not itf.active:
                continue
            is_target = itf.target == name
            src_idx = [k for k, s in enumerate(itf.sources) if s == name]
            if is_target or src_idx:
                e = itf.eps(self.populations)
                if is_target:
                    drift = drift - itf.pi_I * e
                for k in src_idx:
                    drift = drift + itf.source_feedback(k, e)
        return drift / p.tau

    def rate_W(self, name: str) -> torch.Tensor:
        return self.populations[name].rate_W()

    def solve_steady(self, name: str) -> torch.Tensor:
        """Adiabatic elimination of a fast interface target (infinite timescale separation):

            (pi_S S_S + (sum_i pi_Ii) I) x* = sum_i pi_Ii y_i

        For a single additive interface this is x* = pi_I (pi_I I + pi_S S_S)^-1 y (spec 8);
        summing over interfaces also covers the separate-error control (two single-source edges).
        """
        p = self.populations[name]
        A = p.pi * p.S_op()
        rhs = torch.zeros_like(p.x)
        for itf in self.interfaces:
            if itf.active and itf.target == name:
                A = A + itf.pi_I * p.I
                rhs = rhs + itf.pi_I * itf.y(self.populations)
        return torch.linalg.solve(A, rhs)

    # ----- integrator -----
    def step(
        self,
        dt: float,
        gen: Optional[torch.Generator] = None,
        mode: str = "full",
        substeps: int = 1,
        noise: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """One integration step, in the faithful order perception -> environment -> learning.

        mode="adiabatic": targets jump to their steady state each step (matches the novelty-operator
        theory); mode="full": targets take `substeps` explicit-Euler steps (watch them relax).
        Environment (source-only) populations take an Euler step with Euler-Maruyama noise, then
        renormalize. Plastic populations then learn, and re-zero their diagonal.

        `noise` optionally supplies a pre-drawn *unit* (standard-normal) vector per population name,
        used instead of an internal draw and then scaled by `sigma_xi/tau*sqrt(dt)`. This lets a
        caller impose correlated exploration noise across teachers (spec ablation C); when omitted
        each noisy population draws its own independent vector from `gen`.
        """
        targets = self.targets()
        sources = self.sources()

        # 1) perception on interface targets
        for name, p in self.populations.items():
            if name in targets:
                if mode == "adiabatic":
                    p.x = self.solve_steady(name)
                elif mode == "full":
                    for _ in range(substeps):
                        p.x = p.x + dt * self.rate_x(name)
                else:
                    raise ValueError(f"unknown mode={mode!r}")

        # 2) environment (source-only) populations: Euler drift + EM noise, then renormalize
        for name, p in self.populations.items():
            if name in sources and name not in targets:
                p.x = p.x + dt * self.rate_x(name)
                if p.sigma_xi > 0:
                    raw = None
                    if noise is not None and name in noise:
                        raw = noise[name]
                    elif gen is not None:
                        raw = torch.randn(p.d, generator=gen, dtype=p.dtype, device=p.device)
                    if raw is not None:
                        p.x = p.x + (p.sigma_xi / p.tau * math.sqrt(dt)) * raw
                p.renorm()

        # 3) learning on plastic populations, then re-zero the diagonal (no autapses)
        for name, p in self.populations.items():
            if p.plastic:
                p.W = p.W + dt * self.rate_W(name)
                p.zero_diag_W()
