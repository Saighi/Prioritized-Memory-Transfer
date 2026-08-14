"""The composable "network of networks" engine: `Population` nodes + `CouplingInterface`
edges assembled into a `MacroNetwork`. Every shipped model is an instance of it — the
two-population transfer (`prioritized_memory_transfer.model`) and the interleaved merge (`prioritized_memory_transfer.interleaved`, reused
by `prioritized_memory_transfer.continual`). The equations are those of `two_population_memory_transfer_model.md`,
factored so the *wiring* is data rather than code.

Conventions:
  - Tied weights: one recurrent matrix `W` per population; top-down uses `M = I - W`,
    bottom-up uses `M^T`. Row-vector application: fwd(W,x) = x@W.T = Wx, bwd(W,x) = x@W = W^Tx.
  - Zero diagonal on every `W` at all times (no autapses); the `I` in `M = I - W` is the
    structural self/leak term, not a synapse.
  - Reversed precision (the sleep/wake knob) is the interface's *signed* source-side
    precision `rho`: < 0 sleep/replay (drive-to-disagree, the transfer regime), > 0
    wake/recall. The target side always uses an ordinary positive precision `pi_I`.
  - Optional unit nonlinearity (`Population(act=...)`): the self error becomes
    `eps = x - W f(x)`, so `M = I - W` generalizes to the state-dependent Jacobian
    `J = I - W diag(f'(x))` (`Population.jac`). `act=None` is the linear model and takes
    exactly the same code path as before.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .activations import ActivationSpec, is_linear, resolve_activation
from .memory import zero_diag


# --------------------------------------------------------------------- shared knobs
def resolve_signed_precision(
    raw,
    *,
    guard: float,
    safety: float,
    exact_saddle: bool = False,
    pi_pos: Optional[float] = None,
) -> float:
    """Resolve the signed source-side precision (`rho`; called `pi_ST` in the two-population
    model and the paper) from a config value. Precedence:

      - `exact_saddle`  -> the zero-sum value `-pi_pos` (the target-side precision negated;
                           the exact-saddle regime `pi_ST = -pi_TS`);
      - `"auto"`        -> the reversed default `-safety * guard`, safely inside the caller's
                           stability guard;
      - explicit float  -> as given (signed: <0 sleep/replay, >0 wake).
    """
    if exact_saddle:
        return -float(pi_pos)
    if isinstance(raw, str) and raw == "auto":
        return -float(safety) * float(guard)
    return float(raw)


def seed_state(
    p: "Population",
    U: Optional[torch.Tensor],
    gen: Optional[torch.Generator],
    r: Optional[float] = None,
) -> None:
    """Seed population `p` with a random state: draw standard normal, optionally project onto
    `span(U)` (orthonormal columns; pass `None` to skip), renormalize to `r` (default: the
    population's leash `p.r`, or 1.0 if unleashed)."""
    x = torch.randn(p.d, generator=gen, dtype=p.dtype, device=p.device)
    if U is not None and U.shape[1] > 0:
        x = U @ (U.transpose(-2, -1) @ x)
    rr = r if r is not None else (p.r if p.r is not None else 1.0)
    p.x = x * (rr / x.norm().clamp_min(1e-12))


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

    `act` optionally rectifies (or otherwise squashes) what the units *transmit*: the self error
    becomes `eps = x - W f(x)` instead of `M x`, and everything downstream is derived from the
    Jacobian `J = I - W diag(f'(x))` rather than from `M`. `act=None` (the default) is the linear
    model and follows the identical code path it always did. Note the leash is doing real work
    here: it forbids the all-silent state, which under a rectifying `f` with no bias is itself a
    zero-error memory and would otherwise swallow the dynamics.
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
        act: ActivationSpec = None,
    ) -> None:
        self.name = name
        self.W = W                      # recurrent weights, zero diagonal
        self.pi = float(pi)
        self.tau = float(tau)
        self.plastic = bool(plastic)
        self.eta = float(eta)
        self.sigma_xi = float(sigma_xi)
        self.r = None if r is None else float(r)

        self.act = act
        self.linear = is_linear(act)
        self._f, self._f_prime = resolve_activation(act)

        self.d = W.shape[0]
        self.dtype = W.dtype
        self.device = W.device
        self.I = torch.eye(self.d, dtype=self.dtype, device=self.device)
        self.x: Optional[torch.Tensor] = None

    # ----- unit nonlinearity -----
    def f(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Transmitted output f(x) of the units (identity in the linear model). Defaults to the
        live state; pass `x` to evaluate anywhere (e.g. at a stored pattern)."""
        return self._f(self.x if x is None else x)

    def f_prime(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Elementwise f'(x) — the diagonal of the transmission Jacobian."""
        return self._f_prime(self.x if x is None else x)

    # ----- tied operators -----
    @property
    def M(self) -> torch.Tensor:
        """Mismatch operator M = I - W. Exact for a linear population; for a nonlinear one it is
        the operator of the *all-active* cone (use `jac` for the local one)."""
        return self.I - self.W

    def jac(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Jacobian of the self error, J = d eps/d x = I - W diag(f'(x)).

        Equals `M` in the linear model; under a rectifying `f` it is the `M` operator of the
        activation cone the state currently sits in. Falls back to `M` when there is no state
        yet (build time), i.e. to the all-active cone.
        """
        if self.linear:
            return self.M
        x = self.x if x is None else x
        if x is None:
            return self.M
        # W diag(g): scale column j by g_j
        return self.I - self.W * self.f_prime(x).unsqueeze(-2)

    def S_op(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Self-surprise operator S = J^T J (= M^T M in the linear model)."""
        J = self.jac(x)
        return J.transpose(-2, -1) @ J

    # ----- self error / energy / rates (external clamp v = 0) -----
    def eps_self(self) -> torch.Tensor:
        """Self prediction error x - W f(x), zero on this population's own memory manifold
        (= M x in the linear model)."""
        if self.linear:
            return fwd(self.M, self.x)
        return self.x - fwd(self.W, self.f())

    def F_self(self) -> torch.Tensor:
        """Self free energy 0.5 pi ||eps_self||^2."""
        return 0.5 * self.pi * (self.eps_self() ** 2).sum(-1)

    def rate_self(self) -> torch.Tensor:
        """The self term of the state rate: -pi J^T eps_self (= -pi S x when linear), before
        dividing by tau."""
        e = self.eps_self()
        if self.linear:
            return -self.pi * bwd(self.M, e)
        # J^T e = e - diag(f'(x)) W^T e
        return -self.pi * (e - self.f_prime() * bwd(self.W, e))

    def rate_W(self) -> torch.Tensor:
        """dW/dt = eta pi eps_self f(x)^T (Hebbian on the self-error and the *transmitted*
        activity; f(x) = x in the linear model). The diagonal is re-zeroed by the network after
        the update, so it is not removed here."""
        return self.eta * self.pi * outer(self.eps_self(), self.f())

    # ----- probes -----
    def relax(
        self,
        x0: torch.Tensor,
        *,
        known: Optional[torch.Tensor] = None,
        cue: Optional[torch.Tensor] = None,
        n_steps: int = 1500,
        dt: float = 0.2,
    ) -> torch.Tensor:
        """Descend this population's OWN free energy from `x0`, with no interface input.

        This is the recall query: with `known` (a boolean mask) and `cue`, the flagged units are
        clamped to the cue at every step and the rest descend `F` onto the memory manifold —
        projected gradient flow. Without a mask it is free relaxation, which answers "is `x0` in
        the basin of a memory?".

        Exact for a nonlinear population (it descends the true `F` via `rate_self`, no
        linearization). `x0` may be batched as `(B, d)`; `cue` broadcasts over the batch. This is
        a probe: the population's live state is restored before returning.
        """
        if x0.shape[-1] != self.d:
            raise ValueError(f"x0 must have trailing dimension {self.d} (got {tuple(x0.shape)}).")
        if (known is None) != (cue is None):
            raise ValueError("pass `known` and `cue` together, or neither.")
        if known is not None and known.dtype != torch.bool:
            raise TypeError(f"known must be a boolean mask (got {known.dtype}).")

        saved = self.x
        try:
            x = x0.clone()
            if known is not None:
                x[..., known] = cue[known]
            for _ in range(int(n_steps)):
                self.x = x
                x = x + (dt / self.tau) * self.rate_self()
                if known is not None:
                    x[..., known] = cue[known]
            return x
        finally:
            self.x = saved

    # ----- state maintenance -----
    def renorm(self) -> None:
        if self.r is None:
            return
        n = self.x.norm(dim=-1, keepdim=True)
        self.x = self.x * (self.r / n.clamp_min(1e-12))

    def zero_diag_W(self) -> None:
        self.W = zero_diag(self.W)


# ------------------------------------------------------------------------------- edge
@dataclass
class CouplingInterface:
    """One coupling error ``eps = x_target - x_source`` between two populations.

    Rate contributions:
      - to the target: -pi_I * eps   (ordinary positive precision)
      - to the source: +rho * eps    (signed source-side precision; <0 sleep)
    """

    target: str
    source: str
    pi_I: float
    rho: float
    active: bool = True

    def y(self, pops: Dict[str, Population]) -> torch.Tensor:
        return pops[self.source].x

    def eps(self, pops: Dict[str, Population]) -> torch.Tensor:
        return pops[self.target].x - self.y(pops)

    def source_feedback(self, e: torch.Tensor) -> torch.Tensor:
        return self.rho * e


# --------------------------------------------------------------------------- the graph
class MacroNetwork:
    """A graph of `Population` nodes wired by binary `CouplingInterface` edges.

    Assembles each population's state rate from its own self term plus every interface it takes
    part in, and steps the whole system with one integrator (`step`). A population that is the
    target of some interface is a *perception* node (it can be eliminated adiabatically); a
    population that only feeds interfaces is an *environment* node (it gets exploration noise and
    the amplitude leash). This mirrors `prioritized_memory_transfer.model.simulate` exactly for the two-population case.
    """

    def __init__(self, populations: List[Population], interfaces: List[CouplingInterface]) -> None:
        if not populations:
            raise ValueError("MacroNetwork needs at least one population.")
        names = [p.name for p in populations]
        if len(set(names)) != len(names):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"population names must be unique (duplicates: {duplicates!r}).")
        self.populations: Dict[str, Population] = {p.name: p for p in populations}
        self.interfaces: List[CouplingInterface] = list(interfaces)
        ref = populations[0]
        for itf in self.interfaces:
            if itf.target not in self.populations:
                raise ValueError(f"interface target {itf.target!r} is not a population.")
            target = self.populations[itf.target]
            if itf.source not in self.populations:
                raise ValueError(f"interface source {itf.source!r} is not a population.")
            source = self.populations[itf.source]
            if source.d != target.d:
                raise ValueError(
                    f"interface {itf.source!r}->{itf.target!r} needs equal dimensions "
                    f"(got {source.d} and {target.d})."
                )
        self.d = ref.d
        self.dtype = ref.dtype
        self.device = ref.device

    # ----- roles (only ACTIVE interfaces count; inactive edges are transparent) -----
    def targets(self) -> set:
        return {itf.target for itf in self.interfaces if itf.active}

    def sources(self) -> set:
        return {itf.source for itf in self.interfaces if itf.active}

    # ----- assembled rates -----
    def rate_x(self, name: str) -> torch.Tensor:
        """Full dx/dt for population `name`: self term + all active interface contributions, over tau."""
        p = self.populations[name]
        drift = p.rate_self()
        for itf in self.interfaces:
            if not itf.active:
                continue
            is_target = itf.target == name
            is_source = itf.source == name
            if is_target or is_source:
                e = itf.eps(self.populations)
                if is_target:
                    drift = drift - itf.pi_I * e
                if is_source:
                    drift = drift + itf.source_feedback(e)
        return drift / p.tau

    def rate_W(self, name: str) -> torch.Tensor:
        return self.populations[name].rate_W()

    def solve_steady(self, name: str) -> torch.Tensor:
        """Adiabatic elimination of a fast interface target (infinite timescale separation):

            (pi_S S_S + (sum_i pi_Ii) I) x* = sum_i pi_Ii y_i

        For a single interface this is x* = pi_I (pi_I I + pi_S S_S)^-1 y;
        summing over interfaces also covers the separate-error control (two single-source edges).

        Linear targets only: with a unit nonlinearity the steady state is not a linear solve, and
        pretending otherwise would silently integrate the wrong dynamics — use `mode="full"`.
        """
        active_inputs = [
            itf for itf in self.interfaces if itf.active and itf.target == name
        ]
        if not active_inputs:
            raise ValueError(f"population {name!r} has no active incoming interface to solve.")
        p = self.populations[name]
        if not p.linear:
            raise ValueError(
                f"population {name!r} has a nonlinear activation ({p.act!r}), so its steady "
                "state is not a linear solve; run with mode='full' (optionally raising "
                "SimConfig.s_substeps) instead of mode='adiabatic'."
            )
        A = p.pi * p.S_op()
        rhs = torch.zeros_like(p.x)
        for itf in active_inputs:
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

        `noise` optionally supplies a pre-drawn *unit* (standard-normal) vector per population
        name, scaled by `sigma_xi/tau*sqrt(dt)`; when omitted each noisy population draws its
        own independent vector from `gen`.
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
