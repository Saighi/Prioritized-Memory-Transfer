"""The composable "network of networks" engine: `Population` nodes + `CouplingInterface`
edges assembled into a `MacroNetwork`. Every shipped model is an instance of it — the
two-population transfer (`src.model`) and the interleaved merge (`src.interleaved`, reused
by `src.continual`). The equations are those of `two_population_memory_transfer_model.md`,
factored so the *wiring* is data rather than code.

Conventions:
  - Tied weights: one recurrent matrix `W` per population; top-down uses `M = I - W`,
    bottom-up uses `M^T`. Row-vector application: fwd(W,x) = x@W.T = Wx, bwd(W,x) = x@W = W^Tx.
  - Zero diagonal on every `W` at all times (no autapses); the `I` in `M = I - W` is the
    structural self/leak term, not a synapse.
  - Reversed precision (the sleep/wake knob) is the interface's *signed* source-side
    precision `rho`: < 0 sleep/replay (drive-to-disagree, the transfer regime), > 0
    wake/recall. The target side always uses an ordinary positive precision `pi_I`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

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
        if not isinstance(name, str) or not name.strip():
            raise ValueError("population name must be a non-empty string.")
        if not isinstance(W, torch.Tensor) or W.ndim != 2 or W.shape[0] != W.shape[1]:
            shape = None if not isinstance(W, torch.Tensor) else tuple(W.shape)
            raise ValueError(f"population W must be a square matrix (got {shape}).")
        if W.shape[0] < 1 or not W.is_floating_point():
            raise ValueError(f"population W must be a non-empty floating matrix (got {W.dtype}).")
        if not bool(torch.isfinite(W).all()):
            raise ValueError(f"population {name!r} has NaN or infinite weights.")
        diag_tol = 10 * torch.finfo(W.dtype).eps
        if float(torch.diagonal(W).abs().max()) > diag_tol:
            raise ValueError(f"population {name!r} weights must have a zero diagonal.")
        if not math.isfinite(float(pi)) or pi <= 0:
            raise ValueError(f"population {name!r} pi must be finite and > 0 (got {pi!r}).")
        if not math.isfinite(float(tau)) or tau <= 0:
            raise ValueError(f"population {name!r} tau must be finite and > 0 (got {tau!r}).")
        if not math.isfinite(float(eta)) or eta < 0:
            raise ValueError(f"population {name!r} eta must be finite and >= 0 (got {eta!r}).")
        if not math.isfinite(float(sigma_xi)) or sigma_xi < 0:
            raise ValueError(
                f"population {name!r} sigma_xi must be finite and >= 0 (got {sigma_xi!r})."
            )
        if r is not None and (not math.isfinite(float(r)) or r <= 0):
            raise ValueError(f"population {name!r} r must be finite and > 0 (got {r!r}).")
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
class CouplingInterface:
    """One coupling prediction/error edge (supports one or several summed sources).

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

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("interface target must be a non-empty population name.")
        if not self.sources or any(not isinstance(s, str) or not s.strip() for s in self.sources):
            raise ValueError("interface sources must contain non-empty population names.")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError(f"interface contains duplicate sources: {self.sources!r}.")
        if len(self.alpha) != len(self.sources):
            raise ValueError(
                f"interface alpha length must match sources "
                f"({len(self.alpha)} != {len(self.sources)})."
            )
        if any(not math.isfinite(float(a)) for a in self.alpha):
            raise ValueError("interface alpha values must be finite.")
        if self.C is not None and len(self.C) != len(self.sources):
            raise ValueError(
                f"interface C length must match sources ({len(self.C)} != {len(self.sources)})."
            )
        if not math.isfinite(float(self.pi_I)) or self.pi_I <= 0:
            raise ValueError(f"interface pi_I must be finite and > 0 (got {self.pi_I!r}).")
        if not math.isfinite(float(self.rho)):
            raise ValueError(f"interface rho must be finite (got {self.rho!r}).")

    def _map_in(self, k: int, x: torch.Tensor) -> torch.Tensor:
        """Apply C_k to a source state: C_k x = x @ C_k^T (row-vector convention)."""
        Ck = None if self.C is None else self.C[k]
        return x if Ck is None else x @ Ck.transpose(-2, -1)

    def _map_back(self, k: int, e: torch.Tensor) -> torch.Tensor:
        """Apply C_k^T to a target-space error: C_k^T e = e @ C_k (row-vector convention)."""
        Ck = None if self.C is None else self.C[k]
        return e if Ck is None else e @ Ck

    def y(self, pops: Dict[str, Population]) -> torch.Tensor:
        """Combined prediction y = sum_k alpha_k C_k x_k."""
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
    """A graph of `Population` nodes wired by `CouplingInterface` hyper-edges.

    Assembles each population's state rate from its own self term plus every interface it takes
    part in, and steps the whole system with one integrator (`step`). A population that is the
    target of some interface is a *perception* node (it can be eliminated adiabatically); a
    population that only feeds interfaces is an *environment* node (it gets exploration noise and
    the amplitude leash). This mirrors `src.model.simulate` exactly for the two-population case.
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
        for p in populations[1:]:
            if p.dtype != ref.dtype or p.device != ref.device:
                raise ValueError(
                    "all populations in a MacroNetwork must share dtype and device "
                    f"(expected {ref.dtype}/{ref.device}, got {p.dtype}/{p.device} for {p.name!r})."
                )
        for itf in self.interfaces:
            if itf.target not in self.populations:
                raise ValueError(f"interface target {itf.target!r} is not a population.")
            target = self.populations[itf.target]
            for k, source_name in enumerate(itf.sources):
                if source_name not in self.populations:
                    raise ValueError(f"interface source {source_name!r} is not a population.")
                source = self.populations[source_name]
                Ck = None if itf.C is None else itf.C[k]
                if Ck is None:
                    if source.d != target.d:
                        raise ValueError(
                            f"identity interface {source_name!r}->{itf.target!r} needs equal "
                            f"dimensions (got {source.d} and {target.d}); provide C[{k}]."
                        )
                else:
                    if not isinstance(Ck, torch.Tensor) or Ck.shape != (target.d, source.d):
                        shape = None if not isinstance(Ck, torch.Tensor) else tuple(Ck.shape)
                        raise ValueError(
                            f"C[{k}] for {source_name!r}->{itf.target!r} must have shape "
                            f"({target.d}, {source.d}) (got {shape})."
                        )
                    if Ck.dtype != ref.dtype or Ck.device != ref.device:
                        raise ValueError(
                            f"C[{k}] for {source_name!r}->{itf.target!r} must match network "
                            f"dtype/device ({ref.dtype}, {ref.device})."
                        )
                    if not bool(torch.isfinite(Ck).all()):
                        raise ValueError(
                            f"C[{k}] for {source_name!r}->{itf.target!r} contains NaN or infinity."
                        )
        self.d = ref.d
        self.dtype = ref.dtype
        self.device = ref.device

    def _require_state(self, name: str) -> torch.Tensor:
        if name not in self.populations:
            raise ValueError(f"unknown population {name!r}.")
        p = self.populations[name]
        if p.x is None:
            raise ValueError(f"population {name!r} has no state; initialize x before stepping.")
        if p.x.shape != (p.d,) or p.x.dtype != p.dtype or p.x.device != p.device:
            raise ValueError(
                f"population {name!r} state must have shape ({p.d},), dtype {p.dtype}, "
                f"and device {p.device}."
            )
        if not bool(torch.isfinite(p.x).all()):
            raise ValueError(f"population {name!r} state contains NaN or infinity.")
        return p.x

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
        self._require_state(name)
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
        self._require_state(name)
        return self.populations[name].rate_W()

    def solve_steady(self, name: str) -> torch.Tensor:
        """Adiabatic elimination of a fast interface target (infinite timescale separation):

            (pi_S S_S + (sum_i pi_Ii) I) x* = sum_i pi_Ii y_i

        For a single interface this is x* = pi_I (pi_I I + pi_S S_S)^-1 y;
        summing over interfaces also covers the separate-error control (two single-source edges).
        """
        self._require_state(name)
        active_inputs = [
            itf for itf in self.interfaces if itf.active and itf.target == name
        ]
        if not active_inputs:
            raise ValueError(f"population {name!r} has no active incoming interface to solve.")
        p = self.populations[name]
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
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError(f"dt must be finite and > 0 (got {dt!r}).")
        if mode not in {"full", "adiabatic"}:
            raise ValueError(f"mode must be 'full' or 'adiabatic' (got {mode!r}).")
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError(f"substeps must be an integer >= 1 (got {substeps!r}).")
        targets = self.targets()
        sources = self.sources()
        required = targets | sources | {
            name for name, p in self.populations.items() if p.plastic
        }
        for name in required:
            self._require_state(name)
        if noise is not None:
            unknown = set(noise) - set(self.populations)
            if unknown:
                raise ValueError(f"noise provided for unknown populations: {sorted(unknown)!r}.")
            for name, raw in noise.items():
                p = self.populations[name]
                if raw.shape != (p.d,) or raw.dtype != p.dtype or raw.device != p.device:
                    raise ValueError(
                        f"noise[{name!r}] must have shape ({p.d},), dtype {p.dtype}, "
                        f"and device {p.device}."
                    )

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
