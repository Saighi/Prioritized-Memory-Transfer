"""The two-population predictive-coding model (teacher T above student S) and its run loop.

A thin instance of the engine in `prioritized_memory_transfer.macro`: `build_system` wires two `Population`s and one
`CouplingInterface` (`y = x_T`, `eps_TS = x_S - x_T`, target precision `pi_TS`, signed source
precision `pi_ST`); `TwoPopModel` is a facade over that `MacroNetwork` exposing the historical
surface (`W_T/W_S/x_T/x_S`, `eps_*`, `F_*`, `novelty_operator`, `rate_*`). `simulate` runs it
and records a `History`. Full derivations: `two_population_memory_transfer_model.md`.

Invariants: both `W` matrices keep a ZERO DIAGONAL at all times (no autapses; the `I` in
`M = I - W` is the structural self/leak term, not a synapse).

Reversed precision (the sleep/wake mechanism): the student always *descends* its free energy;
the teacher's interface term is `+pi_ST * eps_TS` with `pi_ST` *signed* —
  - wake / recall  (`pi_ST > 0`): the teacher minimizes the interface error; no transfer.
  - sleep / replay (`pi_ST < 0`): reversed precision, the teacher *maximizes* the interface
    error (drive-to-disagree) — the transfer drive. Default is negative.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch

from .artifacts import TwoPopBuildInfo
from .config import ModelConfig, SimConfig
from .macro import (  # noqa: F401 (re-export)
    CouplingInterface,
    MacroNetwork,
    Population,
    bwd,
    fwd,
    outer,
    resolve_signed_precision,
)
from .memory import build_memory, make_patterns

if TYPE_CHECKING:
    from .history import History


class TwoPopModel:
    """Facade over a 2-node `MacroNetwork` (populations "T", "S" + one coupling interface).

    Exposes the historical two-population surface — states, errors, energies, the novelty
    operator, and the three instantaneous rates (dx_T/dt, dx_S/dt, dW_S/dt) — by delegating to
    the underlying engine, so existing notebooks, diagnostics and viz keep working unchanged.
    """

    def __init__(
        self,
        W_T: torch.Tensor,
        cfg: ModelConfig,
        pi_ST: float,
        patterns: torch.Tensor,
    ) -> None:
        self.cfg = cfg
        self.dtype = W_T.dtype
        self.device = W_T.device
        self.d = W_T.shape[0]
        self.I = torch.eye(self.d, dtype=self.dtype, device=self.device)

        # scalar mirrors (read by diagnostics.check_gradients / circulation)
        self.pi_T = float(cfg.pi_T)
        self.pi_S = float(cfg.pi_S)
        self.pi_TS = float(cfg.pi_TS)
        self.pi_ST = float(pi_ST)
        self.tau_T = float(cfg.tau_T)
        self.tau_S = float(cfg.tau_S)
        self.eta = float(cfg.eta)
        self.sigma_xi = float(cfg.sigma_xi)
        self.r0 = float(cfg.r0)
        if patterns.ndim != 2 or patterns.shape[0] != self.d:
            raise ValueError(
                f"patterns must have shape ({self.d}, P) (got {tuple(patterns.shape)})."
            )
        if patterns.dtype != self.dtype or patterns.device != self.device:
            raise ValueError("patterns must match W_T dtype and device.")
        self.patterns = patterns              # (d, P), unit-norm columns

        # the engine: T (frozen, noisy, leashed) above S (plastic, driven by the interface)
        T = Population("T", W_T, self.pi_T, self.tau_T, plastic=False,
                       sigma_xi=self.sigma_xi, r=self.r0)
        S = Population("S", torch.zeros_like(W_T), self.pi_S, self.tau_S,
                       plastic=True, eta=self.eta)
        itf = CouplingInterface(target="S", sources=["T"], alpha=[1.0],
                                pi_I=self.pi_TS, rho=self.pi_ST)
        self.macro = MacroNetwork([T, S], [itf])

    # ----- engine handles -----
    @property
    def _T(self) -> Population:
        return self.macro.populations["T"]

    @property
    def _S(self) -> Population:
        return self.macro.populations["S"]

    @property
    def _itf(self) -> CouplingInterface:
        return self.macro.interfaces[0]

    # ----- weights / states as views on the engine -----
    @property
    def W_T(self) -> torch.Tensor:
        return self._T.W

    @property
    def M_T(self) -> torch.Tensor:
        return self._T.M

    @property
    def W_S(self) -> torch.Tensor:
        return self._S.W

    @W_S.setter
    def W_S(self, value: torch.Tensor) -> None:
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (self.d, self.d)
            or value.dtype != self.dtype
            or value.device != self.device
        ):
            shape = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
            raise ValueError(
                f"W_S must have shape ({self.d}, {self.d}), dtype {self.dtype}, "
                f"and device {self.device} (got shape {shape})."
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError("W_S contains NaN or infinite values.")
        if float(torch.diagonal(value).abs().max()) > 10 * torch.finfo(value.dtype).eps:
            raise ValueError("W_S must have a zero diagonal.")
        self._S.W = value

    @property
    def M_S(self) -> torch.Tensor:
        return self._S.M

    @property
    def x_T(self) -> Optional[torch.Tensor]:
        return self._T.x

    @x_T.setter
    def x_T(self, value: torch.Tensor) -> None:
        self._T.x = self._validated_state("x_T", value)

    @property
    def x_S(self) -> Optional[torch.Tensor]:
        return self._S.x

    @x_S.setter
    def x_S(self, value: torch.Tensor) -> None:
        self._S.x = self._validated_state("x_S", value)

    def _validated_state(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (self.d,)
            or value.dtype != self.dtype
            or value.device != self.device
        ):
            shape = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
            raise ValueError(
                f"{name} must have shape ({self.d},), dtype {self.dtype}, "
                f"and device {self.device} (got shape {shape})."
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains NaN or infinite values.")
        return value

    # ----- errors -----
    def eps_T(self) -> torch.Tensor:
        return self._T.eps_self()

    def eps_TS(self) -> torch.Tensor:
        return self._itf.eps(self.macro.populations)   # x_S - x_T

    def eps_S(self) -> torch.Tensor:
        return self._S.eps_self()

    # ----- energies -----
    def F_T(self) -> torch.Tensor:
        return self._T.F_self()

    def F_S(self) -> torch.Tensor:
        return 0.5 * self.pi_TS * (self.eps_TS() ** 2).sum(-1) \
            + 0.5 * self.pi_S * (self.eps_S() ** 2).sum(-1)

    def Phi(self) -> torch.Tensor:
        return self.F_T() - self.F_S()

    # ----- novelty operator N_S and S's steady state (fast-S reduction) -----
    def S_S(self) -> torch.Tensor:
        return self._S.S_op()

    def novelty_operator(self) -> torch.Tensor:
        """N_S = pi_S S_S (pi_TS I + pi_S S_S)^-1 ; eps_TS* = -N_S x_T at S's steady state."""
        from .diagnostics import novelty_operator   # lazy import (avoid cycle)
        return novelty_operator(self.S_S(), self.pi_TS, self.pi_S)

    def solve_xS_steady(self, x_T: torch.Tensor) -> torch.Tensor:
        """x_S* = pi_TS (pi_TS I + pi_S S_S)^-1 x_T (adiabatic elimination of fast S)."""
        Amat = self.pi_TS * self.I + self.pi_S * self.S_S()
        return self.pi_TS * torch.linalg.solve(Amat, x_T)

    # ----- instantaneous rates (assembled by the engine) -----
    def rate_x_T_det(self) -> torch.Tensor:
        """Deterministic part of tau_T dx_T/dt, divided by tau_T."""
        return self.macro.rate_x("T")

    def rate_x_S(self) -> torch.Tensor:
        return self.macro.rate_x("S")

    def rate_W_S(self) -> torch.Tensor:
        """dW_S/dt = eta pi_S eps_S x_S^T (Hebbian on S's self-error and activity)."""
        return self._S.rate_W()

    # ----- state maintenance -----
    def renorm_x_T(self) -> None:
        self._T.renorm()

    def zero_diag_W_S(self) -> None:
        self._S.zero_diag_W()

    def reset_state(self, generator: Optional[torch.Generator] = None) -> None:
        self._S.x = torch.zeros(self.d, dtype=self.dtype, device=self.device)
        x = torch.randn(self.d, generator=generator, dtype=self.dtype, device=self.device)
        self._T.x = x * (self.r0 / x.norm().clamp_min(1e-12))

    def pretrain(self, subset: Sequence[int]) -> None:
        """Make S already 'know' a subset of patterns by setting W_S to the zero-diagonal
        covPCN solution for those patterns (so M_S nulls them -> they are 'known')."""
        indices = list(subset)
        if len(set(indices)) != len(indices):
            raise ValueError(f"pretrain_subset contains duplicate indices: {indices!r}.")
        if any(not isinstance(i, int) or i < 0 or i >= self.patterns.shape[1] for i in indices):
            raise ValueError(
                f"pretrain_subset indices must lie in [0, {self.patterns.shape[1] - 1}] "
                f"(got {indices!r})."
            )
        Msub = self.patterns[:, indices]
        self._S.W = build_memory(Msub).W
        self.zero_diag_W_S()


def build_system(cfg: ModelConfig) -> tuple[TwoPopModel, TwoPopBuildInfo]:
    """Build patterns + frozen W_T, resolve pi_ST against the spectral gap, and return
    (model, info). `info` carries the spectral gap, guard values, and the manifold basis."""
    from .diagnostics import manifold_basis, spectral_gap   # lazy import (avoid cycle)

    M = make_patterns(cfg)
    memory = build_memory(M)
    W_T = memory.W
    M_T = memory.M_op
    S_T = M_T.transpose(-2, -1) @ M_T
    sigma2_min = spectral_gap(S_T)

    # conservative stability threshold: |pi_ST| < pi_T sigma2_min (not a paper theorem;
    # archived analysis in docs/Paper/maths/additional_proofs_not_in_paper.md). The "auto"
    # default sits at pi_ST_safety of it (before this fix the pi_T factor was dropped,
    # which was only correct at the default pi_T = 1).
    guard_safe = cfg.pi_T * sigma2_min
    pi_ST = resolve_signed_precision(
        cfg.pi_ST, guard=guard_safe, safety=cfg.pi_ST_safety,
        exact_saddle=cfg.exact_saddle, pi_pos=cfg.pi_TS,
    )

    model = TwoPopModel(W_T, cfg, pi_ST=pi_ST, patterns=M)
    U_T = manifold_basis(M_T)

    guard_scalar = cfg.pi_T * sigma2_min * (cfg.pi_TS + cfg.pi_S) / cfg.pi_S   # aligned-case bound
    info = TwoPopBuildInfo(
        patterns=M,
        memory=memory,
        S_T=S_T,
        sigma2_min=sigma2_min,
        guard_safe=guard_safe,
        guard_scalar=guard_scalar,
        pi_ST=pi_ST,
        exact_saddle=bool(cfg.exact_saddle),
        U_T=U_T,
        precision_ok=cfg.pi_TS > cfg.pi_S,
        pi_ST_ok=abs(pi_ST) < guard_safe,
        pi_ST_ok_aligned=abs(pi_ST) < guard_scalar,
    )
    return model, info


def simulate(
    model: TwoPopModel,
    sim: SimConfig,
    info: Optional[TwoPopBuildInfo] = None,
) -> "History":
    """Run a two-population transfer simulation and record observables into a `History`.

    Two integration modes (both handled by the engine's `MacroNetwork.step`):
      - "full":      explicit Euler on the fast student (watch it relax); teacher noise uses
                     Euler-Maruyama sqrt(dt) scaling.
      - "adiabatic": the student jumps to its exact steady state each step (infinite timescale
                     separation); faster and matches the novelty-operator theory cleanly.
    The regime is the sign of `pi_ST`: < 0 = sleep/replay = transfer.
    """
    from .history import History   # lazy import (History references TwoPopModel)

    cfg = model.cfg
    gen = torch.Generator(device=model.device).manual_seed(cfg.seed + 12345)

    model.reset_state(gen)
    if sim.pretrain_subset is not None:
        model.pretrain(sim.pretrain_subset)

    U_T = info.U_T if info is not None else None
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
