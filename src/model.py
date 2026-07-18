"""The two-population predictive-coding model (teacher T above student S in the hierarchy).

This is now a thin instance of the composable engine in `pmt.macro`: `build_system` wires two
`Population`s and one `AdditiveInterface`, and `TwoPopModel` is a backward-compatible **facade**
over that `MacroNetwork` — every attribute (`W_T`, `W_S`, `x_T`, `x_S`, ...) and method
(`eps_TS`, `F_S`, `novelty_operator`, `rate_*`, ...) reads or writes the underlying nodes/edge,
so the equations and the public API are unchanged.

Tied-weight convention (architecture invariant):
  Each population stores ONE recurrent weight matrix (`W_T` frozen, `W_S` plastic). The
  top-down (prediction) path uses `M = I - W`; the bottom-up (error-projection) path uses
  `M^T` — the SAME matrix transposed, never an independent tensor. The T<->S interface is
  hard-wired identity both ways, so the single shared signal is the interface error
  `eps_TS = x_S - x_T` (it sits at the lower / student level: T supplies the top-down
  prediction, S carries the residual). In engine terms it is the additive interface
  `y = 1 * x_T`, `eps = x_S - y`, with target-side precision `pi_TS` and signed source-side
  precision `pi_ST`.

Both `W_T` and `W_S` keep a ZERO DIAGONAL at all times (no autapses). The `I` inside
`M = I - W` is the structural self/leak term, not a synapse, so `diag(M) = 1` is correct.

Reversed precision (the sleep/wake mechanism). The student always *descends* its free energy
(perception). The teacher's interface term is `+pi_ST * eps_TS`, where `pi_ST` is the teacher's
*signed* interface precision — there is no separate gate:
  - wake / recall  (`pi_ST > 0`): an ordinary precision; the teacher *minimizes* the interface
    error (chases the student). Both descend — no transfer.
  - sleep / replay (`pi_ST < 0`): a **reversed (negative) precision**; the teacher *maximizes*
    the interface error (drive-to-disagree), toward states the student cannot yet predict — the
    transfer drive.
Transfer is simulated in sleep, so `pi_ST` defaults to a negative value.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from .config import ModelConfig
from .macro import (  # noqa: F401 (re-export)
    AdditiveInterface,
    MacroNetwork,
    Population,
    bwd,
    fwd,
    outer,
    resolve_signed_precision,
)
from .memory import build_W_T, make_patterns, zero_diag


class TwoPopModel:
    """Facade over a 2-node `MacroNetwork` (populations "T", "S" + one additive interface).

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
        self.patterns = patterns              # (d, P), unit-norm columns

        # the engine: T (frozen, noisy, leashed) above S (plastic, driven by the interface)
        T = Population("T", W_T, self.pi_T, self.tau_T, plastic=False,
                       sigma_xi=self.sigma_xi, r=self.r0)
        S = Population("S", torch.zeros_like(W_T), self.pi_S, self.tau_S,
                       plastic=True, eta=self.eta)
        itf = AdditiveInterface(target="S", sources=["T"], alpha=[1.0],
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
    def _itf(self) -> AdditiveInterface:
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
        self._S.W = value

    @property
    def M_S(self) -> torch.Tensor:
        return self._S.M

    @property
    def x_T(self) -> Optional[torch.Tensor]:
        return self._T.x

    @x_T.setter
    def x_T(self, value: torch.Tensor) -> None:
        self._T.x = value

    @property
    def x_S(self) -> Optional[torch.Tensor]:
        return self._S.x

    @x_S.setter
    def x_S(self, value: torch.Tensor) -> None:
        self._S.x = value

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
        S = self.S_S()
        Amat = self.pi_TS * self.I + self.pi_S * S
        return self.pi_S * S @ torch.linalg.inv(Amat)

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
        Msub = self.patterns[:, list(subset)]
        self._S.W = build_W_T(Msub, self.cfg)   # reuse the same faithful construction
        self.zero_diag_W_S()


def build_system(cfg: ModelConfig):
    """Build patterns + frozen W_T, resolve pi_ST against the spectral gap, and return
    (model, info). `info` carries the spectral gap, guard values, and the manifold basis."""
    from .diagnostics import manifold_basis, spectral_gap   # lazy import (avoid cycle)

    M = make_patterns(cfg)
    W_T = build_W_T(M, cfg)
    M_T = torch.eye(cfg.d, dtype=cfg.dtype, device=cfg.device) - W_T
    S_T = M_T.transpose(-2, -1) @ M_T
    sigma2_min = spectral_gap(S_T)

    # conservative guard (Lemma 2): |pi_ST| < pi_T sigma2_min. The "auto" default sits at
    # pi_ST_safety of it (before this fix the pi_T factor was dropped, which was only
    # correct at the default pi_T = 1).
    guard_safe = cfg.pi_T * sigma2_min
    pi_ST = resolve_signed_precision(
        cfg.pi_ST, guard=guard_safe, safety=cfg.pi_ST_safety,
        exact_saddle=cfg.exact_saddle, pi_pos=cfg.pi_TS,
    )

    model = TwoPopModel(W_T, cfg, pi_ST=pi_ST, patterns=M)
    U_T = manifold_basis(M_T)

    guard_scalar = cfg.pi_T * sigma2_min * (cfg.pi_TS + cfg.pi_S) / cfg.pi_S   # aligned-case bound
    info = {
        "kind": "two_pop",
        "patterns": M,
        "W_T": W_T,
        "S_T": S_T,
        "sigma2_min": sigma2_min,
        "guard_safe": guard_safe,                # |pi_ST| < pi_T sigma2_min : conservative (Lemma 2)
        "guard_scalar": guard_scalar,            # |pi_ST| < pi_T sigma2_min (pi_TS+pi_S)/pi_S : aligned case
        "pi_ST": pi_ST,                          # signed (negative in the replay regime)
        "exact_saddle": bool(cfg.exact_saddle),
        "U_T": U_T,                              # (d, k) orthonormal basis of ker M_T
        "manifold_dim": U_T.shape[1],
        "memory_residual": (M_T @ M).norm(dim=0).max().item(),
        "precision_ok": cfg.pi_TS > cfg.pi_S,
        "pi_ST_ok": abs(pi_ST) < guard_safe,          # conservative guard, matches the paper's Lemma 2
        "pi_ST_ok_aligned": abs(pi_ST) < guard_scalar,  # looser aligned-case bound
    }
    return model, info
