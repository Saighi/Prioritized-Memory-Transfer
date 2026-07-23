"""Typed construction results shared by the model builders.

These dataclasses replace loosely-typed ``info`` dictionaries.  They make the scientific
diagnostics part of the public contract: callers can inspect not just the fitted weights, but
also whether the requested memories were actually representable by a zero-diagonal network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .config import InterleavedConfig


@dataclass(frozen=True)
class MemoryBuildResult:
    """Result of fitting one zero-diagonal covariance-PCN memory.

    ``condition_number`` is the worst conditioning of the per-neuron regression problems,
    which is the numerically relevant quantity under the no-autapse constraint.
    """

    W: torch.Tensor
    M_op: torch.Tensor
    residuals: torch.Tensor
    max_residual: float
    pattern_rank: int
    manifold_rank: int
    condition_number: float
    representable: bool
    tolerance: float


@dataclass(frozen=True)
class TwoPopBuildInfo:
    """Derived quantities for a two-population system."""

    patterns: torch.Tensor
    memory: MemoryBuildResult
    S_T: torch.Tensor
    sigma2_min: float
    guard_safe: float
    guard_scalar: float
    pi_ST: float
    exact_saddle: bool
    U_T: torch.Tensor
    precision_ok: bool
    pi_ST_ok: bool
    pi_ST_ok_aligned: bool
    kind: str = "two_pop"

    @property
    def W_T(self) -> torch.Tensor:
        return self.memory.W

    @property
    def manifold_dim(self) -> int:
        return self.U_T.shape[1]

    @property
    def memory_residual(self) -> float:
        return self.memory.max_residual


@dataclass(frozen=True)
class InterleavedBuildInfo:
    """Derived quantities and wiring metadata for an interleaved two-teacher merge."""

    cfg: InterleavedConfig
    M1: torch.Tensor
    M2: torch.Tensor
    memory1: MemoryBuildResult
    memory2: MemoryBuildResult
    U1: torch.Tensor
    U2: torch.Tensor
    U_Sigma: torch.Tensor
    overlap: int
    r_Sigma: int
    capacity_ok: bool
    sigma1_min: float
    sigma2_min: float
    rho1: float
    rho2: float
    target: str
    order: Tuple[str, ...]
    bases: Dict[str, torch.Tensor]
    seed: int
    kind: str = "interleaved"

    @property
    def W1(self) -> torch.Tensor:
        return self.memory1.W

    @property
    def W2(self) -> torch.Tensor:
        return self.memory2.W

    @property
    def rank1(self) -> int:
        return self.U1.shape[1]

    @property
    def rank2(self) -> int:
        return self.U2.shape[1]
