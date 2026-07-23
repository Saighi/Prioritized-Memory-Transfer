"""Configuration dataclasses: `ModelConfig` (two-pop network), `InterleavedConfig` (two-teacher
merge), `ContinualConfig` (buffer/synthesis/storage loop), `SimConfig` (running a simulation).
Raw user knobs only — derived quantities (spectral gap, resolved signed precisions) are
computed by the builders.

The load-bearing knob is the *signed* source-side precision (`pi_ST` / `rho`): > 0 wake
(ordinary precision, no transfer), < 0 sleep/replay (reversed precision, the drive-to-disagree
that transfers memory). There is no separate gate: the phase *is* the sign.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch


def _positive(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and > 0 (got {value!r}).")


def _nonnegative(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be finite and >= 0 (got {value!r}).")


def _signed_or_auto(name: str, value: Union[float, str]) -> None:
    if isinstance(value, str):
        if value != "auto":
            raise ValueError(f"{name} must be a finite number or 'auto' (got {value!r}).")
        return
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite (got {value!r}).")


@dataclass
class ModelConfig:
    # --- sizes ---
    d: int = 48                      # neurons per population (state dimension)
    P: int = 6                       # number of stored memories in T (needs P <= d-1)

    # --- precisions / gains (operating regime: pi_TS > pi_S; |pi_ST| < pi_T * sigma2_min,
    #     the stability threshold — not a paper theorem; archived analysis in
    #     docs/Paper/maths/additional_proofs_not_in_paper.md) ---
    pi_T: float = 1.0                # teacher self-precision (how much T trusts its own recurrent model)
    pi_S: float = 0.5                # student self-precision (how much S trusts its own model)
    pi_TS: float = 1.0               # student's interface precision (how much S trusts T's activity)
    pi_ST: Union[float, str] = "auto"   # teacher's SIGNED interface precision. <0 sleep/replay (reversed),
                                        # >0 wake. "auto" -> reversed default -pi_ST_safety*pi_T*sigma2_min.
    pi_ST_safety: float = 0.5        # fraction of the conservative guard (pi_T*sigma2_min) when "auto"
    exact_saddle: bool = False       # if True, override pi_ST = -pi_TS (the exact-saddle / zero-sum regime)

    # --- time constants & learning rate (tau_S << tau_T << 1/eta) ---
    tau_T: float = 10.0
    tau_S: float = 1.0
    eta: float = 0.02

    # --- noise & amplitude leash ---
    sigma_xi: float = 0.05           # std of white noise on T
    r0: float = 1.0                  # fixed norm to which ||x_T|| is renormalized

    # --- construction choices ---
    pattern_kind: str = "orthonormal"   # "orthonormal" | "correlated" (exact low-rank only at zero noise)
    corr_rank: Optional[int] = None     # effective rank for correlated patterns (default P//2)
    corr_noise: float = 0.1             # additive noise for correlated patterns

    # --- bookkeeping ---
    seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        if type(self.d) is not int or type(self.P) is not int:
            raise ValueError(f"d and P must be integers (got d={self.d!r}, P={self.P!r}).")
        if self.d < 2:
            raise ValueError(f"d must be >= 2 (got {self.d}).")
        if self.P < 1:
            raise ValueError(f"P must be >= 1 (got {self.P}).")
        if self.P > self.d - 1:
            raise ValueError(
                "ModelConfig keeps generated pattern sets below the zero-diagonal capacity, "
                f"so it needs P <= d-1 (got P={self.P}, d={self.d})."
            )
        if self.pattern_kind not in {"orthonormal", "correlated"}:
            raise ValueError(
                f"pattern_kind must be 'orthonormal' or 'correlated' (got {self.pattern_kind!r})."
            )
        if self.corr_rank is not None and (
            type(self.corr_rank) is not int or not 1 <= self.corr_rank <= self.P
        ):
            raise ValueError(f"corr_rank must be an integer in [1, P] (got {self.corr_rank!r}).")
        _nonnegative("corr_noise", self.corr_noise)
        for name in ("pi_T", "pi_S", "pi_TS", "tau_T", "tau_S", "r0"):
            _positive(name, getattr(self, name))
        for name in ("eta", "sigma_xi"):
            _nonnegative(name, getattr(self, name))
        _signed_or_auto("pi_ST", self.pi_ST)
        if not 0 < self.pi_ST_safety <= 1:
            raise ValueError(
                f"pi_ST_safety must lie in (0, 1] (got {self.pi_ST_safety})."
            )
        if self.pi_TS <= self.pi_S:
            # not fatal, but it violates the precision guard (confabulation risk)
            warnings.warn(
                f"precision guard pi_TS > pi_S violated (pi_TS={self.pi_TS}, pi_S={self.pi_S})."
            )


@dataclass
class InterleavedConfig:
    """Configuration for interleaved subspace addition (`prioritized_memory_transfer.interleaved`): two frozen teachers
    `T1`, `T2` merged into a plastic synthesis network `S` by rehearsing ONE teacher per replay
    bout. Each bout is a plain two-network reversed-precision transfer, so `S` learns the
    subspace sum `U1 + U2` without any teacher cross-term.
    """
    # --- sizes & teacher geometry ---
    d: int = 64
    rank1: int = 4                   # dim U1 = ker M_T1
    rank2: int = 4                   # dim U2 = ker M_T2
    geometry: str = "shared"         # "orthogonal" | "shared" | "oblique"
    overlap: int = 2                 # dim(U1 ∩ U2) when geometry == "shared"
    principal_angle: float = 0.35    # principal angle (rad) when geometry == "oblique"

    # --- precisions (guards: pi_I > pi_S ; |rho| below the structure guard) ---
    pi_T1: float = 1.0               # teacher-1 self-precision
    pi_T2: float = 1.0               # teacher-2 self-precision
    pi_S: float = 0.5                # synthesis self-precision
    pi_I: float = 1.0                # synthesis interface precision (target side, positive)
    rho: Union[float, str] = "auto"  # SIGNED teacher-side precision: <0 sleep/replay, >0 wake.
    rho_safety: float = 0.5          # fraction of the structure guard used when rho == "auto"
    exact_saddle: bool = False       # if True, rho = -pi_I (the zero-sum condition)

    # --- time constants & learning rate (tau_S << tau_T1,tau_T2 << 1/eta) ---
    tau_T1: float = 10.0
    tau_T2: float = 10.0
    tau_S: float = 1.0
    eta: float = 0.005

    # --- noise & amplitude leash ---
    sigma_xi1: float = 0.05          # teacher-1 exploration noise std
    sigma_xi2: float = 0.05          # teacher-2 exploration noise std
    r1: float = 1.0                  # fixed norm for ||x1||
    r2: float = 1.0                  # fixed norm for ||x2||

    # --- construction ---
    # --- bookkeeping ---
    seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        if any(type(v) is not int for v in (self.d, self.rank1, self.rank2, self.overlap)):
            raise ValueError("d, rank1, rank2, and overlap must be integers.")
        if self.d < 2:
            raise ValueError(f"d must be >= 2 (got {self.d}).")
        if self.rank1 < 1 or self.rank2 < 1:
            raise ValueError(
                f"rank1 and rank2 must be >= 1 (got {self.rank1}, {self.rank2})."
            )
        if self.rank1 > self.d - 1 or self.rank2 > self.d - 1:
            raise ValueError(
                f"Need rank <= d-1 per teacher (got rank1={self.rank1}, rank2={self.rank2}, d={self.d})."
            )
        if self.geometry not in {"orthogonal", "shared", "oblique"}:
            raise ValueError(
                f"geometry must be 'orthogonal', 'shared', or 'oblique' (got {self.geometry!r})."
            )
        if self.geometry == "orthogonal" and self.rank1 + self.rank2 > self.d:
            raise ValueError(
                f"orthogonal geometry needs rank1+rank2 <= d "
                f"(got {self.rank1}+{self.rank2} > {self.d})."
            )
        if self.geometry == "shared":
            if not 0 <= self.overlap <= min(self.rank1, self.rank2):
                raise ValueError(
                    "shared geometry needs overlap in "
                    f"[0, min(rank1, rank2)] (got {self.overlap})."
                )
            if self.rank1 + self.rank2 - self.overlap > self.d:
                raise ValueError(
                    "shared geometry needs rank1+rank2-overlap <= d "
                    f"(got {self.rank1}+{self.rank2}-{self.overlap} > {self.d})."
                )
        if self.geometry == "oblique":
            if self.rank1 + self.rank2 > self.d:
                raise ValueError(
                    f"oblique geometry needs rank1+rank2 <= d "
                    f"(got {self.rank1}+{self.rank2} > {self.d})."
                )
            if not 0 < self.principal_angle <= math.pi / 2:
                raise ValueError(
                    "principal_angle must lie in (0, pi/2] for oblique geometry "
                    f"(got {self.principal_angle})."
                )
        for name in ("pi_T1", "pi_T2", "pi_S", "pi_I", "tau_T1", "tau_T2", "tau_S", "r1", "r2"):
            _positive(name, getattr(self, name))
        for name in ("eta", "sigma_xi1", "sigma_xi2"):
            _nonnegative(name, getattr(self, name))
        _signed_or_auto("rho", self.rho)
        if not 0 < self.rho_safety <= 1:
            raise ValueError(f"rho_safety must lie in (0, 1] (got {self.rho_safety}).")
        if self.pi_I <= self.pi_S:
            warnings.warn(
                f"precision guard pi_I > pi_S violated (pi_I={self.pi_I}, pi_S={self.pi_S})."
            )


@dataclass
class ContinualConfig:
    """Configuration for the continual-learning loop (`prioritized_memory_transfer.continual`): a stream of memories is
    consolidated one at a time through a fast **Buffer**, a transient **Synthesis** workspace, and
    a slow **Storage** network, so Storage accumulates the whole stream without catastrophic
    forgetting. Each cycle: write the new memory into the buffer (one-shot covPCN), consolidate
    Buffer+Storage -> Synthesis by **interleaved rehearsal** (alternate rehearsing Buffer and
    Storage one at a time — never summed — so single-memory / correlated memories merge without
    crosstalk), then download Synthesis -> Storage. The Buffer (hippocampus) is the ONLY network
    that ever learns from an actual memory; everything downstream learns purely by replay.
    """
    # --- stream (correlated by default: the regime where interleaving matters) ---
    d: int = 32
    n_memories: int = 5
    memory_kind: str = "correlated"  # "correlated" (low-rank + overlaps) | "random" unit vectors

    # --- precisions (guards: pi_I > pi_S ; |rho| below the structure guard) ---
    pi_teacher: float = 1.0          # self-precision of the frozen networks in each phase
    pi_S: float = 0.5                # plastic (learner) self-precision
    pi_I: float = 1.0                # interface precision (target side, positive)
    rho: Union[float, str] = "auto"  # SIGNED source-side precision: <0 sleep/replay
    rho_safety: float = 0.9          # fraction of the structure guard used when rho == "auto"

    # --- time constants & learning rate (tau_S << tau_teacher << 1/eta) ---
    tau_teacher: float = 10.0
    tau_S: float = 1.0
    eta: float = 0.05

    # --- noise & amplitude leash ---
    sigma_xi: float = 0.12           # exploration noise on the frozen (source) networks
    r: float = 1.0                   # fixed norm for source states

    # --- interleaved replay schedule ---
    consolidate_bouts: int = 12      # alternating Buffer/Storage rehearsal bouts per consolidation
    download_bouts: int = 5          # single-teacher (Synthesis->Storage) bouts
    bout_steps: int = 4000           # steps per replay bout
    dt: float = 0.5
    mode: str = "adiabatic"

    # --- loop wiring ---
    synthesis_warm_start: bool = True    # start Synthesis as the current Storage (fast increment)
    storage_support: bool = True         # interleave Storage in as a rehearsal co-teacher

    # --- bookkeeping ---
    seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        if type(self.d) is not int or type(self.n_memories) is not int:
            raise ValueError(
                f"d and n_memories must be integers "
                f"(got d={self.d!r}, n_memories={self.n_memories!r})."
            )
        if self.d < 2:
            raise ValueError(f"d must be >= 2 (got {self.d}).")
        if self.n_memories < 1:
            raise ValueError(f"n_memories must be >= 1 (got {self.n_memories}).")
        if self.memory_kind not in {"correlated", "random"}:
            raise ValueError(
                f"memory_kind must be 'correlated' or 'random' (got {self.memory_kind!r})."
            )
        for name in ("pi_teacher", "pi_S", "pi_I", "tau_teacher", "tau_S", "r", "dt"):
            _positive(name, getattr(self, name))
        for name in ("eta", "sigma_xi"):
            _nonnegative(name, getattr(self, name))
        _signed_or_auto("rho", self.rho)
        if not 0 < self.rho_safety <= 1:
            raise ValueError(f"rho_safety must lie in (0, 1] (got {self.rho_safety}).")
        for name in ("consolidate_bouts", "download_bouts", "bout_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1 (got {value!r}).")
        if self.mode not in {"full", "adiabatic"}:
            raise ValueError(f"mode must be 'full' or 'adiabatic' (got {self.mode!r}).")
        if self.pi_I <= self.pi_S:
            warnings.warn(f"precision guard pi_I > pi_S violated (pi_I={self.pi_I}, pi_S={self.pi_S}).")


@dataclass
class SimConfig:
    n_steps: int = 60000
    dt: float = 0.5
    mode: str = "full"               # "full" (integrate S) | "adiabatic" (S at steady state)
    s_substeps: int = 1              # inner fast S-steps per outer step (full mode only)
    record_every: int = 100
    n_weight_snapshots: int = 6      # how many W_S heatmap snapshots to keep over the run
    pretrain_subset: Optional[Sequence[int]] = None  # pattern indices S already "knows"
    bout_steps: int = 3000           # steps per replay bout in the interleaved merge (prioritized_memory_transfer.interleaved)
    progress: bool = True

    def __post_init__(self) -> None:
        if type(self.n_steps) is not int or self.n_steps < 1:
            raise ValueError(f"n_steps must be an integer >= 1 (got {self.n_steps!r}).")
        _positive("dt", self.dt)
        if self.mode not in {"full", "adiabatic"}:
            raise ValueError(f"mode must be 'full' or 'adiabatic' (got {self.mode!r}).")
        for name in ("s_substeps", "record_every", "bout_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1 (got {value!r}).")
        if type(self.n_weight_snapshots) is not int or self.n_weight_snapshots < 0:
            raise ValueError(
                "n_weight_snapshots must be an integer >= 0 "
                f"(got {self.n_weight_snapshots!r})."
            )
