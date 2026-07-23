"""Configuration dataclasses: `ModelConfig` (two-pop network), `InterleavedConfig` (two-teacher
merge), `ContinualConfig` (buffer/synthesis/storage loop), `SimConfig` (running a simulation).
Raw user knobs only — derived quantities (spectral gap, resolved signed precisions) are
computed by the builders.

The load-bearing knob is the *signed* source-side precision (`pi_ST` / `rho`): > 0 wake
(ordinary precision, no transfer), < 0 sleep/replay (reversed precision, the drive-to-disagree
that transfers memory). There is no separate gate: the phase *is* the sign.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import torch


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
    pattern_kind: str = "orthonormal"   # "orthonormal" (rank P) | "correlated" (rank < P)
    W_T_kind: str = "covpcn"            # "covpcn" (faithful, learned, zero-diag) | "projector"
    corr_rank: Optional[int] = None     # effective rank for correlated patterns (default P//2)
    corr_noise: float = 0.1             # additive noise for correlated patterns

    # --- bookkeeping ---
    seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        if self.P > self.d - 1:
            raise ValueError(
                f"Need P <= d-1 for an exact zero-diagonal fit (got P={self.P}, d={self.d})."
            )
        if self.pi_TS <= self.pi_S:
            # not fatal, but it violates the precision guard (confabulation risk)
            import warnings
            warnings.warn(
                f"precision guard pi_TS > pi_S violated (pi_TS={self.pi_TS}, pi_S={self.pi_S})."
            )


@dataclass
class InterleavedConfig:
    """Configuration for interleaved subspace addition (`src.interleaved`): two frozen teachers
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
    W_T_kind: str = "covpcn"         # frozen-teacher construction ("covpcn" | "projector")

    # --- bookkeeping ---
    seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        if self.rank1 > self.d - 1 or self.rank2 > self.d - 1:
            raise ValueError(
                f"Need rank <= d-1 per teacher (got rank1={self.rank1}, rank2={self.rank2}, d={self.d})."
            )
        if self.pi_I <= self.pi_S:
            import warnings
            warnings.warn(
                f"precision guard pi_I > pi_S violated (pi_I={self.pi_I}, pi_S={self.pi_S})."
            )


@dataclass
class ContinualConfig:
    """Configuration for the continual-learning loop (`src.continual`): a stream of memories is
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
    revisit: Optional[Sequence[int]] = None

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
    W_T_kind: str = "covpcn"             # frozen-network construction ("covpcn" | "projector")

    # --- bookkeeping ---
    seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        if self.pi_I <= self.pi_S:
            import warnings
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
    bout_steps: int = 3000           # steps per replay bout in the interleaved merge (src.interleaved)
    progress: bool = True
