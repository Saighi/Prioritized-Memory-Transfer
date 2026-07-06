"""Configuration dataclasses for the prioritized-memory-transfer model.

Two configs:
  - ModelConfig: everything that defines the network (sizes, precisions, gains, how
    patterns and W_T are built). Raw user knobs only; derived quantities such as the
    spectral gap and the resolved pi_ST are computed in `model.build_system`.
  - SimConfig: everything about *running* the simulation (steps, dt, integration mode,
    recording, optional pretraining).

Reversed precision (the load-bearing object). The teacher's interface precision `pi_ST` is a
*signed* quantity. `pi_ST > 0` is an ordinary precision (wake / recall: the teacher minimizes
the interface error, chasing the student). `pi_ST < 0` is a **reversed (negative) precision**
(sleep / replay: the teacher *maximizes* the interface error — the drive-to-disagree that
transfers memory). A negative precision is not a standard inverse-covariance (those are ≥ 0);
it is introduced deliberately as the single knob that *selects* wake vs sleep. There is no separate
±1 gate: the phase *is* the sign of `pi_ST`.

Nothing here imports torch-heavy machinery beyond the dtype handle, so configs are cheap
to construct and easy to serialize/print.
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

    # --- precisions / gains (guards: pi_TS > pi_S; |pi_ST| < spectral gap) ---
    pi_T: float = 1.0                # teacher self-precision (how much T trusts its own recurrent model)
    pi_S: float = 0.5                # student self-precision (how much S trusts its own model)
    pi_TS: float = 1.0               # student's interface precision (how much S trusts T's activity)
    pi_ST: Union[float, str] = "auto"   # teacher's SIGNED interface precision. <0 sleep/replay (reversed),
                                        # >0 wake. "auto" -> reversed default -pi_ST_safety*sigma2_min.
    pi_ST_safety: float = 0.5        # fraction of the spectral gap used for |pi_ST| when pi_ST == "auto"
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
class AdditiveSynthesisConfig:
    """Configuration for the additive three-network memory-synthesis model
    (`three_network_additive_memory_synthesis.md`): two frozen teachers `T1`, `T2` summed into
    `y = alpha1 x1 + alpha2 x2`, a common error `eps_Sigma = x_S - y` driving a plastic synthesis
    network `S`. An ergonomic front-end; `pmt.additive.build_additive_synthesis` turns it into a
    `MacroNetwork` (three `Population`s + one `AdditiveInterface`).
    """
    # --- sizes & teacher geometry (spec section 20) ---
    d: int = 64
    rank1: int = 4                   # dim U1 = ker M_T1
    rank2: int = 4                   # dim U2 = ker M_T2
    geometry: str = "shared"         # "orthogonal" | "shared" | "oblique"
    overlap: int = 2                 # dim(U1 ∩ U2) when geometry == "shared"
    principal_angle: float = 0.35    # principal angle (rad) when geometry == "oblique"

    # --- precisions (guards: pi_I > pi_S ; |rho| below the structure guard, spec section 17) ---
    pi_T1: float = 1.0               # teacher-1 self-precision
    pi_T2: float = 1.0               # teacher-2 self-precision
    pi_S: float = 0.5                # synthesis self-precision
    pi_I: float = 1.0                # synthesis interface precision (target side, positive)
    rho: Union[float, str] = "auto"  # SIGNED teacher-side precision: <0 sleep/replay, >0 wake.
    rho_safety: float = 0.5          # fraction of the structure guard used when rho == "auto"
    exact_saddle: bool = False       # if True, rho = -pi_I (spec section 7 zero-sum condition)

    # --- additive coupling ---
    alpha1: float = 1.0
    alpha2: float = 1.0

    # --- time constants & learning rate (tau_S << tau_T1,tau_T2 << 1/eta) ---
    tau_T1: float = 10.0
    tau_T2: float = 10.0
    tau_S: float = 1.0
    eta: float = 0.005

    # --- noise & amplitude leash ---
    sigma_xi1: float = 0.05          # teacher-1 exploration noise std
    sigma_xi2: float = 0.05          # teacher-2 exploration noise std
    correlated_noise: bool = False   # if True, teachers share one noise draw (ablation C)
    r1: float = 1.0                  # fixed norm for ||x1|| (when norm_constraint)
    r2: float = 1.0                  # fixed norm for ||x2||
    norm_constraint: bool = True     # if False, drop the renorm leash (ablation F)

    # --- wiring & construction ---
    separate_errors: bool = False    # if True, two single-source interfaces (ablation A / section 3)
    init_on_manifold: bool = True    # initialize each teacher inside its own memory subspace
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
    """Configuration for the continual-learning loop (`pmt.continual`): a stream of memories is
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
    bout_steps: int = 3000           # steps per replay bout in the interleaved merge (pmt.interleaved)
    progress: bool = True
