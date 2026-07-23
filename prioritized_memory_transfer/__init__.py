"""Prioritized Memory Transfer between predictive-coding associative memories.

Three models, one engine. The composable "network of networks" engine lives in `prioritized_memory_transfer.macro`
(`Population` + `CouplingInterface` + `MacroNetwork`); every model is an instance of it,
and networks always communicate two at a time:

    # two-population transfer (teacher T -> student S)
    from prioritized_memory_transfer import ModelConfig, SimConfig, build_system, simulate
    model, info = build_system(ModelConfig())
    hist = simulate(model, SimConfig(), info)

    # interleaved subspace addition (frozen T1, T2 -> plastic S, one teacher per bout)
    from prioritized_memory_transfer import InterleavedConfig, SimConfig, build_interleaved_synthesis, simulate_interleaved
    macro, info = build_interleaved_synthesis(InterleavedConfig())
    hist = simulate_interleaved(macro, SimConfig(mode="adiabatic"), info)

    # continual learning (buffer -> synthesis -> storage over a memory stream)
    from prioritized_memory_transfer import ContinualConfig, ContinualLearner
    hist = ContinualLearner(ContinualConfig()).run()

Visualization lives in `prioritized_memory_transfer.viz_static` / `prioritized_memory_transfer.viz_interactive` / `prioritized_memory_transfer.viz_eigenspace`
(two-population) and `prioritized_memory_transfer.viz_interleaved` / `prioritized_memory_transfer.viz_continual`; all are imported lazily
so the core has no plotting dependency.
"""
from .config import ContinualConfig, InterleavedConfig, ModelConfig, SimConfig
from .artifacts import InterleavedBuildInfo, MemoryBuildResult, TwoPopBuildInfo
from .macro import CouplingInterface, MacroNetwork, Population
from .model import TwoPopModel, build_system, simulate, fwd, bwd, outer
from .interleaved import build_interleaved_synthesis, simulate_interleaved, interleave_merge
from .continual import ContinualLearner
from .memory import make_patterns, make_teacher_subspaces, build_memory, zero_diag, memory_residual
from .recall import AssociativeMemory, RecallTrace, make_mask
from .history import ContinualHistory, History, InterleavedHistory
from . import diagnostics

__all__ = [
    # configs
    "ModelConfig",
    "InterleavedConfig",
    "ContinualConfig",
    "SimConfig",
    # typed construction results
    "MemoryBuildResult",
    "TwoPopBuildInfo",
    "InterleavedBuildInfo",
    # engine (the LEGO layer)
    "Population",
    "CouplingInterface",
    "MacroNetwork",
    # two-population model
    "TwoPopModel",
    "build_system",
    # interleaved subspace addition
    "build_interleaved_synthesis",
    "simulate_interleaved",
    "interleave_merge",
    # continual learning
    "ContinualLearner",
    # dynamics / recording
    "simulate",
    "History",
    "ContinualHistory",
    "InterleavedHistory",
    "diagnostics",
    # memory / weight construction
    "make_patterns",
    "make_teacher_subspaces",
    "build_memory",
    "zero_diag",
    "memory_residual",
    # single-network associative recall (clamped query)
    "AssociativeMemory",
    "RecallTrace",
    "make_mask",
    # tied tensor ops
    "fwd",
    "bwd",
    "outer",
]
