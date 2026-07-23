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
from .model import build_system, simulate, fwd, bwd, outer
from .interleaved import build_interleaved_synthesis, simulate_interleaved
from .continual import ContinualLearner
from .recall import AssociativeMemory, RecallTrace, make_mask
from . import diagnostics

__all__ = [
    # configs
    "ModelConfig",
    "InterleavedConfig",
    "ContinualConfig",
    "SimConfig",
    # two-population model
    "build_system",
    # interleaved subspace addition
    "build_interleaved_synthesis",
    "simulate_interleaved",
    # continual learning
    "ContinualLearner",
    # dynamics / recording
    "simulate",
    "diagnostics",
    # single-network associative recall (clamped query)
    "AssociativeMemory",
    "RecallTrace",
    "make_mask",
    # tied tensor ops
    "fwd",
    "bwd",
    "outer",
]
