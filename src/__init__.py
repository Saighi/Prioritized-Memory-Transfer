"""Prioritized Memory Transfer between predictive-coding associative memories.

Three models, one engine. The composable "network of networks" engine lives in `src.macro`
(`Population` + `CouplingInterface` + `MacroNetwork`); every model is an instance of it,
and networks always communicate two at a time:

    # two-population transfer (teacher T -> student S)
    from src import ModelConfig, SimConfig, build_system, simulate
    model, info = build_system(ModelConfig())
    hist = simulate(model, SimConfig(), info)

    # interleaved subspace addition (frozen T1, T2 -> plastic S, one teacher per bout)
    from src import InterleavedConfig, SimConfig, build_interleaved_synthesis, simulate_interleaved
    macro, info = build_interleaved_synthesis(InterleavedConfig())
    hist = simulate_interleaved(macro, SimConfig(mode="adiabatic"), info)

    # continual learning (buffer -> synthesis -> storage over a memory stream)
    from src import ContinualConfig, ContinualLearner
    hist = ContinualLearner(ContinualConfig()).run()

Visualization lives in `src.viz_static` / `src.viz_interactive` / `src.viz_eigenspace`
(two-population) and `src.viz_interleaved` / `src.viz_continual`; all are imported lazily
so the core has no plotting dependency.
"""
from .config import ContinualConfig, InterleavedConfig, ModelConfig, SimConfig
from .macro import CouplingInterface, MacroNetwork, Population
from .model import TwoPopModel, build_system, fwd, bwd, outer
from .interleaved import build_interleaved_synthesis, simulate_interleaved, interleave_merge
from .continual import ContinualLearner
from .memory import make_patterns, make_teacher_subspaces, build_W_T, zero_diag, memory_residual
from .recall import AssociativeMemory, RecallTrace, make_mask
from .dynamics import simulate
from .history import ContinualHistory, History, InterleavedHistory
from . import diagnostics

__all__ = [
    # configs
    "ModelConfig",
    "InterleavedConfig",
    "ContinualConfig",
    "SimConfig",
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
    "build_W_T",
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
