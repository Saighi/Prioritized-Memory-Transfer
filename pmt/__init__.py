"""pmt — Prioritized Memory Transfer between predictive-coding associative memories.

Two models, one engine. The composable "network of networks" engine lives in `pmt.macro`
(`Population` + `AdditiveInterface` + `MacroNetwork`); both shipped models are instances of it:

    # two-population transfer (teacher T -> student S)
    from pmt import ModelConfig, SimConfig, build_system, simulate
    model, info = build_system(ModelConfig())
    hist = simulate(model, SimConfig(), info)

    # additive three-network synthesis (frozen T1, T2 -> plastic S)
    from pmt import AdditiveSynthesisConfig, SimConfig, build_additive_synthesis, simulate
    macro, info = build_additive_synthesis(AdditiveSynthesisConfig())
    hist = simulate(macro, SimConfig(mode="adiabatic"), info)

Visualization lives in `pmt.viz_static` / `pmt.viz_interactive` (two-population) and
`pmt.viz_additive` (additive); they are imported lazily so the core has no plotting dependency.
"""
from .config import AdditiveSynthesisConfig, ContinualConfig, ModelConfig, SimConfig
from .macro import AdditiveInterface, MacroNetwork, Population
from .model import TwoPopModel, build_system, fwd, bwd, outer
from .additive import build_additive_synthesis, simulate_additive
from .interleaved import build_interleaved_synthesis, simulate_interleaved, interleave_merge
from .continual import ContinualLearner
from .memory import make_patterns, make_teacher_subspaces, build_W_T, zero_diag, memory_residual
from .recall import AssociativeMemory, RecallTrace, make_mask
from .dynamics import simulate
from .history import AdditiveHistory, ContinualHistory, History, InterleavedHistory
from . import diagnostics

__all__ = [
    # configs
    "ModelConfig",
    "AdditiveSynthesisConfig",
    "ContinualConfig",
    "SimConfig",
    # engine (the LEGO layer)
    "Population",
    "AdditiveInterface",
    "MacroNetwork",
    # two-population model
    "TwoPopModel",
    "build_system",
    # additive three-network model
    "build_additive_synthesis",
    "simulate_additive",
    # interleaved merge
    "build_interleaved_synthesis",
    "simulate_interleaved",
    "interleave_merge",
    # continual learning
    "ContinualLearner",
    # dynamics / recording
    "simulate",
    "History",
    "AdditiveHistory",
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
