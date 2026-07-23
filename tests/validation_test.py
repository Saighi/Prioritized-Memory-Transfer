"""Fast boundary-validation checks. Run directly with:

    conda run -n pytorch --no-capture-output python tests/validation_test.py
"""
import torch

from prioritized_memory_transfer import (
    ContinualConfig,
    CouplingInterface,
    InterleavedBuildInfo,
    InterleavedConfig,
    MacroNetwork,
    MemoryBuildResult,
    ModelConfig,
    Population,
    SimConfig,
    TwoPopBuildInfo,
    build_interleaved_synthesis,
    build_memory,
    build_system,
    simulate_interleaved,
)


def rejects(exc_type, fn, contains: str = "") -> None:
    try:
        fn()
    except exc_type as exc:
        assert contains in str(exc), (contains, str(exc))
    else:
        raise AssertionError(f"expected {exc_type.__name__}")


# Removed options fail at the dataclass/API boundary instead of silently doing nothing.
rejects(TypeError, lambda: ModelConfig(W_T_kind="projector"))
rejects(TypeError, lambda: ContinualConfig(revisit=[0, 1, 0]))

# P <= d-1 alone is not sufficient: a one-hot unit cannot predict itself from silent peers.
one_hot = torch.zeros(4, 1, dtype=torch.float64)
one_hot[0, 0] = 1.0
rejects(ValueError, lambda: build_memory(one_hot), "not representable")

# Builder outputs are typed and carry the memory-fit diagnostics.
_, two_info = build_system(ModelConfig(d=12, P=3))
assert isinstance(two_info, TwoPopBuildInfo)
assert isinstance(two_info.memory, MemoryBuildResult)
assert two_info.memory.representable

macro_i, inter_info = build_interleaved_synthesis(
    InterleavedConfig(d=12, rank1=2, rank2=2, geometry="shared", overlap=1)
)
assert isinstance(inter_info, InterleavedBuildInfo)

# Geometry and simulation budgets are checked before low-level tensor indexing or partial bouts.
rejects(
    ValueError,
    lambda: InterleavedConfig(d=5, rank1=4, rank2=4, geometry="oblique"),
    "rank1+rank2",
)
rejects(
    ValueError,
    lambda: simulate_interleaved(
        macro_i,
        SimConfig(n_steps=10, bout_steps=6, progress=False),
        inter_info,
    ),
    "exact multiple",
)

# Graph names, interface arity, and endpoint existence are checked at construction.
z = torch.zeros(3, 3, dtype=torch.float64)
rejects(
    ValueError,
    lambda: MacroNetwork(
        [Population("A", z.clone(), 1.0, 1.0), Population("A", z.clone(), 1.0, 1.0)],
        [],
    ),
    "unique",
)
rejects(
    ValueError,
    lambda: CouplingInterface(
        target="S", sources=["T"], alpha=[], pi_I=1.0, rho=-0.5
    ),
    "alpha length",
)
rejects(
    ValueError,
    lambda: MacroNetwork(
        [Population("S", z.clone(), 1.0, 1.0)],
        [
            CouplingInterface(
                target="S", sources=["missing"], alpha=[1.0], pi_I=1.0, rho=-0.5
            )
        ],
    ),
    "not a population",
)

print("VALIDATION TEST PASSED")
