"""Optional pointwise output nonlinearities for a `Population`.

A population predicts itself with `W f(x)`: the recurrent weights read the *transmitted output*
of each unit, not its state. `f = identity` (``act=None``) is the linear model the paper
analyses; ``"relu"`` makes the very same network piecewise linear — one linear model per
activation pattern, with `W_A` (inactive columns zeroed) in place of `W`.

Only these two are provided. Rectification is the biologically motivated choice (cortical
pyramidal cells are roughly threshold-linear) and it is also the only kind of nonlinearity that
stays meaningful here: it is positively homogeneous, `f(cx) = c f(x)` for `c > 0`, so it behaves
identically at every amplitude. A nonlinearity with an intrinsic scale (tanh, sigmoid, softplus)
is probed only near zero once the state is normalized — per-unit activity is `~r0/sqrt(d)` — and
so degenerates to the identity as `d` grows, which makes it a poor test of nonlinearity.

Each entry is a pair `(f, f_prime)` of elementwise functions. `f_prime` supplies the diagonal
Jacobian `diag(f'(x))` needed by the state rate (`J = I - W diag(f'(x))`) and by the Hebbian
rule (whose presynaptic factor is `f(x)`, not `x`). `resolve_activation` also accepts a custom
`(f, f_prime)` pair directly, so anything else can still be passed in without living here.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch

Activation = Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]

LINEAR_NAMES = {"identity", "linear", "none"}


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _identity_prime(x: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(x)


def _relu(x: torch.Tensor) -> torch.Tensor:
    return x.clamp_min(0.0)


def _relu_prime(x: torch.Tensor) -> torch.Tensor:
    # subgradient 0 at x == 0, matching torch.relu's autograd convention
    return (x > 0).to(x.dtype)


_REGISTRY: dict = {
    "identity": (_identity, _identity_prime),
    "linear": (_identity, _identity_prime),
    "none": (_identity, _identity_prime),
    "relu": (_relu, _relu_prime),
}

ActivationSpec = Union[None, str, Activation]


def available_activations() -> tuple:
    """Names accepted by `resolve_activation`."""
    return tuple(sorted(_REGISTRY))


def is_linear(act: ActivationSpec) -> bool:
    """True when `act` denotes the identity, i.e. the linear model the paper analyses."""
    if act is None:
        return True
    if isinstance(act, str):
        return act.lower() in LINEAR_NAMES
    return False


def resolve_activation(act: ActivationSpec) -> Activation:
    """Return `(f, f_prime)` for an activation spec.

    Accepts `None` / ``"identity"`` / ``"linear"`` (the linear model), a registered name (see
    `available_activations()`), or an `(f, f_prime)` pair of elementwise callables.
    """
    if act is None:
        return _REGISTRY["identity"]
    if isinstance(act, str):
        key = act.lower()
        if key not in _REGISTRY:
            raise ValueError(
                f"unknown activation {act!r}; expected one of {available_activations()} "
                "or a (f, f_prime) pair of callables."
            )
        return _REGISTRY[key]
    if isinstance(act, tuple) and len(act) == 2 and all(callable(c) for c in act):
        return act  # type: ignore[return-value]
    raise ValueError(
        f"activation must be None, one of {available_activations()}, or a (f, f_prime) pair of callables "
        f"(got {act!r})."
    )
