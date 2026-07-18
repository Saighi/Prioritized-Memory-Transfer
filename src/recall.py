"""src.recall — one linear associative memory, queried by clamping units.

The standalone, single-network counterpart to the transfer models. ONE predictive-coding
population stores a set of pictures as memories in its flat memory manifold and completes a
partial cue by minimizing its free energy.

The physics (exactly the single-population special case of `src.macro`, no interfaces):

  - Weights `W` are built from the stored pictures `M = [m_1 ... m_P]` (columns) with the same
    zero-diagonal covariance-PCN / projector construction the rest of the package uses
    (`src.memory.build_W_T`). The mismatch operator is `M_op = I - W` and the self-surprise
    operator is `S = M_op^T M_op`.
  - Every stored picture lies in the memory manifold `ker(M_op)` (its self prediction error is
    ~0), so the network's free energy

        F(x) = 0.5 * pi * || M_op x ||^2

    is a smooth quadratic bowl whose zero-floor is the memory manifold.
  - Recall clamps the *known* units to a partial cue and lets the *free* units descend F by
    projected gradient flow

        tau * xdot = -pi * S x ,   x_known held fixed,

    so the state slides downhill onto the manifold, completing the picture. F decreases
    monotonically to ~0 and the trajectory settles at the stored memory consistent with the cue.

`AssociativeMemory` wraps the construction, the energy, the closed-form completion, and the
recorded gradient-flow query. `make_mask` builds the occlusion patterns (which units are known).
Kept separate from the transfer engine so the didactic recall notebook has a small, tested API.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from .memory import build_W_T


# --------------------------------------------------------------------------- occlusion masks
def make_mask(
    shape: Tuple[int, int],
    kind: str = "bottom",
    frac: float = 0.5,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> torch.Tensor:
    """Return a boolean vector `known` of length `h*w` (row-major) — True where a unit is CLAMPED
    to the cue, False where it is free to be filled in.

    `kind`:
      - "top"/"bottom"/"left"/"right": that fraction `frac` of rows/columns is known;
      - "center": a central box covering `frac` of the area is known;
      - "border": the complementary frame is known (center is hidden);
      - "random": a random fraction `frac` of pixels is known (reproducible via `seed`).
    """
    h, w = shape
    known = torch.zeros(h, w, dtype=torch.bool, device=device)
    if kind == "top":
        known[: max(1, int(round(frac * h))), :] = True
    elif kind == "bottom":
        known[h - max(1, int(round(frac * h))) :, :] = True
    elif kind == "left":
        known[:, : max(1, int(round(frac * w)))] = True
    elif kind == "right":
        known[:, w - max(1, int(round(frac * w))) :] = True
    elif kind in ("center", "border"):
        ih = int(round((1.0 - frac) * h / 2.0))
        iw = int(round((1.0 - frac) * w / 2.0))
        known[ih : h - ih, iw : w - iw] = True
        if kind == "border":
            known = ~known
    elif kind == "random":
        gen = torch.Generator(device=device).manual_seed(seed)
        known = torch.rand(h, w, generator=gen, device=device) < frac
    else:
        raise ValueError(f"unknown mask kind={kind!r}")
    return known.reshape(-1)


@dataclass
class RecallTrace:
    """Recorded gradient-flow recall. All arrays are torch tensors on the memory's device.

    steps        (T,)      integration step index at each record
    X            (T, d)    full state x(t)
    F            (T,)      free energy F(x(t))  (monotone decreasing)
    resid        (T,)      ||M_op x(t)||  (distance-to-manifold, sqrt(2F/pi))
    overlap      (T, P)    cosine of x(t) with each stored picture
    occupancy    (T,)      fraction of x(t)'s norm lying inside the memory manifold
    x0           (d,)      initial (masked) state
    x_star       (d,)      closed-form completion (the fixed point)
    known        (d,)      clamp mask used
    """

    steps: torch.Tensor
    X: torch.Tensor
    F: torch.Tensor
    resid: torch.Tensor
    overlap: torch.Tensor
    occupancy: torch.Tensor
    x0: torch.Tensor
    x_star: torch.Tensor
    known: torch.Tensor


class AssociativeMemory:
    """A single linear covariance-PCN network storing pictures `patterns` (a (d, P) matrix of
    columns) as memories, queried by clamping units.

    `pi` is the self-precision (an overall gain on F; it does not change the fixed point).
    `W_kind` is "covpcn" (the faithful learned, zero-diagonal weights the project uses) or
    "projector" (the ideal zero-diagonal projector onto span(patterns)).
    """

    def __init__(
        self,
        patterns: torch.Tensor,
        pi: float = 1.0,
        W_kind: str = "covpcn",
        ridge: float = 1e-8,
        tol: float = 1e-6,
    ) -> None:
        self.patterns = patterns                       # (d, P)
        self.d, self.P = patterns.shape
        self.pi = float(pi)
        self.dtype = patterns.dtype
        self.device = patterns.device
        self.I = torch.eye(self.d, dtype=self.dtype, device=self.device)

        # zero-diagonal weights from the stored pictures (reuse the package builder)
        self.W = build_W_T(patterns, SimpleNamespace(W_T_kind=W_kind), ridge=ridge)
        self.M_op = self.I - self.W                    # mismatch operator
        self.S = self.M_op.transpose(-2, -1) @ self.M_op   # self-surprise operator

        # orthonormal basis of the stored-picture span (the interpretable "memory subspace")
        U, s, _ = torch.linalg.svd(patterns, full_matrices=False)
        self.pattern_span = U[:, s > tol * s.max()]    # (d, r), r = rank of the stored set

        # orthonormal basis of the actual zero-floor manifold ker(M_op)
        evals, evecs = torch.linalg.eigh(self.S)
        self.manifold = evecs[:, evals < tol]          # (d, m)

    # ---------------------------------------------------------------- energies / geometry
    def free_energy(self, x: torch.Tensor) -> torch.Tensor:
        """F(x) = 0.5 pi ||M_op x||^2 (supports an optional leading batch dim)."""
        eps = x @ self.M_op.transpose(-2, -1)
        return 0.5 * self.pi * (eps ** 2).sum(-1)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """||M_op x|| — Euclidean distance from x to the memory manifold."""
        return (x @ self.M_op.transpose(-2, -1)).norm(dim=-1)

    def occupancy(self, x: torch.Tensor) -> torch.Tensor:
        """Fraction of ||x|| lying inside ker(M_op) (1.0 == fully on the manifold)."""
        proj = (x @ self.manifold) @ self.manifold.transpose(-2, -1)
        return proj.norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-12)

    def overlap(self, x: torch.Tensor) -> torch.Tensor:
        """Cosine of x with each stored picture -> (..., P)."""
        cols = self.patterns / self.patterns.norm(dim=0, keepdim=True).clamp_min(1e-12)
        return (x @ cols) / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def project(self, x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """Coordinates of x in the columns of `basis` (d, k) -> (..., k)."""
        return x @ basis

    # ------------------------------------------------------------------ clamped recall
    def recall_steady(self, cue: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        """Closed-form completion: the state that minimizes F with `x_known == cue_known`.

        Splitting S into free (u) / known (k) blocks, the stationary condition
        (S x)_u = 0 gives  S_uu x_u = -S_uk cue_k.
        """
        u = (~known).nonzero(as_tuple=True)[0]
        k = known.nonzero(as_tuple=True)[0]
        x = cue.clone()
        if u.numel() == 0:
            return x
        S_uu = self.S.index_select(0, u).index_select(1, u)
        S_uk = self.S.index_select(0, u).index_select(1, k)
        rhs = -S_uk @ cue.index_select(0, k)
        x[u] = torch.linalg.solve(S_uu, rhs)
        return x

    def recall(
        self,
        cue: torch.Tensor,
        known: torch.Tensor,
        *,
        x0: Optional[torch.Tensor] = None,
        fill: float = 0.0,
        n_steps: int = 4000,
        dt: float = 0.2,
        tau: float = 1.0,
        record_every: int = 20,
    ) -> RecallTrace:
        """Run the projected gradient-flow query and record the trajectory.

        Dynamics:  x <- x - (dt*pi/tau) * S x,  then re-impose x_known = cue_known each step.
        The known units are clamped to the cue; the free units descend F onto the manifold.
        `x0` overrides the initial free-unit values (default: `fill`, i.e. a blank cue).
        """
        cue = cue.to(self.dtype)
        if x0 is None:
            x = torch.full((self.d,), float(fill), dtype=self.dtype, device=self.device)
            x[known] = cue[known]
        else:
            x = x0.clone().to(self.dtype)
            x[known] = cue[known]
        x0_rec = x.clone()

        coeff = dt * self.pi / tau
        steps, X, F, resid, overlap, occ = [], [], [], [], [], []

        def record(i):
            steps.append(i)
            X.append(x.clone())
            F.append(self.free_energy(x))
            resid.append(self.residual(x))
            overlap.append(self.overlap(x))
            occ.append(self.occupancy(x))

        record(0)
        for i in range(1, n_steps + 1):
            x = x - coeff * (x @ self.S)              # S symmetric: x@S == S x
            x[known] = cue[known]                     # clamp
            if i % record_every == 0 or i == n_steps:
                record(i)

        return RecallTrace(
            steps=torch.tensor(steps, device=self.device),
            X=torch.stack(X),
            F=torch.stack(F),
            resid=torch.stack(resid),
            overlap=torch.stack(overlap),
            occupancy=torch.stack(occ),
            x0=x0_rec,
            x_star=self.recall_steady(cue, known),
            known=known,
        )
