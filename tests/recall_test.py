"""Self-check of the single-network associative-recall API (`src.recall`). Run with:
    conda run -n pytorch python tests/recall_test.py

Covers, on a small synthetic pattern set (no dataset download):
  (A) stored pictures lie in the memory manifold ker(M_op)  (covPCN is faithful);
  (B) closed-form clamped completion recovers a stored picture from a partial cue;
  (C) the recorded gradient-flow query descends F monotonically to the closed-form fixed point;
  (D) make_mask produces the advertised known-unit counts.
"""
import torch

from src import AssociativeMemory, make_mask

torch.manual_seed(0)
DT = torch.float64

# small "images": 12x12, P=8 random unit-norm pictures (P <= d-1)
SIDE, P = 12, 8
d = SIDE * SIDE
raw = torch.rand(d, P, dtype=DT)                 # non-negative, like pixels
patterns = raw / raw.norm(dim=0, keepdim=True)

mem = AssociativeMemory(patterns, pi=1.0, W_kind="covpcn")

# (A) patterns are memories: M_op m_p ~ 0, and the manifold has the expected dimension
res = (mem.M_op @ patterns).norm(dim=0).max().item()
print(f"(A) manifold dim={mem.manifold.shape[1]} (expect {P}); max ||M_op m_p||={res:.1e}")
assert mem.manifold.shape[1] == P, "memory manifold dim != number of independent patterns"
assert res < 1e-5, "stored pictures are not in ker(M_op)"

# (B) closed-form completion from a half cue recovers the exact stored picture
known = make_mask((SIDE, SIDE), "bottom", frac=0.5, dtype=DT)
errs = []
for p in range(P):
    xstar = mem.recall_steady(patterns[:, p], known)
    errs.append((xstar - patterns[:, p]).norm().item() / patterns[:, p].norm().item())
errs = torch.tensor(errs)
print(f"(B) clamped completion rel-err mean={errs.mean():.1e} max={errs.max():.1e}")
assert errs.max() < 1e-6, "clamped completion did not recover the stored picture"

# (C) gradient-flow query: monotone F, lands on the closed-form fixed point, moves onto the manifold
tr = mem.recall(patterns[:, 3], known, n_steps=3000, dt=0.2, tau=1.0, record_every=50)
mono = bool((tr.F[1:] <= tr.F[:-1] + 1e-9).all())
print(f"(C) F {tr.F[0]:.3f}->{tr.F[-1]:.2e} monotone={mono}; "
      f"occupancy {tr.occupancy[0]:.2f}->{tr.occupancy[-1]:.2f}; "
      f"||x_end - x_star||={(tr.X[-1] - tr.x_star).norm():.1e}")
assert mono, "free energy did not decrease monotonically"
assert tr.F[-1] < 1e-4 * tr.F[0], "free energy did not reach the manifold floor"
assert (tr.X[-1] - tr.x_star).norm() < 1e-6, "dynamics did not converge to the closed-form fixed point"
assert tr.occupancy[-1] > 0.999, "converged state is not inside the memory manifold"

# (D) mask counts
for kind, frac, exp in [("bottom", 0.5, d // 2), ("left", 0.25, (SIDE // 4) * SIDE)]:
    k = make_mask((SIDE, SIDE), kind, frac=frac, dtype=DT)
    print(f"(D) mask {kind} frac={frac}: known={int(k.sum())} (expect {exp})")
    assert int(k.sum()) == exp, "mask known-count wrong"

print("\nrecall_test: all checks passed.")
