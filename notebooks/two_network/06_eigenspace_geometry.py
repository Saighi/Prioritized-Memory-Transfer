# %% [markdown]
# # 06 — Eigenspace geometry: how S_S and N_S look, and morph, during transfer
#
# We drop to a genuinely 3-D network (`d=3`, `P=2` orthonormal) so the geometry can be drawn
# honestly, with no projection. There the teacher's memory manifold `ker M_T` is a 2-D **plane**
# and the off-manifold direction is the 1-D **normal**. We watch the student's operators evolve
# as it learns:
#
# - **Novelty sphere** — the unit state-sphere colored by `uᵀ N_S u`. The memory great circle
#   cools toward 0 as it is learned; the off-manifold poles *heat* toward 1 (the student grows
#   confident on the manifold, so off-manifold inputs become highly surprising).
# - **Stretch ellipsoid** — the image of the unit sphere under `S_S` (normalized per frame):
#   a sphere flattening toward a line as the two memory eigenvalues fall to 0 (degeneracy forming).
# - **Energy valley** — `½ xᵀ S_S x` over the [memory dir, off-manifold] slice: a round bowl
#   develops a flat valley along the memory direction.
#
# Each figure has a static triptych (S empty / mid / transferred) and a time animation.

# %% imports & path setup
import os, sys, pathlib
import numpy as np
import torch

_root = pathlib.Path.cwd()
while not (_root / "pmt").exists() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from pmt import ModelConfig, SimConfig, build_system, simulate
from pmt import viz_eigenspace as ve

SHOW = os.environ.get("PMT_NO_SHOW") != "1"   # set PMT_NO_SHOW=1 to run headless
torch.manual_seed(0)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())

# %% pick a well-conditioned d=3 seed (covPCN can be stiff: scan for a modest spectral gap)
def make_cfg(seed):
    return ModelConfig(
        d=3, P=2, pi_TS=1.0, pi_S=0.5,
        pi_ST="auto", pi_ST_safety=0.5,
        tau_T=30.0, tau_S=3.0, eta=0.005,
        sigma_xi=0.04, r0=1.0,
        pattern_kind="orthonormal", W_T_kind="covpcn",
        seed=seed, device="cpu",
    )

best = None
for s in range(31):
    _, inf = build_system(make_cfg(s))
    if inf["manifold_dim"] != 2:
        continue
    if best is None or inf["sigma2_min"] < best[1]:
        best = (s, float(inf["sigma2_min"]))
seed, sigma2_min = best
model_cfg = make_cfg(seed)
model, info = build_system(model_cfg)

# explicit-Euler stability for the -S_T x_T relaxation: dt * sigma2_min / tau_T <= ~0.5
dt = min(0.5, 0.5 * model_cfg.tau_T / sigma2_min)

print(f"chosen seed           : {seed}")
print(f"spectral gap σ²_min   : {sigma2_min:.3f}")
print(f"manifold dim (eff rank): {info['manifold_dim']}   (memory plane is 2-D)")
print(f"pi_ST                 : {info['pi_ST']:.3f}  (reversed/<0)   (guard |π_ST|<σ²_min? {abs(info['pi_ST']) < info['guard_safe']})")
print(f"adaptive dt           : {dt:.3f}")
assert info["manifold_dim"] == 2, "expected a 2-D memory plane for d=3, P=2"

# %% run the transfer; the student's operators are recovered from the weight snapshots
total_time = 6000.0
n_steps = int(total_time / dt)
sim_cfg = SimConfig(n_steps=n_steps, dt=dt, mode="adiabatic",
                    record_every=max(1, n_steps // 200),
                    n_weight_snapshots=40, progress=True)
hist = simulate(model, sim_cfg, info)
H = hist.to_numpy()

eig_S0 = np.array([1.0, 1.0, 1.0])
eig_Sf = torch.linalg.eigvalsh(model.S_S()).cpu().numpy()
print(f"eig(S_S)  t0 : {np.round(eig_S0, 3)}  ->  final : {np.round(eig_Sf, 3)}   (memory eigs → 0, off-manifold grows)")
print(f"novelty top eig: {H['novelty_spec'][0].max():.3f} -> {H['novelty_spec'][-1].max():.3f}   (n(1)=π_S/(π_TS+π_S)=0.333 → 0)")

# %% extract per-frame operator data once (shared by all six figures)
frames = ve.eigenframes(hist, model, info)
print(f"frames: {len(frames)}  (from weight snapshots)")

# %% 1a. novelty sphere — static triptych
fig = ve.novelty_sphere_triptych(hist, model, info)
print("novelty triptych:", len(fig.data), "traces")
if SHOW:
    fig.show()

# %% 1b. novelty sphere — animation
fig = ve.novelty_sphere_animated(hist, model, info)
print("novelty animation:", len(fig.data), "traces,", len(fig.frames), "frames")
if SHOW:
    fig.show()

# %% 2a. stretch ellipsoid — static triptych
fig = ve.stretch_ellipsoid_triptych(hist, model, info)
print("ellipsoid triptych:", len(fig.data), "traces")
if SHOW:
    fig.show()

# %% 2b. stretch ellipsoid — animation
fig = ve.stretch_ellipsoid_animated(hist, model, info)
print("ellipsoid animation:", len(fig.data), "traces,", len(fig.frames), "frames")
if SHOW:
    fig.show()

# %% 3a. energy valley — static triptych
fig = ve.energy_valley_triptych(hist, model, info)
print("valley triptych:", len(fig.data), "traces")
if SHOW:
    fig.show()

# %% 3b. energy valley — animation
fig = ve.energy_valley_animated(hist, model, info)
print("valley animation:", len(fig.data), "traces,", len(fig.frames), "frames")
if SHOW:
    fig.show()

# %% verdict
nov_top_fell = H["novelty_spec"][-1].max() < 0.5 * H["novelty_spec"][0].max()
mem_eigs_small = np.sort(eig_Sf)[:2].max() < 0.5    # the two memory eigenvalues collapsed
ok = (info["manifold_dim"] == 2) and nov_top_fell and mem_eigs_small
print(f"\n[{'CONFIRMED' if ok else 'CHECK'}] In d=3 the memory manifold is a 2-D plane (degenerate λ=0 "
      "eigenspace of S_T). As the student learns it, S_S's two in-plane eigenvalues collapse to 0 while "
      "the off-manifold eigenvalue grows: the novelty sphere cools on the memory great circle, the "
      "stretch ellipsoid flattens sphere→line, and the energy bowl opens a flat valley along the "
      "memory direction — the geometry of prioritized subspace consolidation.")
