"""Headless check that all figures build. Run:
    conda run -n pytorch python tests/viz_test.py
Outputs PNGs + one HTML under tests/ for eyeballing.
"""
import matplotlib
matplotlib.use("Agg")

from src import ModelConfig, SimConfig, build_system, simulate
from src import viz_static as vs
from src import viz_interactive as vi

model, info = build_system(ModelConfig(d=40, P=5, seed=1))
hist = simulate(
    model,
    SimConfig(n_steps=12000, dt=0.5, mode="adiabatic", record_every=100,
              n_weight_snapshots=5, progress=False),
    info,
)

fig = vs.dashboard(hist, model, info)
fig.savefig("tests/_dashboard.png", dpi=80, bbox_inches="tight")
fig2 = vs.weight_snapshots(hist, model)
fig2.savefig("tests/_weights.png", dpi=80, bbox_inches="tight")

f3 = vi.staircase(hist)
f4 = vi.raster(hist)
f5 = vi.trajectory_3d(hist, model, info)
f5.write_html("tests/_traj.html")

print("VIZ OK  static panels saved; plotly trace counts:",
      [len(f.data) for f in (f3, f4, f5)], "frames:", len(f5.frames))

# --- eigenspace-geometry figures (tiny d=3 network: plane=manifold, line=off-manifold) ---
from src import viz_eigenspace as ve

_best = None
for s in range(11):
    _, _inf3 = build_system(ModelConfig(d=3, P=2, seed=s))
    if _inf3["manifold_dim"] == 2 and (_best is None or _inf3["sigma2_min"] < _best[1]):
        _best = (s, float(_inf3["sigma2_min"]))
seed3, s2min = _best
m3, info3 = build_system(ModelConfig(d=3, P=2, seed=seed3))
dt3 = min(0.5, 0.5 * 10.0 / s2min)                      # explicit-Euler stability for -S_T x_T
h3 = simulate(m3, SimConfig(n_steps=int(2000 / dt3), dt=dt3, mode="adiabatic",
              record_every=max(1, int(2000 / dt3) // 40), n_weight_snapshots=12, progress=False), info3)

eig_figs = [ve.novelty_sphere_triptych(h3, m3, info3), ve.novelty_sphere_animated(h3, m3, info3),
            ve.stretch_ellipsoid_triptych(h3, m3, info3), ve.stretch_ellipsoid_animated(h3, m3, info3),
            ve.energy_valley_triptych(h3, m3, info3), ve.energy_valley_animated(h3, m3, info3)]
ve.novelty_sphere_animated(h3, m3, info3).write_html("tests/_eig_novelty.html")
assert all(len(f.data) > 0 for f in eig_figs), "an eigenspace figure has no traces"
assert all(len(eig_figs[i].frames) > 0 for i in (1, 3, 5)), "animated eigenspace figs need frames"
print("EIGViz OK  d=3 seed", seed3, "| traces:", [len(f.data) for f in eig_figs],
      "| frames:", [len(eig_figs[i].frames) for i in (1, 3, 5)])
