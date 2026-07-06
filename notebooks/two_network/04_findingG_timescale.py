# %% [markdown]
# # Selection is spectral, not a "race in time"
#
# Known-vs-novel discrimination is decided by whether the student's weights can *null* a
# direction (a spectral fact via `N_S`), provided the student is fast (`τ_S ≪ τ_T`). A
# genuinely speed-dependent "race" only re-emerges as `τ_S → τ_T`.
#
# **Test.** Pretrain the student on a subset (the *known* memories). For each ratio `τ_S/τ_T`,
# clamp the teacher at a known pattern and at a novel pattern, relax the student for one
# T-timescale, and record the residual mismatch `‖ε_TS‖`. Prediction: in the fast-S regime a
# known pattern is explained away (`‖ε_TS‖→0`) while a novel one is not
# (`→ n(1)=π_S/(π_TS+π_S)`), giving large discrimination; as `τ_S→τ_T`, the student can no
# longer keep up even on known patterns, so the discrimination collapses.

# %% setup
import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

_root = pathlib.Path.cwd()
while not (_root / "pmt").exists() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from pmt import ModelConfig
from pmt import experiments as ex

sns.set_theme(context="notebook", style="whitegrid")

base = ModelConfig(d=32, P=4, pi_TS=1.0, pi_S=0.5, seed=2)
n1 = base.pi_S / (base.pi_TS + base.pi_S)     # novel-direction residual in the fast limit

# %% sweep tau_S/tau_T
ratios = np.geomspace(0.01, 1.0, 16)
r, res_known, res_novel = ex.timescale_discrimination(
    base, ratios, known_idx=[0, 1], novel_idx=[2, 3], tau_T=10.0)
discrimination = res_novel - res_known

# %% figure
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

ax[0].semilogx(r, res_novel, "o-", c="C3", lw=1.8, label="novel pattern  ‖ε_TS‖")
ax[0].semilogx(r, res_known, "o-", c="C0", lw=1.8, label="known pattern  ‖ε_TS‖")
ax[0].axhline(n1, ls=":", c="C3", alpha=0.7, label=f"spectral limit n(1)={n1:.2f}")
ax[0].axhline(0.0, ls=":", c="C0", alpha=0.7, label="spectral limit 0 (explained away)")
ax[0].set(title="Residual mismatch after one T-timescale of S relaxation",
          xlabel=r"$\tau_S/\tau_T$  (S slower →)", ylabel="‖ε_TS‖")
ax[0].legend(fontsize=8)

ax[1].semilogx(r, discrimination, "o-", c="C4", lw=2)
ax[1].set(title="Known/novel discrimination  (novel − known)",
          xlabel=r"$\tau_S/\tau_T$", ylabel="discrimination")
ax[1].annotate("sharp & flat\n(spectral regime)", xy=(0.02, discrimination[1]),
               xytext=(0.02, discrimination[1] * 0.6), fontsize=9, color="C4")
ax[1].annotate("collapses as\nτ_S → τ_T", xy=(0.7, discrimination[-1]),
               xytext=(0.2, discrimination[-1] * 0.5 + 0.05),
               arrowprops=dict(arrowstyle="->"), fontsize=9, color="C4")
fig.suptitle("Spectral selection in the fast-S regime, dynamic only near τ_S≈τ_T",
             y=1.02, fontsize=13)
fig.tight_layout()
fig

# %% verdict
flat = np.std(discrimination[r < 0.15]) < 0.05
collapse = discrimination[-1] < 0.5 * discrimination[0]
ok = flat and collapse
print(f"discrimination(fast) ≈ {discrimination[0]:.3f}, (τ_S≈τ_T) ≈ {discrimination[-1]:.3f}")
print(f"\n[{'CONFIRMED' if ok else 'CHECK'}] discrimination is high and ~flat across the fast-S "
      "regime (known explained away ⇒ spectral) and collapses only as τ_S→τ_T — so selection is "
      "structural, not a speed race.")
