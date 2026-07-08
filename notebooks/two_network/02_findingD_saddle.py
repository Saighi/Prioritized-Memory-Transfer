# %% [markdown]
# # The saddle is exact only when π_ST = −π_TS
#
# In the sleep/replay regime the teacher uses a **reversed (negative) precision** `π_ST < 0` on
# the interface error. The coupled `(x_T, x_S)` flow is the gradient flow of a single potential
# `Φ = F_T − F_S` (the teacher descends, the student ascends) **only when the teacher's reversed
# precision exactly mirrors the student's: `π_ST = −π_TS`** (a zero-sum game). Otherwise there is
# no such potential: the flow has a rotational ("circulation") component a height function can
# never produce.
#
# **Test.** The saddle mismatch is `C = ‖ ∂f_T/∂x_S + (∂f_S/∂x_T)ᵀ ‖`. The math predicts
# `C = |π_ST + π_TS|·√d` exactly — a perfect V bottoming at `π_ST = −π_TS`. We sweep the signed
# precision `π_ST` (the reversed-precision axis), compute `C` by autograd, and overlay the theory.

# %% setup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pmt import ModelConfig
from pmt import experiments as ex

sns.set_theme(context="notebook", style="whitegrid")

# %% sweep the signed pi_ST and compare to theory
base = ModelConfig(d=48, P=6, pi_TS=1.0, pi_S=0.5, seed=0)
pi_STs = np.linspace(-2.5, 0.5, 41)        # the reversed-precision axis (sleep is pi_ST < 0)
k, emp, theo = ex.sweep_circulation(base, pi_STs)

max_err = float(np.max(np.abs(emp - theo)))
print(f"max |empirical − theory|  = {max_err:.2e}   (machine precision ⇒ exact match)")
print(f"empirical circulation at π_ST=−π_TS={-base.pi_TS}: {emp[np.argmin(abs(k+base.pi_TS))]:.2e}")

# %% figure
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

ax[0].plot(k, theo, "-", c="C0", lw=2, label=r"theory  $|\pi_{ST}+\pi_{TS}|\sqrt{d}$")
ax[0].plot(k, emp, "o", c="C3", ms=5, label="simulation (autograd)")
ax[0].axvline(-base.pi_TS, ls="--", c="k", alpha=0.5)
ax[0].annotate("π_ST = −π_TS\n(exact saddle)", xy=(-base.pi_TS, 0), xytext=(-base.pi_TS + 0.15, 1.5),
               arrowprops=dict(arrowstyle="->"), fontsize=9)
ax[0].set(title="Saddle mismatch (circulation) vs the reversed precision π_ST",
          xlabel="π_ST  (reversed precision, sleep < 0)", ylabel=r"$C$")
ax[0].legend()

# residual of the match (log scale) — flat at machine epsilon
ax[1].semilogy(k, np.abs(emp - theo) + 1e-18, "o-", c="C2", ms=4)
ax[1].set(title="|simulation − theory|  (≈ machine epsilon everywhere)",
          xlabel="π_ST", ylabel="absolute error")
fig.suptitle("Single-potential saddle holds iff π_ST = −π_TS", y=1.02, fontsize=13)
fig.tight_layout()
fig

# %% verdict
verdict = "CONFIRMED" if max_err < 1e-9 else "CHECK"
print(f"\n[{verdict}] circulation = |π_ST+π_TS|·√d to {max_err:.1e}; it vanishes only at π_ST=−π_TS, "
      "so the exact single-functional saddle exists only where the teacher's reversed precision "
      "is the exact negative of the student's.")
