# %% [markdown]
# # Dreaming MNIST — replay as a movie
#
# Store `N_MEMORIES` MNIST digits in the teacher, run the sleep/replay transfer, and animate the
# teacher's activity `f(x_T)` as an image sequence. That sequence *is* the dream: the teacher is
# wandering its memory manifold, pushed by the reversed precision toward whatever the student has
# not yet learned, so the frames should drift between stored digits and blends of them.
#
# Set up to be swapped: `ACTIVATION` and `N_MEMORIES` are the two knobs. `N_STEPS` is a plain
# number — no automatic scaling with `P`, tune it yourself.
#
# **Speed notes.** The cost is dominated by `build_memory`, which does one least-squares fit per
# neuron: `d` solves of a `(d-1)×(d-1)` system. At `SIDE=14` (`d=196`) that is a second or two; at
# `SIDE=28` (`d=784`) it is minutes. The replay loop below is deliberately hand-rolled instead of
# using `History`, because `History` records the novelty spectrum every step, which inverts a
# `d×d` matrix each time — pure waste here (and it is the diagnostic we discarded anyway).

# %% imports & setup
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

import matplotlib
if os.environ.get("PMT_NO_SHOW") == "1":
    matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from prioritized_memory_transfer import ModelConfig
from prioritized_memory_transfer.activations import resolve_activation
from prioritized_memory_transfer.memory import build_memory
from prioritized_memory_transfer.model import TwoPopModel

SHOW = os.environ.get("PMT_NO_SHOW") != "1"
_root = Path(__file__).resolve().parents[2]
DT = torch.float64
torch.manual_seed(0)

# ------------------------------------------------------------------ knobs
ACTIVATION = "relu"        # "relu" | None (linear)
N_MEMORIES = 3             # how many digits to store (needs N_MEMORIES <= d-1)
SIDE = 14                  # 14 -> d=196 (fast) ; 28 -> d=784 (slow build)
N_STEPS = 100000            # plain knob, no auto-scaling
DT_STEP = 0.5              # integration step
N_FRAMES = 200             # frames in the movie
FPS = 20
OUT = Path(__file__).with_name(f"dreams_{ACTIVATION or 'linear'}_{N_MEMORIES}.gif")

d = SIDE * SIDE
assert N_MEMORIES <= d - 1, f"need N_MEMORIES <= d-1 = {d-1}"

# %% [markdown]
# ## 1. Load the digits
#
# MNIST average-pooled to `SIDE x SIDE`, one digit per memory cycling through the classes. Pixel
# intensities are **nonnegative**, so for a rectifying activation `f(m_p) = m_p` holds
# automatically — MNIST is a natural fit for the relu case with no extra work.

# %% load
WANT = [i % 10 for i in range(N_MEMORIES)]     # cycle 0..9, repeating as needed


def _load_mnist():
    import torchvision
    ds = torchvision.datasets.MNIST(root=str(_root / "data"), train=True, download=True)
    imgs, labels = ds.data.to(DT) / 255.0, ds.targets
    sel, used = [], {}
    for i in range(imgs.shape[0]):
        lb = int(labels[i])
        if WANT.count(lb) > used.get(lb, 0):
            sel.append(i); used[lb] = used.get(lb, 0) + 1
        if len(sel) == len(WANT):
            break
    X = Fn.avg_pool2d(imgs[sel].unsqueeze(1), 28 // SIDE).squeeze(1)
    return X.reshape(len(sel), -1).T, [int(labels[i]) for i in sel]


def _load_digits_fallback():
    from sklearn.datasets import load_digits
    dg = load_digits()
    imgs = torch.tensor(dg.images, dtype=DT) / 16.0
    sel, used = [], {}
    for i in range(imgs.shape[0]):
        lb = int(dg.target[i])
        if WANT.count(lb) > used.get(lb, 0):
            sel.append(i); used[lb] = used.get(lb, 0) + 1
        if len(sel) == len(WANT):
            break
    return imgs[sel].reshape(len(sel), -1).T, [int(dg.target[i]) for i in sel]


try:
    raw, labels = _load_mnist()
except Exception as exc:                       # no torchvision / no download
    print("MNIST unavailable, falling back to sklearn 8x8 digits:", repr(exc))
    SIDE = 8; d = SIDE * SIDE
    raw, labels = _load_digits_fallback()

patterns = raw / raw.norm(dim=0, keepdim=True)          # (d, P) unit-norm columns
P = patterns.shape[1]
print(f"stored P={P} digits {labels} | d={d} units ({SIDE}x{SIDE}) | "
      f"RMS activity per unit {float(patterns.pow(2).mean(0).sqrt().mean()):.3f}")

img = lambda v: np.asarray(v.detach()).reshape(SIDE, SIDE)

fig0, axs = plt.subplots(1, P, figsize=(1.15 * P, 1.5))
for p in range(P):
    axs[p].imshow(img(patterns[:, p]), cmap="magma")
    axs[p].set_title(str(labels[p]), fontsize=9)
    axs[p].axis("off")
fig0.suptitle("stored memories", y=1.06)
fig0.tight_layout()
fig0

# %% [markdown]
# ## 2. Build the two-population system
#
# `build_system` generates its own random patterns, so it is bypassed here: fit the teacher
# directly on the digits with `build_memory`, then hand the weights to `TwoPopModel`.

# %% build
cfg = ModelConfig(
    d=d, P=P,
    pi_TS=1.0, pi_S=0.5,
    pi_ST=-0.5,                     # fixed precisions, no "auto"
    tau_T=10.0, tau_S=1.0, eta=0.005,
    sigma_xi=0.05, r0=1.0,
    pattern_kind="nonneg", activation=ACTIVATION,
    seed=0, device="cpu", dtype=DT,
)
mem = build_memory(patterns, act=ACTIVATION)     # the slow step; see the speed note above
model = TwoPopModel(mem.W, cfg, pi_ST=float(cfg.pi_ST), patterns=patterns)
f_act, _ = resolve_activation(ACTIVATION)

resid = float((patterns - mem.W @ f_act(patterns)).norm(dim=0).max())
print(f"teacher fit: max_p ||m_p - W_T f(m_p)|| = {resid:.2e}  (memories are fixed points)")
print(f"f(m_p) == m_p ? {bool(torch.allclose(f_act(patterns), patterns, atol=1e-10))}")

# %% [markdown]
# ## 3. Replay
#
# Hand-rolled loop: step the engine, keep `x_T` every `stride` steps. Nothing else is recorded, so
# this is as fast as the dynamics allow.

# %% run
gen = torch.Generator().manual_seed(cfg.seed + 12345)
model.reset_state(gen)
stride = max(1, N_STEPS // N_FRAMES)
frames, energies = [], []
for step in range(N_STEPS):
    model.macro.step(DT_STEP, gen, mode="full")
    if step % stride == 0:
        frames.append(model.x_T.detach().clone())
        energies.append(float(model.pattern_energy().max()))
X = torch.stack(frames)                                   # (F, d) raw states
D = f_act(X)                                              # (F, d) what the units transmit
coef = np.linalg.lstsq(patterns.numpy(), X.numpy().T, rcond=None)[0].T   # (F, P)
print(f"{len(frames)} frames | worst memory F_S {energies[0]:.4f} -> {energies[-1]:.4f}")

# %% [markdown]
# ## 4. The movie
#
# Left: the dream, i.e. what the teacher's units transmit, `f(x_T)`. Right: the memory
# coordinates of that state, so you can read which stored digit (or blend) is being replayed.
# relu gives nonnegative frames; the linear model can go negative, so the
# colour scale switches to a diverging map automatically.

# %% animate
Dn = D.numpy()
if Dn.min() < -1e-9:
    cmap, vmin, vmax = "RdBu_r", -np.abs(Dn).max(), np.abs(Dn).max()
else:
    cmap, vmin, vmax = "magma", 0.0, Dn.max()
cmax = float(np.abs(coef).max())

fig, (axi, axb) = plt.subplots(1, 2, figsize=(8.4, 3.6),
                               gridspec_kw={"width_ratios": [1, 1.5]})
im = axi.imshow(Dn[0].reshape(SIDE, SIDE), cmap=cmap, vmin=vmin, vmax=vmax)
axi.axis("off")
bars = axb.bar(range(P), coef[0], color="C0")
axb.axhline(0, c="k", lw=0.8, alpha=0.5)
axb.set(ylim=(-cmax * 1.05, cmax * 1.05), xlabel="stored memory", ylabel="$c_p$")
axb.set_xticks(range(P))
axb.set_xticklabels([str(l) for l in labels], fontsize=8)
title = fig.suptitle("")


def update(k):
    im.set_data(Dn[k].reshape(SIDE, SIDE))
    for p, b in enumerate(bars):
        b.set_height(coef[k, p])
        b.set_color("C3" if coef[k, p] < 0 else "C0")
    best = int(np.argmax(np.abs(coef[k])))
    title.set_text(f"{ACTIVATION or 'linear'} dream — frame {k}/{len(Dn)}   "
                   f"closest memory: {labels[best]}  ($c$={coef[k, best]:+.2f})")
    return [im, title, *bars]


anim = animation.FuncAnimation(fig, update, frames=len(Dn), interval=1000 // FPS, blit=False)
anim.save(OUT, writer=animation.PillowWriter(fps=FPS))
print(f"saved {OUT}")
plt.close(fig)

# %% show inline (VS Code interactive window)
if SHOW:
    from IPython.display import Image, display
    display(Image(filename=str(OUT)))

# %% [markdown]
# ## 5. Can the student be queried?
#
# The dream shows what the teacher replays; this shows whether the **student** actually stored it.
# Occlude part of each digit, clamp the visible pixels, let the student relax its own free energy
# (`model.recall`), and look at what it fills in. Clamping is what makes the query well posed —
# the memory manifold is degenerate, so an unclamped relaxation has no preferred destination on it.

# %% recall
from prioritized_memory_transfer import make_mask

MASK_KIND = "bottom"       # "bottom" | "top" | "left" | "right" | "center" | "random"
SHOW_FRAC = 0.5            # fraction revealed in the picture below


def complete(m, known):
    """Clamp the visible pixels to the cue, start blank, settle the student alone."""
    return model.recall(torch.zeros(d, dtype=m.dtype), known=known, cue=m)


print(f"landing distance  ||x_inf - m_p|| / ||m_p||   ({MASK_KIND} mask)")
for frac in (0.25, 0.5, 0.75):
    kn = make_mask((SIDE, SIDE), MASK_KIND, frac)
    err = [float((complete(patterns[:, p], kn) - patterns[:, p]).norm() / patterns[:, p].norm())
           for p in range(P)]
    print(f"   {frac:.0%} revealed : mean {np.mean(err):.3f}   per digit {np.round(err, 3)}")

known = make_mask((SIDE, SIDE), MASK_KIND, SHOW_FRAC)
rows = []
for p in range(P):
    m = patterns[:, p]
    cue = m.clone()
    cue[~known] = 0.0
    rows.append((m, cue, complete(m, known)))

allv = torch.stack([v for r in rows for v in r]).numpy()
if allv.min() < -1e-9:
    qcmap, qlo, qhi = "RdBu_r", -np.abs(allv).max(), np.abs(allv).max()
else:
    qcmap, qlo, qhi = "magma", 0.0, allv.max()

figq, axq = plt.subplots(3, P, figsize=(1.15 * P, 4.0), squeeze=False)
for p, (m, cue, out) in enumerate(rows):
    for r, v in enumerate((m, cue, out)):
        axq[r][p].imshow(img(v), cmap=qcmap, vmin=qlo, vmax=qhi)
        axq[r][p].set(xticks=[], yticks=[])
        axq[r][p].grid(False)
    axq[0][p].set_title(str(labels[p]), fontsize=9)
for r, lab in enumerate(("stored", "cue", "recalled")):
    axq[r][0].set_ylabel(lab, fontsize=9)
figq.suptitle(f"student queried with {SHOW_FRAC:.0%} of each digit ({MASK_KIND} mask)", y=1.02)
figq.tight_layout()
figq

# %% [markdown]
# ## 6. A strip of frames, for a figure
#
# The movie is for looking at; this is what goes in a paper.

# %% filmstrip
K = 10
picks = np.linspace(0, len(Dn) - 1, K).astype(int)
figs, axs = plt.subplots(1, K, figsize=(1.15 * K, 1.6))
for a, k in zip(axs, picks):
    a.imshow(Dn[k].reshape(SIDE, SIDE), cmap=cmap, vmin=vmin, vmax=vmax)
    a.set_title(f"{labels[int(np.argmax(np.abs(coef[k])))]}", fontsize=9)
    a.axis("off")
figs.suptitle(f"{ACTIVATION or 'linear'} replay, {K} frames across the run", y=1.08)
figs.tight_layout()
figs

# %% [markdown]
# ## Try next
# - **Activations**: compare `ACTIVATION = "relu"` against `None`. Expect relu to keep the
#   coefficients nonnegative (blue bars rare) because rectification confines the state to the
#   patterns' cone, while the linear model roams the full span and dreams negative blends — which
#   look like ghostly inverted digits.
# - **Load**: raise `N_MEMORIES`. The dominance of a few digits in the bar chart is the thing to
#   watch — replay tends to park rather than tour.
# - **Resolution**: `SIDE=28` for full MNIST, at a much slower `build_memory`.
