# %% [markdown]
# # One linear associative memory — storing pictures, recall by clamping units
#
# **Background demo, not a paper figure** — pedagogical companion to the transfer ladder.
#
# Run cell-by-cell in VS Code (select the **`pytorch`** conda interpreter as the kernel).
#
# This notebook steps *out* of the two-network transfer story and looks at a **single**
# predictive-coding associative memory in isolation — the elementary object the rest of the
# project is built from. We store a handful of pictures (MNIST digits) as memories, then **query
# the network with a fragment of a picture** by clamping the "seen" units and letting the rest of
# the network fill itself in. Two things get visualized:
#
# 1. the **trajectory of network activity** as it slides onto the memory subspace, and
# 2. the **free energy** falling toward zero as the query is answered.
#
# ### The network
# One population of $d$ units with a frozen, zero-diagonal recurrent weight matrix $W$ built from
# the stored pictures $M=[m_1\dots m_P]$ (the same covariance-PCN construction the teachers use
# elsewhere in `src`). Its mismatch operator is $M_{\mathrm{op}} = I - W$ and its **free energy** is
#
# $$
# F(x) \;=\; \tfrac{1}{2}\,\pi\,\lVert M_{\mathrm{op}}\,x\rVert^2 .
# $$
#
# Every stored picture satisfies $M_{\mathrm{op}}\,m_p \approx 0$, so the pictures span the
# **memory manifold** $\ker(M_{\mathrm{op}})$ — the flat, zero-energy floor of $F$.
#
# ### Recall = clamp part of the picture, then descend $F$
# Split the units into a **known** set $K$ (clamped to the cue) and a **free** set $U$. Hold
# $x_K$ fixed and let the free units follow the gradient flow
#
# $$
# \tau\,\dot{x}_U \;=\; -\,\pi\,\bigl(M_{\mathrm{op}}^{\top} M_{\mathrm{op}}\,x\bigr)_U ,
# \qquad x_K \equiv \text{cue} ,
# $$
#
# which is projected gradient descent on $F$. The state slides downhill onto the manifold,
# completing the picture; $F$ decreases monotonically to $\sim 0$. The fixed point is closed-form,
# $\;S_{UU}\,x_U = -\,S_{UK}\,x_K\;$ with $S = M_{\mathrm{op}}^{\top}M_{\mathrm{op}}$ — we check the
# dynamics land exactly on it.

# %% imports & path setup
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

_root = Path(__file__).resolve().parents[2]        # repo root (for the MNIST data dir)

SHOW = os.environ.get("PMT_NO_SHOW") != "1"        # set PMT_NO_SHOW=1 to run headless

import matplotlib
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prioritized_memory_transfer import AssociativeMemory, make_mask

torch.manual_seed(0)
DT = torch.float64
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())

# %% [markdown]
# ## 1. Load pictures and pick the memories to store
#
# We use MNIST, average-pooled to **14×14** (so $d=196$ units — small enough that the covPCN
# weights build in a moment, big enough that the digits stay clearly legible). We store one clean
# example of each digit 0–9 plus a repeated 3 and 8, for **P = 12** memories. (If MNIST can't be
# downloaded, we fall back to scikit-learn's 8×8 `load_digits`.)

# %% load & downsample
SIDE = 14                                          # pooled image side -> d = SIDE*SIDE
d = SIDE * SIDE
WANT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 8]        # digit classes to store (P = 12)


def _load_mnist():
    import torchvision
    ds = torchvision.datasets.MNIST(root=str(_root / "data"), train=True, download=True)
    imgs = ds.data.to(DT) / 255.0                  # (N, 28, 28) in [0, 1]
    labels = ds.targets
    sel, used = [], {}
    for i in range(imgs.shape[0]):
        lb = int(labels[i])
        if WANT.count(lb) > used.get(lb, 0):
            sel.append(i); used[lb] = used.get(lb, 0) + 1
        if len(sel) == len(WANT):
            break
    X = Fn.avg_pool2d(imgs[sel].unsqueeze(1), 28 // SIDE).squeeze(1)   # (P, SIDE, SIDE)
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
    patterns_raw, mem_labels = _load_mnist()
except Exception as e:
    print("MNIST unavailable, using sklearn digits (8x8):", repr(e))
    SIDE = 8; d = SIDE * SIDE
    patterns_raw, mem_labels = _load_digits_fallback()

patterns = patterns_raw / patterns_raw.norm(dim=0, keepdim=True)     # unit-norm columns (d, P)
P = patterns.shape[1]
print(f"stored P={P} pictures of digits {mem_labels}; d={d} units ({SIDE}x{SIDE})")


def to_img(vec):
    """Reshape a length-d state to a (SIDE, SIDE) numpy image."""
    return np.asarray(vec.detach()).reshape(SIDE, SIDE)


# %% [markdown]
# ## 2. Build the memory and check it is faithful
#
# `AssociativeMemory` builds the zero-diagonal covPCN weights from the pictures. The memory
# manifold $\ker(M_{\mathrm{op}})$ should come out exactly $P$-dimensional (the pictures are
# linearly independent), and each stored picture should have essentially zero self-error.

# %% build
mem = AssociativeMemory(patterns, pi=1.0)

S_eig = torch.linalg.eigvalsh(mem.S)
resid = (mem.M_op @ patterns).norm(dim=0).max().item()
print(f"memory manifold dim  : {mem.manifold.shape[1]}   (= P = {P})")
print(f"max_p ||M_op m_p||    : {resid:.2e}   (~0 => pictures live on the manifold)")
print(f"S = M_opᵀM_op spectrum: min-nonzero {float(S_eig[S_eig > 1e-9].min()):.3f}, "
      f"max {float(S_eig.max()):.3f}   (sets the gradient-flow step: dt·π < 2/λ_max)")
assert mem.manifold.shape[1] == P and resid < 1e-4

# %% [markdown]
# ## 3. The stored gallery
# The $P$ pictures the network holds — every one of these is a zero-energy point of $F$.

# %% gallery
ncol = 6
nrow = int(np.ceil(P / ncol))
fig_gal, axes = plt.subplots(nrow, ncol, figsize=(1.5 * ncol, 1.5 * nrow))
for j, ax in enumerate(axes.ravel()):
    if j < P:
        ax.imshow(to_img(patterns[:, j]), cmap="gray_r")
        ax.set_title(f"m{j}: “{mem_labels[j]}”", fontsize=9)
    ax.axis("off")
fig_gal.suptitle("Stored memories (the flat floor of F)", y=1.02)
fig_gal.tight_layout()
fig_gal

# %% [markdown]
# ## 4. Query the network with a fragment of a picture
#
# We take one stored digit, **hide the bottom half** (those units are free), and clamp the top
# half to the cue. Then we run the gradient-flow recall and watch the free units fill in.

# %% choose a query and run recall
target = 2                                          # which stored memory to cue (index into the gallery)
known = make_mask((SIDE, SIDE), "top", frac=0.5)     # clamp the TOP half; fill the bottom
cue = patterns[:, target]

trace = mem.recall(cue, known, fill=0.0, n_steps=1000, dt=0.2, tau=1.0, record_every=10)

print(f"clamped {int(known.sum())}/{d} units; freed {d - int(known.sum())}")
print(f"F: {float(trace.F[0]):.4f} -> {float(trace.F[-1]):.2e}   "
      f"(monotone: {bool((trace.F[1:] <= trace.F[:-1] + 1e-9).all())})")
print(f"distance to manifold ||M_op x||: {float(trace.resid[0]):.3f} -> {float(trace.resid[-1]):.2e}")
print(f"manifold occupancy: {float(trace.occupancy[0]):.2f} -> {float(trace.occupancy[-1]):.3f}")
print(f"recovered picture rel-error: "
      f"{float((trace.X[-1] - cue).norm() / cue.norm()):.2e}   (dynamics vs closed form: "
      f"{float((trace.X[-1] - trace.x_star).norm()):.1e})")

# %% [markdown]
# ## 5. The reconstruction filmstrip
# Left to right: the true picture, the clamped cue (hidden units in red), then the network state
# $x(t)$ as the query is answered, ending at the completed picture.

# %% filmstrip
def show_cue(ax, cue, known):
    img = to_img(cue).copy()
    img[~to_img(known).astype(bool)] = np.nan       # hidden units -> NaN
    cmap = plt.cm.gray_r.copy(); cmap.set_bad("tab:red")
    ax.imshow(img, cmap=cmap); ax.axis("off")


frame_idx = np.unique(np.linspace(0, len(trace.steps) - 1, 6).astype(int))
fig_film, axes = plt.subplots(1, len(frame_idx) + 2, figsize=(2.0 * (len(frame_idx) + 2), 2.3))
axes[0].imshow(to_img(cue), cmap="gray_r"); axes[0].set_title("target", fontsize=10); axes[0].axis("off")
show_cue(axes[1], cue, known); axes[1].set_title("cue (clamp)", fontsize=10)
for ax, i in zip(axes[2:], frame_idx):
    ax.imshow(to_img(trace.X[i]), cmap="gray_r"); ax.axis("off")
    ax.set_title(f"t={int(trace.steps[i])}", fontsize=10)
fig_film.suptitle(f"Recall of memory m{target} (“{mem_labels[target]}”) by clamping the top half", y=1.05)
fig_film.tight_layout()
fig_film

# %% [markdown]
# ## 6. Free energy being minimized during the query
# The whole query is one long descent of $F$. On the left, $F(x(t))$ and the distance to the
# memory manifold $\lVert M_{\mathrm{op}}x\rVert$ fall to zero (log scale); on the right, the
# fraction of the state that lies *inside* the memory manifold rises to 1.

# %% free-energy descent (plotly)
def energy_figure(trace):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    t = trace.steps.cpu().numpy()
    F = trace.F.cpu().numpy(); res = trace.resid.cpu().numpy(); occ = trace.occupancy.cpu().numpy()
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "free energy F(x) and distance to manifold", "manifold occupancy"))
    fig.add_trace(go.Scatter(x=t, y=np.maximum(F, 1e-16), name="F(x)",
                             line=dict(color="crimson", width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.maximum(res, 1e-16), name="‖M_op x‖",
                             line=dict(color="steelblue", width=2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=occ, name="occupancy", line=dict(color="seagreen", width=3),
                             showlegend=False), row=1, col=2)
    fig.update_yaxes(type="log", title_text="energy / distance (log)", row=1, col=1)
    fig.update_yaxes(range=[0, 1.02], title_text="fraction inside ker(M_op)", row=1, col=2)
    fig.update_xaxes(title_text="integration step", row=1, col=1)
    fig.update_xaxes(title_text="integration step", row=1, col=2)
    fig.update_layout(template="plotly_white", height=420,
                      title="F is minimized as the query is answered")
    return fig


fig_E = energy_figure(trace)
if SHOW:
    fig_E.show()

# %% [markdown]
# ## 7. The trajectory of network activity in the memory subspace
#
# We project the state onto the first three principal directions of the stored pictures (a 3-D
# window into the memory subspace). Each stored digit is a labelled landmark; the red path is the
# network state as it relaxes. It starts at the masked cue (off to the side, partly *outside* the
# subspace) and travels to land exactly on the target memory's landmark. Press **▶** or drag the
# slider to scrub the query in time.

# %% memory-subspace trajectory (plotly, animated)
def trajectory_figure(mem, trace, patterns, mem_labels, target, max_frames=80):
    import plotly.graph_objects as go
    mu = patterns.mean(dim=1, keepdim=True)                 # (d,1)
    U, _, _ = torch.linalg.svd(patterns - mu, full_matrices=False)
    B3 = U[:, :3]                                           # (d,3) top principal directions
    land = ((patterns - mu).T @ B3).cpu().numpy()          # (P,3) landmark coords
    path = ((trace.X - mu.T) @ B3).cpu().numpy()           # (T,3) trajectory coords
    t = trace.steps.cpu().numpy()

    landmarks = go.Scatter3d(
        x=land[:, 0], y=land[:, 1], z=land[:, 2], mode="markers+text",
        text=[f"{l}" for l in mem_labels], textposition="top center",
        marker=dict(size=5, color="royalblue"), name="stored memories")
    tgt = go.Scatter3d(x=[land[target, 0]], y=[land[target, 1]], z=[land[target, 2]],
                       mode="markers", marker=dict(size=11, color="royalblue",
                       symbol="diamond", line=dict(color="black", width=2)),
                       name=f"target m{target}")
    base = go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines",
                        line=dict(width=5, color=t, colorscale="Hot"), opacity=0.7, name="x(t) path")
    start = go.Scatter3d(x=[path[0, 0]], y=[path[0, 1]], z=[path[0, 2]], mode="markers",
                         marker=dict(size=7, color="black"), name="cue (t=0)")
    head = go.Scatter3d(x=[path[0, 0]], y=[path[0, 1]], z=[path[0, 2]], mode="markers",
                        marker=dict(size=8, color="red"), name="x(t)")

    idx = np.unique(np.linspace(0, len(t) - 1, min(max_frames, len(t))).astype(int))
    frames = [go.Frame(data=[go.Scatter3d(x=[path[i, 0]], y=[path[i, 1]], z=[path[i, 2]],
                             mode="markers", marker=dict(size=8, color="red"))],
                       traces=[4], name=f"{int(t[i])}") for i in idx]

    fig = go.Figure(data=[landmarks, tgt, base, start, head], frames=frames)
    fig.update_layout(
        title="Network activity relaxing onto the memory subspace",
        template="plotly_white", height=640,
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=0.05, buttons=[
            dict(label="▶ play", method="animate",
                 args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True)]),
            dict(label="⏸ pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.1, len=0.85, y=0.0, currentvalue=dict(prefix="step = "),
                      steps=[dict(method="animate", label=f"{int(t[i])}",
                             args=[[f"{int(t[i])}"], dict(mode="immediate",
                                   frame=dict(duration=0, redraw=True))]) for i in idx])])
    return fig


fig_traj = trajectory_figure(mem, trace, patterns, mem_labels, target)
if SHOW:
    fig_traj.show()

# %% [markdown]
# ## 8. Which memory is the network committing to?
# The cosine overlap of $x(t)$ with every stored picture over the course of the query. The target
# memory's curve climbs to 1 while the others stay put — the network *decides* as it descends.

# %% overlap raster (plotly)
def overlap_figure(trace, mem_labels, target):
    import plotly.graph_objects as go
    t = trace.steps.cpu().numpy()
    ov = trace.overlap.cpu().numpy()                        # (T, P)
    fig = go.Figure()
    for p in range(ov.shape[1]):
        is_t = (p == target)
        fig.add_trace(go.Scatter(x=t, y=ov[:, p], mode="lines",
                      name=f"m{p} “{mem_labels[p]}”" + (" (target)" if is_t else ""),
                      line=dict(width=4 if is_t else 1.5,
                                color="crimson" if is_t else None)))
    fig.update_layout(template="plotly_white", height=440,
                      title="cos(x(t), m_p) — the network commits to the cued memory",
                      xaxis_title="integration step", yaxis_title="cosine overlap")
    return fig


fig_ov = overlap_figure(trace, mem_labels, target)
if SHOW:
    fig_ov.show()

# %% [markdown]
# ## 9. Different fragments, same memory
# Pattern completion works from *any* informative fragment. Here the same digit is recalled from
# five different occlusions — top, bottom, left, a small central patch, and 50% random pixels.
# Each row: the cue (hidden units red) and the completed reconstruction.

# %% occlusion montage
masks = [("top", 0.5), ("bottom", 0.5), ("left", 0.5), ("center", 0.35), ("random", 0.5)]
fig_occ, axes = plt.subplots(len(masks), 2, figsize=(4.2, 2.0 * len(masks)))
for row, (kind, frac) in enumerate(masks):
    k = make_mask((SIDE, SIDE), kind, frac=frac)
    tr = mem.recall(cue, k, n_steps=1000, dt=0.2, record_every=50)
    show_cue(axes[row, 0], cue, k)
    axes[row, 0].set_title(f"{kind} ({int(k.sum())} known)", fontsize=9)
    axes[row, 1].imshow(to_img(tr.X[-1]), cmap="gray_r"); axes[row, 1].axis("off")
    err = float((tr.X[-1] - cue).norm() / cue.norm())
    axes[row, 1].set_title(f"recall (err {err:.0e})", fontsize=9)
fig_occ.suptitle(f"Completing m{target} from different fragments", y=1.01)
fig_occ.tight_layout()
fig_occ

# %% [markdown]
# ## 10. When the cue is too small: blends, not memories
#
# A linear memory recovers the point of the memory *subspace* closest to the cue. Because that
# subspace is $P$-dimensional, a cue needs on the order of $P$ well-placed known units to pin a
# single memory; above that threshold recall is essentially exact, below it the closest subspace
# point is a **blend** of several stored pictures. We shrink the cue across that threshold
# ($P=$ the number of stored memories) and watch recall degrade from a clean digit into a
# superposition — the honest failure mode of a *linear* associative memory (nonlinear /
# winner-take-all dynamics are what would break the tie and snap to one picture).

# %% shrinking-cue sweep
def random_known(k, seed=0):
    """A boolean mask with exactly k known (clamped) units."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(d, generator=g)[:k]
    m = torch.zeros(d, dtype=torch.bool)
    m[idx] = True
    return m


counts = [40, 20, 12, 8, 4]                         # known-pixel counts straddling P = 12
fig_amb, axes = plt.subplots(1, len(counts) + 1, figsize=(2.0 * (len(counts) + 1), 2.3))
axes[0].imshow(to_img(cue), cmap="gray_r"); axes[0].set_title("target", fontsize=10); axes[0].axis("off")
for ax, kk in zip(axes[1:], counts):
    k = random_known(kk, seed=1)
    xstar = mem.recall_steady(cue, k)
    ax.imshow(to_img(xstar), cmap="gray_r"); ax.axis("off")
    err = float((xstar - cue).norm() / cue.norm())
    ax.set_title(f"{kk} px\nerr {err:.2f}", fontsize=9)
fig_amb.suptitle(f"Shrinking the cue across the P={P} threshold: exact recall → blend", y=1.05)
fig_amb.tight_layout()
fig_amb

# %% [markdown]
# ## Try next
# - **Cue another digit**: change `target` (0…{P-1}) and the mask `kind` in §4, then rerun 4–8.
# - **Fewer / more memories**: edit `WANT` in §1 (keep `P ≤ d-1`); a fuller memory has a
#   higher-dimensional manifold, so smaller cues start to blend sooner (§10).
# - **Precision `pi`**: it scales `F` and the flow speed but not the fixed point — raising it just
#   sharpens the descent (watch §6). The step guard is `dt·pi < 2/λ_max(S)`.
# - **Noise**: add a small `sigma` term to the flow (in `prioritized_memory_transfer.recall.AssociativeMemory.recall`) to
#   see the state jitter within the flat manifold — the free on-manifold diffusion discussed in the
#   single-network precision/curiosity analysis.
# - **Representability**: try different dense pattern sets and inspect
#   `mem.memory_build.max_residual` / `condition_number`. Construction fails early when the
#   no-autapse network cannot faithfully store the requested columns.
