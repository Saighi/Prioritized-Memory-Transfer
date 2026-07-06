# Understanding a Predictive-Coding Associative Memory (from the ground up)

A slow, unrolled walk through **why the teacher's self-dynamics look the way they do** — why there are
two terms, why one of them "completes" the other into a proper downhill slide, when the forces cancel,
and why the network ends up *flat along its memories but springy away from them*. (The same logic
governs either population's self-relaxation; we use the teacher as the running example.)

We keep **everything written out neuron-by-neuron** (no compact matrix shorthand), and every
technical word is explained the first time it shows up. We use a tiny **2-neuron** network the whole
way through so you can plug in numbers.

---

## 0. The little glossary (read once)

- **value neuron** `x_i` — the activity of neuron `i` (a number). The thing that changes over time.
- **error neuron** `ε_i` — a partner neuron that reports neuron `i`'s *prediction error* (also a number).
- **weight** `W_ij` — how strongly neuron `j` is used to **predict** neuron `i`. No self-weights: `W_ii = 0`.
- **prediction of neuron `i`** — what the *other* neurons think `x_i` should be: `Σ_{j≠i} W_ij x_j`
  (multiply each other neuron by its weight and add up).
- **transpose** — just "use `W_ji` where you'd use `W_ij`", i.e. read the weight table the other way.
- **gradient** — for a function `F(x_1,…,x_d)`, the list of slopes `∂F/∂x_i`. It points **uphill**;
  the negative of it points **downhill** (the steepest way to *decrease* `F`).
- **gradient descent** — "roll downhill": move each `x_i` by `−∂F/∂x_i`. Then `F` can only go down.
- **energy / Lyapunov function** — a single number `F` attached to the whole network that the dynamics
  keeps pushing *down*. If a dynamics is gradient descent on some `F`, that `F` can never increase →
  the system must settle, never blow up or circle forever.
- **null space** (of a transformation) — the set of inputs it sends to **zero**. Here: the states with
  *zero error everywhere* — the stored memories.
- **eigenvalue / eigenvector** — a special direction `v` that a transformation only **stretches**
  (doesn't rotate), `→ λ v`; the number `λ` is the stretch factor.
- **symmetric matrix** — a weight table equal to its own transpose (`Q_ij = Q_ji`).
- **positive semi-definite (PSD)** — a symmetric table `Q` such that the energy `½ Σ_ij Q_ij v_i v_j ≥ 0`
  for *every* `v`. Picture: the energy surface is a **bowl** (or flat), it never curves *downward*. All
  its eigenvalues are `≥ 0` (every special direction is stretched by a non-negative amount).

---

## 1. The tiny network we'll use

Two value neurons `x_1, x_2`, two weights (diagonal is zero):

```
W_12 = 2     (neuron 2 predicts neuron 1, with weight 2)
W_21 = 0.5   (neuron 1 predicts neuron 2, with weight 0.5)
```

The **error neurons** are "my value minus what the others predict me to be":

```
ε_1 = x_1 − W_12 x_2 = x_1 − 2 x_2
ε_2 = x_2 − W_21 x_1 = x_2 − 0.5 x_1
```

A **stored memory** is a state where *every error is zero* (the network is perfectly self-consistent):
`ε_1 = 0` and `ε_2 = 0`. For our numbers that means `x_1 = 2 x_2`, e.g.

```
memory:  (x_1, x_2) = (2, 1)      →  ε_1 = 2 − 2·1 = 0,   ε_2 = 1 − 0.5·2 = 0  ✓
```

So the memory here is the whole **line** `x_1 = 2 x_2` (any multiple of `(2,1)` is also error-free —
keep that in mind, it matters at the end).

---

## 2. The equation, written per neuron

The teacher's self-dynamics (ignoring the student's drive and the noise for now) say each value neuron is pushed by:

```
τ · dx_i/dt  =   − ε_i             ← term ① "fix my own error"
               + Σ_{j≠i} W_ji ε_j  ← term ② "fix the errors I cause in the neurons I predict"
```

For our 2-neuron net this is exactly:

```
field on neuron 1:  −ε_1 + W_21 ε_2 = −ε_1 + 0.5 ε_2
field on neuron 2:  −ε_2 + W_12 ε_1 = −ε_2 + 2   ε_1
```

("field" = the push on that neuron, the right-hand side.)

Two things to notice about the index in term ②: it's `W_ji`, **not** `W_ij`. That's the transpose, and
it's there for a concrete reason — `W_ji` is exactly the weight with which neuron `i` *helps predict*
neuron `j` (look back at `ε_2 = x_2 − 0.5 x_1`: the `0.5 = W_21` is how much neuron 1 feeds the
prediction of neuron 2). So term ② collects **the errors of the neurons you help predict**, each
weighted by how much you contribute to them.

So every neuron wears two hats at once:
- **Hat ① (being predicted):** "I have my own error `ε_i`; shrink it." → `−ε_i`
- **Hat ② (predicting others):** "I also feed other neurons' predictions; if *they* are wrong and it's
  partly my fault, help fix them." → `+ Σ W_ji ε_j`

---

## 3. "Term ① alone is already a memory" — true!

Write out term ① using the definition of `ε_i`:

```
−ε_i = −(x_i − Σ_{j≠i} W_ij x_j) = −x_i + Σ_{j≠i} W_ij x_j
            └ leak toward 0 ┘   └ pull toward the prediction ┘
```

So term ① alone does two intuitive things: a gentle **leak** pulling `x_i` toward 0, plus a **pull
toward what the other neurons predict** `x_i` should be. Its resting points are exactly `x_i = `(its
prediction)`, i.e. all errors zero — the memories. **This by itself is the classic recurrent
associative memory** (the Hopfield-style "relax toward the stored pattern"). Your instinct was right.

**The catch:** term ① alone is only *reliable* if the weight table is symmetric (`W_ij = W_ji`). Our
example isn't (`W_12 = 2 ≠ 0.5 = W_21`). With a lopsided table, "roll using term ① only" is not rolling
downhill on anything — it can spiral or even run away, because there's no energy guaranteed to
decrease. That's where term ② comes in.

---

## 4. Why term ② "completes it into a gradient" (the key idea)

Here is the cleanest way to see it. Give the whole network a single **energy** = the total squared
error (how inconsistent the network currently is):

```
F = ½ (ε_1² + ε_2² + … )      (for our net: F = ½(ε_1² + ε_2²))
```

We'd *like* the dynamics to **roll downhill on `F`** — i.e. `dx_i/dt = −∂F/∂x_i` — because then `F`
keeps dropping and the network must settle onto a low-error state (a memory).

So let's honestly compute the slope `∂F/∂x_1` for our 2-neuron net. The trick: **`x_1` sits inside
*two* error neurons**, so changing it changes `F` through both.

```
ε_1 = x_1 − 2 x_2        → if x_1 goes up by 1, ε_1 goes up by 1      (∂ε_1/∂x_1 = +1)
ε_2 = x_2 − 0.5 x_1      → if x_1 goes up by 1, ε_2 goes DOWN by 0.5  (∂ε_2/∂x_1 = −0.5)
```

Chain rule (slope of `½ε²` is `ε` times the slope of `ε`):

```
∂F/∂x_1 =  ε_1 · (∂ε_1/∂x_1)  +  ε_2 · (∂ε_2/∂x_1)
        =  ε_1 · (+1)         +  ε_2 · (−0.5)
        =  ε_1 − 0.5 ε_2
```

Therefore the true downhill move is:

```
dx_1/dt = −∂F/∂x_1 = −ε_1 + 0.5 ε_2
```

**Look at the two pieces of the slope:**
- `ε_1·(+1)` — how `x_1` affects **its own** error. This is where **term ① (`−ε_1`)** comes from.
- `ε_2·(−0.5)` — how `x_1` affects **neuron 2's** error (the one it helps predict). This is where
  **term ② (`+0.5 ε_2`)** comes from.

So **term ① is only *half* the slope.** If you used term ① alone, you'd be ignoring the collateral
damage `x_1` does to neuron 2's prediction — you wouldn't actually be going downhill on `F`, just on a
"fake" slope that happens to ignore other neurons. **Adding term ② supplies the missing half**, and
the sum is exactly the real gradient. *That* is what "completes it into a gradient" means: with both
terms the dynamics is genuine gradient descent on the total error `F`, so `F` can only fall and the
network is guaranteed to settle — for **any** weight table, symmetric or not.

(In compact shorthand people write the two terms together as `−Mᵀ M x`, but that hides exactly the
thing we just unrolled: `−Mᵀ` is "account for the errors you cause in others.")

---

## 5. When does the push cancel? (examples)

The push on a neuron is a **tug-of-war**: term ① wants to shrink *your* error, term ② wants you to help
shrink *others'* errors, and they can pull opposite ways. The neuron sits still when they balance.

**(a) Reading the balance.** Take neuron 1: push = `−ε_1 + 0.5 ε_2`. Suppose for a moment the error
neurons happened to read `ε_1 = 1` and `ε_2 = 2`:

```
push on 1 = −(1) + 0.5·(2) = −1 + 1 = 0
```

Zero! Even though neuron 1 *is* in error (`ε_1 = 1`), it doesn't move — because moving to fix its own
error would worsen neuron 2's error by just as much. In words: **you only correct yourself to the
extent your own error outweighs the error you'd create in the neurons you predict.** That's your
insight, and it's exactly the tug-of-war between term ① and term ②.

**(b) A real resting state — the memory.** The errors aren't free numbers, though: they're produced by
the state (`ε_1 = x_1 − 2x_2`, `ε_2 = x_2 − 0.5x_1`). Plug in the memory `(2,1)`:

```
ε_1 = 0,  ε_2 = 0   →   push on 1 = −0 + 0.5·0 = 0,   push on 2 = −0 + 2·0 = 0
```

Both neurons sit still because **every error is zero**. This is the honest, full cancellation.

**(c) Off the memory — it doesn't sit still.** Take `(x_1, x_2) = (1, 0)`:

```
ε_1 = 1 − 0   = 1
ε_2 = 0 − 0.5 = −0.5
push on 1 = −1 + 0.5·(−0.5) = −1.25   (x_1 pushed down)
push on 2 = −(−0.5) + 2·1   = +2.5    (x_2 pushed up)
```

Both pushes are nonzero, and they point the state straight back toward the memory line `x_1 = 2x_2`.
No resting here.

**The important truth:** for **one** neuron the push can momentarily balance (example a), but the
errors are all generated by the *same* state, so they're tied together. The only way **every** neuron's
push cancels **at the same time** is when **all errors are zero** — i.e. on a stored memory. (In our
tiny net the errors are so tightly linked that even a single neuron's push reaches zero only on the
memory; in bigger nets one neuron can momentarily balance while the whole thing still drifts, but
*simultaneous* stillness still means all-errors-zero.) So: **the network's only rest states are the
memories**, exactly because that's the only place the whole tug-of-war goes slack.

---

## 6. The payoff: flat along the memories, springy away from them

Now zoom out to the energy landscape `F = ½(ε_1² + ε_2² + …)` — the "total inconsistency" as a height
over the space of states. The dynamics just rolls downhill on it. What does this surface look like?

- **Along the memory line** (`x_1 = 2x_2` here; in general the whole set of error-free states): every
  point has `ε = 0`, so `F = 0`. It's a **flat valley floor** — the height doesn't change as you slide
  from one memory to another (or to any blend of them). Flat floor ⇒ **zero slope ⇒ zero push**. This
  is why **at equilibrium there is no pull toward any *particular* memory**: they're all at the same
  lowest height, equally good, so nothing prefers one over another.
  - *Why is the floor flat and not a set of separate dimples?* Because the network is **linear**: if
    `(2,1)` is error-free, so is `(4,2)`, `(−2,−1)`, and every multiple/blend. There are *many*
    self-consistent states, and the network has no reason to favor any — it only removes inconsistency.
    (This flat floor of equally-valid memories is exactly the "it stores a **subspace**, not separate
    patterns" point from the analysis.)

- **Away from the memories** (any direction that makes some `ε ≠ 0`): `F > 0`, the ground rises, so the
  slope points back down → a **restoring push** (a spring pulling you back to the floor). The steeper
  the rise, the stronger the damping.

The reason the rise is *always* a bowl going **up** (a spring), never a dome going **down** (a
run-away), is that the energy `F = ½ Σ ε_k²` is a sum of squares — it can't be negative. In the
glossary's language, the "curvature table" of `F` is **positive semi-definite (PSD)**: flat in the
memory directions (curvature 0), curving upward in all the others (curvature > 0), never downward.
That single property *is* "damping (or nothing) in every direction, anti-damping in none."

**So your hypothesis is exactly right.** The two-hats structure (term ① + term ②) is precisely what
makes the dynamics descend the total-error energy `F = ½ Σ ε²`. That energy is **flat wherever the
network is self-consistent** (the memories → no push, just a flat sea of equally-valid states) and
**rises in every inconsistent direction** (→ a restoring spring, the damping). Damping toward the
patterns' subspace, flatness along it, and no attractive pull toward any single pattern — all three
come from the same fact: **this network doesn't chase a goal, it only erases inconsistency**, and there
are many equally-consistent states to rest in.

---

## 7. One-paragraph summary

Each neuron is both *predicted by* others (hat ①: shrink my own error, `−ε_i`) and a *predictor of*
others (hat ②: help shrink the errors I feed into, `+Σ W_ji ε_j`). Term ① alone is the classic
relax-to-a-memory move but isn't guaranteed to behave; term ② adds the *other half of the slope* of the
total error `F = ½ Σ ε²`, so the two together are honest downhill motion (gradient descent) that always
settles. The forces all cancel only where every error is zero — the memories — and because the model is
linear there is a whole **flat** set of such states. Hence: a smooth bowl that is flat along the memory
subspace (no pull toward any one memory) and springy (damped) away from it.
