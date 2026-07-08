# Derivation curriculum — the ownable math of the paper

**How this works.** You derive each result by hand on paper. I give you the goal, what's given, and
the target statement, then pose one step at a time as a question. You report your paper work; I
verify, hint, or correct, and only then advance. When a result is proved, we confirm it numerically
against the matching code diagnostic. This file is the live tracker.

**Status legend:** `not-started` → `deriving` (in progress) → `proved` (paper done, I verified) →
`verified` (numerically confirmed in code).

---

## Progress tracker

| # | Result | Status | Code check |
|---|---|---|---|
| T1 | Matrix differentiation (toolkit) | proved | — |
| T2 | PSD ordering (toolkit) | proved (sandwich instinct: vᵀSv=‖Mv‖²) | — |
| T3 | Projections: subspace + sphere tangent (toolkit) | proved | — |
| T4 | Rayleigh flow / eigenbasis decoupling (toolkit) | proved | — |
| L1 | **Proposition 1** — student inference & learning descend `F_S` | verified (err 0.0) | `check_gradients` |
| L2 | **Lemma 1** — fast-student elimination → novelty operator `N_S` | verified (learned→0, unlearned→0.36, id 1e-15) | `novelty_operator` |
| L3 | Novelty spectrum `n(μ)` and monotonicity `n'(μ)>0` | verified (eig map 7e-16, monotone) | eig of `N_S` |
| L3b | **Corollary 1** — surprise identity: `F_S^eq(x_T) = (π_TS/2)x_TᵀN_Sx_T` (settled F_S = novelty score) | verified (err 1.8e-15; to derive by hand) | `surprise_identity_error` |
| L4 | **Theorem 1** — the student's push `u = |π_ST|N_S x_T` steers the teacher up the *student's* surprise `F_S^eq` (steepest local ascent) | verified (rates ∝ n_k: learned→0, novel→+) | `novelty_operator` |
| L5 | **Lemma 2** — off-manifold normal stability (guard; keeps teacher on its manifold — numerical) | not-started (author solo) | `spectral_gap`, `manifold_leakage` |
| L6 | **Proposition 3** — exact saddle iff `π_ST = −π_TS` (active-inference-reminiscent) | not-started (author solo) | `circulation → 0` |

**Detailed, fully-justified proofs** of Prop 1, Lemma 1, Corollary 1, Theorem 1 (every transpose /
inverse / commutation spelled out): [detailed_proofs.md](detailed_proofs.md). **Intuitive, no-step-skipped
geometric version** of Theorem 1 (a picture per step): [intuitive_proof.md](intuitive_proof.md).
**Removed from the paper:** Proposition 2 (interleaving union) — union & continual learning are now
**empirical** (figures), no theorem. The **old self-extinction corollary** — folded into Theorem 1 Consequence 2
as a one-line remark (`|π_ST|n_k ≤ (|π_ST|π_S/π_TS)‖M_S u_k‖²`), not a standalone result. (The name
**Corollary 1** is now used for the *surprise identity*, L3b — a different, new result.)
**Future work:** the restricted-novelty / manifold analysis (operator `A_S = U_Tᵀ N_S U_T`) — which
surprising directions dominate once the teacher's own geometry is folded in.

---

## Notation (fixed for the whole curriculum)

- $d$ neurons per population; $x_T, x_S \in \mathbb{R}^d$ teacher/student states.
- $W_T, W_S$ recurrent weights (zero diagonal); $M_T = I - W_T$, $M_S = I - W_S$.
- Self-error operators $S_T = M_T^\top M_T$, $S_S = M_S^\top M_S$ (symmetric PSD; weight-level →
  novelty side of the terminology; the codebase calls them "self-surprise operators").
- Errors: self $\varepsilon_T = M_T x_T$, $\varepsilon_S = M_S x_S$; interface $\varepsilon_{TS} = x_S - x_T$.
- Precisions: $\pi_T, \pi_S > 0$ (self); $\pi_{TS} > 0$ (student interface); $\pi_{ST}$ signed
  (teacher interface — sleep $<0$, wake $>0$).
- Free energies: $F_S = \tfrac{\pi_{TS}}{2}\|\varepsilon_{TS}\|^2 + \tfrac{\pi_S}{2}\|\varepsilon_S\|^2$, and $F_T = \tfrac{\pi_T}{2}\|\varepsilon_T\|^2$.
- $U_T$ = orthonormal basis of the teacher memory manifold $\ker M_T$; $P_T = U_T U_T^\top$.
- Novelty operator $N_S = \pi_S S_S(\pi_{TS} I + \pi_S S_S)^{-1}$; restricted $A_S = U_T^\top N_S U_T$.
- Settled student surprise $F_S^{\mathrm{eq}}(x_T) = \min_{x_S} F_S(x_S, x_T) = \tfrac{\pi_{TS}}{2} x_T^\top N_S x_T$ (Corollary 1).
- **Terminology:** *novelty* = weight-level (operator $N_S$, eigenvalues $n_k$: which directions the
  student can't predict); *surprise* = state-level scalar (free energies $F_S$, $F_T$). Corollary 1 is
  the bridge; every surprise Theorem 1 climbs is the student's ("teacher's surprise" only ever means $F_T$).

---

## Toolkit warm-ups

### T1 — Matrix differentiation
**Target.**

$$
\nabla_x \tfrac12\|Ax\|^2 = A^\top A x, \qquad \nabla_W \tfrac12\|x - Wx\|^2 = -(x - Wx)x^\top .
$$

Derive both by the perturbation method (the (G1)/(G2) definitions in detailed_proofs.md "The
atoms"): expand f(x+δ) / f(W+Δ) exactly, isolate the linear term, match the template (for the
matrix case via the extraction rule aᵀΔb = ⟨abᵀ, Δ⟩). Then do the weight gradient once more
coordinate-by-coordinate (Kronecker delta) as a check — it also yields the Hebbian wire-by-wire
reading.

### T2 — PSD ordering
**Target.** $A \preceq B \iff v^\top A v \le v^\top B v \ \forall v$. Connect to eigenvalues for
symmetric matrices. (Used for the extinction bound and the guard.)

### T3 — Projections
**Target.** $P_T = U_T U_T^\top$ projects onto the memory subspace; $P_{x^\perp} = I - \dfrac{xx^\top}{\|x\|^2}$
projects onto the sphere's tangent space (removes the radial component).

### T4 — Rayleigh flow in an eigenbasis
**Target.** For symmetric $A$ with $A v_i = \lambda_i v_i$ and $y = \sum_i a_i v_i$, the flow
$\dot y = (I - \tfrac{yy^\top}{r_0^2})Ay$ gives

$$
\dot a_i = (\lambda_i - \bar\lambda)a_i, \quad \bar\lambda = \frac{y^\top A y}{r_0^2}, \qquad
\frac{d}{dt}\log\left|\frac{a_i}{a_j}\right| = \lambda_i - \lambda_j .
$$

So components with larger eigenvalues grow relative to smaller ones — the whole power-iteration idea.

---

## Lessons (target statements)

### L1 — Proposition 1: student inference & learning descend `F_S`
**Given.** $F_S$ above; student perception updates $x_S$, learning updates $W_S$ with $\mathrm{diag}\,W_S = 0$.
**Target.** $\dot x_S = -\nabla_{x_S} F_S$ (exact gradient descent); the zero-diagonal weight update
is *projected* gradient descent on $F_S$ over the zero-diagonal subspace ($F_S$ still decreases).
**Prereqs.** T1, T3. **Code check.** `check_gradients`.

### L2 — Lemma 1: fast-student elimination produces the novelty operator
**Given.** Fast student ($\tau_S \to 0$): set $\dot x_S = 0$.
**Target.** $x_S^{*} = \pi_{TS}(\pi_{TS} I + \pi_S S_S)^{-1} x_T$ and $\varepsilon_{TS} = -N_S x_T$ with
$N_S = \pi_S S_S(\pi_{TS} I + \pi_S S_S)^{-1}$; and $N_S U_T = 0$ on learned directions.
**Prereqs.** linear solve, positive-definite inverse. **Code check.** `novelty_operator`.

### L3 — Novelty spectrum and monotonicity
**Given.** $S_S v = \mu v$.
**Target.** $N_S v = n(\mu) v$ with $n(\mu) = \dfrac{\pi_S \mu}{\pi_{TS} + \pi_S \mu} \in [0,1)$, and
$n'(\mu) = \dfrac{\pi_S \pi_{TS}}{(\pi_{TS} + \pi_S \mu)^2} > 0$. Endpoints: $n(0)=0$ (learned), $n$ increasing.
**Prereqs.** spectral decomposition, one-variable calculus. **Code check.** eigenvalues of `N_S`.

### L3b — Corollary 1: the surprise identity (settled F_S = novelty score)
**Given.** Lemma 1's equilibrium and the complement identity $\pi_{TS} B^{-1} = I - N_S$, i.e.
$x_S^* = (I - N_S)x_T$ (the student settles on the part of the teacher's state it can predict).
**Target.** $F_S^{\mathrm{eq}}(x_T) = \min_{x_S} F_S(x_S, x_T) = \tfrac{\pi_{TS}}{2} x_T^\top N_S x_T$,
hence $\nabla_{x_T} F_S^{\mathrm{eq}} = \pi_{TS} N_S x_T$. Per direction the interface term carries
$n_k^2$, the self term $n_k(1-n_k)$, summing to the plain novelty reading $n_k$. Second proof via the
**envelope theorem** (for $g(\theta) = \min_x f(x,\theta)$: $g' = \partial f/\partial\theta$ at the
minimizer, because $\partial f/\partial x = 0$ there — stated, proved and toy-checked in the Cor 1
remark of detailed_proofs.md): only the explicit $x_T$-dependence counts.
**Why it matters.** It shows Theorem 1's quadratic score is not an invented quantity — it is the
model's own $F_S$, the student's surprise, settled; this is what Theorem 1 climbs and what Prop 3's
saddle subtracts. **Prereqs.** L2, L3. **Code check.** `diagnostics.surprise_identity_error`
(settled $F_S$ vs $(π_{TS}/2)x_T^\top N_S x_T$; err ~1.8e-15, asserted in `tests/smoke_test.py`).

### L4 — Theorem 1: the student steers the teacher up the *student's* surprise
**Given.** Sleep ($\pi_{ST} < 0$); **fast student state** (x_S at its Lemma-1 equilibrium) **+ frozen
student weights**. From Lemma 1 + the teacher's equation, the student's push is $u = |\pi_{ST}| N_S x_T$
(no manifold, no projection). N_S has student rulers $u_k$ with novelty eigenvalues $n_k$ ($N_S u_k = n_k u_k$).
**Target — THE CORE (two moves + a comparison, short).** Move 1: isolate the push from the teacher's
equation. Move 2: $\nabla F_S^{\mathrm{eq}} = \pi_{TS} N_S x_T$ (Corollary 1 + gradient of a quadratic
form). Compare: $u = (|\pi_{ST}|/\pi_{TS})\nabla F_S^{\mathrm{eq}}$, so the teacher's dynamics
restricted to the student's push is **exact gradient ascent on the student's settled variational free
energy**, at every point ("steepest ascent" is one clause — standard fact, no proof in the paper).
**Consequences (downstream, clearly separated).** C1 *prioritization:* diagonalize
($N_S = U D_n U^\top$, $c = U^\top x_T$) — $\dot c_k = (|\pi_{ST}|/\tau_T) n_k c_k$, learned frozen,
most-novel dominates; not needed for the core, it is what the experiments measure. C2 *monotone climb
+ self-limiting (absorbs the old self-extinction corollary):*
$|\pi_{ST}| n_k \le (|\pi_{ST}|\pi_S/\pi_{TS})\|M_S u_k\|^2$ — the push on a direction dies as the
student learns it. C3 *saddle preview:* the full flow is $-\nabla F_T + (|\pi_{ST}|/\pi_{TS})\nabla F_S^{\mathrm{eq}}$,
which is $-\nabla(F_T - F_S^{\mathrm{eq}})$ iff $|\pi_{ST}| = \pi_{TS}$ — Prop 3's condition, one line.
Caveats: local ascent (not global summit); the push rescales but does not seed a zero component (noise
seeds it). No "teacher's surprise" anywhere — the teacher's own surprise is $F_T$ (the self-pull).
**Prereqs.** T1 (gradient of quadratic form), L3b. **Code check.** `surprise_identity_error` (the
core's $F_S^{\mathrm{eq}}$), `novelty_operator` (C1: n_k learned→0, novel→+).

### L5 — Lemma 2: conservative off-manifold normal stability (the guard)
**Given.** $q \in U_T^\perp$; sleep operator $-\pi_T S_T + |\pi_{ST}| N_S$.
**Target.** $q^\top(-\pi_T S_T + |\pi_{ST}| N_S)q \le -(\pi_T \sigma_{\min}^2 - |\pi_{ST}|)\|q\|^2$, so
$|\pi_{ST}| < \pi_T \sigma_{\min}^2$ makes the normal–normal block strictly contracting. Caveat: this is
NOT manifold invariance (cross-block $Q_T N_S P_T$ leaks). **Prereqs.** Rayleigh quotient, eigenvalue bound.
**Code check.** `spectral_gap`, `manifold_leakage`.

### L6 — Proposition 3: exact `F_T − F_S` saddle condition
**Given.** $\Phi = F_T - F_S$; teacher/student mobilities; tangent projection on the teacher sphere.
(Falls out naturally now: Theorem 1's core showed the student's push is $(|\pi_{ST}|/\pi_{TS})\nabla F_S^{\mathrm{eq}}$
— a *gradient of the student's surprise* — which is exactly what lets a single potential $\Phi$
generate the flow, and already yields the condition $|\pi_{ST}| = \pi_{TS}$ on the teacher's state
equation.)
**Target.** $\Phi$ generates the sleep dynamics iff $\pi_{ST} = -\pi_{TS}$ (Schwarz symmetry of mixed
partials), with the three qualifications (mobilities; projected $W_S$ ascent; Riemannian gradient on
the sphere). Reading: the teacher descends its own surprise $F_T$ while ascending the student's $F_S$
(and the student, descending $F_S$ by Prop 1, *ascends* $\Phi$) — a genuine minimax on one potential.
Framed as active-inference-*reminiscent*, not identical. **Prereqs.** T1, L3b, mixed partials.
**Code check.** `circulation → 0`.
