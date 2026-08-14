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
| L2 | **Lemma 1** — mode-by-mode settling → novelty operator `N_S` emerges (dynamics-first; prioritization inside) | **derived by hand 2026-07-19**; verified (learned→0, unlearned→0.36, id 1e-15) | `novelty_operator` |
| L3 | Novelty spectrum `n(μ)` and monotonicity `n'(μ)>0` | verified (eig map 7e-16, monotone) | eig of `N_S` |
| L3b | **Corollary 1** — surprise identity: `F_S^eq(x_T) = (π_TS/2)x_TᵀN_Sx_T` (settled F_S = novelty score) | verified (err 1.8e-15; to derive by hand) | `surprise_identity_error` |
| L4 | **Theorem 1** — the push `u = π_ST ε_TS` is exact gradient ascent on `F_S^eq` (sign identity + envelope; no `N_S`) | verified (`surprise_identity_error` 1.8e-15) | `surprise_identity_error` |
| L4b | **Prioritization remark** — on the landscape, `ċ_k = (|π_ST|/τ_T)n_k c_k`: learned frozen, novel amplified, self-limiting | verified (rates ∝ n_k: learned→0, novel→+) | `novelty_operator` |
| L5 | ~~Lemma 2 (guard)~~ — **cut from the paper 2026-07-20**: replaced by the operating choice \|π_ST\| ≪ π_T (code default: half the stability threshold) + numerical containment | archived in additional_proofs_not_in_paper.md — do not reintroduce | `spectral_gap`, `manifold_leakage` |
| L6 | **Proposition 2** — exact saddle iff `π_ST = −π_TS` (active-inference-reminiscent) | proof drafted in final_proofs.md §8 (iff via Schwarz; author to check) | `circulation → 0` |

**Detailed, fully-justified proofs** of Prop 1, Lemma 1, Corollary 1, Theorem 1 (every transpose /
inverse / commutation spelled out): [detailed_proofs.md](detailed_proofs.md). **Intuitive, no-step-skipped
geometric version** of Theorem 1 (a picture per step): [intuitive_proof.md](intuitive_proof.md).
**Removed from the paper:** the old *interleaving-union* proposition — union & continual learning are
now **empirical** (figures), no theorem; the number **Proposition 2** is reused for the *saddle* (L6).
The **old self-extinction corollary** — folded into the prioritization remark as a one-line bound
(`|π_ST|n_k ≤ (|π_ST|π_S/π_TS)‖M_S u_k‖²`), not a standalone result. (The name **Corollary 1** is now
used for the *surprise identity*, L3b — a different, new result.)
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
symmetric matrices. (Used for the extinction bound.)

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

### L2 — Lemma 1: the fast student, mode by mode (dynamics-first — THE default route)
**Given.** The student's state ODE; $S_S$ symmetric with eigenframe $U$, dials $\mu_k = \|M_S u_k\|^2$;
frozen weights (frozen frame).
**Target — five moves, no inverses.** (1) Rotate the student's ODE into the frame: one scalar
tug-of-war per mode, $\tau_S \dot s_k = -\pi_{TS}(s_k - c_k) - \pi_S \mu_k s_k$. (2) Settle each
mode by dividing by the positive number $\pi_{TS} + \pi_S\mu_k$: $s_k^* = (1-n_k)c_k$,
$e_k = -n_k c_k$, with $n(\mu) = \pi_S\mu/(\pi_{TS}+\pi_S\mu)$. (3) Study the dial: $n \in [0,1)$,
$n(0)=0$, increasing, $n(\mu) \le (\pi_S/\pi_{TS})\mu$. (4) Reassemble: $\varepsilon_{TS} = -N_S x_T$
and $x_S^* = (I-N_S)x_T$ with $N_S := U\,\mathrm{diag}(n_k)\,U^\top$ — the operator is *born* as the
name of the reassembly; properties (symmetric, PSD, kernel = learned) by construction; closed form
$\pi_S S_S(\pi_{TS}I+\pi_S S_S)^{-1}$ by matching the action on each $u_k$. (5) The teacher's law,
per mode: $\tau_T \dot c_k = |\pi_{ST}| n_k c_k$ — **prioritization inside the lemma** (learned
frozen, most-novel dominates, quadratic self-extinction via (3)).
**Prereqs.** spectral theorem (rotate–scale–rotate), coordinate read-off, one-variable algebra.
**Status: derived by hand, dynamics-first (2026-07-19).** **Code check.** `novelty_operator`.

### L3 — Novelty spectrum and monotonicity
**Given.** $S_S v = \mu v$.
**Target.** $N_S v = n(\mu) v$ with $n(\mu) = \dfrac{\pi_S \mu}{\pi_{TS} + \pi_S \mu} \in [0,1)$, and
$n'(\mu) = \dfrac{\pi_S \pi_{TS}}{(\pi_{TS} + \pi_S \mu)^2} > 0$. Endpoints: $n(0)=0$ (learned), $n$ increasing.
**Prereqs.** spectral decomposition, one-variable calculus. **Code check.** eigenvalues of `N_S`.

### L3b — Corollary 1: the bridge (settled F_S = novelty score; the two routes meet)
**Given.** Lemma 1's per-mode data ($s_k^* = (1-n_k)c_k$, $e_k = -n_k c_k$, the scalar identity
$\pi_S\mu_k(1-n_k) = \pi_{TS} n_k$); ONE new atom — a squared norm is the sum of squared
coordinates in an orthonormal frame.
**Target.** $F_S^{\mathrm{eq}}(x_T) = \min_{x_S} F_S(x_S, x_T) = \tfrac{\pi_{TS}}{2} x_T^\top N_S x_T$,
hence $\nabla_{x_T} F_S^{\mathrm{eq}} = \pi_{TS} N_S x_T$. Proof = a three-line dial-by-dial
continuation of L2: per mode the interface term carries $n_k^2$, the self term
$\mu_k(1-n_k)^2 \to n_k(1-n_k)$ (absorb $\mu$ into $n$), summing to the plain novelty reading $n_k$.
**Why it matters — the triangle.** Theorem 1 (energy route, no $N_S$) and Lemma 1 (dynamics route,
no free energy) are two derivations sharing no steps; this corollary is where they provably meet:
$(|\pi_{ST}|/\pi_{TS})\,\nabla F_S^{\mathrm{eq}} = |\pi_{ST}| N_S x_T$ = Lemma 1's reassembled push.
Also the taxonomy sentence (novelty = spectrum of surprise) and the explicit landscape the figures
draw. **Prereqs.** L2. **Code check.** `diagnostics.surprise_identity_error`
(settled $F_S$ vs $(π_{TS}/2)x_T^\top N_S x_T$; err ~1.8e-15, asserted in `tests/smoke_test.py`).

### L4 — Theorem 1: the push is gradient ascent on the student's surprise (sign identity + envelope)
**Given.** Sleep ($\pi_{ST} < 0$); fast student state + frozen student weights. The push is read
directly off the teacher's equation: $u = \pi_{ST}\,\varepsilon_{TS}$. **No Lemma 1, no $N_S$.**
**Target — two steps.** *Step 1, the sign identity (any $x_S$, settled or not):* only the interface
term of $F_S$ contains $x_T$, so $\nabla_{x_T} F_S = -\pi_{TS}\varepsilon_{TS}$, hence
$u = (|\pi_{ST}|/\pi_{TS})\,\nabla_{x_T}F_S|_{x_S}$ — teacher and student feel one shared interface
energy through opposite-signed precisions; the sign of $\pi_{ST}$ IS the wake/sleep switch
($\pi_{ST}>0$: descent, no transfer). *Step 2, the envelope theorem:* $F_S$ is strictly convex in
$x_S$ (Hessian $\pi_{TS}I + \pi_S S_S \succ 0$), so the settled state is unique and
$F_S^{\mathrm{eq}} = \min_{x_S} F_S$ is well defined; differentiating through the minimizer is free
($\nabla_{x_S}F_S = 0$ there), so $\nabla F_S^{\mathrm{eq}} = -\pi_{TS}\varepsilon_{TS}|_{\text{settled}}$
and $u = (|\pi_{ST}|/\pi_{TS})\nabla F_S^{\mathrm{eq}}$: **exact gradient ascent on the student's
settled VFE**, every point ("steepest ascent" = one clause, standard fact).
**Downstream (separate results, not part of L4):** the landscape is $(π_{TS}/2)x_T^\top N_S x_T$
(L2 + L3b); prioritization on it — $\dot c_k = (|\pi_{ST}|/\tau_T) n_k c_k$, learned frozen,
self-limiting $|\pi_{ST}| n_k \le (|\pi_{ST}|\pi_S/\pi_{TS})\|M_S u_k\|^2$ — is L4b (the remark);
the saddle iff $\pi_{ST} = -\pi_{TS}$ is L6 (Prop 2). Caveats: local ascent; no seeding (noise
seeds); no "teacher's surprise" anywhere — the teacher's own surprise is $F_T$ (the self-pull).
**Prereqs.** L1 (the interface gradient), envelope theorem (stated + proved in detailed_proofs.md,
Thm 1 Step 2). **Code check.** `surprise_identity_error` ($F_S^{\mathrm{eq}}$ to 1.8e-15);
`novelty_operator` for L4b (n_k: learned→0, novel→+).

### L5 — (cut from the paper) manifold containment as operating choice + measurement
**Decision (2026-07-20).** The guard lemma (off-manifold normal stability under
$|\pi_{ST}| < \pi_T \sigma_{\min}^2$) is **not in the paper**. The paper instead states the
operating choice — reversed precision kept well below the teacher's self-precision (code default
`pi_ST = "auto"`: half the stability threshold, `pi_ST_safety = 0.5`) — and verifies containment
numerically. The full statement, proof, cross-block/leakage analysis, and a bounded-leakage
corollary are archived in
[additional_proofs_not_in_paper.md](additional_proofs_not_in_paper.md) (reviewer-rebuttal
material only — do not reintroduce into the paper).
**Code check.** `spectral_gap`, `manifold_leakage`, `offmanifold_growth`, `terminal_occupancy`.

### L6 — Proposition 2: exact `F_T − F_S` saddle condition
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
