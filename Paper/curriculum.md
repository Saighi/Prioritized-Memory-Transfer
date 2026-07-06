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
| T3 | Projections: subspace + sphere tangent (toolkit) | partial (zero-diag done; sphere/subspace in L4) | — |
| T4 | Rayleigh flow in an eigenbasis (toolkit) | not-started (in L4) | — |
| L1 | **Proposition 1** — student inference & learning descend `F_S` | verified (err 0.0) | `check_gradients` |
| L2 | **Lemma 1** — fast-student elimination → novelty operator `N_S` | verified (learned→0, unlearned→0.36, id 1e-15) | `novelty_operator` |
| L3 | Novelty spectrum `n(μ)` and monotonicity `n'(μ)>0` | verified (eig map 7e-16, monotone) | eig of `N_S` |
| L4 | **Theorem 1** — Rayleigh ascent on `A_S = U_Tᵀ N_S U_T` | deriving | `restricted_novelty_spectrum` |
| L5 | **Corollary 1** — self-extinction `vᵀN_Sv ≤ (π_S/π_TS)‖M_Sv‖²` | not-started | `transfer_deficit` |
| L6 | **Lemma 2** — off-manifold normal stability | not-started | `spectral_gap`, `manifold_leakage` |
| L7 | **Proposition 2** — interleaving union `range(C_mix)=Σ range(C_k)` | not-started | `subspace_sum_basis` |
| L8 | **Proposition 3** — exact saddle iff `π_ST = −π_TS` | not-started | `circulation → 0` |

---

## Notation (fixed for the whole curriculum)

- $d$ neurons per population; $x_T, x_S \in \mathbb{R}^d$ teacher/student states.
- $W_T, W_S$ recurrent weights (zero diagonal); $M_T = I - W_T$, $M_S = I - W_S$.
- Self-surprise operators $S_T = M_T^\top M_T$, $S_S = M_S^\top M_S$ (symmetric PSD).
- Errors: self $\varepsilon_T = M_T x_T$, $\varepsilon_S = M_S x_S$; interface $\varepsilon_{TS} = x_S - x_T$.
- Precisions: $\pi_T, \pi_S > 0$ (self); $\pi_{TS} > 0$ (student interface); $\pi_{ST}$ signed
  (teacher interface — sleep $<0$, wake $>0$).
- Free energies: $F_S = \tfrac{\pi_{TS}}{2}\|\varepsilon_{TS}\|^2 + \tfrac{\pi_S}{2}\|\varepsilon_S\|^2$, and $F_T = \tfrac{\pi_T}{2}\|\varepsilon_T\|^2$.
- $U_T$ = orthonormal basis of the teacher memory manifold $\ker M_T$; $P_T = U_T U_T^\top$.
- Novelty operator $N_S = \pi_S S_S(\pi_{TS} I + \pi_S S_S)^{-1}$; restricted $A_S = U_T^\top N_S U_T$.

---

## Toolkit warm-ups

### T1 — Matrix differentiation
**Target.**

$$
\nabla_x \tfrac12\|Ax\|^2 = A^\top A x, \qquad \nabla_W \tfrac12\|x - Wx\|^2 = -(x - Wx)x^\top .
$$

Derive each coordinate-by-coordinate once; then gradient steps become routine.

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
**Target.** $x_S^\* = \pi_{TS}(\pi_{TS} I + \pi_S S_S)^{-1} x_T$ and $\varepsilon_{TS} = -N_S x_T$ with
$N_S = \pi_S S_S(\pi_{TS} I + \pi_S S_S)^{-1}$; and $N_S U_T = 0$ on learned directions.
**Prereqs.** linear solve, positive-definite inverse. **Code check.** `novelty_operator`.

### L3 — Novelty spectrum and monotonicity
**Given.** $S_S v = \mu v$.
**Target.** $N_S v = n(\mu) v$ with $n(\mu) = \dfrac{\pi_S \mu}{\pi_{TS} + \pi_S \mu} \in [0,1)$, and
$n'(\mu) = \dfrac{\pi_S \pi_{TS}}{(\pi_{TS} + \pi_S \mu)^2} > 0$. Endpoints: $n(0)=0$ (learned), $n$ increasing.
**Prereqs.** spectral decomposition, one-variable calculus. **Code check.** eigenvalues of `N_S`.

### L4 — Theorem 1: spectral prioritization of the manifold-restricted sleep drive
**Given.** Sleep ($\pi_{ST} < 0$); restrict to the manifold $x_T = U_T y$, $\|y\| = r_0$; fast student.
**Target.** $\tau_T \dot y = |\pi_{ST}|\,(I - \tfrac{yy^\top}{r_0^2}) A_S\, y$ with $A_S = U_T^\top N_S U_T \succeq 0$;
hence $\tfrac{d}{dt}(\tfrac12 y^\top A_S y) \ge 0$ (Rayleigh ascent). Consequences: state approaches a
dominant eigenspace of $A_S$; learned directions are neutral; larger restricted novelty → stronger
amplification; degeneracy → eigenspace not a named direction. **Does not claim completion.**
**Prereqs.** T2, T3, T4. **Code check.** `restricted_novelty_spectrum`.

### L5 — Corollary 1: replay gain extinguishes with squared student residual
**Given.** $N_S = \pi_S S_S(\pi_{TS} I + \pi_S S_S)^{-1}$.
**Target.** $N_S \preceq \dfrac{\pi_S}{\pi_{TS}} S_S$, so for unit $v$: $v^\top N_S v \le \dfrac{\pi_S}{\pi_{TS}}\|M_S v\|^2$.
Thus the drive vanishes $O(\|M_S v\|^2)$; at full transfer $M_S U_T = 0 \Rightarrow N_S U_T = 0 \Rightarrow A_S = 0$
(deterministic drift vanishes on a *set* of sphere equilibria). **Prereqs.** T2. **Code check.** `transfer_deficit`.

### L6 — Lemma 2: conservative off-manifold normal stability
**Given.** $q \in U_T^\perp$; sleep operator $-\pi_T S_T + |\pi_{ST}| N_S$.
**Target.** $q^\top(-\pi_T S_T + |\pi_{ST}| N_S)q \le -(\pi_T \sigma_{\min}^2 - |\pi_{ST}|)\|q\|^2$, so
$|\pi_{ST}| < \pi_T \sigma_{\min}^2$ makes the normal–normal block strictly contracting. Caveat: this is
NOT manifold invariance (cross-block $Q_T N_S P_T$ leaks). **Prereqs.** Rayleigh quotient, eigenvalue bound.
**Code check.** `spectral_gap`, `manifold_leakage`.

### L7 — Proposition 2: interleaving guarantees the teacher-subspace union
**Given.** Second moments $C_k = \mathbb{E}[x_k x_k^\top]$; interleaved bouts with weights $p_k > 0$.
**Target.** $C_{\mathrm{mix}} = \sum_k p_k C_k$ and $\mathrm{range}(C_{\mathrm{mix}}) = \sum_k \mathrm{range}(C_k)$
(union of subspaces $U_\Sigma = U_1 + U_2$, capacity $r_\Sigma \le d-1$). **Prereqs.** second-moment
range algebra, ranks. **Code check.** `subspace_sum_basis`, `mixture_min_eig`.

### L8 — Proposition 3: exact `F_T − F_S` saddle condition
**Given.** $\Phi = F_T - F_S$; teacher/student mobilities; tangent projection on the teacher sphere.
**Target.** $\Phi$ generates the sleep dynamics iff $\pi_{ST} = -\pi_{TS}$ (Schwarz symmetry of mixed
partials), with the three qualifications (mobilities; projected $W_S$ ascent; Riemannian gradient on
the sphere). Framed as active-inference-*reminiscent*, not identical. **Prereqs.** T1, mixed partials.
**Code check.** `circulation → 0`.
