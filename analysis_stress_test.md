# Analysis of the Two-Population Memory-Transfer Model
### Full reasoning, written to be followed without a heavy math background

This document derives *why* each claim in `two_population_memory_transfer_model.md` holds. I assume you know basic linear algebra notation loosely but not deeply, so **every time a non-obvious step, symbol, or concept appears, I stop and explain it in plain words first, then show the math.** Read the "Primer" once; after that the sections are self-contained.

---

## 0. Primer — the vocabulary you need

Read this once. Everything later refers back to it.

**Vector and matrix basics**
- `x ∈ ℝ^d` means `x` is a list of `d` numbers (a point in `d`-dimensional space). Here it's the activity of `d` neurons.
- A matrix `W ∈ ℝ^{d×d}` is a `d×d` grid of numbers. "Applying" it to a vector, `Wx`, produces a new vector — a linear transformation (rotate/stretch/mix the coordinates).
- `Wᵀ` ("W transpose") flips the matrix across its diagonal: row `i` becomes column `i`. If `W` connects neuron `j → i`, then `Wᵀ` connects `i → j`. It's the "reverse wiring."
- `I` = the identity matrix: `I x = x` (does nothing). Think of it as the number 1 for matrices.
- `‖x‖` = the *length* (norm) of the vector: `‖x‖² = x·x = Σ_i x_i²`. It's just Pythagoras in `d` dimensions. `‖x‖²` is the squared length.
- `xᵀy` (a row times a column) = the **dot product** `Σ_i x_i y_i`, a single number measuring overlap/alignment of two vectors.
- `x yᵀ` (a column times a row) = an **outer product**, a whole `d×d` matrix. (Order matters: `xᵀy` is a number, `x yᵀ` is a matrix.)

**Eigenvalues / eigenvectors (used in the stability argument)**
- For a matrix `S`, a vector `v` with `S v = λ v` is an **eigenvector**: `S` doesn't rotate `v`, it only scales it by the number `λ` (the **eigenvalue**). Eigenvectors are the "natural axes" of the transformation.
- If all eigenvalues of `S` are ≥ 0, `S` is called **positive semi-definite** — geometrically it never flips a vector to point "backwards," and `xᵀ S x ≥ 0` always.
- The **null space** of `S` = all vectors `v` with `S v = 0` (the directions `S` completely flattens). The **smallest nonzero eigenvalue** is the weakest non-flattened direction — the "spectral gap."

**Calculus on many variables (used everywhere)**
- A **functional / potential** `F(x)` is a rule that takes a vector and returns one number (e.g. an energy). Picture a landscape: height `F` over the plane of all `x`.
- The **gradient** `∇F` (or `∂F/∂x`) is the vector pointing in the direction of steepest *increase* of `F`, with length = how steep. Its components are the partial derivatives `∂F/∂x_i`.
- **Gradient descent** = "roll downhill": `dx/dt = −∇F`. The system moves to reduce `F`. A **Lyapunov function** is just a quantity that only ever decreases along the dynamics — proof that the system settles down rather than blowing up. If `dx/dt = −∇F`, then `F` is automatically a Lyapunov function.
- Two standard derivative facts we'll reuse (you can take these on faith):
  - `∂/∂x (½‖x‖²) = x`. (Slope of "½·length²" is the position itself — like `d/dx(½x²)=x` in 1-D.)
  - Chain rule for a linear map: `∂/∂x (½‖A x‖²) = Aᵀ A x`. (The `Aᵀ` appears because differentiating "pushes the map back through in reverse," which is exactly what transpose means.)

**Predictive-coding objects in this model**
- `M_T = I − W_T`, `M_S = I − W_S`: "mismatch operators." `M x` measures how far `x` is from being a fixed point of the recurrent weights `W` (i.e. from `Wx = x`). If `x` is a stored pattern, `Wx = x` so `Mx = 0`.
- **No autapses (stated up front).** Both `W_T` and `W_S` have a **zero diagonal**: entry `(i,i)` would be a neuron synapsing onto itself (an *autapse*), and the model forbids it. T's diagonal is frozen at zero; S's is re-zeroed after every learning step. This is a modeling assumption from the start, and it is load-bearing — see §6 for why allowing autapses would let the memory cheat.
- The three errors are just three linear maps of the states:
  - `ε_T = M_T x_T` — T's distance from its own memories.
  - `ε_TS = x_S − x_T` — the T–S disagreement at the student level.
  - `ε_S = M_S x_S` — S's distance from its own memories.
- "On the **manifold**" = the state lies in the set of stored patterns (a flat subspace here, because the model is *linear*). On it, the self-error is zero. "Off-manifold" = noise directions, where the self-error is nonzero and gets damped.

That's the whole toolkit. Now the analysis.

---

## 1. What the analysis establishes

I treat the model as a set of mathematical claims and verify each by direct computation. The headline results, in order of importance:

- **The VFE mapping is exact and derived.** The student's perception and learning are gradient descent on a single free energy `F_S` (§2).
- **The transfer mechanism works.** Adiabatic elimination of the fast student gives the **novelty operator** `N_S`; the drive sees known directions as zero and novel directions as a persistent residual (§3).
- **The sleep saddle is conditional.** The sleep flow is the gradient flow of a *single* functional `Φ = F_T − F_S` **only when `π_ST = −π_TS`**, and even then only as a *projected* saddle on the sphere `‖x_T‖=r₀`. Otherwise it is a non-potential flow (§4).
- **The structure guard has a tight form.** The safe default `|π_ST| < π_T·σ²_min` is conservative; the exact boundary is the operator inequality `|π_ST| N_S ≺ π_T S_T` (§5).
- **Learning is *projected* gradient descent** under the zero-diagonal (no-autapse) constraint — still a valid descent (§6).
- **Selection is spectral, with no winner-take-all.** In the mandated fast-S regime the known/novel split is structural; equally-unlearned directions are degenerate (§7, §8).
- **The linear model transfers a *subspace*.** Steps in the staircase count the *effective rank* of the memory set, not necessarily `P`; discrete episodic replay needs an added nonlinearity (§9).

§8 proves the core positive claim analytically — the drive prefers unknown *directions* to known ones, no simulation required — and §9 draws out the crucial consequence that those directions form a *subspace*, not a list of named memories.

---

## 2. The VFE mapping is exact

**The claim.** S's free energy is `F_S = (π_TS/2)‖ε_TS‖² + (π_S/2)‖ε_S‖²`, and S's perception and learning are gradient descent on it.

**Why this `F_S` shape (intuition).** `F_S` is a sum of two squared errors, each weighted by a *precision* (how much you trust that channel — a big weight means "this error matters a lot"). The first term punishes disagreeing with the top-down prediction `x_T` (weight `π_TS`); the second punishes disagreeing with S's own learned model (weight `π_S`). Minimizing this trades off "believe what you see" against "believe what you know." This is literally the negative-log-probability of a Gaussian model — the standard predictive-coding energy.

**Perception — check it's exact gradient descent.** We compute the gradient of `F_S` with respect to the state `x_S`, using the two derivative facts from the primer.

- First term, `(π_TS/2)‖x_S − x_T‖²`. The thing inside depends on `x_S` as `(x_S − x_T)`. Using `∂/∂x(½‖x‖²)=x`:
  `∂/∂x_S [(π_TS/2)‖x_S − x_T‖²] = π_TS (x_S − x_T) = π_TS ε_TS.`
- Second term, `(π_S/2)‖M_S x_S‖²`. Use the linear-map rule `∂/∂x(½‖Ax‖²)=AᵀAx` with `A = M_S`:
  `∂/∂x_S [(π_S/2)‖M_S x_S‖²] = π_S M_Sᵀ M_S x_S = π_S M_Sᵀ ε_S.`
- Add them: `∂F_S/∂x_S = π_TS ε_TS + π_S M_Sᵀ ε_S.`

Gradient descent means `dx_S/dt = −∂F_S/∂x_S` (roll downhill), times the time constant:
`τ_S dx_S/dt = −π_TS ε_TS − π_S M_Sᵀ ε_S.`
**This is exactly the model's perception equation.** The plus/minus signs and both precisions land perfectly — the student is pulled toward the top-down prediction (`−π_TS ε_TS = +π_TS(x_T − x_S)`) and toward its own prior.

**Learning — check it too.** Now differentiate `F_S` with respect to the *weights* `W_S`. Only the second term depends on them, through `ε_S = x_S − W_S x_S`. Writing it out coordinate by coordinate and differentiating (the algebra: `∂/∂W_S(½‖x_S − W_S x_S‖²) = −ε_S x_Sᵀ` — the outer product appears because each weight `W_{ij}` couples error component `i` to activity component `j`):
`∂F_S/∂W_S = −π_S ε_S x_Sᵀ.`
Gradient descent: `dW_S/dt = −∂F_S/∂W_S = +η π_S ε_S x_Sᵀ`. **Exactly the model's Hebbian rule.**

So the "VFE = exact and derived" status is fully justified — with one qualifier in §6 (the zero-diagonal projection). In the **wake** phase the teacher *also* descends, so wake is pure joint VFE minimisation; reversing the sign of `π_ST` (sleep) is what converts that bowl into the exploratory drive analysed next.

---

## 3. The transfer mechanism, and why it works

This is the heart of the model. The key move is **adiabatic elimination**: because S is much faster than T (`τ_S ≪ τ_T`), at any moment S has essentially already settled to its equilibrium for the current `x_T`. So we can "freeze" T, solve S's steady state, and see what drive T then feels in sleep.

**Solve S's steady state.** Set `τ_S dx_S/dt = 0`:
`π_TS (x_S − x_T) = −π_S M_Sᵀ M_S x_S`,  i.e.  `π_TS x_T = (π_TS I + π_S M_Sᵀ M_S) x_S`,
⟹ `x_S = π_TS (π_TS I + π_S M_Sᵀ M_S)⁻¹ x_T.`

(The `(...)⁻¹` is a matrix inverse — the matrix analog of dividing. It exists because the bracket is `π_TS I` plus a positive-semidefinite matrix, so all its eigenvalues are ≥ `π_TS > 0`.)

**The residual mismatch S leaves behind.** Plug back into `ε_TS = x_S − x_T`:
`ε_TS = −(π_TS I + π_S M_Sᵀ M_S)⁻¹ · π_S M_Sᵀ M_S · x_T.`

**Name the pieces (this is the cleanest object in the whole model).** Write `S_T = M_TᵀM_T` and `S_S = M_SᵀM_S` (both symmetric, eigenvalues ≥ 0 — read them as "how surprised T / S is, per direction"). Then the residual is `ε_TS = −N_S x_T` with the **novelty operator**
`N_S = π_S S_S (π_TS I + π_S S_S)⁻¹.`
`N_S` is the student-dependent filter that turns T's raw state into "the part of it S cannot explain." It shares `S_S`'s eigenvectors, and along an eigen-direction with `S_S`-eigenvalue `μ ≥ 0` it is just a number `n(μ) = π_S μ /(π_TS + π_S μ)` — a gain between 0 and 1. (Almost everything below is most cleanly stated through `N_S`.)

Now read this in the two cases, recalling that in sleep the drive on T is `π_ST ε_TS` with the **reversed precision** `π_ST < 0`; since `ε_TS = −N_S x_T`, that drive is `−π_ST N_S x_T = +|π_ST| N_S x_T`:

- **Known pattern.** S has learned it ⟹ `M_S` flattens it ⟹ `μ = 0` ⟹ `ε_TS = 0`. **No drive.** T feels nothing; the memory is "explained away."
- **Novel pattern.** S is empty there ⟹ `W_S = 0` ⟹ `M_S = I` ⟹ `μ = 1` ⟹ `ε_TS = −[π_S/(π_TS + π_S)] · x_T`. A *persistent residual* survives. The reversed-precision drive `π_ST ε_TS = +|π_ST|·[π_S/(π_TS+π_S)] x_T` (`π_ST<0`) **pushes `x_T` further along that same novel direction** — positive feedback.

So the discrimination is real and built into the algebra. Note the residual fraction `π_S/(π_TS+π_S)` is **less than 1** (since `π_TS > π_S > 0` by the precision guard) — S "partially copies" even a novel pattern. This `<1` factor matters in §5.

**What renormalization actually does (it is *not* winner-take-all over named memories).** Positive feedback alone would blow `x_T` up. Rescaling `‖x_T‖` back to `r₀` every step converts "grow" into "*rotate toward the fastest-growing directions*" — mathematically power iteration, which converges to the **dominant eigen-*space*** of the growth operator. If that space is one-dimensional (a unique strongest direction) you get a single clean winner; but when several unlearned directions share the *same* growth rate — exactly the case for equally-novel patterns — power iteration does **not** single one out, it leaves `x_T` free to drift within their span. So T is driven onto the *unknown subspace*, not onto a particular named memory. This is the heart of §9.

**Why it terminates.** Once S has learned all P patterns, `M_S` flattens the whole manifold, so on-manifold `ε_TS → 0` and `ε_S → 0`: no drive, no learning. Off-manifold noise still creates some `ε_TS`, but T's intrinsic damping `−π_T M_Tᵀ M_T` (eigenvalues ≥ `π_T σ²_min > 0` there) suppresses it as long as the structure guard holds. So the end state is "T diffuses gently on the manifold, nothing is learned" — a genuine fixed point of the learning, and the correct "done" signal.

---

## 4. The sleep saddle is exact only when `π_ST = −π_TS`

A natural and elegant reading of the sleep dynamics is that the *entire* `(x_T, x_S, W_S)` flow is **the gradient flow of a single functional**, with the teacher descending and the student ascending. That reading is exact, but only when the teacher's reversed precision is the exact negative of the student's.

### 4.1 What a "saddle / minimax" is, intuitively

Imagine one landscape `Φ` (a single height function over all the variables). A **minimax / saddle** dynamic is two players on that *one* landscape who disagree about up:
- Player T always rolls **downhill** (`dx_T/dt = −∂Φ/∂x_T`) — minimizing.
- Player S always climbs **uphill** (`dx_S/dt = +∂Φ/∂x_S`) — maximizing.

Like a horse's saddle: from front-to-back it curves down (a valley for T), from side-to-side it curves up (a ridge for S). The resting point is the saddle's seat. This is a strong, clean structure — *if* a single `Φ` really generates both players' motion.

### 4.2 The correct functional is `Φ = F_T − F_S`

We already know (§2) that S's perception and learning are gradient *descent* on `F_S`. "S ascends `Φ`" therefore means `Φ` must contain `−F_S` (ascending a negative is descending the positive). The only `x_T`-dependent piece left for T is its self-energy `F_T = (π_T/2)‖ε_T‖²`. So the candidate is:
```
Φ = F_T − F_S = (π_T/2)‖M_T x_T‖² − (π_TS/2)‖ε_TS‖² − (π_S/2)‖ε_S‖².
```
Check S (ascending): `+∂Φ/∂x_S = −π_TS ε_TS − π_S M_Sᵀ ε_S` ✅ and `+∂Φ/∂W_S = π_S ε_S x_Sᵀ` ✅. **Both of S's equations come out exactly right.**

### 4.3 But T only matches if `π_ST = −π_TS`

Now check T on this `Φ`. T's actual interface drive is `+π_ST ε_TS = +π_ST(x_S − x_T)` (with `π_ST` the signed precision); the student's is `−π_TS ε_TS = −π_TS(x_S − x_T)`. The clean way to pin when these two come from a single `Φ` is the cross-derivative test (the nested signs in `∂Φ/∂x_T` are easy to slip on — exactly the trap that makes the condition *look* like `π_ST = −π_TS`; the symmetry test settles it unambiguously).

**Cross-derivative symmetry.** For *any* genuine landscape `Φ`, the "cross slopes" must be consistent: how T's slope changes as you move S must mirror how S's slope changes as you move T (the mixed second derivatives are equal — the same "no circulation / curl-free" condition that decides whether a force field has a potential at all). For a saddle (T descends `Φ`, S ascends it) the off-diagonal Jacobian blocks must cancel under transpose: `∂f_T/∂x_S + (∂f_S/∂x_T)ᵀ = 0`. Read off the two coupling terms:
- T's drive `+π_ST(x_S − x_T)` gives `∂f_T/∂x_S = +π_ST·I`.
- S's drive `−π_TS(x_S − x_T)` gives `∂f_S/∂x_T = +π_TS·I`.
- Cancellation requires `π_ST·I + π_TS·I = 0`, i.e. **`π_ST = −π_TS`** — the teacher's reversed precision is the exact negative of the student's.

If `π_ST ≠ −π_TS`, the cross-slopes don't match, **no `Φ` exists**, and the flow is *not* a gradient/saddle flow at all. It's a genuinely non-conservative coupled system (it can have rotational/circulating components a potential can never produce). The mismatch is purely in the coupling ratio, so no choice of the time constants `τ` can fix it (`τ_T` rescales T's *entire* right-hand side by one factor, leaving the ratio untouched).

### 4.4 At `π_ST = −π_TS`, the clean zero-sum game

When `π_ST = −π_TS`, `Φ` collapses to the clean **zero-sum game `Φ = F_T − F_S`**: T minimizes `F_T − F_S` (lower its own error, *raise* S's surprise), S minimizes `F_S`. That is a textbook adversarial saddle. Geometrically `Φ` is a **bowl** along directions S can predict (T is pinned — flat) and a **hill** along novel directions (T slides off — the replay drive); S's learning turns each hill into a bowl.

**One more caveat — the renormalization.** Even at `π_ST = −π_TS`, the exact-saddle statement is about the *unconstrained* ODE. The model also renormalizes `‖x_T‖ = r₀` every step, and that radial rescaling is **not** generated by `Φ`. So the precise object is a **saddle flow of `Φ` restricted to the sphere**: T's velocity is the `Φ`-gradient *projected onto the tangent of the sphere*, via `P_{x_T^⊥} = I − x_T x_Tᵀ/‖x_T‖²` (the operator that strips off any radial, norm-changing component). The saddle picture is right; just read it as living on the sphere `‖x_T‖ = r₀`, not in full `ℝ^d`.

### 4.5 Why this matters (it's not pedantry)

§8 of the spec makes a careful, valuable distinction: the EFE/active-inference reading is honestly labeled "constructed / structural / not derived," while the minimax reading is exact *theorem-grade* — but it carries a hidden condition (`π_ST = −π_TS`) that the rest of the model leaves free (indeed it gives `π_ST` and `π_TS` separate names, separate jobs, and separate guards, which signals they're meant to be independent). So either adopt `π_ST = −π_TS` (and enjoy a real theorem), or read Reading 1 as "structural," on the same honest footing as the EFE story. The spec offers the former as an option and defaults to the latter.

---

## 5. The structure guard — safe default, tight boundary, operator form

**The safe default.** `|π_ST| < π_T·σ²_min` keeps T from amplifying noise in sleep, where `σ²_min` is the smallest nonzero eigenvalue of `M_Tᵀ M_T` (the spectral gap).

**Set up the stability question.** Take a pure noise (off-manifold) direction and ask: does T's state grow or shrink there in sleep? Using the adiabatic S-steady-state from §3, T's linearized sleep motion along such a direction is
`τ_T dx_T/dt ≈ (−π_T λ + |π_ST|·[π_S/(π_TS + π_S)]) x_T,`
where:
- `−π_T λ` is T's intrinsic damping; off-manifold `λ ≥ σ²_min > 0` (this is what "spectral gap" buys you — every noise direction is damped at least this hard).
- `|π_ST|·[π_S/(π_TS+π_S)]` is the destabilizing drive — note the **`<1` attenuation factor** `π_S/(π_TS+π_S)`, because (§3) S partially copies even novel/noise directions, so T doesn't feel the full `|π_ST|`.

**The tight boundary.** The direction is stable (shrinks) when damping beats drive:
`π_T λ > |π_ST|·π_S/(π_TS + π_S)`, worst case `λ = σ²_min`, giving
`|π_ST| < π_T·σ²_min · (π_TS + π_S)/π_S.`

Since `(π_TS+π_S)/π_S > 1`, this true threshold is **larger** (more permissive) than the safe default `|π_ST| < π_T·σ²_min`. The safe default is *sufficient* (always safe) but conservative — it throws away the attenuation factor. That's a defensible choice: `|π_ST| < π_T·σ²_min` is **S-independent**, so it holds no matter what S has learned, which is exactly what you want for a guarantee.

**The fully general boundary (operator form).** The scalar `|π_ST| < π_T·σ²_min·(π_TS+π_S)/π_S` secretly assumes the novelty operator `N_S` and T's surprise operator `S_T` share eigenvectors, so everything is a scalar per axis. In general they don't — T's noise directions and S's novelty directions can be tilted relative to each other. The exact, assumption-free condition is the **operator inequality** on the off-manifold subspace (where `S_T` is invertible):
`|π_ST| N_S ≺ π_T S_T`,   equivalently   `|π_ST| < π_T / λ_max(S_T^{−1/2} N_S S_T^{−1/2})`.
("`X ≺ Y`" means "Y − X has all-positive eigenvalues," i.e. damping `π_T S_T` beats drive `|π_ST| N_S` in *every* direction at once, not just on average.) Because `0 ⪯ N_S ≺ I` always, picking `|π_ST| < π_T·σ²_min` satisfies this no matter what S has learned — which is exactly why it is the safe default. The scalar threshold above is just the special aligned-eigenvector case of this matrix inequality.

---

## 6. Learning is *projected* gradient descent (and why no-autapse is mandatory)

**Neither population has autapses.** As noted in the primer, both `W_T` and `W_S` carry a **zero diagonal** — entry `(i,i)` would be the synapse from neuron `i` onto itself (an *autapse*), and the model forbids it. T's diagonal is baked in and frozen; S's is re-zeroed after every learning step so it can never drift away from zero.

**Why autapses would break it (not just biological tidiness).** If `W_S[i,i]` were allowed, a neuron could satisfy its own prediction-error by predicting *itself*: set `W_S[i,i] ≈ 1` and the self-error component `ε_S,i` collapses to zero with no actual association learned. The unit then looks "explained" while having stored nothing relational — the memory degenerates into a bank of trivial self-loops. Pinning the diagonal to zero forces each neuron to be explained by the *others*, which is exactly what makes S store genuine inter-neuron memories. So no-autapse is load-bearing — it's what makes a covPCN memory a memory. (The same logic protects T.)

**What the constraint costs the gradient story.** Given the zero-diagonal constraint, S's learning is **not** literally the unconstrained gradient of `F_S`. The true gradient `ε_S x_Sᵀ` has nonzero diagonal entries; deleting them is exactly the *orthogonal projection* of the gradient onto the subspace of zero-diagonal matrices. That subspace is linear (flat), so projected descent still lowers `F_S` monotonically — `F_S` stays a Lyapunov function and learning still self-extinguishes as `ε_S → 0`. Nothing breaks; the precise phrase is **projected** gradient descent on `F_S`. (Perception puts no constraint on `x_S`, so it *is* exact — §2 stands.)

---

## 7. Spectral selection (not a speed race), and the ordering question

**In the required regime, the known/novel split is spectral.** The model *requires* `τ_S ≪ τ_T` — S is so fast it's essentially always at steady state. And at steady state (§3), the known-vs-novel distinction is decided entirely by whether `M_S` flattens the direction (`μ = 0` vs `μ = 1`) — a **structural / spectral** property, independent of how fast S runs. A *genuinely dynamic* (speed-dependent) effect only shows up if you raise `τ_S` toward `τ_T` — which is precisely the ablation in §7 of the spec. So: spectral in-regime, dynamic only as you leave it.

**"Next-most-novel" is underdetermined — and there is no winner-take-all.** Every pattern S hasn't seen *at all* has the identical novelty gain `n(1) = π_S/(π_TS+π_S)` (§3), so there is no graded "how novel" signal to rank them — *and* no winner-take-all to pick one out either: with equal growth rates the renormalized flow simply diffuses inside the degenerate subspace (see §3 and §9). What actually breaks the tie is the slow learning itself — whichever direction noise happens to load `x_T` onto when `W_S` starts updating gets extinguished first, and that direction need not be a named pattern. The experiment that pre-trains a subset and checks that held-out directions transfer first tests only **binary** novelty (seen vs unseen), which the model genuinely delivers.

---

## 8. Proof: the drive prefers unknown directions over known ones (no simulation needed)

This is the rigorous backbone of §3. **Claim:** that T's state is pushed toward directions S has *not* learned, and not toward ones it already has, is a *theorem* — provable straight from the equations. Only the tie-break *among several equally-unknown* directions actually needs simulation.

Everything hinges on one quantity per direction: its **growth rate** inside `x_T`. "Driven toward direction `p`" means precisely: *if a small amount of `p` is present in `x_T`, does it grow?* Positive growth rate → attracted to it; zero or negative → not.

**Assumptions (kept explicit; the qualitative result survives weaker forms — only the constants move):**
- *(A1)* the stored patterns `m_p` are orthonormal and are the null directions of `M_T` (`M_T m_p = 0`) — clean axes.
- *(A2)* `M_S` nulls a pattern S has learned (its `M_SᵀM_S`-eigenvalue `μ_p = 0`) and acts as the identity on one it hasn't (`μ_p = 1`); partial learning gives `μ_p ∈ (0,1)`.
- *(A3)* the structure guard holds: `|π_ST| < π_T·σ²_min·(π_TS+π_S)/π_S`.
- *(A4)* S is fast enough to sit at its steady state (the `τ_S ≪ τ_T` regime), so we may use the `ε_TS` formula from §3.

**Step 1 — coordinates.** Write `x_T = Σ_q a_q m_q + (off-manifold part)`, where `a_p = m_pᵀ x_T` is "how much of pattern `p`" is currently in T. We track each `a_p`.

**Step 2 — T's own dynamics are silent along the memories.** Project the intrinsic relaxation `−π_T M_Tᵀ M_T x_T` onto `m_p`:
`m_pᵀ M_Tᵀ M_T x_T = (M_T m_p)ᵀ (M_T x_T) = 0`   because `M_T m_p = 0`.
So T's self-relaxation neither feeds nor drains any stored direction — it only damps the off-manifold part. *Every* memory-direction motion comes from the reversed-precision drive `π_ST ε_TS` (`π_ST<0`, plus noise). This is "T is flat," made precise.

**Step 3 — the drive's growth rate, known vs unknown.** Insert S's steady state `ε_TS = −N_S x_T` and project onto `m_p` (an eigen-axis, so the operator is just a number there). The drive is `π_ST ε_TS = −π_ST N_S x_T` (with `π_ST<0`, that is `+|π_ST| N_S x_T`), so:
```
τ_T · da_p/dt = g_p · a_p ,     g_p = −π_ST · π_S μ_p / (π_TS + π_S μ_p) = |π_ST| · π_S μ_p / (π_TS + π_S μ_p).
```
Read it off:

| direction | `μ_p` | growth rate `g_p` |
|---|---|---|
| known pattern | 0 | **0** |
| unknown pattern | 1 | **`|π_ST| π_S/(π_TS+π_S) > 0`** |
| off-manifold noise | (1) | `−π_T λ + |π_ST| π_S/(π_TS+π_S)`, with `λ ≥ σ²_min` → **< 0** under (A3) |

These three lines are the heart of it: known = exactly zero, unknown = strictly positive, noise = strictly negative. The inequalities are algebraic — true for *all* valid parameters, nothing to simulate.

**Step 4 — from "grows faster" to "the state actually goes there."** A positive growth rate would blow up without the leash; renormalizing `‖x_T‖ = r₀` converts growth into a **competition on a sphere**. The renormalized flow is
`ẋ_T = G x_T − (x_Tᵀ G x_T / r₀²) x_T`,
where `G = −π_T S_T − π_ST N_S` (with `π_ST<0`, the `−π_ST N_S = +|π_ST| N_S` term grows) is the symmetric operator whose eigenvalues are the `g_p` above. This is the textbook Rayleigh-quotient / power-iteration flow: it provably converges to the eigenspace of the *largest* eigenvalue of `G`. The largest `g_p` is the unknown value (`> 0`); known patterns sit at `0`, noise below `0`. Therefore `x_T` is driven onto the **unknown-pattern subspace** and away from both known patterns and noise. ∎ (a convergence theorem, not a numerical observation.) Note the conclusion is about a *subspace*: within it, equally-unlearned directions are degenerate and the flow does **not** single out a named pattern — §9 draws out why that matters.

**A graded order once learning is underway.** `g_p = |π_ST| π_S μ_p/(π_TS+π_S μ_p)` is **strictly increasing in `μ_p`**, and `μ_p` is literally *how much of pattern `p` S still fails to predict*. So once patterns are partly learned, the **least-encoded one always gets the strongest push** — a genuine graded "finish the least-known first" ordering, defined by *current learning state* rather than fixed identity.

**The drive shuts itself off (termination, spectral form).** Take a direction `v` S is currently learning and suppose it is approximately an eigenvector of `W_S` with eigenvalue `w` (`W_S v = w v`). Then `M_S v = (1−w)v`, so `S_S v = (1−w)² v`, and the novelty gain there is `n_v = π_S(1−w)² / (π_TS + π_S(1−w)²)`. As learning drives `w → 1`, `n_v → 0` **quadratically** in the residual `(1−w)`. So the instant a direction is learned, its replay drive collapses and T moves on — the precise statement of "the system goes to the next memory," and the reason consolidation self-terminates.

**What still needs simulation (not provable by this argument):**
- the tie-break *among fully-unseen directions* — all have `μ = 1`, identical `g`, so which one goes first is set by the noise `ξ` plus competition; genuinely stochastic.
- dwell times and the exact staircase shape — these couple the slow `W_S` learning to the fast loop.
- robustness when (A1)–(A4) are relaxed: non-orthogonal/overlapping patterns, `μ_p` strictly between 0 and 1, finite (not infinite) timescale separation.

---

## 9. The linear model transfers a *subspace*, not discrete memories

This is the most important conceptual point. The intuitive narrative ("noise stirs T → S learns one unlearned memory at a time → `F_S` falls in `P` discrete steps") is not, in the **linear** covPCN, a guarantee. Here's why, and what the precise claim is.

**Why "memories" aren't individuated in a linear model.** T's memories are the null directions of `M_T`: `M_T m_p = 0`. But `M_T` is *linear*, so if `M_T m_1 = 0` and `M_T m_2 = 0`, then `M_T(α m_1 + β m_2) = 0` for any `α, β`. Every linear combination of stored patterns is *also* a perfect memory of T. So T does not hold `P` isolated attractors — it holds a flat **memory subspace** `ker M_T`, and *no point inside it is special*. (This is the deliberate "T is flat" choice of §1, followed to its logical end.)

**What the dynamics therefore transfer.** From the proof (§8), T is driven onto the *unlearned part* of that subspace, and within it (§3) every still-novel direction carries the identical gain `n(1)`. The growth operator `G = −π_T S_T − π_ST N_S` (with `π_ST<0`, the `−π_ST N_S = +|π_ST| N_S` term grows) restricted to the unlearned subspace is a *multiple of the identity* — **fully degenerate**. Renormalization (a power iteration) converges to that whole subspace, not to a basis vector of it. Which specific directions get carved off and learned first is set by noise plus the order learning happens to extinguish them — and a learned direction can be an arbitrary *rotation* of the named patterns. So S learns the **span** of T's manifold, direction by direction, **not** pattern by pattern.

**The precise statement of the staircase.** The `F_S` (or novelty-spectrum) curve still falls in steps, but:
> the number of steps is the **effective rank** of the memory subspace (the count of independent directions S still lacks), *not* the raw pattern count `P`; and each step is a newly-acquired independent *direction*, not necessarily a named memory `m_p`.

For uncorrelated (orthonormal) patterns the effective rank equals `P`, so you may *see* `P` steps — but that's a coincidence of the geometry, not a guarantee that each step is one labelled memory. For **correlated** patterns the rank is lower, so there are **fewer than `P`** steps, and learning one memory automatically reduces novelty on its correlated neighbours.

**The sharp observable.** Let `U_T` be an orthonormal basis of T's memory manifold (`ker M_T = span(U_T)`). Track the **restricted novelty spectrum**
`U_Tᵀ N_S(t) U_T`,
a small matrix whose eigenvalues say exactly which parts of T's manifold S still cannot explain. Transfer is complete when its largest eigenvalue `→ 0`. Because it measures *directions*, it is robust to the basis ambiguity above and is the right way to read the staircase for correlated memory sets.

**This is a feature, not a failure — it names the base model:**
> the linear version is **prioritized *subspace* consolidation**, driven by the student's novelty operator `N_S`.

To get prioritized **episodic** (discrete, one-pattern-at-a-time) replay, you must add something that *individuates* patterns — a nonlinearity / sparsity / soft-WTA term / point-attractor structure in T or S. That is the natural next extension, and it cleanly separates "what the linear model provably does" from "what the full memory-consolidation story wants." (Note this is a *different* nonlinearity question from §10 of the spec, which is about T's landscape being rugged enough to need planning; here it's about T's memories being discrete enough to be individuated.)

---

## 10. How to confirm all this numerically

Each result has a concrete, testable prediction; the notebooks under `notebooks/` and the probes under `tests/` carry them out:
- **Saddle exactness** (`02_findingD_saddle.py`): sweep the signed precision `π_ST` and compute the "circulation" (the antisymmetric part of the system's Jacobian, `∂f_T/∂x_S + (∂f_S/∂x_T)ᵀ`). It is ≈ 0 **only when `π_ST = −π_TS`**, and equals `|π_ST + π_TS|·√d` exactly.
- **Stability guard** (`03_findingE_stability.py`): sweep `|π_ST|` (the reversed-precision magnitude) at fixed precisions; the noise-chasing onset occurs near `π_T·σ²_min·(π_TS+π_S)/π_S`, *above* `π_T·σ²_min`. With deliberately mis-aligned `S_T` and `N_S` eigenbases, the onset tracks the operator bound, not the scalar one.
- **Spectral selection** (`04_findingG_timescale.py`): sweep `τ_S/τ_T`; binary novel-vs-known sorting stays sharp deep into the fast-S regime (spectral), while any graded/ordering effect sharpens only near `τ_S ≈ τ_T`.
- **Subspace transfer** (`05_findingH_subspace.py`): run with **correlated** patterns and track `U_Tᵀ N_S U_T`. The number of clean staircase steps equals the **effective rank** of the memory set — *fewer than `P`* — and the learned directions need not align with the named `m_p`. A soft-WTA/nonlinearity would restore one-pattern-per-step.
- **Eigenspace geometry** (`06_eigenspace_geometry.py`): in a `d=3` network, watch `S_S`'s memory eigenvalues collapse to 0 and the novelty sphere cool on the memory great circle.

*Future extension:* replace the raw-mismatch reversed-precision drive `u = π_ST ε_TS` (`π_ST<0`) with a **learnability-weighted** drive — weight `ε_TS` by how much S's uncertainty can still shrink in that direction — so the action chases *learning progress* rather than mere surprise. That sharpens the active-inference reading by separating surprise ≠ learning-progress ≠ information-gain, and is the natural partner to the nonlinear/episodic extension of §9.
