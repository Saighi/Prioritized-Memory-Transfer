# The algebra & calculus cheat-sheet for the proofs

Everything used in [detailed_proofs.md](detailed_proofs.md) (Prop 1, Lemma 1, Corollary 1, Theorem 1),
organized in three layers:

1. **Rules** (§1–§10) — the raw algebra facts, grouped by topic, each with where it is used.
2. **Proof patterns** (§11) — the recurring *moves* that combine those rules; proofs are made of these.
3. **House identities** (§12) — the model-specific formulas to know cold.

Rows marked **★** are the "know by heart" set; they are collected into a flash-card list in §13.

Two master mnemonics cover most accidents before they happen:
(1) **transpose and inverse both flip the order of a product**;
(2) **matrices don't commute unless one is built from the other (or they share an eigenbasis) — default to "no."**

---

## 1. Transpose (rule R1 of the proofs)

| Rule | Note | Used |
|---|---|---|
| ★ (AB)ᵀ = Bᵀ Aᵀ | **order flips** | everywhere |
| (ABC)ᵀ = Cᵀ Bᵀ Aᵀ | flips the whole chain | Prop 1 S1 |
| (Aᵀ)ᵀ = A | transpose twice = back | — |
| (A + B)ᵀ = Aᵀ + Bᵀ | splits over sums | I − N_S symmetric (Cor 1 S3) |
| (A⁻¹)ᵀ = (Aᵀ)⁻¹ | transpose & inverse commute | B⁻¹ symmetric (Lemma 1 S7) |
| ★ a scalar equals its own transpose: xᵀy = yᵀx, and more generally s = sᵀ for any 1×1 | the trick that merges the two middle terms of a quadratic expansion | Thm 1 core, Cor 1 |
| xᵀAy = yᵀAᵀx; if A symmetric, xᵀAy = yᵀAx | "slide A across by transposing it" | Thm 1 C1 (u_kᵀN_S = n_k u_kᵀ) |

## 2. Products — what you may always do

| Rule | Note | Used |
|---|---|---|
| A(BC) = (AB)C | **associative** — regroup freely, no permission needed | everywhere |
| A(B + C) = AB + AC, (A+B)C = AC + BC | distributes both ways | Lemma 1 S4, Cor 1 |
| c·AB = (cA)B = A(cB) | **scalars float anywhere** | precisions π float freely |
| ★ AB ≠ BA | **NOT commutative** in general — the default assumption | — |

## 3. Commuting — when AB = BA *is* allowed

| Case | Example from the proofs |
|---|---|
| one factor is I or c·I | π_TS I commutes with everything |
| ★ one is a polynomial / power / inverse of the other | S_S and B = π_TS I + π_S S_S, hence S_S and B⁻¹ (Lemma 1 S6) |
| they share an orthonormal eigenbasis (⟺ AB = BA, both symmetric) | N_S, S_S, B⁻¹, (I − N_S): all built from S_S, all commute (Cor 1 S3) |
| both diagonal in the same basis | same thing, coordinates view |
| **anything else** | **assume NO** |

**The family rule (worth internalizing):** every matrix built from S_S alone — powers, inverses of
shifts, N_S, I − N_S — lives in one happy family: all symmetric, all commuting, all sharing S_S's
eigenvectors, each just relabeling the eigenvalues (§7). Inside the family you may reorder freely.
That is why Lemma 1, Corollary 1 and Theorem 1 feel like scalar algebra.

## 4. Inverses

| Rule | Note | Used |
|---|---|---|
| (AB)⁻¹ = B⁻¹A⁻¹ | order flips, like transpose | — |
| ★ A⁻¹A = AA⁻¹ = I | a matrix commutes with **its own** inverse | complement identity |
| invertible ⟺ no zero eigenvalue | det = product of eigenvalues | Lemma 1 S3 |
| eigen-action of the inverse: Av = λv ⇒ A⁻¹v = (1/λ)v | inverse = "one over" per eigenvalue | Lemma 1 S3 |
| only square + full-rank has an inverse | rectangular matrices: none | — |

## 5. Symmetric matrices — when is a product symmetric?

| Fact | Condition | Used |
|---|---|---|
| Aᵀ = A | definition | — |
| A + B, A − B, cA symmetric | whenever A, B are | I − N_S (Cor 1) |
| ★ AᵀA and AAᵀ | **always** symmetric, for any A (Gram) | S_S, S_T |
| product AB of two symmetric matrices is symmetric | **iff they commute** | N_S = π_S S_S B⁻¹ (Lemma 1 S7) |
| congruence UᵀXU symmetric | whenever X is — any U, even rectangular | A_S = U_TᵀN_SU_T |
| inverse of a symmetric matrix is symmetric | (A⁻¹)ᵀ = (Aᵀ)⁻¹ = A⁻¹ | B⁻¹ |

## 6. PSD and the sandwich instinct

| Fact | Note | Used |
|---|---|---|
| ★ vᵀ(MᵀM)v = ‖Mv‖² ≥ 0 | the **sandwich instinct**: see MᵀM between vᵀ and v, read a squared norm | Lemma 1 S5, Cor 1 S3 |
| hence MᵀM ⪰ 0 always | its eigenvalues are μ = ‖Mv‖²/‖v‖² ≥ 0 | S_S, S_T PSD |
| same kernel: Mx = 0 ⟺ MᵀMx = 0 | learned direction: M_S u = 0 ⇒ S_S u = 0 ⇒ N_S u = 0 | Lemma 1 S5, S8 |
| A + cI shifts every eigenvalue by c | B = π_TS I + π_S S_S has eigenvalues π_TS + π_S μ ≥ π_TS > 0 | Lemma 1 S3 |
| PSD + no zero eigenvalue ⇒ invertible | why B⁻¹ exists | Lemma 1 S3 |
| A ⪯ B means vᵀAv ≤ vᵀBv ∀v | compare eigenvalues in a shared basis | extinction bound N_S ⪯ (π_S/π_TS)S_S (Thm 1 C2) |
| Rayleigh bounds: λ_min ≤ xᵀAx/‖x‖² ≤ λ_max | eigenvalue sandwiching a quadratic form | Lemma 2 (guard) |

## 7. The spectral toolkit — treating a matrix as independent scalars

The license for all of it is **symmetry**.

| Fact | Note | Used |
|---|---|---|
| ★ **spectral theorem**: A symmetric ⇒ A = U D Uᵀ — rotation U (orthonormal eigenvector columns, UᵀU = UUᵀ = I), diagonal D (real eigenvalues), rotation back | read right to left: rotate into the eigen-frame, scale each coordinate by its dial, rotate back; products collapse via the middle UᵀU = I | Lemma 1 S3b, Thm 1 C1 |
| coordinates: y = Σ c_k u_k with c_k = u_kᵀy | **dotting with a unit eigenvector reads off the component** | c_k = u_kᵀx_T |
| no cross terms: for orthonormal u_k, quadratic forms split per direction | yᵀAy = Σ λ_k c_k² | Cor 1 per-direction table |
| ★ **spectral mapping**: f(A) has the same eigenvectors, eigenvalues f(λ_k) — for f = powers, inverse, polynomials, shifts | one function of the matrix = the same function of each eigenvalue | n(μ) = π_S μ/(π_TS + π_S μ) |
| linear ODE decouples: ẏ = Ay, A = UDUᵀ ⇒ c = Uᵀy obeys ċ = Dc ⇒ ċ_k = λ_k c_k ⇒ c_k(t) = c_k(0)e^{λ_k t} | rotate coordinates → middle UᵀU evaporates → diagonal system = d independent scalar ODEs → solve, rotate back; fully worked in Thm 1 C1 on the scaffold of Lemma 1 S3b | Thm 1 C1 (rates |π_ST|n_k) |
| ratio grows: c_i/c_j ∝ e^{(λ_i−λ_j)t} | power iteration; why the most novel direction wins | novelty prioritization |
| two symmetric A, B share one eigenbasis ⟺ AB = BA | simultaneous diagonalization | the S_S family (§3) |

## 8. Gradients — the calculus layer

This is the entire calculus needed for the proofs. Everything reduces to one method (§11, P1) plus
these results.

| Fact | Statement | Used |
|---|---|---|
| ★ gradient DEFINITION (G1) | f(x+δ) = f(x) + ∇fᵀδ + O(‖δ‖²) — expand exactly, isolate the δ-linear term, transpose its coefficient: that column IS the gradient. Components are the partials: (∇f)ᵢ = ∂f/∂xᵢ (take δ = εeᵢ) — one template match computes all d partials at once. Directional rate: ∇fᵀe per unit length along unit e | every gradient in the proofs |
| ★ matrix gradient (G2) | f(W+Δ) = f(W) + ⟨∇_W f, Δ⟩ + O(‖Δ‖²), Frobenius ⟨A,B⟩ = Σ AᵢⱼBᵢⱼ; **extraction rule**: aᵀΔb = ⟨a bᵀ, Δ⟩ ⇒ gradient is the outer product a bᵀ | Prop 1 S5 |
| ★ half-squared-norm (R2) | ∇ᵤ ½‖u‖² = u | Prop 1 S1 |
| ★ affine version (T1) | ∇ₓ ½‖Ax‖² = AᵀAx; a constant shift drops: ∇ₓ ½‖x − c‖² = x − c | Prop 1 S2 (both F_S terms), ∇F_T |
| ★ quadratic form (A symmetric) | ∇ₓ ½ xᵀAx = Ax | ∇F_S^eq = π_TS N_S x_T (Thm 1 core) |
| sum rule | ∇(f + g) = ∇f + ∇g | F_S = interface + self |
| weight gradient of the model | ∇_W ½‖x − Wx‖² = −(x − Wx)xᵀ = −ε xᵀ — via (G2): ε(W+Δ) = ε − Δx, extraction rule on −εᵀΔx (coordinate/Kronecker check optional, Prop 1 S6) | Prop 1 S5 |
| ★ chain rule along a flow | dF/dt = ∇F · ẋ — (G1) with δ = ẋ dt (Frobenius version for matrices: dF/dt = ⟨∇_W F, Ẇ⟩) | Prop 1 S8, Thm 1 C2 |
| ★ Cauchy–Schwarz → steepest ascent | ∇F·e ≤ ‖∇F‖ for unit e, equality iff e ∥ ∇F: **the gradient is the steepest-ascent direction** (standard; one clause in the paper) | Thm 1 expository block |
| positive rescaling keeps the direction | c > 0 ⇒ c∇F points along ∇F; ascent property unchanged | u = (|π_ST|/π_TS)∇F_S^eq |
| ★ envelope theorem ("free lunch at a minimum") | for g(θ) = min_x f(x, θ): ∇_θ g = ∂f/∂θ evaluated at the minimizer — the minimizer's motion contributes nothing because the slope there is zero. Stated + proved + toy example in the Cor 1 remark of detailed_proofs.md | Cor 1 remark (∇F_S^eq) |
| gradient flow descends | ẋ = −∇F ⇒ dF/dt = −‖∇F‖² ≤ 0 (ascent: flip both signs) | Prop 1, Thm 1 C2 |

## 9. Projections (zero-diagonal and sphere-tangent)

| Rule | Note | Used |
|---|---|---|
| Frobenius inner product | ⟨X, Y⟩ = Σᵢⱼ XᵢⱼYᵢⱼ = tr(XᵀY) — treats matrices as long vectors | Prop 1 S7 |
| P² = P | **idempotent** — project twice = once | Prop 1 S8 |
| ⟨PX, Y⟩ = ⟨X, PY⟩ | self-adjoint (orthogonal projection) | Prop 1 S8 |
| ★ ⟨G, PG⟩ = ‖PG‖² ≥ 0 | insert P twice, move one across: projected gradient descent still descends | Prop 1 S8 |
| P₀(X) = X − diag(X) | projector onto zero-diagonal matrices | learning rule |
| P⊥ = I − yyᵀ/‖y‖² | projector onto a sphere's tangent plane (kills the radial part) | Prop 3 (sphere) |

## 10. Kernels, norms, orthonormal maps

| Fact | Note | Used |
|---|---|---|
| ker(M) = ker(MᵀM) | the "flat"/learned directions of S = MᵀM are exactly Mv = 0 | learned ⇒ N_S u = 0 |
| orthonormal columns U (UᵀU = I_r): ‖Uy‖ = ‖y‖ | norms preserved | manifold coordinates |
| but UUᵀ = P ≠ I when U is rectangular | the fat product is a **projector** — you cannot cancel U | the U_T trap |
| (m×n)(n×p) only | inner dimensions must match — check shapes before anything else | — |

---

## 11. Proof patterns — the recurring moves

These are the actual "plays." Each proof is a short sequence of them.

**P1 ★ — Perturb and read the linear term (how every gradient is computed).**
To differentiate a scalar f(x): replace x → x + δ, expand *exactly*, and collect the term linear in
δ. It always looks like (something)ᵀδ; that "something" is the gradient — this is the (G1)
definition, not a trick. Matrix version: perturb W → W + Δ, and read the linear term through the
(G2) extraction rule aᵀΔb = ⟨a bᵀ, Δ⟩ — the gradient is the outer product. *Used:* Prop 1 S1
(‖Ax‖²), Prop 1 S5 (the weight gradient), Thm 1 core (xᵀN_Sx). The ½ in front of squared norms exists
precisely to cancel the factor 2 from the two equal cross terms.

**P2 ★ — The sandwich instinct.**
Whenever MᵀM appears between a vector and its transpose, read a squared norm: vᵀMᵀMv = ‖Mv‖².
Instantly gives PSD, kernels, and per-direction costs. *Used:* Lemma 1 S5, Cor 1 S3, Lemma 2.

**P3 ★ — Insert the identity where you need a common factor.**
Write x = B⁻¹Bx, or I = BB⁻¹, to make two terms share a left factor you can then pull out — the
matrix version of "multiply by 1 cleverly." *Used:* Lemma 1 S4 (ε_TS), complement identity.

**P4 — Factor the unknown, then invert.**
Collect all terms in the unknown vector into (matrix)·x = rhs, prove the matrix has no zero
eigenvalue (usually: PSD + positive shift, §6), then left-multiply by the inverse. *Used:* Lemma 1
S2–S3. This is "solving a linear system" in matrix clothing.

**P5 — Kronecker collapse (differentiating w.r.t. one entry — the optional coordinate check).**
Each matrix entry is an independent variable: ∂(W)_{ik}/∂(W)_{ab} = δ_{ia}δ_{kb}. Chain-rule the
scalar, then let each delta kill one sum. Reassemble the entries into an outer product. The
perturbation route (P1 matrix version) is shorter; this one confirms it entry by entry and gives
the physical wire-by-wire (Hebbian) reading. *Used:* Prop 1 S6 (check of ∇_W F_S = −π_S ε_S x_Sᵀ).

**P6 — Stay in the family (eigen-slide).**
All matrices built from one symmetric S commute and share its eigenbasis (§3, §7). Inside that
family, reorder products freely, and evaluate any expression per-eigenvalue as scalar algebra:
matrix identity ⟺ one scalar identity for each μ. *Used:* Lemma 1 S6–S7, Cor 1 S3, n(μ).

**P7 — Diagonalize the dynamics (rotate coordinates, then solve per axis).**
The full recipe, worked in detail in Thm 1 C1 on the scaffold of Lemma 1 S3b: (1) spectral theorem
in three-matrix form: A = U D Uᵀ — rotation U (eigenvector columns, UᵀU = UUᵀ = I), diagonal
dial-board D (eigenvalues), rotation back; (2) change coordinates c(t) = Uᵀy(t) — U constant in
time, so ċ = Uᵀẏ; (3) rotate the equation ẏ = Ay by left-multiplying with Uᵀ: the middle UᵀU
evaporates, leaving ċ = Dc; (4) a **diagonal** system IS d independent scalar equations
ċ_k = λ_k c_k — the off-diagonal zeros of D are the visible **no-mixing** proof; (5) solve each,
c_k(t) = c_k(0)e^{λ_k t}, and rotate back y = Uc. The same move decouples quadratic forms:
yᵀAy = cᵀDc = Σ λ_k c_k² (Cor 1). *Used:* Thm 1 C1 (ċ_k = (|π_ST|/τ_T)n_k c_k), Cor 1.

**P8 — Idempotent + self-adjoint ⇒ never uphill.**
For an orthogonal projection P: ⟨G, PG⟩ = ⟨PG, PG⟩ = ‖PG‖² ≥ 0 (insert P² = P, move one P across).
So following a *projected* gradient still descends the true objective. *Used:* Prop 1 S8.

**P9 ★ — Zero slope at a minimum (the envelope theorem).**
If x*(θ) minimizes f(x, θ) in x, then differentiating F(θ) = f(x*(θ), θ) *through* the minimizer
costs nothing: the ∂f/∂x factor vanishes by the definition of x*. Only the explicit θ-dependence
survives. (Name: F is the *lower envelope* of the curve family θ ↦ f(x, θ), one curve per frozen x;
envelope and touching curve are tangent — same slope. Full statement, two-line proof, and a d = 1
toy check live in the Cor 1 remark of detailed_proofs.md.) *Used:* Cor 1 remark — one line to
∇F_S^eq = π_TS N_S x_T.

**P10 — Complement split (predicted + novel = whole).**
From BB⁻¹ = I with B = π_TS I + π_S S_S: π_TS B⁻¹ = I − N_S. Any expression containing B⁻¹ or
S_SB⁻¹ can be rewritten in terms of N_S and I − N_S — the "novel part" and the "predicted part."
*Used:* Lemma 1 remark (x_S* = (I − N_S)x_T), Cor 1 S3 (π_S S_S(I − N_S) = π_TS N_S).

**P11 — Gradient bookkeeping of a flow.**
To interpret a dynamics ẋ = (stuff): try to write each force as ±∇(some scalar). Once every force
is a gradient, the flow is descent/ascent on a combined potential, and matching the coefficients
gives exact conditions. *Used:* Prop 1 (definition of the dynamics), Thm 1 C3 (saddle condition
|π_ST| = π_TS).

---

## 12. House identities — the model's own formulas (know cold)

| # | Identity | In words |
|---|---|---|
| H1 | M = I − W, S = MᵀM | mismatch, then its Gram = self-error operator (weight-level) |
| H2 | ε_T = M_T x_T, ε_S = M_S x_S, ε_TS = x_S − x_T | the three errors |
| H3 | F_S = (π_TS/2)‖ε_TS‖² + (π_S/2)‖ε_S‖², F_T = (π_T/2)‖ε_T‖² | surprise = free energy (state-level); F_S student's, F_T teacher's |
| H4 | B = π_TS I + π_S S_S; N_S = π_S S_S B⁻¹ | the novelty operator (weight-level) |
| H5 ★ | π_TS B⁻¹ = I − N_S ⟺ x_S* = (I − N_S)x_T | complement identity: the student settles on what it can predict |
| H6 ★ | ε_TS = −N_S x_T (at the fast-student equilibrium) | the residual drive = the novel part of the teacher's state |
| H7 ★ | n(μ) = π_S μ/(π_TS + π_S μ) ∈ [0,1); n(0) = 0; n′ > 0; n(μ) ≤ (π_S/π_TS)μ | novelty eigenvalues: 0 = learned, increasing in self-error, linearly bounded |
| H8 ★ | F_S^eq(x_T) = min over x_S of F_S = (π_TS/2) x_Tᵀ N_S x_T | **surprise identity**: settled surprise = novelty score (per direction: n² + n(1−n) = n) |
| H9 | ∇_{x_S}F_S = π_TS ε_TS + π_S M_Sᵀ ε_S; ∇_{W_S}F_S = −π_S ε_S x_Sᵀ | the student's two gradients (Prop 1) |
| H10 | ∇_{x_T}F_T = π_T S_T x_T; ∇_{x_T}F_S^eq = π_TS N_S x_T | the teacher-side gradients |
| H11 ★ | u = |π_ST| N_S x_T = (|π_ST|/π_TS) ∇F_S^eq | the push = the student's surprise gradient (Thm 1) |
| H12 ★ | τ_T ẋ_T = −∇F_T + (|π_ST|/π_TS)∇F_S^eq; single potential Φ = F_T − F_S iff π_ST = −π_TS | descend own surprise, ascend the student's; exact saddle = mobility matching |

**Terminology guard:** *novelty* = weight-level (N_S, n_k — which directions the student can't
predict); *surprise* = state-level scalar (F_S, F_T — how bad the current state feels). H8 is the
bridge. "Teacher's surprise" only ever means F_T.

---

## 13. The flash-card list — if you memorize only these

Algebra reflexes:
1. (AB)ᵀ = BᵀAᵀ — transpose flips order (and so does inverse).
2. A scalar is its own transpose: xᵀAy = yᵀAᵀx.
3. Matrices don't commute — except within a "family" built from one symmetric matrix (polynomials, shifts, inverses), which shares one eigenbasis and behaves like scalars.
4. Sandwich instinct: vᵀMᵀMv = ‖Mv‖² ≥ 0 — Gram matrices are symmetric PSD, same kernel as M.
5. Spectral theorem: symmetric ⇒ orthonormal eigenbasis; then quadratic forms split (Σλc²) and linear ODEs decouple (ċ = λc).
6. Spectral mapping: f(A) acts as f(λ) on each eigendirection.

Calculus reflexes:
7. Gradients by perturbation (G1/G2): f(x+δ) = f(x) + ∇fᵀδ + O(δ²) — the coefficient of δ, transposed, IS the gradient; matrices likewise with ⟨∇_W f, Δ⟩ and the extraction rule aᵀΔb = ⟨abᵀ, Δ⟩. Know the three outputs: ∇½‖Ax‖² = AᵀAx; ∇½‖x − c‖² = x − c; ∇½xᵀAx = Ax (A symmetric).
8. dF/dt = ∇F · ẋ; gradient flow ẋ = −∇F gives dF/dt = −‖∇F‖² ≤ 0.
9. Cauchy–Schwarz: the gradient is the steepest-ascent direction; positive scalars don't change that.
10. Envelope theorem: differentiating through a minimizer is free — only explicit dependence counts.
11. Orthogonal projection: P² = P, Pᵀ = P, hence ⟨G, PG⟩ = ‖PG‖² ≥ 0 — projected descent still descends.

House reflexes:
12. x_S* = (I − N_S)x_T and ε_TS = −N_S x_T — copy the predicted, leave the novel.
13. F_S^eq = (π_TS/2)x_TᵀN_Sx_T — settled surprise = novelty score.
14. u = (|π_ST|/π_TS)∇F_S^eq, and Φ = F_T − F_S is exact iff π_ST = −π_TS.
