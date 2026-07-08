# Detailed proofs — Proposition 1, Lemma 1, Corollary 1, Theorem 1

**Purpose.** These are the fully-spelled-out proofs of the core results, written so that *every*
matrix operation — every transpose, every inverse, every time two matrices are swapped (commuted),
every collapse of a sum by a Kronecker delta — is justified explicitly and in the most apprehendable
way. Nothing is left to "it is easy to see." A companion cheat-sheet of the algebra rules used here is
in [cheatsheet.md](cheatsheet.md).

---

## Notation

- `d` neurons per population; teacher state x_T, student state x_S, both in d-dimensional space.
- Weights W_T (frozen), W_S (plastic), both with **zero diagonal** (no autapses).
- Mismatch operators:

$$
M_T = I - W_T, \qquad M_S = I - W_S .
$$

- Self-error operators (symmetric, positive semidefinite — proved in Lemma 1, step 5; these are
  weight-level objects, on the *novelty* side of the terminology below — the codebase still calls
  them "self-surprise operators"):

$$
S_T = M_T^\top M_T, \qquad S_S = M_S^\top M_S .
$$

- Errors:

$$
\varepsilon_T = M_T\, x_T, \qquad \varepsilon_S = M_S\, x_S, \qquad \varepsilon_{TS} = x_S - x_T .
$$

- Precisions: π_T, π_S > 0 (self); π_TS > 0 (student interface); π_ST signed (teacher interface;
  sleep means π_ST < 0).
- Student free energy:

$$
F_S = \frac{\pi_{TS}}{2}\,\|\varepsilon_{TS}\|^2 + \frac{\pi_S}{2}\,\|\varepsilon_S\|^2 ,
$$

  teacher free energy:

$$
F_T = \frac{\pi_T}{2}\,\|\varepsilon_T\|^2 .
$$

### Terminology — novelty vs. surprise (kept strictly apart)

These proofs use two words that are easy to blur, so we fix them once, because they live at two
different levels:

- **Novelty** is *structural* — a property of the student's **weights**, independent of any state.
  It lives in the novelty operator N_S (Lemma 1), whose eigenvalues n_k ∈ [0,1) grade each
  *direction* of state space by how badly the student's weights fail to predict it (n_k = 0: fully
  learned; larger: more novel). We call the n_k **novelty eigenvalues**. No state x appears anywhere
  inside N_S.
- **Surprise** is a *scalar felt in the moment* — a property of a **state**, given the weights. It is
  measured by the free energies: F_S is the *student's* surprise, F_T the *teacher's*. (In predictive
  coding, free energy is the formal stand-in for surprise.)

The two levels meet in **Corollary 1** below: once the student's state has settled, the student's
surprise F_S equals the novelty-weighted energy of the teacher's state. So novelty is the *spectrum*,
surprise is the *score* — and every surprise that Theorem 1 climbs is the **student's**. The teacher's
own surprise F_T is a different object doing a different job: it only ever enters the dynamics
through the teacher's self-pull, and it is never the quantity the push ascends.

### The atoms — the complete list of facts the derivations are built from

Every step of every proof below reduces to one of these. Each proof opens with a **Toolkit** line
saying exactly which atoms it uses; the cheat-sheet ([cheatsheet.md](cheatsheet.md)) is the full
reference, section numbers cited as §.

**(R1) Transpose reverses a product:** (AB)ᵀ = BᵀAᵀ. Corollary used constantly: a scalar equals its
own transpose, so xᵀAy = yᵀAᵀx. (§1)

**(G1) What a vector gradient IS — the perturbation definition.** For a scalar function f of a
vector x, the gradient is *defined* as the unique vector ∇f(x) such that

$$
f(x + \delta) \;=\; f(x) \;+\; \nabla f(x)^\top \delta \;+\; O(\|\delta\|^2) .
$$

**This is the formula that links every perturbation to every gradient in this document.** The
method it dictates: expand f(x + δ) *exactly*, collect the term linear in δ, and match it to the
template ∇fᵀδ — whatever column vector r makes the linear term read rᵀδ **is** the gradient. (If
the linear term shows up as (row vector)·δ, the gradient is that row transposed into a column, by
R1.) Two consequences, used whenever a "rate" appears:

- *(components = the partial derivatives — what a gradient actually computes)* take δ = ε·e_i,
  where e_i is the i-th standard basis vector (all zeros, a single 1 in slot i): the template gives
  f(x + εe_i) − f(x) = ε·∇fᵀe_i + O(ε²) = ε·(∇f)_i + O(ε²). Divide by ε and let ε → 0 — that limit
  is *the definition of the partial derivative*. So

$$
\big(\nabla f(x)\big)_i \;=\; \frac{\partial f}{\partial x_i}(x) :
$$

  the gradient is nothing mysterious — it is the column of the d partial derivatives, and matching
  the (G1) template is just a way of computing **all d of them at once** instead of one at a time;
- *(directional rate)* step by ε along a **unit** direction e, i.e. δ = εe: then
  f(x + εe) − f(x) = ε·(∇fᵀe) + O(ε²), so **∇fᵀe is the per-unit-length rate of change of f in
  direction e** (for e = e_i this recovers the partial derivative — partials are just the axis-aligned
  directional rates);
- *(rate along a trajectory — the chain rule)* along x(t), take δ = ẋ·dt: then
  f(x + ẋ dt) − f(x) = (∇Fᵀẋ)·dt + O(dt²), so **dF/dt = ∇Fᵀẋ = ∇F·ẋ**.

**(R2) First payoff of G1:** ∇ₓ ½‖x‖² = x, and more generally ∇ₓ ½‖Ax‖² = AᵀAx (derived in Prop 1,
Step 1) and ∇ₓ ½xᵀAx = Ax for symmetric A (derived in Theorem 1's core, Move 2). (§8)

**(G2) What a matrix gradient IS.** Same definition, with the Frobenius inner product
⟨A, B⟩ = Σᵢⱼ AᵢⱼBᵢⱼ = tr(AᵀB) playing the role of the dot product: ∇_W f is the unique matrix with

$$
f(W + \Delta) \;=\; f(W) \;+\; \big\langle \nabla_W f,\ \Delta \big\rangle \;+\; O(\|\Delta\|^2) .
$$

The one **extraction rule** needed to read matrix gradients off: for column vectors a, b,

$$
a^\top \Delta\, b \;=\; \big\langle a\, b^\top,\ \Delta \big\rangle ,
$$

proved by writing out the sums: aᵀΔb = Σᵢⱼ aᵢ Δᵢⱼ bⱼ = Σᵢⱼ (a bᵀ)ᵢⱼ Δᵢⱼ. So whenever the
linear-in-Δ term is a scalar of the shape aᵀΔb, the gradient is the outer product a bᵀ. As with
(G1), the entries are the partial derivatives — take Δ = ε·E_{ab} (the one-hot matrix, a single 1
in entry (a, b)) and the template gives (∇_W f)_{ab} = ∂f/∂(W)_{ab}. (§8)

**The remaining atoms**, each introduced in full where first used:

- the **sandwich**: vᵀMᵀMv = ‖Mv‖² ≥ 0 (Lemma 1, Step 5; §6);
- the **spectral theorem** in three-matrix form — a symmetric matrix is A = U D Uᵀ, a *rotation*
  (U, orthonormal eigenvector columns, UᵀU = UUᵀ = I), a *diagonal dial-board* (D, the
  eigenvalues), and the rotation back — with **spectral mapping** (functions of A keep the frame
  and act on the dials). Built in full in Lemma 1, Step 3b (rotate–scale–rotate reading,
  coordinate change c = Uᵀx, middle-UᵀU evaporation in products), and used to decouple quadratic
  forms (Corollary 1) and dynamics (Theorem 1, Consequence 1) (§7);
- the **eigenvalue shift** A + cI and "invertible ⟺ no zero eigenvalue" (Lemma 1, Step 3; §4, §6);
- the **commuting family**: everything built from S_S commutes and shares its eigenbasis
  (Lemma 1, Step 6; §3);
- the **Frobenius projection facts**: P² = P, ⟨PX, Y⟩ = ⟨X, PY⟩, hence ⟨G, PG⟩ = ‖PG‖² ≥ 0
  (Prop 1, Steps 7–8; §9);
- **Cauchy–Schwarz**: a·e ≤ ‖a‖‖e‖, equality iff parallel (stated and proved in Theorem 1's
  expository block — tutorial material, one clause in the paper; §8);
- the **envelope theorem** (stated and proved in Corollary 1's remark; §8).

---

## Timescales — the two distinct "student is fixed" assumptions

The proofs hold two different variables of the student fixed, on two different timescales. They are
**not** the same assumption; keeping them separate is what makes the framing correct. The model has a
nested separation

$$
\tau_S \ \ll\ \tau_T \ \ll\ 1/\eta ,
$$

- **fastest — the student STATE x_S** (time constant τ_S): relaxes essentially instantly;
- **middle — the teacher STATE x_T** (time constant τ_T): moves slowly compared to x_S;
- **slowest — the student WEIGHTS W_S** (learning rate η): change negligibly over either state motion.

Consequently:

- **Lemma 1 uses the student STATE being FAST.** Because x_S equilibrates almost instantly, it is
  *already settled* at every teacher configuration; we set the state rate to zero (ẋ_S = 0) and solve
  for x_S. The student's state is fixed here **because it is fast, not slow.** The leftover interface
  error ε_TS = −N_S x_T is the residual of the *battle* at that equilibrium between the interface term
  (weight π_TS, "copy the teacher, x_S = x_T") and the self term (weight π_S, "stay on my own manifold,
  M_S x_S = 0"): zero on directions where the two agree (learned), nonzero where they conflict
  (unlearned). That residual is reinjected as the drive on the teacher.
- **Theorem 1 uses the student WEIGHTS being SLOW (frozen).** Over the timescale on which the teacher
  explores, W_S is effectively constant, so the operators S_S, N_S, A_S are fixed matrices. This — not
  any slowness of the state — is the "frozen student" in Theorem 1.

In one line: **fast student state (Lemma 1) + frozen student weights (Theorem 1).** The coupled process
in which W_S actually changes (so A_S drifts and the teacher re-targets the next-novel direction) is
the learning dynamics, left to the empirical results.

---

## Proposition 1 — student inference and learning descend F_S

**Toolkit:** (R1); (G1) for both vector gradients; (G2) + the extraction rule for the weight
gradient; the Frobenius projection facts (§9) for the zero-diagonal constraint; the trajectory
consequence of (G1) for "descent never increases F_S". Patterns (cheatsheet §11): P1, P8.

**Claim.** The student's state relaxation is exact gradient descent on F_S, and the zero-diagonal
weight update is *projected* gradient descent on F_S (so F_S never increases under either).

### Part A — inference (the gradient with respect to x_S)

**Step 1 — the one differentiation rule we need.** For a fixed matrix A and the scalar

$$
g(x) = \tfrac12\,\|A x\|^2 = \tfrac12\,(A x)^\top (A x) ,
$$

perturb x → x + δ and expand exactly:

$$
g(x + \delta) = \tfrac12\,(A x + A\delta)^\top (A x + A\delta) = \tfrac12\,(Ax)^\top(Ax) + (Ax)^\top(A\delta) + \tfrac12\,(A\delta)^\top(A\delta) .
$$

The middle (linear-in-δ) term is (Ax)-transpose·A·δ. Using **(R1)**, (Ax)-transpose = x-transpose·A-transpose, so the linear term is

$$
x^\top A^\top A\, \delta .
$$

Now match this to the **(G1) template**. G1 says the expansion must read

$$
g(x + \delta) = g(x) + \nabla g^\top \delta + O(\|\delta\|^2),
$$

so the gradient is identified by forcing the linear term into the shape ∇gᵀδ:

$$
x^\top A^\top A\, \delta \;\stackrel{!}{=}\; \nabla g^\top\, \delta \quad\Longrightarrow\quad \nabla g^\top = x^\top A^\top A \quad\Longrightarrow\quad \nabla g = \big(x^\top A^\top A\big)^\top = A^\top A\, x ,
$$

the last step by **(R1)** applied twice ((AB C)ᵀ = CᵀBᵀAᵀ, with (Aᵀ)ᵀ = A). Hence

$$
\nabla_x\, \tfrac12\|Ax\|^2 = A^\top A\, x .
$$

*In terms of derivatives (what we just computed):* by the components-are-partials fact of (G1),
this single move produced the whole column of d partial derivatives at once,

$$
\frac{\partial}{\partial x_i}\, \tfrac12\|Ax\|^2 = \big(A^\top A\, x\big)_i .
$$

Sanity check in dimension 1, where A is just a number a: d/dx ½(ax)² = a²x — matches AᵀAx. ✓

This three-move sequence — *expand exactly, isolate the δ-linear term, transpose its coefficient* —
is how **every** gradient in this document is obtained; we will cite it as "matching the (G1)
template" from here on.

*(Why the ½ was there: the linear term came out with coefficient 1, not 2, because we expanded a
product of two identical factors — the two cross terms (Ax)ᵀ(Aδ) and (Aδ)ᵀ(Ax) are equal scalars,
summing to 2, which the ½ cancels. The expansion used here is the four-move
**unfold → distribute → merge → refold** of a squared norm; it is written out in maximal detail,
one move per line, in Step 2's interface term below — read that version first if any move here
felt jumped.)*

**Step 2 — apply it to each term of F_S.** F_S has two terms. Differentiate each with respect to x_S,
holding x_T fixed (x_T is the top-down input the student receives).

*Interface term — perturbed directly (no pattern, no shift argument).* Perturb x_S → x_S + δ while
x_T stays fixed (x_T is the top-down input the student receives). Watch what happens to the error
itself:

$$
(x_S + \delta) - x_T = (x_S - x_T) + \delta = \varepsilon_{TS} + \delta ,
$$

— the perturbation passes straight onto the error, untouched. Now expand the squared norm, every
move shown. **Unfold** the norm into a transpose product (that is what a squared norm *is*:
‖v‖² = vᵀv):

$$
\|\varepsilon_{TS} + \delta\|^2 = (\varepsilon_{TS} + \delta)^\top (\varepsilon_{TS} + \delta) .
$$

**Distribute** — the transpose splits over the sum ((a+b)ᵀ = aᵀ + bᵀ), and then the product
distributes term by term, exactly like (p+q)(r+s) = pr + ps + qr + qs:

$$
(\varepsilon_{TS}^\top + \delta^\top)(\varepsilon_{TS} + \delta) = \varepsilon_{TS}^\top \varepsilon_{TS} \;+\; \varepsilon_{TS}^\top \delta \;+\; \delta^\top \varepsilon_{TS} \;+\; \delta^\top \delta .
$$

**Merge the two cross terms.** Both are scalars (row times column = a 1×1 number), and a scalar
equals its own transpose, so

$$
\delta^\top \varepsilon_{TS} = \big(\delta^\top \varepsilon_{TS}\big)^\top = \varepsilon_{TS}^\top \delta \qquad\text{(R1 on the product)},
$$

and the two middle terms add up to 2 ε_TSᵀδ. **Refold** the two pure squares back into norms
(εᵀε = ‖ε‖², δᵀδ = ‖δ‖²):

$$
\|\varepsilon_{TS} + \delta\|^2 = \|\varepsilon_{TS}\|^2 + 2\,\varepsilon_{TS}^\top \delta + \|\delta\|^2 .
$$

Multiply through by the prefactor π_TS/2 — the ½ cancels the 2 of the merged cross terms (this is
the entire reason the ½ is in F_S):

$$
\frac{\pi_{TS}}{2}\,\|\varepsilon_{TS} + \delta\|^2 = \frac{\pi_{TS}}{2}\,\|\varepsilon_{TS}\|^2 + \pi_{TS}\,\varepsilon_{TS}^\top \delta + \frac{\pi_{TS}}{2}\,\|\delta\|^2 .
$$

The linear term already sits in the (G1) template shape ∇fᵀδ, so

$$
\nabla_{x_S}\, \tfrac{\pi_{TS}}{2}\|x_S - x_T\|^2 = \pi_{TS}\,\varepsilon_{TS} .
$$

*In terms of derivatives (what we just computed):* component i of this gradient is the partial
derivative

$$
\frac{\partial}{\partial x_{S,i}}\, \frac{\pi_{TS}}{2} \sum_j \big(x_{S,j} - x_{T,j}\big)^2 = \pi_{TS}\,\big(x_{S,i} - x_{T,i}\big) ,
$$

because only the j = i term of the sum contains x_{S,i} (chain rule on one square: ½·2·(inside)·1).
That is entry i of π_TS ε_TS — the template computed all d of these partials in one move. ✓

*Self term.* Here

$$
\tfrac{\pi_S}{2}\,\|\varepsilon_S\|^2 = \tfrac{\pi_S}{2}\,\|M_S\, x_S\|^2 ,
$$

which is exactly the Step-1 form with A = M_S. So

$$
\nabla_{x_S}\, \tfrac{\pi_S}{2}\|M_S x_S\|^2 = \pi_S\, M_S^\top M_S\, x_S = \pi_S\, M_S^\top \varepsilon_S .
$$

*In terms of derivatives:* component i is the ordinary chain rule over the sum of squares —
ε_{S,j} depends on x_{S,i} through the entry (M_S)_{ji}, so

$$
\frac{\partial}{\partial x_{S,i}}\, \frac{\pi_S}{2}\sum_j \varepsilon_{S,j}^2 = \pi_S \sum_j \varepsilon_{S,j}\, \frac{\partial \varepsilon_{S,j}}{\partial x_{S,i}} = \pi_S \sum_j \varepsilon_{S,j}\, (M_S)_{ji} ,
$$

which is entry i of π_S M_Sᵀ ε_S (the transpose appears precisely because we sum over the *row*
index j of M_S — bottom-up error feedback travels the wires backwards). ✓

**Step 3 — sum and identify the dynamics.** Adding the two gradients:

$$
\nabla_{x_S} F_S = \pi_{TS}\,\varepsilon_{TS} + \pi_S\, M_S^\top \varepsilon_S .
$$

The student's state equation is defined to descend this gradient:

$$
\tau_S\, \dot x_S = -\nabla_{x_S} F_S = -\pi_{TS}\,\varepsilon_{TS} - \pi_S\, M_S^\top \varepsilon_S .
$$

This is exactly the model's student state dynamics, so **inference is exact gradient descent on F_S.**

### Part B — learning (the gradient with respect to W_S)

**Step 4 — only one term depends on W_S.** The interface term ½‖x_S − x_T‖² contains no W_S, so its
W_S-derivative is zero. Only the self term ½‖M_S x_S‖² = ½‖x_S − W_S x_S‖² depends on W_S.

**Step 5 — perturb the weights (the same method as Step 1, now with (G2)).** No coordinates
needed: perturb the whole matrix, W_S → W_S + Δ, and track the error *exactly*:

$$
\varepsilon_S(W_S + \Delta) = x_S - (W_S + \Delta)\, x_S = \big(x_S - W_S x_S\big) - \Delta\, x_S = \varepsilon_S - \Delta\, x_S .
$$

Expand the self term exactly — it is the same three-term expansion as Step 1, with u = ε_S and the
perturbation −Δx_S:

$$
\frac{\pi_S}{2}\,\|\varepsilon_S - \Delta x_S\|^2 = \frac{\pi_S}{2}\,\|\varepsilon_S\|^2 \;-\; \pi_S\, \varepsilon_S^\top \Delta\, x_S \;+\; \frac{\pi_S}{2}\,\|\Delta x_S\|^2 .
$$

(As in Step 1, the two equal cross terms summed to 2, cancelling the ½.) The last term is O(‖Δ‖²)
and is dropped by the (G2) definition. The linear-in-Δ term, −π_S ε_Sᵀ Δ x_S, has exactly the shape
aᵀΔb of the **extraction rule** (with a = −π_S ε_S, b = x_S), so it reads

$$
-\,\pi_S\, \varepsilon_S^\top \Delta\, x_S \;=\; \big\langle -\pi_S\, \varepsilon_S\, x_S^\top,\ \Delta \big\rangle .
$$

Matching the **(G2) template** f(W+Δ) = f(W) + ⟨∇_W f, Δ⟩ + O(‖Δ‖²):

$$
\nabla_{W_S} F_S = -\,\pi_S\, \varepsilon_S\, x_S^\top .
$$

*In terms of derivatives:* by (G2)'s components-are-partials fact, entry (a, b) of this matrix is
the partial derivative ∂F_S/∂(W_S)_{ab} = −π_S ε_{S,a} x_{S,b} — Step 6 recomputes exactly that
partial, entry by entry, and matches.

**Step 6 — the same, wire by wire (optional check — and the biological reading).** The coordinate
route confirms Step 5 and explains *why the rule is Hebbian*. Each weight is an independent
variable, ∂(W_S)_{ik}/∂(W_S)_{ab} = δ_{ia}δ_{kb}, so differentiating
ε_{S,i} = x_{S,i} − Σ_k (W_S)_{ik} x_{S,k} with respect to the single entry (W_S)_{ab} gives
−δ_{ia} x_{S,b}; chaining through ½Σ_i ε_{S,i}² and letting the delta collapse the sum to i = a:

$$
\frac{\partial F_S}{\partial (W_S)_{ab}} = -\,\pi_S\, \varepsilon_{S,a}\, x_{S,b} ,
$$

which is entry (a, b) of the outer product −π_S ε_S x_Sᵀ — the same answer. *Read physically:
(W_S)_{ab} is the wire from neuron b to neuron a; only neuron a's error feels it, and x_{S,b} is the
presynaptic signal on that wire. So gradient descent on F_S is exactly the Hebbian prescription
"postsynaptic error × presynaptic activity", wire by wire.*

**Step 7 — impose the zero-diagonal constraint (projection).** *This is standard, not ours to prove:
the fact that projecting a gradient onto a linear subspace still gives a descent direction is the
classical gradient-projection method (Levitin & Polyak 1966), and the specific zero-diagonal
constraint is inherited from the covPCN substrate we build on, where the recurrent weights are
zero-diagonal and learning is defined to keep them so (Tang et al. 2023). We include the short
algebra only for self-containedness. The cleanest reading needs no theorem at all: the diagonal
entries are* not parameters of the model *— there are no autapses by architecture — so the genuine
parameters are the off-diagonal entries, and descent on them is ordinary unconstrained gradient
descent; the projection P_0 below is merely how that same update looks when written as a full
matrix. With that said, the two-line verification:*

W_S must keep a zero diagonal, so the
admissible weight changes form the linear subspace Z of zero-diagonal matrices. Equip d×d matrices
with the Frobenius inner product

$$
\langle X, Y\rangle = \sum_{i,j} X_{ij} Y_{ij} = \operatorname{tr}(X^\top Y) .
$$

The orthogonal projection onto Z simply zeros the diagonal:

$$
P_0(X) = X - \operatorname{diag}(X) .
$$

It is an orthogonal projection because Z (zero-diagonal) and its complement (diagonal matrices) share
no nonzero entries, so any diagonal D and any Z-member satisfy ⟨D, Z-member⟩ = 0. Two properties
follow, each checkable in one line:

$$
P_0^2 = P_0 \quad(\text{zeroing an already-zero diagonal changes nothing}),
$$

$$
\langle P_0 X, Y\rangle = \sum_{i\neq j} X_{ij}Y_{ij} = \langle X, P_0 Y\rangle \quad(\text{self-adjoint}).
$$

**Step 8 — projected descent still decreases F_S.** The learning rule follows the *projected* gradient:

$$
\dot W_S = -\eta\, P_0\big(\nabla_{W_S} F_S\big), \qquad \eta > 0 .
$$

The rate of change of F_S along this update is the Frobenius inner product of the true gradient with
the direction moved — the matrix version of (G1)'s trajectory rate, with ⟨·,·⟩ in place of the dot
product (written out, it is the chain rule dF_S/dt = Σ_{ij} (∂F_S/∂W_ij)(dW_ij/dt)):

$$
\dot F_S = \big\langle \nabla_{W_S} F_S,\ \dot W_S \big\rangle = -\eta\,\big\langle \nabla_{W_S} F_S,\ P_0(\nabla_{W_S} F_S)\big\rangle .
$$

Now use idempotency (insert a second P_0: P_0 G = P_0(P_0 G)) then self-adjointness (move one P_0
across the inner product), with G = ∇_{W_S}F_S:

$$
\big\langle G, P_0 G\big\rangle = \big\langle G, P_0(P_0 G)\big\rangle = \big\langle P_0 G, P_0 G\big\rangle = \|P_0 G\|_F^2 .
$$

A Frobenius norm squared is a sum of real squares, hence ≥ 0. Therefore

$$
\dot F_S = -\eta\,\|P_0(\nabla_{W_S} F_S)\|_F^2 \ \le\ 0 ,
$$

with equality only when the gradient is purely diagonal (no permitted, i.e. off-diagonal, direction
can lower F_S). So **learning is projected gradient descent on F_S.** Together with Part A, Proposition
1 is proved. ∎

*(Numerical check: `diagnostics.check_gradients` returns err_xS = err_WS = 0.0 — the code implements
the dynamics as these gradients, so the match is exact.)*

---

## Lemma 1 — fast-student elimination produces the novelty operator

**Toolkit:** (R1); the sandwich (§6); the eigenvalue shift + "invertible ⟺ no zero eigenvalue"
(§4, §6); the commuting family (§3). Patterns: P2 (sandwich), P3 (insert the identity),
P4 (factor and invert), P6 (eigen-slide).

**Claim.** When the student is fast (relaxes to its F_S-minimum while x_T is held fixed), its
equilibrium state and the leftover interface error are

$$
x_S^{*} = \pi_{TS}\,(\pi_{TS} I + \pi_S S_S)^{-1} x_T, \qquad \varepsilon_{TS} = -N_S\, x_T,
$$

with the **novelty operator**

$$
N_S = \pi_S\, S_S\,(\pi_{TS} I + \pi_S S_S)^{-1} .
$$

N_S is symmetric and positive semidefinite, and N_S annihilates every direction the student has
already learned.

**Step 1 — impose the steady state.** "Fast student" means the student sits at ∇_{x_S}F_S = 0. Using
the Part-A gradient (with M_S-transpose·M_S = S_S):

$$
\pi_{TS}\,(x_S - x_T) + \pi_S\, S_S\, x_S = 0 .
$$

**Step 2 — solve the linear system for x_S.** Expand and move the pure-x_T term to the right (it
becomes positive):

$$
\pi_{TS}\, x_S + \pi_S\, S_S\, x_S = \pi_{TS}\, x_T .
$$

Factor x_S out on the left (it is a matrix times x_S):

$$
(\pi_{TS} I + \pi_S S_S)\, x_S = \pi_{TS}\, x_T .
$$

Left-multiply by the inverse (existence justified in Step 3):

$$
x_S^{*} = \pi_{TS}\,(\pi_{TS} I + \pi_S S_S)^{-1} x_T .
$$

**Step 3 — why the inverse exists (eigenvalue argument).** Let B = π_TS I + π_S S_S. If v is an
eigenvector of S_S with eigenvalue μ (S_S v = μ v), then

$$
B\, v = (\pi_{TS} I + \pi_S S_S)\, v = \pi_{TS} v + \pi_S (\mu v) = (\pi_{TS} + \pi_S \mu)\, v .
$$

So B has the same eigenvectors as S_S with eigenvalues π_TS + π_S μ. Since μ ≥ 0 (Step 5), π_TS > 0,
π_S > 0, every eigenvalue of B is ≥ π_TS > 0 — none is zero, so B is invertible (a matrix is
invertible iff no eigenvalue is zero, because det = product of eigenvalues). The inverse acts on that
eigenvector as one over the eigenvalue:

$$
B^{-1} v = \frac{1}{\pi_{TS} + \pi_S \mu}\, v .
$$

**Step 3b — the diagonalized picture: rotate, scale, rotate back (built once here; reused verbatim
in Corollary 1 and Theorem 1).** S_S is symmetric (Step 5), so the **spectral theorem** applies in
its matrix form: collect the orthonormal eigenvectors u_1, …, u_d as the **columns** of one matrix,
and their eigenvalues into a **diagonal** matrix,

$$
U = \big[\, u_1 \;\big|\; u_2 \;\big|\; \cdots \;\big|\; u_d \,\big], \qquad D_\mu = \mathrm{diag}(\mu_1, \dots, \mu_d), \qquad\text{then}\qquad S_S = U\, D_\mu\, U^\top .
$$

Three facts to unpack, each elementary.

*(i) U is a rotation.* Compute UᵀU entry by entry: entry (k, l) is (row k of Uᵀ)·(column l of U)
= u_kᵀu_l. On the diagonal (k = l) this is ‖u_k‖² = 1 (unit vectors); off the diagonal it is 0
(perpendicular vectors). So the whole matrix is

$$
U^\top U = I ,
$$

and since U is square with independent columns, also UUᵀ = I: **Uᵀ is the inverse of U.**
Geometrically, U rotates the standard axes onto the student's axes and Uᵀ = U⁻¹ rotates back; a
rotation stretches nothing.

*(ii) Read the three-matrix product right to left: rotate → scale → rotate back.* Apply
S_S = U D_μ Uᵀ to any vector x, one factor at a time:

1. **Uᵀx — rotate into the student's axes.** Row k of Uᵀ is u_kᵀ, so component k of Uᵀx is u_kᵀx:
   the coordinate of x along the k-th student axis. Call the coordinate vector

$$
c = U^\top x \qquad (\text{and back: } x = U c) .
$$

2. **D_μ c — scale each coordinate on its own.** Row k of a diagonal matrix is all zeros except
   μ_k in slot k, so component k of D_μ c is μ_k·c_k and nothing else. **The no-mixing fact is
   *visible in the shape of the matrix*: the off-diagonal zeros are exactly the statement that
   coordinate k in never influences coordinate l out.**
3. **U(·) — rotate back** to the original frame.

So "apply S_S" means: read the coordinates on the student's axes, multiply each by its own μ_k,
reassemble. Sanity check on an eigenvector u_l: Uᵀu_l = e_l (u_l's coordinates are 1 on axis l,
0 elsewhere — that is (i) again); D_μ e_l = μ_l e_l (diagonal scaling); U(μ_l e_l) = μ_l u_l
(column l of U). Net: S_S u_l = μ_l u_l ✓ — and agreeing on every vector of a basis pins the
matrix down.

*(iii) The whole family = the SAME two rotations with a different dial-board in the middle.*
For sums, use I = U I Uᵀ (rotate, do nothing, rotate back) and factor U, Uᵀ out:

$$
B = \pi_{TS}\, I + \pi_S\, S_S = U\,\big(\pi_{TS} I + \pi_S D_\mu\big)\, U^\top = U\, D_B\, U^\top , \qquad D_B = \mathrm{diag}\big(\pi_{TS} + \pi_S \mu_k\big) .
$$

For products, sandwich two members and watch the middle evaporate —

$$
\big(U D_1 U^\top\big)\big(U D_2 U^\top\big) = U\, D_1\, \underbrace{(U^\top U)}_{=\,I}\, D_2\, U^\top = U\,\big(D_1 D_2\big)\, U^\top ,
$$

where diagonal matrices multiply slot by slot: D_1D_2 = diag(d_{1,k}·d_{2,k}). **This one-line
evaporation of the middle UᵀU is the engine behind every "no mixing" statement in this
document.** Inverses come for free: B⁻¹ = U D_B⁻¹ Uᵀ with D_B⁻¹ = diag(1/(π_TS + π_S μ_k)) —
check by the product rule just proved: B B⁻¹ = U (D_B D_B⁻¹) Uᵀ = U I Uᵀ = I ✓. Hence the rest of
the family, all in the same frame:

$$
N_S = \pi_S\, S_S\, B^{-1} = U\,\big(\pi_S D_\mu D_B^{-1}\big)\, U^\top = U\, \mathrm{diag}(n_k)\, U^\top , \qquad n_k = \frac{\pi_S\, \mu_k}{\pi_{TS} + \pi_S \mu_k} ,
$$

$$
I - N_S = U\, \mathrm{diag}(1 - n_k)\, U^\top .
$$

**The picture to keep:** every matrix in the family is the *same pair of rotations* — into the
student's axes and back — with a different diagonal dial-board in the middle (dials μ_k,
π_TS + π_S μ_k, its reciprocal, n_k, 1 − n_k). Acting with any of them on any vector: rotate to
coordinates, turn each coordinate by that member's dial, rotate back. The off-diagonal zeros of
the middle matrix *are* the guarantee that nothing ever leaks between axes.

**Step 4 — the leftover interface error.** Compute ε_TS = x_S\* − x_T. Write the bare x_T as
B-inverse·B·x_T (that product is the identity), so both terms share a left factor of B-inverse:

$$
\varepsilon_{TS} = x_S^{*} - x_T = B^{-1}\big[\,\pi_{TS} I - (\pi_{TS} I + \pi_S S_S)\,\big] x_T .
$$

Simplify the bracket, distributing the minus over both pieces of B:

$$
\pi_{TS} I - (\pi_{TS} I + \pi_S S_S) = \pi_{TS} I - \pi_{TS} I - \pi_S S_S = -\,\pi_S S_S .
$$

The identity terms cancel, leaving

$$
\varepsilon_{TS} = B^{-1}(-\pi_S S_S)\, x_T = -\,\pi_S\, B^{-1} S_S\, x_T .
$$

**Step 5 — S_S is symmetric and positive semidefinite (μ ≥ 0), and the "learned ⇒ zero" fact.** For
any vector v, sandwich S_S:

$$
v^\top S_S\, v = v^\top M_S^\top M_S\, v = (M_S v)^\top (M_S v) = \|M_S v\|^2 \ \ge\ 0 ,
$$

using (R1) to regroup v-transpose·M_S-transpose = (M_S v)-transpose. For an eigenvector this reads
μ‖v‖² = ‖M_S v‖², so μ = ‖M_S v‖² / ‖v‖² ≥ 0. Moreover S_S is symmetric because
(M_S-transpose·M_S)-transpose = M_S-transpose·M_S by (R1). Finally, a **learned** direction u is one
the student can predict, i.e. M_S u = 0; then

$$
S_S\, u = M_S^\top M_S\, u = M_S^\top (0) = 0 .
$$

**Step 6 — S_S commutes with B-inverse, giving the official form of N_S.** First, S_S commutes with B,
by direct expansion (S_S commutes with I and with itself):

$$
S_S\, B = \pi_{TS} S_S + \pi_S S_S^2 = B\, S_S .
$$

Now pass the commutation through the inverse by sandwiching S_S·B = B·S_S with B-inverse on both
sides:

$$
B^{-1}(S_S B)B^{-1} = B^{-1}(B S_S)B^{-1} \ \Longrightarrow\ B^{-1} S_S = S_S\, B^{-1} ,
$$

where the left side collapsed via B·B-inverse = I and the right via B-inverse·B = I. Therefore, in
Step 4, we may slide S_S to the left of B-inverse:

$$
\varepsilon_{TS} = -\,\pi_S\, S_S\, B^{-1}\, x_T = -\,N_S\, x_T, \qquad N_S = \pi_S\, S_S\, (\pi_{TS} I + \pi_S S_S)^{-1} .
$$

**Step 7 — N_S is symmetric and PSD.** Symmetric: N_S is the product of the two symmetric matrices
S_S and B-inverse, and a product of symmetric matrices is symmetric **iff they commute** — which they
do by Step 6. Explicitly, using (R1) and B-inverse symmetric (the inverse of a symmetric matrix is
symmetric):

$$
(S_S B^{-1})^\top = (B^{-1})^\top S_S^\top = B^{-1} S_S = S_S B^{-1} .
$$

PSD: in the shared eigenbasis, N_S has eigenvalues π_S μ / (π_TS + π_S μ) = n(μ), each ≥ 0 since μ ≥ 0.

**Step 8 — N_S annihilates learned directions.** For a learned u (M_S u = 0), Step 5 gives S_S u = 0,
so using the commuting form,

$$
N_S\, u = \pi_S\, B^{-1} S_S\, u = \pi_S\, B^{-1}(0) = 0, \qquad\text{hence}\quad \varepsilon_{TS} = -N_S\, u = 0 .
$$

So a teacher pointing along a learned direction produces no interface error — no drive. Lemma 1 is
proved. ∎

*(Numerical check: with the student pre-trained on a subset, `model.novelty_operator()` annihilates
the learned patterns (‖N_S m‖ ~ 1e-10) and passes the unlearned ones (~0.35), and the identity
ε_TS = −N_S x_T holds to ~1e-15.)*

### Remark — the complement identity ("copied + missing = whole")

Lemma 1 hides a one-line dividend that we will use in Corollary 1, and that makes the equilibrium
transparent. Start from the fact that a matrix times its own inverse is the identity, B·B⁻¹ = I,
write B out in full, and distribute the B⁻¹ over the sum:

$$
(\pi_{TS} I + \pi_S S_S)\, B^{-1} = I \quad\Longrightarrow\quad \pi_{TS}\, B^{-1} + \pi_S S_S B^{-1} = I .
$$

The second piece is exactly N_S (its definition, Step 6). Move it to the right:

$$
\pi_{TS}\, B^{-1} = I - N_S .
$$

Substituting this into the equilibrium x_S\* = π_TS B⁻¹ x_T (Step 2) gives the equilibrium in its
most readable form:

$$
x_S^{*} = (I - N_S)\, x_T .
$$

Read it aloud: **the student settles on exactly the part of the teacher's state that it can
predict** — it copies the learned content (the I) and drops the novel content (the −N_S). The
interface error is then immediate, with no inverse in sight:

$$
\varepsilon_{TS} = x_S^{*} - x_T = (I - N_S)\, x_T - x_T = -\,N_S\, x_T ,
$$

recovering Step 4. Per direction (using the eigenbasis of Step 3): the student copies the fraction
(1 − n_k) of each component of x_T and leaves the fraction n_k behind as error. Learned direction:
copy everything, no error. Fully novel direction: copy almost nothing, almost all error.

---

## Corollary 1 — the surprise identity: the settled student's F_S *is* the novelty score

**Toolkit:** the complement identity (Lemma 1's remark; pattern P10); (R1); the sandwich (§6); the
commuting family (§3). For the closing remarks only: (G1) and the envelope theorem (P9).

**Claim (fast student, exactly as in Lemma 1).** Let the student settle (x_S = x_S\*) while the
teacher sits at x_T, and read off the value of the student's free energy there. The settled state is
the Lemma-1 equilibrium x_S\*(x_T) — fully determined by x_T — so this value is a function of the
teacher's state alone. **Definition:**

$$
F_S^{\mathrm{eq}}(x_T) \;=\; F_S\big(x_S^{*}(x_T),\, x_T\big) .
$$

*(A bonus fact, recorded now but NOT used by the proof below: the equilibrium is in fact the unique
**minimum** of F_S over x_S — F_S is a strictly convex quadratic in x_S, since its Hessian is
B = π_TS I + π_S S_S, positive definite by Lemma 1, Step 3 — so equivalently
F_S^eq(x_T) = min over x_S of F_S(x_S, x_T). The direct proof below only plugs the equilibrium in;
this "min" characterization becomes load-bearing solely in the envelope-theorem remark at the end,
and it is what the paper's variational reading refers to — F_S^eq is the free energy after inference
has run.)*

Then

$$
F_S^{\mathrm{eq}}(x_T) \;=\; \frac{\pi_{TS}}{2}\, x_T^\top N_S\, x_T .
$$

In words: **the student's surprise about the teacher's current state** is the teacher's state weighed
by the student's novelty operator. This is the bridge announced in the Terminology note — the novelty
eigenvalues are the spectrum of the surprise.

**Why we want it.** Theorem 1 will show that the teacher is pushed along N_S x_T. Without this
corollary, that direction is "the gradient of some quadratic score ½ x_TᵀN_S x_T" — a quantity we
would have to *name into existence* just for the theorem. With it, the score is nothing new: it is
F_S itself, the model's own surprise functional, evaluated with the fast student settled. The theorem
then says something with real content: *the student pushes the teacher up the student's own
surprise.* No separate "teacher's surprise" is ever defined — the teacher's surprise is F_T, and it
plays its own role (the self-pull).

**Step 1 — what to plug in.** At the settled state, by Lemma 1 and its complement identity:

$$
x_S^{*} = (I - N_S)\, x_T, \qquad \varepsilon_{TS} = -\,N_S\, x_T, \qquad \varepsilon_S = M_S\, x_S^{*} .
$$

F_S has two terms (interface + self); we evaluate each on these and add.

**Step 2 — the interface term.** Its squared norm, using (R1) to regroup and the symmetry of N_S
(Lemma 1, Step 7; the two minus signs multiply to +1):

$$
\|\varepsilon_{TS}\|^2 = (-N_S x_T)^\top(-N_S x_T) = x_T^\top N_S^\top N_S\, x_T = x_T^\top N_S^2\, x_T .
$$

So the interface term contributes

$$
\frac{\pi_{TS}}{2}\,\|\varepsilon_{TS}\|^2 = \frac{\pi_{TS}}{2}\, x_T^\top N_S^2\, x_T .
$$

**Step 3 — the self term.** First the sandwich instinct (Lemma 1, Step 5):

$$
\|\varepsilon_S\|^2 = \|M_S\, x_S^{*}\|^2 = (x_S^{*})^\top M_S^\top M_S\, x_S^{*} = (x_S^{*})^\top S_S\, x_S^{*} .
$$

Insert x_S\* = (I − N_S)x_T on both sides of S_S. The left copy needs a transpose,
((I − N_S)x_T)ᵀ = x_Tᵀ(I − N_S)ᵀ = x_Tᵀ(I − N_S), where the last step used that I and N_S are both
symmetric (so their difference is too). Hence

$$
\frac{\pi_S}{2}\,\|\varepsilon_S\|^2 = \frac{\pi_S}{2}\, x_T^\top (I - N_S)\, S_S\, (I - N_S)\, x_T .
$$

Now the one clever move, and it is just the complement identity read backwards: I − N_S = π_TS B⁻¹,
so

$$
\pi_S\, S_S\,(I - N_S) = \pi_{TS}\,\big(\pi_S\, S_S\, B^{-1}\big) = \pi_{TS}\, N_S ,
$$

by the definition of N_S. Therefore (S_S and N_S commute with I − N_S, all being built from S_S —
Lemma 1, Step 6):

$$
\frac{\pi_S}{2}\,(I - N_S)\, S_S\, (I - N_S) = \frac{\pi_{TS}}{2}\,(I - N_S)\, N_S = \frac{\pi_{TS}}{2}\,\big(N_S - N_S^2\big) ,
$$

where the last equality just distributed the product. So the self term contributes

$$
\frac{\pi_S}{2}\,\|\varepsilon_S\|^2 = \frac{\pi_{TS}}{2}\, x_T^\top \big(N_S - N_S^2\big)\, x_T .
$$

**Step 4 — add, and watch the squares cancel.**

$$
F_S^{\mathrm{eq}} = \frac{\pi_{TS}}{2}\, x_T^\top\big[\,N_S^2 + N_S - N_S^2\,\big]\, x_T = \frac{\pi_{TS}}{2}\, x_T^\top N_S\, x_T . \qquad \blacksquare
$$

**The per-direction picture — a second, independent proof, dial by dial (why the cancellation is
not an accident).** This time we do not reuse the operator algebra above: we recompute both terms
of F_S from scratch in the student's frame, and watch the same answer appear. Rotate into the
student's axes (Lemma 1, Step 3b): c = Uᵀx_T are the teacher's coordinates, and the family is
diagonal there — N_S carries dials n_k, S_S carries dials μ_k, I − N_S carries dials 1 − n_k. Two
frame facts we use repeatedly: a rotation preserves norms (‖v‖² = sum of squared coordinates in the
frame), and a quadratic form with a diagonal middle matrix is a pure per-axis sum of squares —
component k of D_a c is a_k c_k (row k of a diagonal matrix touches only slot k), so

$$
c^\top D_a\, c = \sum_k c_k \cdot \big(a_k\, c_k\big) = \sum_k a_k\, c_k^2 ,
$$

with no cross term c_k c_l ever: it would have to come from an off-diagonal entry of D_a, and those
are zero.

*Term 1 — the interface error, dial by dial.* The settled interface error is ε_TS = −N_S x_T
(Step 1), whose k-th coordinate in the frame is −n_k c_k (apply N_S: scale coordinate k by its
dial). Norm = sum of squared coordinates:

$$
\frac{\pi_{TS}}{2}\,\|\varepsilon_{TS}\|^2 = \frac{\pi_{TS}}{2} \sum_k n_k^2\, c_k^2 .
$$

*Term 2 — the self error, dial by dial.* The settled state x_S\* = (I − N_S)x_T has coordinates
(1 − n_k)c_k. The sandwich gives ‖M_S x_S\*‖² = (x_S\*)ᵀ S_S x_S\*, a quadratic form with dials μ_k:

$$
\frac{\pi_S}{2}\,\|M_S\, x_S^{*}\|^2 = \frac{\pi_S}{2} \sum_k \mu_k\, (1 - n_k)^2\, c_k^2 .
$$

*The one computation with content — absorb μ_k into n_k.* Two facts read straight off the
definition n_k = π_S μ_k/(π_TS + π_S μ_k):

$$
1 - n_k = \frac{(\pi_{TS} + \pi_S \mu_k) - \pi_S \mu_k}{\pi_{TS} + \pi_S \mu_k} = \frac{\pi_{TS}}{\pi_{TS} + \pi_S \mu_k} , \qquad \pi_S\, \mu_k = n_k\, \big(\pi_{TS} + \pi_S \mu_k\big) .
$$

Substitute both into the self-term dial:

$$
\pi_S\, \mu_k\, (1 - n_k)^2 = n_k\,\big(\pi_{TS} + \pi_S \mu_k\big) \cdot \frac{\pi_{TS}^2}{\big(\pi_{TS} + \pi_S \mu_k\big)^2} = \pi_{TS}\; n_k \cdot \frac{\pi_{TS}}{\pi_{TS} + \pi_S \mu_k} = \pi_{TS}\; n_k\, (1 - n_k) .
$$

So the self term is (π_TS/2) Σ_k n_k(1 − n_k) c_k² — the raw self-error dial μ_k has been fully
absorbed into the novelty dial n_k.

*Add, dial by dial:*

$$
F_S^{\mathrm{eq}} = \frac{\pi_{TS}}{2} \sum_k \big[\, n_k^2 + n_k(1 - n_k) \,\big]\, c_k^2 = \frac{\pi_{TS}}{2} \sum_k n_k\, c_k^2 = \frac{\pi_{TS}}{2}\, x_T^\top N_S\, x_T ,
$$

the last equality rotating back (cᵀ diag(n_k) c = x_Tᵀ N_S x_T). Same identity, second proof —
the operator route above is this arithmetic done without ever choosing a frame. Per direction, in
units of (π_TS/2)c_k²:

| term | contribution | reading |
|---|---|---|
| interface ½π_TS‖ε_TS‖² | n_k² | the student left a fraction n_k of the component unmatched; squared cost |
| self ½π_S‖ε_S‖² | n_k(1 − n_k) | the student's own prediction error on the (1 − n_k) it *did* copy |
| **total** | **n_k** | the plain novelty reading |

However the equilibrium splits the penalty between "failing to copy the teacher" and "failing to
predict myself," the two costs always recombine into exactly n_k: a learned direction (n_k = 0)
costs nothing, a novel one costs its novelty. The tug-of-war of Lemma 1 has a price, and the price
*is* the novelty.

**Remark — why N_S and not S_S: surprise before vs. after inference.** One might expect the
student's free energy to be expressed through its self-error operator S_S rather than N_S. It is —
N_S = π_S S_S (π_TS I + π_S S_S)⁻¹ is S_S pushed through a specific function — and the inverse in
that function is the footprint of the *settling*. Compare:

- **Clamped student** (force x_S = x_T, no settling): ε_TS = 0 and

$$
F_S^{\text{clamped}} = \frac{\pi_S}{2}\, x_T^\top S_S\, x_T
$$

  — plain S_S, the **raw** surprise the student would feel if it had to represent the teacher's
  state exactly as given.

- **Settled student**: the compromise x_S\* = (I − N_S)x_T requires *solving a linear system*
  (Lemma 1, Step 2) — solving is where the inverse enters — and the leftover is Corollary 1's
  (π_TS/2) x_Tᵀ N_S x_T.

So the two operators are one quantity at two moments: **S_S is surprise before inference, N_S is
surprise after inference** — and F_S^eq, defined at the minimum, naturally wears the after-inference
operator. Per eigenvalue the discount is explicit: n(μ) = π_S μ/(π_TS + π_S μ) grows like
(π_S/π_TS)μ for nearly-learned directions but saturates at 1 for wildly novel ones — the student
can always cap its bill by dropping a hopeless direction entirely (paying only the interface
penalty). Settling only ever lowers the bill, (π_TS/2)n(μ) ≤ (π_S/2)μ per direction — the same
fact as the extinction bound N_S ⪯ (π_S/π_TS)S_S in Theorem 1, Consequence 2. This is the predictive-coding
slogan made an operator equation: surprise is not the raw mismatch, it is what remains *after
perception has done its best to explain the input away*.

**Remark — a second proof in one line: the envelope theorem, made explicit.** The direct
computation above never leaves linear algebra, and never used the "min" form of the definition.
This remark is where that form earns its keep: a one-line calculus route through a small general
fact worth owning — the **envelope theorem**. We state it, prove it, explain its name, and then
apply it.

*Setup.* A function of two things, f(x, θ). For each fixed θ, optimize away the x and record the
value at the optimum:

$$
g(\theta) \;=\; \min_x\, f(x, \theta) \;=\; f\big(x^{*}(\theta),\, \theta\big),
$$

where x\*(θ) is the minimizer (it moves as θ moves).

*Statement.* The slope of g ignores the motion of the minimizer entirely:

$$
g'(\theta) \;=\; \left.\frac{\partial f}{\partial \theta}\right|_{x = x^{*}(\theta)} ,
$$

i.e. differentiate f in θ **as if x were frozen at the optimum**.

*Proof (two lines, just the chain rule).* g(θ) = f(x\*(θ), θ) depends on θ twice — explicitly, and
through the dragged minimizer. Chain rule on both dependencies:

$$
g'(\theta) \;=\; \underbrace{\left.\frac{\partial f}{\partial x}\right|_{x^{*}}}_{=\,0}\cdot\,\frac{dx^{*}}{d\theta} \;+\; \left.\frac{\partial f}{\partial \theta}\right|_{x^{*}} .
$$

The first factor is zero **by the definition of a minimum**: x\*(θ) is precisely the point where the
slope in x vanishes — the very equation that defines it. So however fast the minimizer moves, its
motion is multiplied by zero: sliding along the bottom of a valley costs nothing to first order
(the "free lunch at a minimum"). ∎

*Why "envelope".* Draw one curve θ ↦ f(x, θ) for each frozen x — a whole family of curves over the
θ-axis. The value function g traces the **lower envelope** of that family: at each θ it sits on
whichever curve is lowest there, and at the touching point the envelope and that curve are
*tangent* — same value, same slope. "Same slope" is exactly the statement above.

*Toy check (our model in d = 1).* Blank student (W_S = 0, so M_S = 1), π_TS = π_S = 2:
f(x_S, x_T) = (x_S − x_T)² + x_S². Minimize in x_S: 2(x_S − x_T) + 2x_S = 0 gives x_S\* = x_T/2 —
the tug-of-war settles halfway. Direct route (plug in): g(x_T) = (x_T/2 − x_T)² + (x_T/2)² = x_T²/2,
so g′ = x_T. Envelope route (freeze): ∂f/∂x_T = −2(x_S − x_T), at x_S\* = x_T/2 this is
−2(−x_T/2) = x_T. Same answer, no plugging-in.

*Application to F_S.* Here f = F_S(x_S, x_T), θ = x_T, and x_S\*(x_T) is the Lemma-1 equilibrium —
whose defining equation (Lemma 1, Step 1) is exactly ∇_{x_S}F_S = 0, the "slope zero at the minimum"
the proof needs. Only the interface term ½π_TS‖x_S − x_T‖² contains x_T explicitly, and by the
Step-1 rule of Prop 1 (with the roles of the two vectors swapped, which flips the sign) its
x_T-gradient is π_TS(x_T − x_S). Freeze x_S at the optimum:

$$
\nabla_{x_T} F_S^{\mathrm{eq}} = \left.\frac{\partial F_S}{\partial x_T}\right|_{x_S = x_S^{*}} = \pi_{TS}\,(x_T - x_S^{*}) = -\,\pi_{TS}\,\varepsilon_{TS} = \pi_{TS}\, N_S\, x_T .
$$

Since F_S^eq and (π_TS/2)x_TᵀN_S x_T both vanish at x_T = 0 and have the same gradient everywhere,
they are equal — Corollary 1 again. Fine print, satisfied here without caveats: the two-line proof
wants an interior, smooth minimum with a well-behaved minimizer; F_S is a *strictly convex
quadratic* in x_S (B = π_TS I + π_S S_S is positive definite, Lemma 1 Step 3), so x_S\*(x_T) is
unique and linear in x_T. The direct computation of Steps 1–4 is this theorem done by hand — and
the gradient it yields is exactly the one Theorem 1 needs.

*(Numerical check: `diagnostics.surprise_identity_error` settles the student at random x_T,
evaluates F_S directly, and compares with (π_TS/2)x_TᵀN_S x_T — they agree to ~1.8e-15, machine
precision; asserted in `tests/smoke_test.py` both before and after learning.)*

---

## Theorem 1 — the student steers the teacher up the *student's* surprise

*(For the geometric, no-step-skipped version — a picture at every step — see
[intuitive_proof.md](intuitive_proof.md).)*

**Toolkit.** *Core:* Lemma 1 (the push read-off), Corollary 1 (the surprise identity and its
gradient — via (G1) or the envelope remark). *Consequences:* the rotate–scale–rotate picture of
Lemma 1, Step 3b for the axis-by-axis prioritization (§7, pattern P7); the trajectory rate of (G1)
and the PSD bound n(μ) ≤ (π_S/π_TS)μ (§6) for the climb/self-limiting; nothing new for the saddle
preview. *Expository only:* Cauchy–Schwarz (proved in the closing block).

**Claim (fast student state + frozen student weights, deterministic; see Timescales above).**
**The core statement:** isolate the student's contribution to the teacher's motion (the push
u = |π_ST| N_S x_T); then the teacher's dynamics under the push alone is **exact gradient ascent on
the student's settled variational free energy**,

$$
\tau_T\, \dot x_T \big|_{\text{push}} \;=\; \frac{|\pi_{ST}|}{\pi_{TS}}\, \nabla_{x_T} F_S^{\mathrm{eq}}(x_T) ,
$$

at every point x_T, with no approximation. Consequences, logically downstream and proved after the
core: (a) *prioritization* — axis by axis, the push amplifies each component of the teacher's state
at a rate proportional to that direction's novelty (learned directions exactly frozen); (b) the
student's surprise never decreases under the push, and the push self-extinguishes direction by
direction as the student learns; (c) the full flow (push + self-pull) is a single-potential saddle
flow on F_T − F_S exactly when π_ST = −π_TS. Caveats, stated openly: this is a *local* ascent from
the current state (not a pointer at the global summit); the push rescales but cannot *seed* a
direction the state does not occupy (noise supplies the seed in the full model); no manifold
restriction is used; and no "teacher's surprise" is ever invoked — the teacher's own surprise F_T
enters only through the self-pull, and the quantity the push climbs is the *student's*.

### The core, in two moves

**Move 1 — isolate the student's push (Lemma 1 + the teacher's equation).** The teacher in sleep obeys

$$
\tau_T\, \dot x_T = -\pi_T\, M_T^\top \varepsilon_T + \pi_{ST}\, \varepsilon_{TS} + \xi .
$$

Substitute the self-term M_Tᵀε_T = M_TᵀM_T x_T = S_T x_T and the fast-student interface error
ε_TS = −N_S x_T (Lemma 1); drop the noise ξ; use π_ST < 0 so −π_ST = |π_ST|:

$$
\tau_T\, \dot x_T = \underbrace{-\pi_T\, S_T\, x_T}_{\text{teacher self-pull}} \; + \; \underbrace{|\pi_{ST}|\, N_S\, x_T}_{\text{student's push } u} .
$$

We study only the student's push,

$$
u = |\pi_{ST}|\, N_S\, x_T ,
$$

which is latent in the equation — no extra construction needed. (The self-pull −π_T S_T x_T is the
teacher holding itself near its *own* memory; that it wins off-manifold is the guard, **Lemma 2**, shown
numerically — a logically separate result.)

**Move 2 — compare with the gradient of the student's settled free energy.** By Corollary 1, the
student's surprise about the teacher's state — its settled free energy, viewed as a function of
where the teacher is — is

$$
F_S^{\mathrm{eq}}(x_T) = \frac{\pi_{TS}}{2}\, x_T^\top N_S\, x_T ,
$$

**not a new quantity invented for the theorem** — the model's own F_S, with the fast student
settled. Its gradient: nudge x_T by δ and match the (G1) template. Expanding the quadratic form,
the two cross terms are equal (N_S symmetric, a scalar equals its own transpose:
δᵀN_Sx_T = x_TᵀN_Sδ), and they sum against the ½:

$$
F_S^{\mathrm{eq}}(x_T + \delta) = F_S^{\mathrm{eq}}(x_T) + \big(\pi_{TS}\, N_S\, x_T\big)^\top \delta + O(\|\delta\|^2) \quad\Longrightarrow\quad \nabla_{x_T} F_S^{\mathrm{eq}} = \pi_{TS}\, N_S\, x_T
$$

(the same gradient the envelope argument gave in Corollary 1's remark; in terms of derivatives,
component i is ∂F_S^eq/∂x_{T,i} = π_TS(N_S x_T)_i). Now compare the two vectors:

$$
u = |\pi_{ST}|\, N_S\, x_T \qquad\text{and}\qquad \nabla_{x_T} F_S^{\mathrm{eq}} = \pi_{TS}\, N_S\, x_T
$$

— the *same* vector up to the positive constant |π_ST|/π_TS. Hence, under the push alone,

$$
\boxed{\ \tau_T\, \dot x_T \big|_{\text{push}} = u = \frac{|\pi_{ST}|}{\pi_{TS}}\, \nabla_{x_T} F_S^{\mathrm{eq}}(x_T) \ } .
$$

**That is the theorem.** At this precise point x_T — every point x_T — the teacher's dynamics,
restricted to the student's contribution, is exact gradient **ascent** on the student's variational
free energy at the settled state. Since a gradient is the direction of locally steepest increase (a
standard fact, essentially the definition — unpacked in the expository block at the end for
completeness), the push steers the teacher toward *maximum student surprise*, locally. Two moves,
both already proved elsewhere (Lemma 1, Corollary 1): the theorem is the comparison. ∎

### Consequences — logically downstream of the core

*(None of the following is needed to prove the core; each unpacks what the ascent does. The first
is the "prioritized" of the paper's title and is what the experiments measure.)*

**Consequence 1 — prioritization, axis by axis (diagonalize the dynamics).**
N_S is symmetric, so the rotate–scale–rotate machinery of Lemma 1, Step 3b applies verbatim:

$$
N_S = U\, D_n\, U^\top , \qquad U = [\,u_1 | \cdots | u_d\,] \text{ a rotation } (U^\top U = U U^\top = I), \qquad D_n = \mathrm{diag}(n_1, \dots, n_d) .
$$

The n_k are the **novelty eigenvalues** of the Terminology note: n_k ∈ [0,1), n_k = 0 exactly on a
learned direction, growing with the student's self-error μ_k = ‖M_S u_k‖² through the increasing
map n(μ) = π_S μ/(π_TS + π_S μ). One crucial observation before touching the dynamics:
**frozen weights ⇒ frozen frame.** W_S is constant on the timescale considered (Timescales), so
S_S, the rotation U, and the dials n_k are all constant in time — only the teacher's state moves.
Now diagonalize the motion in four elementary moves.

*(a) Change coordinates — rotate the state into the student's frame.* Define, at every instant,

$$
c(t) = U^\top x_T(t) \qquad (\text{and back: } x_T(t) = U\, c(t)) ,
$$

the teacher's coordinates on the student's axes (component k is c_k = u_kᵀx_T, row k of Uᵀ).
Because U is **constant in time**, the time derivative slides past it:

$$
\dot c(t) = \frac{d}{dt}\big(U^\top x_T(t)\big) = U^\top \dot x_T(t) .
$$

*(b) Rotate the equation of motion.* Under the push alone the motion is τ_T ẋ_T = u = |π_ST| N_S x_T.
Multiply both sides on the left by Uᵀ (rotating the whole equation into the student's frame), insert
the three-matrix form of N_S, and regroup by associativity:

$$
\tau_T\, U^\top \dot x_T \;=\; |\pi_{ST}|\, U^\top \big(U\, D_n\, U^\top\big)\, x_T \;=\; |\pi_{ST}|\, \underbrace{(U^\top U)}_{=\,I}\, D_n\, \underbrace{\big(U^\top x_T\big)}_{=\,c} \;=\; |\pi_{ST}|\, D_n\, c .
$$

The middle UᵀU evaporated (Step 3b(iii)), and the left side is τ_T ċ by (a). So **in the student's
coordinates the dynamics is diagonal**:

$$
\tau_T\, \dot c \;=\; |\pi_{ST}|\, D_n\, c .
$$

*(c) A diagonal system IS d independent scalar equations (this is where "no mixing" is proved —
by the visible zeros).* Write out row k of the right side: row k of D_n is all zeros except n_k in
slot k, so it reads n_k c_k and touches no other coordinate. The vector equation therefore splits
into

$$
\boxed{\ \tau_T\, \dot c_k = |\pi_{ST}|\, n_k\, c_k \ , \qquad k = 1, \dots, d\ } ,
$$

one autonomous scalar equation per axis — **independent and additive**: the equation for c_k
contains c_k and nothing else. Any coupling between coordinates would have to live in the
off-diagonal entries of the middle matrix, and they are zero. That empty off-diagonal *is* the
no-mixing proof.

*(d) Solve each axis.* Each equation is the elementary growth law ċ = λc with constant rate
λ_k = |π_ST| n_k / τ_T, solved by c(t) = c(0)e^{λt} (check: differentiate, get λ·c(0)e^{λt} = λc ✓;
uniqueness is standard for linear ODEs). Hence

$$
c_k(t) = c_k(0)\, \exp\!\Big(\frac{|\pi_{ST}|\, n_k}{\tau_T}\, t\Big) .
$$

The physical state is recovered by rotating back out of the student's frame:
x_T(t) = U c(t) = Σ_k c_k(t) u_k — each independently-grown coordinate rides its own fixed axis.

Reading: a **learned** axis (n_k = 0) has e⁰ = 1 — its coordinate is exactly frozen, for all time. A
**novel** axis grows exponentially, at a rate proportional to its novelty. And the *ratios* order
the competition: c_k(t)/c_l(t) = (c_k(0)/c_l(0))·e^{(n_k − n_l)|π_ST| t/τ_T}, so among the axes the
state actually occupies, the most novel one eventually dominates the teacher's heading (the
power-iteration effect). That already settles "the student steers the teacher toward novel places,"
using nothing but the push and the diagonalization.

**Consequence 2 — the climb is monotone, and it self-extinguishes.**

Under the push alone (ẋ_T = u/τ_T), the
rate of change of the student's surprise along the motion is given by the trajectory consequence of
(G1) — take δ = ẋ_T·dt in the template, F(x_T + ẋ_T dt) − F(x_T) = (∇Fᵀẋ_T)·dt + O(dt²), and
divide by dt (this *is* the chain rule):

$$
\frac{dF_S^{\mathrm{eq}}}{dt} = \nabla F_S^{\mathrm{eq}} \cdot \dot x_T = \big(\pi_{TS} N_S x_T\big)^\top \frac{|\pi_{ST}|}{\tau_T} N_S x_T = \frac{\pi_{TS}\,|\pi_{ST}|}{\tau_T}\,\|N_S x_T\|^2 \ \ge\ 0 ,
$$

so the student's surprise never decreases, strictly increasing unless N_S x_T = 0 (state fully
learned).
**Self-limiting (this absorbs the old Corollary 1):** the per-direction push rate is bounded by the
student's residual there, since n(μ) ≤ (π_S/π_TS)μ gives

$$
|\pi_{ST}|\, n_k \ \le\ \frac{|\pi_{ST}|\,\pi_S}{\pi_{TS}}\,\|M_S u_k\|^2 .
$$

As the student learns a direction (Proposition 1 drives ‖M_S u_k‖ → 0), the push on it vanishes
quadratically in the residual, and shuts off exactly when the direction is learned (n_k = 0). So the
steering is self-terminating, direction by direction. (Note the pleasing loop: the teacher climbs a
surprise landscape that its own climbing causes the student to flatten — Theorem 1 holds the weights
frozen, but once Proposition 1 is switched back on, every direction the teacher visits gets learned,
its n_k drops, and the landscape deflates under the teacher's feet.)

**Consequence 3 — the exact-saddle condition falls out (preview of Proposition 3).** Write the
full sleep flow of Move 1 with *both* forces as gradients. The self-pull already is one: the
teacher's own surprise F_T = (π_T/2)‖M_T x_T‖² has, by the Step-1 rule of Prop 1 with A = M_T,

$$
\nabla_{x_T} F_T = \pi_T\, M_T^\top M_T\, x_T = \pi_T\, S_T\, x_T ,
$$

so the self-pull is −∇F_T — the teacher descending its *own* surprise. And by the core the push is
(|π_ST|/π_TS)∇F_S^eq — the teacher ascending the *student's* surprise. The whole flow is therefore

$$
\tau_T\, \dot x_T = -\,\nabla_{x_T} F_T \;+\; \frac{|\pi_{ST}|}{\pi_{TS}}\, \nabla_{x_T} F_S^{\mathrm{eq}} .
$$

This is the flow generated by the single saddle potential Φ = F_T − F_S^eq (i.e. τ_T ẋ_T = −∇Φ)
**exactly when the coefficient of the ascent equals 1: |π_ST| = π_TS, that is π_ST = −π_TS** — which
is Proposition 3's exact-saddle condition, obtained here in one line, with a transparent reading:
*the strength with which the teacher climbs the student's surprise must equal the precision with
which that surprise was measured.* (Proposition 3 proper adds the sphere constraint, the mobilities,
and the student's projected W_S ascent; see curriculum L6. This consequence only shows where the
condition comes from.)

### Expository — "steepest ascent", made precise (NOT for the paper)

*(In the manuscript, "the gradient is the direction of locally steepest ascent" is one clause with
no proof — a standard fact, essentially the definition of the gradient. This block unpacks it once,
for the reader of this document, because the core leans on that clause for its interpretation.)*

Why does "along the gradient" mean "steepest ascent"? Two ingredients.

*Ingredient 1 — the rate of change in a direction, from (G1).* Step from x_T by a small amount ε
along a **unit** direction e (i.e. δ = εe, ‖e‖ = 1). The (G1) expansion gives

$$
F_S^{\mathrm{eq}}(x_T + \varepsilon e) - F_S^{\mathrm{eq}}(x_T) = \varepsilon\,\big(\nabla F_S^{\mathrm{eq}} \cdot e\big) + O(\varepsilon^2) ,
$$

so **the per-unit-length rate of increase of the surprise in direction e is the dot product
∇F_S^eq · e** — one number per candidate direction. "Steepest ascent" is now a well-posed
competition: which unit e makes that dot product largest?

*Ingredient 2 — Cauchy–Schwarz decides the competition.* For any two vectors a and e,

$$
a \cdot e \ \le\ \|a\|\,\|e\| , \qquad \text{with equality iff } e \text{ points along } a .
$$

*Why it is true (geometric):* the dot product is a·e = ‖a‖‖e‖ cos θ, where θ is the angle between
the two vectors — and cos θ ≤ 1, with equality exactly when θ = 0, i.e. when e and a point the same
way. *(Algebraic proof, if you prefer no geometry: the quadratic
q(t) = ‖a − te‖² = ‖a‖² − 2t(a·e) + t²‖e‖² is a sum of squares, hence ≥ 0 for every real t; a
quadratic that never goes negative has non-positive discriminant, 4(a·e)² − 4‖a‖²‖e‖² ≤ 0, which is
the inequality squared. Equality forces q(t\*) = 0 at the minimizing t\*, i.e. a = t\*e — parallel.)*

Apply it with a = ∇F_S^eq and a unit e (‖e‖ = 1):

$$
\text{rate in direction } e \;=\; \nabla F_S^{\mathrm{eq}} \cdot e \ \le\ \|\nabla F_S^{\mathrm{eq}}\| , \qquad \text{equality iff } e = \frac{\nabla F_S^{\mathrm{eq}}}{\|\nabla F_S^{\mathrm{eq}}\|} .
$$

So among **all** unit directions the teacher could be pushed in, the largest achievable rate of
surprise increase is ‖∇F_S^eq‖, and it is achieved **only** by the gradient direction — which is
where the push points.

**Conclusion.** The theorem itself is two moves and a comparison: Lemma 1 isolates the push
u = |π_ST|N_S x_T; Corollary 1 gives ∇F_S^eq = π_TS N_S x_T; they are the same vector up to a
positive constant, so the isolated teacher dynamics is exact gradient ascent on the student's
settled variational free energy — at every point, no approximation. Downstream of that core:
per-axis prioritization (Consequence 1 — rates |π_ST|n_k/τ_T, learned frozen, most-novel dominates),
monotone climb with self-extinction (Consequence 2), and the exact-saddle condition π_ST = −π_TS
(Consequence 3). No manifold, compression, or projection — and no "teacher's surprise": the teacher
descends its own F_T while being steered up the student's F_S. Two caveats stated openly, not
buried: the ascent is *local* (not a pointer at the global summit), and the push rescales but does
not *seed* an empty direction (noise seeds it). **Deferred to future work:** *which* of the novel
directions ultimately dominate once the teacher's own memory geometry is folded in (the
restricted-novelty / manifold analysis) — the teacher staying near its manifold is the guard's job,
Lemma 2, shown numerically. ∎

*(Numerical check: u = |π_ST| N_S x_T = (|π_ST|/π_TS)∇F_S^eq with F_S^eq verified to machine
precision by `diagnostics.surprise_identity_error` (Corollary 1) — the core. For Consequence 1,
`model.novelty_operator()` gives N_S with novelty eigenvalues n_k that are 0 on learned directions
and positive on unlearned ones, so the per-axis rates |π_ST|n_k are 0 on learned axes and positive
on novel ones.)*
