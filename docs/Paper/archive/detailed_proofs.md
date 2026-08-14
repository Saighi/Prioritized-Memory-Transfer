# Detailed proofs — Proposition 1, Theorem 1, Lemma 1, Corollary 1, Proposition 2

**Logical order (and reading order):** Proposition 1 (the student descends F_S) →
**Theorem 1** (the student's push on the teacher is gradient ascent on the student's settled
F_S — proved structurally, in two short steps, with *no operator algebra*) → Lemma 1 +
Corollary 1 (what that landscape *is*: the novelty-weighted quadratic, plus the prioritization
remark) → Proposition 2 (the exact saddle, one line from Theorem 1). Manifold containment is
**not a theorem of the paper**: it is an operating choice (|π_ST| kept well below π_T — the code
default is half the stability threshold) plus a numerical verification; an archived
sufficient-condition lemma lives in
[additional_proofs_not_in_paper.md](additional_proofs_not_in_paper.md) and is deliberately kept
out of the paper.

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

One status note inherited from final_proofs §1 and assumed everywhere below: **the dynamics is
the primitive, the free energies are partial descriptors.** In wake every force is a VFE
gradient; in sleep the teacher's reversed-precision interface term is *postulated*, owned by no
bounded free energy — Theorem 1 characterizes it exactly (ascent on the student's settled F_S:
curiosity, not inference) and Proposition 2 recovers it geometrically (the negative half of the
zero-sum saddle at π_ST = −π_TS; active-inference resemblance speculative).

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
Step 1) and ∇ₓ ½xᵀAx = Ax for symmetric A (derived in Corollary 1's gradient remark). (§8)

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

- the **sandwich**: vᵀMᵀMv = ‖Mv‖² ≥ 0 (Lemma 1, Step 1(iii); §6);
- the **spectral theorem** in three-matrix form — a symmetric matrix is A = U D Uᵀ, a *rotation*
  (U, orthonormal eigenvector columns, UᵀU = UUᵀ = I), a *diagonal dial-board* (D, the
  eigenvalues), and the rotation back — with **spectral mapping** (functions of A keep the frame
  and act on the dials). Built in full in Lemma 1, Step 1 (rotate–scale–rotate reading,
  coordinate change c = Uᵀx, middle-UᵀU evaporation in products), and used to decouple quadratic
  forms (Corollary 1) and dynamics (Lemma 1, Steps 2 and 5) (§7);
- the **eigenvalue shift** A + cI and "invertible ⟺ no zero eigenvalue" (Lemma 1, Step 3; §4, §6);
- the **commuting family**: everything built from S_S commutes and shares its eigenbasis
  (§3) — a fact of the legacy operator route; the default proofs use the rotation machinery of
  Lemma 1, Step 1 instead;
- the **Frobenius projection facts**: P² = P, ⟨PX, Y⟩ = ⟨X, PY⟩, hence ⟨G, PG⟩ = ‖PG‖² ≥ 0
  (Prop 1, Steps 7–8; §9);
- the **envelope theorem** (stated and proved in Theorem 1, Step 2; §8, pattern P9).

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

- **The student STATE being FAST (used by Theorem 1, Step 2, and by Lemma 1).** Because x_S
  equilibrates almost instantly, it is *already settled* at every teacher configuration; we set the
  state rate to zero (ẋ_S = 0) and solve for x_S. The student's state is fixed here **because it is
  fast, not slow.** This is what makes the settled free energy F_S^eq(x_T) = min over x_S of F_S a
  well-defined landscape over the teacher's state alone — the object Theorem 1's envelope step needs.
  In Lemma 1, the leftover interface error ε_TS = −N_S x_T is the residual of the *battle* at that
  equilibrium between the interface term (weight π_TS, "copy the teacher, x_S = x_T") and the self
  term (weight π_S, "stay on my own manifold, M_S x_S = 0"): zero on directions where the two agree
  (learned), nonzero where they conflict (unlearned). That residual is reinjected as the drive on the
  teacher.
- **The student WEIGHTS being SLOW (frozen — used by Theorem 1 and by Lemma 1, Steps 2 and 5).** Over
  the timescale on which the teacher explores, W_S is effectively constant, so the landscape F_S^eq —
  and, in the landscape section, the operators S_S, N_S and their eigenframe — are fixed. This — not
  any slowness of the state — is the "frozen student" in Theorem 1.

In one line: **fast student state (the landscape exists) + frozen student weights (the landscape
holds still).** The coupled process in which W_S actually changes (so the landscape deflates and the
teacher re-targets the next-novel direction) is the learning dynamics, left to the empirical results.

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

## Theorem 1 — the student's push is gradient ascent on the student's surprise

*(For the geometric, no-step-skipped version — a picture at every step — see
[intuitive_proof.md](intuitive_proof.md).)*

**Toolkit:** (R1); (G1) — specifically the interface gradient already computed in Prop 1, Step 2;
the sandwich (§6) for one line of convexity; the **envelope theorem**, stated and proved below
(pattern P9). Nothing else. In particular: **no novelty operator, no operator algebra** — Lemma 1
is not used. The theorem is structural, and the structure is a *sign*.

**Claim (fast student state + frozen student weights, deterministic; see Timescales above).**
Isolate the student's contribution to the teacher's sleep motion — the push u = π_ST ε_TS, read
directly off the teacher's equation. Then the teacher's dynamics under the push alone is **exact
gradient ascent on the student's settled variational free energy**

$$
F_S^{\mathrm{eq}}(x_T) \;=\; \min_{x_S}\, F_S(x_S,\, x_T) :
$$

$$
\tau_T\, \dot x_T \big|_{\text{push}} \;=\; \frac{|\pi_{ST}|}{\pi_{TS}}\, \nabla_{x_T} F_S^{\mathrm{eq}}(x_T) ,
$$

at every point x_T, with no approximation. Caveats, stated openly: this is a *local* ascent from
the current state (not a pointer at the global summit); the push rescales but cannot *seed* a
direction the state does not occupy (noise supplies the seed in the full model); no manifold
restriction is used; and no "teacher's surprise" is ever invoked — the teacher's own surprise F_T
enters only through the self-pull, and the quantity the push ascends is the *student's*.

### Step 1 — the sign identity: teacher and student share one interface energy

The teacher in sleep obeys

$$
\tau_T\, \dot x_T = \underbrace{-\,\pi_T\, M_T^\top \varepsilon_T}_{\text{teacher self-pull}} \;+\; \underbrace{\pi_{ST}\, \varepsilon_{TS}}_{\text{student's push } u} \;+\; \xi .
$$

The push u = π_ST ε_TS is latent in the equation — no construction needed. Drop the noise ξ
(deterministic claim) and set the self-pull aside (it is the teacher holding itself near its *own*
memory; that it wins off-manifold is an operating-regime matter — |π_ST| kept well below π_T —
verified numerically, not proved).

Now differentiate F_S with respect to **the teacher's state**, holding x_S wherever it happens to
be — any x_S, settled or not. Only the interface term of F_S contains x_T (the self term
½π_S‖M_S x_S‖² does not), and the interface gradient was already computed in Prop 1, Step 2; with
the roles of the two vectors swapped the sign flips:

$$
\nabla_{x_T} F_S \;=\; \pi_{TS}\,(x_T - x_S) \;=\; -\,\pi_{TS}\, \varepsilon_{TS} .
$$

Compare with the push. Since π_ST < 0 in sleep (so −π_ST = |π_ST|):

$$
u \;=\; \pi_{ST}\, \varepsilon_{TS} \;=\; \frac{-\pi_{ST}}{\pi_{TS}}\,\big(-\pi_{TS}\,\varepsilon_{TS}\big) \;=\; \frac{|\pi_{ST}|}{\pi_{TS}}\, \nabla_{x_T} F_S \Big|_{x_S} .
$$

One line, valid at every instant: **the teacher and the student feel the same shared interface
energy ½π_TS‖x_S − x_T‖² through opposite-signed precisions, so the teacher's drive is the
student's gradient, sign-flipped and rescaled.** The adversarial pairing of the two populations is
not derived — it is *built into the precision antisymmetry*, and the sign of π_ST is the entire
wake/sleep switch: π_ST > 0 would make the same term a *descent* (the teacher accommodating the
student); π_ST < 0 makes it an ascent. Already at this point, with no assumption on x_S, the push
performs instantaneous partial-gradient ascent on F_S in the x_T-slot.

### Step 2 — the envelope theorem: the partial gradient IS the landscape gradient

Step 1's gradient is a *partial* gradient of a functional that also depends on the moving x_S; to
speak of ascent on a fixed landscape over x_T alone, we need the fast-student assumption — and one
small, classical tool worth owning. We state it, prove it, explain its name, and apply it.

*Setup.* A function of two things, f(x, θ). For each fixed θ, optimize away the x and record the
value at the optimum:

$$
g(\theta) \;=\; \min_x\, f(x, \theta) \;=\; f\big(x^{*}(\theta),\, \theta\big),
$$

where x\*(θ) is the minimizer (it moves as θ moves).

*Statement (the envelope theorem).* The slope of g ignores the motion of the minimizer entirely:

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

*Application.* Here f = F_S(x_S, x_T), θ = x_T, and x\*(θ) is the fast student's settled state
x_S\*(x_T) — whose defining equation is exactly ∇_{x_S}F_S = 0 (Prop 1's gradient set to zero), the
"slope zero at the minimum" the proof needs. Fine print, satisfied without caveats: F_S is a
*strictly convex quadratic* in x_S — its Hessian is π_TS I + π_S S_S ⪰ π_TS I ≻ 0, since
S_S = M_Sᵀ M_S is PSD by the sandwich vᵀS_S v = ‖M_S v‖² ≥ 0 — so the settled state is the unique
minimum and F_S^eq(x_T) = min over x_S is well defined and smooth. The envelope theorem then gives,
using Step 1's partial gradient frozen at the optimum:

$$
\nabla_{x_T} F_S^{\mathrm{eq}}(x_T) \;=\; \left.\frac{\partial F_S}{\partial x_T}\right|_{x_S = x_S^{*}} \;=\; -\,\pi_{TS}\,\varepsilon_{TS}\big|_{\text{settled}} .
$$

Combine with Step 1 evaluated at the settled state, and the ε_TS cancels out of sight:

$$
\boxed{\ \tau_T\, \dot x_T \big|_{\text{push}} \;=\; u \;=\; \frac{|\pi_{ST}|}{\pi_{TS}}\, \nabla_{x_T} F_S^{\mathrm{eq}}(x_T) \ } .
$$

**That is the theorem.** At every point x_T, the teacher's dynamics, restricted to the student's
contribution, is exact gradient ascent on the student's variational free energy at the settled
state. Since the gradient is the direction of locally steepest increase (a standard fact,
essentially the definition of the gradient — one clause in the paper), the push steers the teacher
toward *maximum student surprise*, locally. ∎

**Note what was *not* needed.** The novelty operator never appeared: the theorem is the sign
identity plus the envelope theorem, and both are indifferent to what the landscape looks like. The
novelty operator enters *next*, when we ask the natural follow-up question — what **is** the
landscape F_S^eq the teacher is climbing? That is Lemma 1 and Corollary 1's job.

**Consequence recorded for later (Proposition 2).** Step 1 also shows exactly when the ascent
coefficient is 1: |π_ST| = π_TS, i.e. π_ST = −π_TS — the teacher's push is then *precisely* the
reversed gradient of the student's surprise. That is the exact-saddle condition, harvested in
Proposition 2 below.

*(Numerical check: the push π_ST ε_TS equals (|π_ST|/π_TS)∇F_S^eq with F_S^eq verified to machine
precision by `diagnostics.surprise_identity_error` (~1.8e-15, asserted in `tests/smoke_test.py`).)*

---

## Lemma 1 — the fast student, mode by mode: the novelty operator emerges

**Toolkit:** the spectral theorem in rotate–scale–rotate form (§7) — built in full in Step 1;
coordinate read-off by dotting; the sandwich (§6), used once (to see μ_k ≥ 0); one-variable algebra.
**Not used:** no matrix inverse, no commuting arguments, no operator manipulation — the default
route settles one scalar equation per mode and only *names* a matrix at the end. (The coordinate-free
operator route — factor and invert — is a legitimate alternative; it is recorded in the paper file
[final_proofs.md](final_proofs.md) only through its endpoint, the closed form of Step 6.)

**Claim (fast student, any teacher state x_T; the sign of π_ST plays no role).** The student's
settled state and the leftover interface error are

$$
x_S^{*} = (I - N_S)\, x_T , \qquad \varepsilon_{TS} = -\,N_S\, x_T ,
$$

with the **novelty operator** defined spectrally: in the orthonormal eigenbasis U of S_S,

$$
N_S \;=\; U\, \mathrm{diag}\big(n(\mu_k)\big)\, U^\top , \qquad n(\mu) = \frac{\pi_S\, \mu}{\pi_{TS} + \pi_S\, \mu} \ \in\ [0, 1) ,
$$

symmetric, PSD, vanishing exactly on learned directions; equivalently (Step 6)
N_S = π_S S_S (π_TS I + π_S S_S)⁻¹, the closed form the code computes. Moreover — and this now
lives *inside* the lemma — the teacher's per-mode law under the push alone is
τ_T ċ_k = |π_ST| n_k c_k: **prioritization is born here**, before any energy is mentioned.

### Step 1 — the rotation machinery: rotate, scale, rotate back

S_S = M_SᵀM_S is symmetric ((M_SᵀM_S)ᵀ = M_SᵀM_S by (R1)), so the **spectral theorem** applies in
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

Sanity check on an eigenvector u_l: Uᵀu_l = e_l (u_l's coordinates are 1 on axis l, 0 elsewhere);
D_μ e_l = μ_l e_l; U(μ_l e_l) = μ_l u_l. Net: S_S u_l = μ_l u_l ✓.

*(iii) The dials are squared residuals — hence nonnegative.* One sandwich (§6), applied to a unit
eigenvector:

$$
\mu_k \;=\; u_k^\top S_S\, u_k \;=\; u_k^\top M_S^\top M_S\, u_k \;=\; \|M_S\, u_k\|^2 \ \ge\ 0 :
$$

each dial μ_k is the student's **squared self-error along its own axis** — zero exactly on a
**learned** direction (M_S u_k = 0), positive on a novel one. This is the only place the sandwich
is needed.

### Step 2 — the student's dynamics separates into modes

Write the student's state equation (Prop 1, Part A) and rotate the whole equation into the
student's frame. With s = Uᵀx_S and c = Uᵀx_T (both frames share the constant U — frozen weights ⇒
frozen frame), multiply the equation on the left by Uᵀ; the middle UᵀU evaporates exactly as in
Step 1(ii), and S_S turns into its dial-board D_μ:

$$
\tau_S\, \dot s_k \;=\; -\,\pi_{TS}\,\big(s_k - c_k\big) \;-\; \pi_S\, \mu_k\, s_k , \qquad k = 1, \dots, d .
$$

**One scalar ODE per mode, no coupling anywhere** — the equation for s_k contains s_k and c_k only.
(Any coupling would need an off-diagonal entry in the middle matrix, and there are none.) Each mode
is its own little tug-of-war: an interface spring of stiffness π_TS pulling s_k toward the
teacher's coordinate c_k, against a leak of strength π_S μ_k pulling s_k toward 0 (the student's
own model, which predicts nothing along a novel axis).

### Step 3 — settle each mode, and meet the dial n(μ)

Fast student: set ṡ_k = 0 and solve the one-variable equation. Collect the s_k terms and divide by
their coefficient π_TS + π_S μ_k — a strictly positive **number** (π_TS > 0, π_S > 0, μ_k ≥ 0), so
the division is legitimate with no invertibility argument at all:

$$
s_k^{*} \;=\; \frac{\pi_{TS}}{\pi_{TS} + \pi_S\, \mu_k}\; c_k \;=\; \big(1 - n_k\big)\, c_k , \qquad e_k \;=\; s_k^{*} - c_k \;=\; -\,n_k\, c_k ,
$$

where the error coordinate came from putting c_k over the common denominator, and

$$
n_k \;=\; n(\mu_k) , \qquad n(\mu) \;=\; \frac{\pi_S\, \mu}{\pi_{TS} + \pi_S\, \mu} .
$$

**Each mode is copied up to the attenuation factor (1 − n_k), and the fraction n_k is left behind
as interface error** — "copied + missing = whole," one axis at a time.

**The dial, studied as a one-variable function** (each fact is one line):

- *Ends:* n(0) = 0 (learned ⇒ no error at all) and n(μ) → 1 as μ → ∞ (the student gives up the
  axis entirely; the error is the whole coordinate).
- *Never reaches the ceiling:* the denominator strictly exceeds the numerator (π_TS > 0), so
  n(μ) < 1 for every finite μ.
- *Strictly increasing:* rewrite n(μ) = 1 − π_TS/(π_TS + π_S μ) — as μ grows the subtracted
  fraction shrinks, so n rises. (Equivalently n′(μ) = π_S π_TS/(π_TS + π_S μ)² > 0.)
- *The linear bound (self-limiting, used in Step 5):* from π_TS + π_S μ ≥ π_TS, take reciprocals
  (inequality flips) and multiply by π_S μ ≥ 0:

$$
n(\mu) \;\le\; \frac{\pi_S}{\pi_{TS}}\,\mu , \qquad \text{equality only at } \mu = 0 .
$$

  Geometrically, (π_S/π_TS)μ is the tangent to n at the origin (n′(0) = π_S/π_TS), and the
  saturating curve sits below its tangent ever after.

### Step 4 — reassemble the modes: the novelty operator is born

Stack the per-mode results back into vectors. Every vector is the sum of its coordinates times the
axes, so

$$
\varepsilon_{TS} \;=\; \sum_k e_k\, u_k \;=\; -\sum_k n_k\, c_k\, u_k \;=\; -\,U\, D_n\, c \;=\; -\,\big(U\, D_n\, U^\top\big)\, x_T , \qquad D_n = \mathrm{diag}(n_1, \dots, n_d) ,
$$

reading the sum as (rotation)(diagonal)(coordinates) exactly as in Step 1(ii), then substituting
c = Uᵀx_T. The matrix that appears is *named*

$$
N_S \;:=\; U\, D_n\, U^\top \qquad\Longrightarrow\qquad \boxed{\ \varepsilon_{TS} = -\,N_S\, x_T\ } ,
$$

and the same stacking of s_k\* = (1 − n_k)c_k gives

$$
x_S^{*} \;=\; U\,(I - D_n)\,U^\top x_T \;=\; (I - N_S)\, x_T :
$$

**the student settles on an attenuated copy of the teacher's state** — it copies the fraction
(1 − n_k) of each component (all of a learned one, almost none of a hopeless one) and leaves the
fraction n_k behind as error. The novelty operator was never assumed: it is the *name of the
reassembly*.

### Step 5 — the teacher's law, mode by mode: prioritization, born inside the lemma

The teacher's push is u = π_ST ε_TS (read off its equation; self-pull set aside — exactly zero on
the teacher's own manifold, near which the state stays — an operating-regime matter, |π_ST| kept
well below π_T, verified numerically; noise dropped — it
seeds, the push rescales). Its coordinate on axis k, using Step 3's error and π_ST = −|π_ST| in
sleep:

$$
\tau_T\, \dot c_k \;=\; \pi_{ST}\, e_k \;=\; (-|\pi_{ST}|)\cdot(-\,n_k\, c_k) \;=\; |\pi_{ST}|\; n_k\; c_k
$$

— the two minus signs cancel: **reversed precision × backward-pointing error = forward
amplification.** Each mode obeys the elementary growth law ċ = λc, so

$$
c_k(t) \;=\; c_k(0)\, \exp\!\Big(\frac{|\pi_{ST}|\, n_k}{\tau_T}\, t\Big) ,
$$

and the four readings follow, one look each: a **learned** axis (n_k = 0) is exactly frozen, for
all time; between two novel axes the more novel grows strictly faster; the ratio
c_k/c_l ∝ e^{(n_k − n_l)|π_ST|t/τ_T} means the most novel *occupied* axis eventually dominates the
teacher's heading (occupied: a coordinate that is exactly zero stays zero — the push rescales,
noise seeds); and by Step 3's linear bound the rate obeys

$$
|\pi_{ST}|\, n_k \ \le\ \frac{|\pi_{ST}|\,\pi_S}{\pi_{TS}}\,\|M_S u_k\|^2 ,
$$

so as Proposition 1 shrinks a residual, the drive on that axis dies **quadratically** — the
steering self-terminates, direction by direction. In vector form (stack as in Step 4):
τ_T ẋ_T|push = |π_ST| N_S x_T. This is the **prioritized** of the paper's title, derived from the
dynamics alone — no free energy was mentioned anywhere in this lemma.

### Step 6 — properties by construction, and the closed form

Each in one line, from the form N_S = U D_n Uᵀ:

- **Symmetric:** (U D_n Uᵀ)ᵀ = U D_nᵀ Uᵀ = U D_n Uᵀ by (R1) (a diagonal matrix is its own
  transpose). ✓
- **Eigenpairs:** N_S u_k = U D_n (Uᵀu_k) = U (n_k e_k) = n_k u_k — same axes as S_S, dials
  remapped through n(·), *by construction*. ✓
- **PSD, with ‖N_S‖ < 1:** the eigenvalues are the n_k ∈ [0, 1) (Step 3). ✓
- **Kills learned directions:** μ_k = 0 ⇒ n_k = 0 ⇒ N_S u_k = 0 ⇒ ε_TS = 0: a teacher pointing
  along a learned direction produces **no drive at all**. ✓
- **Closed form:** apply π_S S_S (π_TS I + π_S S_S)⁻¹ to one u_k, factor by factor, each acting as
  a number on an eigenvector — S_S as μ_k, the shifted matrix as π_TS + π_S μ_k, its inverse as
  division by that positive number:

$$
\pi_S\, S_S\,\big(\pi_{TS} I + \pi_S S_S\big)^{-1} u_k \;=\; \frac{\pi_S\,\mu_k}{\pi_{TS} + \pi_S\,\mu_k}\; u_k \;=\; n_k\, u_k .
$$

  Same action as N_S on every vector of a basis ⇒ the same matrix. This is the form the code
  computes (`novelty_operator`). ∎

*(Numerical check: with the student pre-trained on a subset, `model.novelty_operator()` annihilates
the learned patterns (‖N_S m‖ ~ 1e-10) and passes the unlearned ones (~0.35), and the identity
ε_TS = −N_S x_T holds to ~1e-15.)*

---

## Corollary 1 — the bridge: the landscape Theorem 1 climbs is the novelty-weighted quadratic

**Toolkit:** Lemma 1's modes (Steps 1–4); ONE new atom — *a rotation preserves norms*, i.e. a
squared norm is the sum of squared coordinates in an orthonormal frame (unavoidable here: this
corollary is about an **energy**, and energies are norms); the scalar identities of Lemma 1,
Step 3. No operator algebra.

**Why this result exists — the triangle.** The theory now contains two derivations that share *no
steps*: the **energy route** (Theorem 1: the push is gradient ascent on F_S^eq — sign identity +
chain rule, no N_S anywhere) and the **dynamics route** (Lemma 1: the push amplifies mode k at
rate |π_ST|n_k — scalar ODEs, no free energy anywhere). Two true stories about one flow. This
corollary is where they **meet**: it computes the landscape explicitly and shows the two stories
are the same fact. That a formalism passes a cross-check between two independent derivations is
the strongest internal evidence it offers — and it is exactly what `surprise_identity_error`
verifies at machine precision.

**Claim (fast student, as in Lemma 1).** The settled value of the student's free energy, as a
function of the teacher's state,

$$
F_S^{\mathrm{eq}}(x_T) \;=\; F_S\big(x_S^{*}(x_T),\, x_T\big) ,
$$

*(the same object Theorem 1 climbs: the equilibrium is the unique minimum of F_S over x_S — strict
convexity was checked in Theorem 1, Step 2 — so equivalently F_S^eq = min over x_S of F_S, the free
energy after inference has run)*, has the closed form

$$
F_S^{\mathrm{eq}}(x_T) \;=\; \frac{\pi_{TS}}{2}\, x_T^\top N_S\, x_T .
$$

In words: **the student's surprise about the teacher's state is the novelty-weighted energy of that
state** — a quadratic bowl, exactly flat along learned directions, curved along novel ones. The
novelty eigenvalues are the spectrum of the surprise; this is the bridge announced in the
Terminology note. No separate "teacher's surprise" is ever defined — the teacher's surprise is F_T,
doing its own job in the self-pull.

**Proof — dial by dial (a three-line continuation of Lemma 1).** Work in the modes. The one new
atom first: for any vector v with coordinates w = Uᵀv, expanding v = Σ_k w_k u_k and using
orthonormality (cross terms die through u_kᵀu_l = 0),

$$
\|v\|^2 = \Big(\sum_k w_k u_k\Big)^{\!\top}\Big(\sum_l w_l u_l\Big) = \sum_{k,l} w_k w_l\,\underbrace{u_k^\top u_l}_{\delta_{kl}} = \sum_k w_k^2 :
$$

a squared norm is the sum of squared coordinates. Now evaluate the two terms of F_S at the settled
state, whose per-mode data Lemma 1, Step 3 already computed:

*Interface term.* The error coordinates are −n_k c_k, so

$$
\frac{\pi_{TS}}{2}\,\|\varepsilon_{TS}\|^2 \;=\; \frac{\pi_{TS}}{2} \sum_k n_k^2\, c_k^2 .
$$

*Self term.* The settled coordinates are (1 − n_k)c_k, and the sandwich gives
‖M_S x_S\*‖² = (x_S\*)ᵀS_S x_S\* — a quadratic form with dials μ_k, hence per mode
μ_k(1 − n_k)²c_k². Absorb μ_k into n_k with the two scalar facts of Lemma 1, Step 3
(1 − n_k = π_TS/(π_TS + π_S μ_k) and π_S μ_k = n_k(π_TS + π_S μ_k)):

$$
\pi_S\, \mu_k\, (1 - n_k)^2 \;=\; n_k\,\big(\pi_{TS} + \pi_S \mu_k\big)\cdot\frac{\pi_{TS}^2}{\big(\pi_{TS} + \pi_S \mu_k\big)^2} \;=\; \pi_{TS}\; n_k\,(1 - n_k) ,
$$

so the self term is (π_TS/2) Σ_k n_k(1 − n_k)c_k².

*Add, dial by dial, and rotate back:*

$$
F_S^{\mathrm{eq}} \;=\; \frac{\pi_{TS}}{2} \sum_k \big[\, n_k^2 + n_k(1 - n_k) \,\big]\, c_k^2 \;=\; \frac{\pi_{TS}}{2} \sum_k n_k\, c_k^2 \;=\; \frac{\pi_{TS}}{2}\, x_T^\top N_S\, x_T . \qquad \blacksquare
$$

Per direction, in units of (π_TS/2)c_k²:

| term | contribution | reading |
|---|---|---|
| interface ½π_TS‖ε_TS‖² | n_k² | the student left a fraction n_k of the component unmatched; squared cost |
| self ½π_S‖ε_S‖² | n_k(1 − n_k) | the student's own prediction error on the (1 − n_k) it *did* copy |
| **total** | **n_k** | the plain novelty reading |

However the equilibrium splits the penalty between "failing to copy the teacher" and "failing to
predict myself," the two costs always recombine into exactly n_k: a learned direction costs
nothing, a novel one costs its novelty. The per-mode tug-of-war of Lemma 1 has a price, and the
price *is* the novelty.

**Remark — the triangle closes (consistency of the two routes).** Differentiate the closed form:
nudge x_T by δ and match the (G1) template (the two cross terms are equal — N_S symmetric, a scalar
equals its own transpose — and sum against the ½):

$$
\nabla_{x_T} F_S^{\mathrm{eq}} \;=\; \pi_{TS}\, N_S\, x_T .
$$

Now compare the three corners. **Theorem 1** (energy route) said the push is
(|π_ST|/π_TS)∇F_S^eq — computed *without knowing the landscape*, as −π_TS ε_TS at the settled
state. **Lemma 1** (dynamics route) said the push is |π_ST| N_S x_T — computed *without any energy*,
by reassembling the per-mode law of its Step 5. And this corollary's gradient makes them equal:

$$
\frac{|\pi_{ST}|}{\pi_{TS}}\,\big(\pi_{TS}\, N_S\, x_T\big) \;=\; |\pi_{ST}|\, N_S\, x_T . \checkmark
$$

Two derivations sharing no steps, one object. *(Numerically: `diagnostics.surprise_identity_error`
settles the student at random x_T, evaluates F_S directly, and compares with (π_TS/2)x_TᵀN_S x_T —
agreement to ~1.8e-15, asserted in `tests/smoke_test.py` before and after learning.)*

**Remark — why N_S and not S_S: surprise before vs. after inference.** One might expect the
student's free energy to be expressed through its self-error operator S_S rather than N_S. It is —
N_S acts in S_S's frame with remapped dials n(μ) — and the remapping is the footprint of the
*settling*. Compare:

- **Clamped student** (force x_S = x_T, no settling): ε_TS = 0 and

$$
F_S^{\text{clamped}} = \frac{\pi_S}{2}\, x_T^\top S_S\, x_T
$$

  — plain S_S, the **raw** surprise the student would feel if it had to represent the teacher's
  state exactly as given.

- **Settled student**: each mode retreats to the compromise (1 − n_k)c_k (Lemma 1, Step 3 — the
  per-mode division is where the reciprocal enters), and the leftover is Corollary 1's
  (π_TS/2) x_Tᵀ N_S x_T.

So the two operators are one quantity at two moments: **S_S is surprise before inference, N_S is
surprise after inference** — and F_S^eq, defined at the minimum, naturally wears the after-inference
operator. Per dial the discount is explicit: n(μ) grows like (π_S/π_TS)μ for nearly-learned
directions but saturates below 1 for wildly novel ones — the student can always cap its bill by
dropping a hopeless axis entirely (paying only the interface penalty). Settling only ever lowers
the bill, (π_TS/2)n(μ) ≤ (π_S/2)μ per direction — the same linear bound as Lemma 1, Step 3. This
is the predictive-coding slogan made an operator equation: surprise is not the raw mismatch, it is
what remains *after perception has done its best to explain the input away*.

---

## Remark — prioritization: the energetic reading (and scope)

*(The prioritization law itself was derived inside Lemma 1, Step 5, from the dynamics alone. This
remark records its second, energetic derivation and the scope notes; the paper's version is
final_proofs.md §7.)*

**The energetic reading.** By Theorem 1 the push is (|π_ST|/π_TS)∇F_S^eq, and by Corollary 1 the
landscape is the quadratic (π_TS/2)x_TᵀN_S x_T — so the push-only flow is gradient ascent on a
quadratic bowl, and ascent on a quadratic decouples along the eigenframe of its matrix: the same
per-mode law τ_T ċ_k = |π_ST| n_k c_k as Lemma 1, Step 5, now read as "each mode climbs its own
parabola, at a rate set by its curvature." Monotonicity needs no separate derivation — any gradient
ascent increases its own potential: dF_S^eq/dt = (|π_ST|/(π_TS τ_T))‖∇F_S^eq‖² ≥ 0, strictly
unless N_S x_T = 0 (fully learned state).

**Scope notes** (shared by both readings): the decoupling concerns the *push alone* — the full
drift adds the self-pull −π_T S_T x_T, which mixes the novelty eigenvectors unless S_T and N_S
commute; the amplitude leash rescales all components equally, so prioritization is a statement
about *ratios* and survives it (final_proofs §7 gives the projected-flow computation); and the
closed loop — once Proposition 1's learning is switched back on, every direction the teacher
visits gets learned, its n_k drops, and **the teacher climbs a landscape that its own climbing
flattens**. That the coupled dynamics actually completes the transfer is demonstrated numerically,
not proved.

---

## Proposition 2 — the exact free-energy saddle

**Toolkit:** Theorem 1; the Step-1 rule of Prop 1 (for ∇F_T). One line each.

**Claim.** The full sleep flow is generated by the single saddle potential Φ = F_T − F_S^eq exactly
when π_ST = −π_TS.

**Proof.** Write the full sleep flow of Theorem 1, Step 1 with *both* forces as gradients. The
self-pull already is one: the teacher's own surprise F_T = (π_T/2)‖M_T x_T‖² has, by the Step-1
rule of Prop 1 with A = M_T,

$$
\nabla_{x_T} F_T = \pi_T\, M_T^\top M_T\, x_T = \pi_T\, S_T\, x_T ,
$$

so the self-pull −π_T S_T x_T is −∇F_T — the teacher descending its *own* surprise. And by
Theorem 1 the push is (|π_ST|/π_TS)∇F_S^eq — the teacher ascending the *student's* surprise. The
whole flow is therefore

$$
\tau_T\, \dot x_T = -\,\nabla_{x_T} F_T \;+\; \frac{|\pi_{ST}|}{\pi_{TS}}\, \nabla_{x_T} F_S^{\mathrm{eq}} .
$$

This is the flow generated by the single saddle potential Φ = F_T − F_S^eq (i.e. τ_T ẋ_T = −∇Φ)
**exactly when the coefficient of the ascent equals 1: |π_ST| = π_TS, that is π_ST = −π_TS** — with
a transparent reading: *the strength with which the teacher climbs the student's surprise must
equal the precision with which that surprise was measured* (mobility matching). At that point the
teacher's push is *precisely the reversed gradient* of the student's surprise — the sign identity
of Theorem 1, Step 1 with coefficient 1. Meanwhile the student, descending F_S (Proposition 1),
ascends Φ: a minimax on one potential. ∎

*(The full proposition adds the sphere constraint, the mobilities, and the student's projected W_S
ascent; see curriculum L6. This section shows where the condition comes from.)*

---

## The chain, in one paragraph

The student descends F_S (Proposition 1). Because teacher and student share one interface energy
through opposite-signed precisions, the student's push on the teacher is the student's own gradient
sign-flipped — and, with the fast student settled, the envelope theorem makes that exact gradient
ascent on the landscape F_S^eq (Theorem 1: a sign identity plus a classical two-line tool, no
operator algebra). Independently, settling the student mode by mode in its own eigenframe — one
scalar tug-of-war per axis, no inverses anywhere — leaves the error −n_k c_k on each axis and hands
the teacher the per-mode law ċ_k ∝ n_k c_k: learned axes frozen, novel axes amplified in proportion
to their novelty, each push extinguishing quadratically as the student learns — the ascent is
*prioritized*, derived from the dynamics alone (Lemma 1, the novelty operator emerging as the name
of the reassembly). The two routes meet in Corollary 1 — the bridge: the landscape is the
novelty-weighted quadratic (π_TS/2)x_TᵀN_S x_T, flat where learned, curved where novel, and its
gradient makes the energy story and the dynamics story one fact. And the full flow is a minimax on
the single potential Φ = F_T − F_S^eq exactly when π_ST = −π_TS (Proposition 2). The teacher
staying near its manifold is an operating-regime matter — the reversed precision is kept well
below the teacher's self-precision (the code default sets |π_ST| to half the stability
threshold) and containment is verified numerically. No manifold restriction,
no compression, no "teacher's surprise" anywhere: the teacher descends its own F_T while being
steered up the student's F_S.

*(Numerical checks: the push π_ST ε_TS = (|π_ST|/π_TS)∇F_S^eq with F_S^eq verified to machine
precision by `diagnostics.surprise_identity_error` (~1.8e-15, asserted in `tests/smoke_test.py`);
`model.novelty_operator()` gives N_S with novelty eigenvalues n_k that are 0 on learned directions
and positive on unlearned ones, so the per-axis rates |π_ST|n_k are 0 on learned axes and positive
on novel ones.)*
