# Additional proofs — NOT in the paper

**Status (decision of 2026-07-20): the guard lemma below is cut from the paper.** Do not
reintroduce it into final_proofs.md, detailed_proofs.md, the outline, or the LaTeX drafts. The
paper instead states an *operating choice* plus a *measurement*: the reversed precision is kept
conservatively small — the code default (`pi_ST = "auto"`, `pi_ST_safety = 0.5` in
`src/config.py`) sets |π_ST| to half the stability threshold π_T σ_min² — and the teacher's
containment on its memory manifold during sleep is verified numerically (`manifold_leakage`,
`offmanifold_growth`, `terminal_occupancy`). Rationale for the cut: the paper is already
mathematically complete; the projector machinery and the mixed-state caveats below add ownership
burden without being needed for any other result.

**Why keep this file:** reviewer-rebuttal ammunition. If a reviewer asks "you flipped a precision
sign — why doesn't the teacher amplify noise off its manifold?", everything below is the
ready-made answer: a sufficient condition (the guard), a proof that the off-manifold subspace is
uniformly contracting under it, and a bounded-leakage corollary for mixed states.

Notation, model, and assumptions are those of [final_proofs.md](final_proofs.md); the toolkit
atoms cited (§ numbers, (R1), sandwich, spectral theorem, Parseval) are those of
[detailed_proofs.md](detailed_proofs.md) and [cheatsheet.md](cheatsheet.md).

---

## The guard lemma — paper-register version

With the student eliminated (Lemma 1), the teacher's deterministic sleep drift is

$$
\tau_T\, \dot x_T \;=\; A\, x_T , \qquad A \;=\; -\,\pi_T\, S_T \;+\; |\pi_{ST}|\, N_S .
$$

The concern is that the growth term |π_ST| N_S could amplify components off the teacher's memory
manifold (noise-chasing). Let σ_min² denote the smallest **nonzero** eigenvalue of S_T (the
spectral gap; recall ker S_T = ker M_T is the manifold).

**Lemma (the guard)** *(assumptions A1–A3).* If

$$
|\pi_{ST}| \;<\; \pi_T\, \sigma_{\min}^2 ,
$$

then the quadratic form of the reduced drift is strictly negative on every nonzero vector of the
off-manifold subspace: for q = Q_T q ≠ 0,

$$
q^\top A\, q \ \le\ -\,\big(\pi_T\, \sigma_{\min}^2 - |\pi_{ST}|\big)\, \|q\|^2 \ <\ 0 ,
$$

and consequently ‖q(t)‖² ≤ ‖q(0)‖² e^{−2γt/τ_T} with γ = π_T σ_min² − |π_ST|, for as long as
the trajectory stays off-manifold.

**Proof.**

*Step 0 — what the quadratic form measures.* Along the reduced flow,

$$
\frac{d}{dt}\,\|q\|^2 \;=\; 2\, q^\top \dot q \;=\; \frac{2}{\tau_T}\, q^\top A\, q ,
$$

so the sign of qᵀAq **is** the instantaneous growth of the state's length: every bound below is
a bound on a growth rate. By linearity, qᵀAq = −π_T (qᵀS_T q) + |π_ST| (qᵀN_S q) — a shrink
rate from the self-pull against a growth rate from the push, raced on the same vector.

*Step 1 — the hypothesis.* q = Q_T q is equivalent (apply P_T, use P_T Q_T = 0) to P_T q = 0,
i.e. u_pᵀq = 0 for every memory basis vector: q is orthogonal to the whole manifold — a state
with zero memory content.

*Step 2 — damping floor.* Expand q in the orthonormal eigenbasis of S_T. By Step 1 its
coefficients a_j on the kernel modes (the manifold, eigenvalue 0) all vanish, so only modes with
σ_j² ≥ σ_min² carry weight, and with Parseval (Σ a_j² = ‖q‖²):

$$
q^\top S_T\, q \;=\; \sum_{j:\,\sigma_j^2 > 0} \sigma_j^2\, a_j^2 \;\ge\; \sigma_{\min}^2\, \|q\|^2 :
$$

the pull shrinks any pure off-manifold state at rate at least π_T σ_min² per unit of ‖q‖² —
none of its length can sit in the force-free kernel.

*Step 3 — drive ceiling.* Expand the same q in the eigenbasis of N_S (the student's frame — in
general **misaligned** with S_T's, since nothing makes the two operators commute; this is why no
common frame, and no per-mode decoupling as in Lemma 1, is available). Every novelty eigenvalue
is strictly below 1 (Lemma 1: n(μ) = 1 − π_TS/(π_TS + π_S μ) < 1 — a fraction of a component
cannot exceed 100%), so with Parseval in this second frame (Σ b_k² = ‖q‖²):

$$
q^\top N_S\, q \;=\; \sum_k n_k\, b_k^2 \;<\; \|q\|^2 :
$$

the push grows any state at rate strictly below |π_ST| per unit of ‖q‖², however novel.

*Step 4 — combine.* The two bounds live in two different frames but are both stated against
‖q‖², which is frame-independent; they add:

$$
q^\top A\, q \ \le\ \big(-\,\pi_T\, \sigma_{\min}^2 + |\pi_{ST}|\big)\, \|q\|^2 .
$$

Worst defense (softest spring) against best attack (saturated novelty dial): if the floor beats
the ceiling, the misalignment of the frames is irrelevant. Under the guard the bracket is −γ < 0,
and Step 0 turns the bound into d‖q‖²/dt ≤ −(2γ/τ_T)‖q‖²; since the bound is uniform over the
off-manifold subspace it applies at every instant the trajectory spends there, and integrating
(Grönwall) gives the exponential decay. ∎

In words: **on the off-manifold subspace, damping beats drive**, exactly when the reversed
precision stays below the teacher's damping floor π_T σ_min².

**Scope.** The lemma bounds the quadratic form on vectors lying *wholly* in the off-manifold
subspace: such a state has negative instantaneous radial growth. It is **not**, by itself, a
contraction theorem for the off-manifold component of a general mixed state — the cross-block
Q_T N_S P_T can inject a normal component from a manifold component — and it is not manifold
invariance. Containment in the full model (cross-coupling, leash, noise) is a numerical result
(`manifold_leakage`). (Sharper version of the same argument, if ever needed:
|π_ST| ‖Q_T N_S Q_T‖ < π_T σ_min² suffices.)

---

## The cross-block: why mixed states leak, and why the leak is structured

Split a general state x = P_T x + Q_T x into memory part m and noise part q. The noise part
obeys

$$
\tau_T\, \dot q \;=\; \big(Q_T A Q_T\big)\, q \;+\; \big(Q_T A P_T\big)\, m ,
$$

and the cross-block reduces to the push alone (S_T annihilates the manifold, so Q_T S_T P_T = 0):

$$
Q_T\, A\, P_T \;=\; |\pi_{ST}|\, Q_T\, N_S\, P_T .
$$

Because the student's eigenframe is misaligned with the teacher's manifold split, N_S applied to
a memory component generically rotates it slightly off-manifold (filtering through tilted axes
rotates): Q_T N_S m ≠ 0 mid-transfer. Two structural facts about this injection:

- **It is deterministic and persistent, not noise:** the same vector Q_T N_S m is fed for as
  long as the state and the student's weights persist — it does not average out, and the
  student's integrating Hebbian rule can accumulate it (a small "ghost" direction learned into
  W_S). This is self-limiting by the same mechanism as the main transfer: learning the ghost
  drops its novelty, which kills the push that sustained the injection. The net cost is a small
  blur of the transferred subspace, not runaway.
- **It vanishes in the aligned cases:** a blank student (N_S ∝ I rotates nothing) and a
  finished/aligned student (student axes inside or perpendicular to the manifold) both give
  Q_T N_S P_T = 0, i.e. exact invariance. Leakage is a mid-transfer, misalignment phenomenon.

**Bounded-leakage corollary (reduced model; "leaky bucket").** From the guard's uniform decay
−γ‖q‖² on the diagonal block, Cauchy–Schwarz with ‖N_S‖ < 1 on the source, and the leash
‖m‖ ≤ r₀:

$$
\frac{d}{dt}\,\|q\| \;\le\; \frac{1}{\tau_T}\,\big(-\,\gamma\,\|q\| + |\pi_{ST}|\, r_0\big)
\qquad\Longrightarrow\qquad
\limsup_{t}\, \|q(t)\| \;\le\; \frac{|\pi_{ST}|\; r_0}{\gamma}
$$

(Grönwall). The bound survives learning: it uses only ‖N_S‖ < 1 (true for every W_S), and γ is
built from frozen teacher quantities. Not stated for the full model because the sphere
renormalization (projected flow adds a radial correction), the noise ξ, and discretization each
break the clean two-liner — which is exactly why containment stays a numerical claim in the
paper.

---

## Fully-detailed pedagogical proof (projector algebra spelled out)

*This is the tutorial-register version formerly in detailed_proofs.md, with the master identity
"the quadratic form IS the derivative of the squared length" made explicit at every use.*

### Step 0 — the reduced drift, derived

During sleep the teacher's deterministic equation (noise dropped, A3; leash set aside — Step 7
returns to it) is

$$
\tau_T\, \dot x_T \;=\; -\,\pi_T\, M_T^\top \varepsilon_T \;+\; \pi_{ST}\, \varepsilon_{TS} .
$$

Substitute the teacher's self-error definition into the first term:

$$
\varepsilon_T = M_T\, x_T \qquad\Longrightarrow\qquad -\,\pi_T\, M_T^\top M_T\, x_T \;=\; -\,\pi_T\, S_T\, x_T .
$$

For the second term, Lemma 1 has eliminated the fast student: at its settled state the interface
error is the novelty-weighted part of the teacher's state, ε_TS = −N_S x_T. Sleep means the
interface precision is reversed, π_ST = −|π_ST|, and the two minus signs cancel:

$$
\pi_{ST}\, \varepsilon_{TS} \;=\; \big(-\,|\pi_{ST}|\big)\big(-\,N_S\, x_T\big) \;=\; +\,|\pi_{ST}|\, N_S\, x_T .
$$

Collecting, the teacher obeys one linear flow under two competing forces:

$$
\tau_T\, \dot x_T \;=\; A\, x_T , \qquad A \;=\; \underbrace{-\,\pi_T\, S_T}_{\text{pull toward the memories}} \;+\; \underbrace{|\pi_{ST}|\, N_S}_{\text{push toward novelty}} .
$$

The push is the mechanism (it seeks out what the student has not learned — Lemma 1, Step 5), but
it is indiscriminate: noise is also unlearned. The lemma asks whether the push can grow
non-memory content, and answers no — under the guard.

### Step 1 — the two projectors, built from scratch

Let u_1, …, u_P be an orthonormal basis of the memory manifold (the kernel of M_T, spanned by
the stored patterns), stacked as the **columns** of U_T, so that

$$
U_T^\top U_T \;=\; I_P
$$

(entry (p, r) of the product is u_pᵀu_r: 1 on the diagonal — unit vectors — and 0 off it —
perpendicular vectors). Careful: the product in the *other* order, U_T U_Tᵀ, is d×d and is
**not** the identity — it is our projector. Define

$$
P_T \;=\; U_T\, U_T^\top , \qquad Q_T \;=\; I - P_T .
$$

Applied to a state x, unpack the matrix product column by column:

$$
P_T\, x \;=\; U_T\,\big(U_T^\top x\big) \;=\; \sum_{p=1}^{P} \big(u_p^\top x\big)\, u_p .
$$

Read right to left: the numbers u_pᵀx are the coordinates of x along each memory direction, and
P_T x rebuilds a vector out of *only* those. **P_T x is the memory content of x — the shadow x
casts on the manifold — and Q_T x = x − P_T x is what is left: the noise content.** Three
formulas, one line each:

*(i) Projecting twice changes nothing* — the orthonormality identity evaporates in the middle:

$$
P_T^2 \;=\; U_T\,\underbrace{U_T^\top U_T}_{=\,I_P}\,U_T^\top \;=\; P_T , \qquad Q_T^2 = (I - P_T)^2 = I - 2P_T + P_T^2 = Q_T .
$$

(The shadow of a shadow is itself.)

*(ii) The two parts are mutually invisible:*

$$
P_T\, Q_T \;=\; P_T - P_T^2 \;=\; 0 .
$$

(Extract the noise content, then ask for its memory content: exactly zero. Same in the other
order.)

*(iii) The split is perpendicular, so Pythagoras applies.* Trivially x = P_T x + Q_T x since
P_T + Q_T = I; and the two pieces are orthogonal — P_T is symmetric (a matrix times its own
transpose read backwards: (U_T U_Tᵀ)ᵀ = U_T U_Tᵀ), so with (ii):

$$
(P_T x)^\top (Q_T x) \;=\; x^\top P_T\, Q_T\, x \;=\; 0 \qquad\Longrightarrow\qquad \|x\|^2 = \|P_T x\|^2 + \|Q_T x\|^2 .
$$

**Every teacher state is uniquely memory-part plus noise-part, at right angles — like a vector's
horizontal and vertical components.**

### Step 2 — the hypothesis decoded: q = Q_T q ≠ 0 means 100% noise, 0% memory

The lemma considers a state q with q = Q_T q. Apply P_T to both sides and use formula (ii):

$$
P_T\, q \;=\; P_T\, Q_T\, q \;=\; 0 .
$$

So the memory content of q is exactly zero. Unpack what that says through Step 1's column
expansion: a linear combination of the independent vectors u_p vanishes only if every coefficient
vanishes individually, so

$$
u_p^\top q \;=\; 0 \;\;\text{for every } p \qquad\Longrightarrow\qquad m^\top q = 0 \;\;\text{for every memory } m
$$

(every memory is a combination of the u_p). **q is perpendicular to every stored pattern and to
every mixture of them — it contains no trace of any memory. That is the precise meaning of
"wholly off-manifold."** The converse holds too: if P_T q = 0 then q = (P_T + Q_T)q = Q_T q. So
the three statements are one and the same:

$$
q = Q_T\, q \quad\Longleftrightarrow\quad P_T\, q = 0 \quad\Longleftrightarrow\quad q \perp \text{every memory} .
$$

Finally q ≠ 0 rules out the trivial vector. **This is the worst-case patient — a state made of
pure hallucination — and the lemma asks whether the sleep dynamics grows it or kills it.**

### Step 3 — the master identity: the sandwich IS the derivative of the length

Track the squared length coordinate-wise and differentiate with the chain rule:

$$
\|q\|^2 \;=\; \sum_i q_i^2 \qquad\Longrightarrow\qquad \frac{d}{dt}\,\|q\|^2 \;=\; \sum_i 2\, q_i\, \dot q_i \;=\; 2\, q^\top \dot q .
$$

Substitute the flow of Step 0:

$$
\dot q \;=\; \frac{1}{\tau_T}\, A\, q \qquad\Longrightarrow\qquad \boxed{\ \frac{d}{dt}\,\|q\|^2 \;=\; \frac{2}{\tau_T}\; q^\top A\, q\ } .
$$

An exact identity, no approximation: **the sandwich qᵀAq is the time-derivative of the squared
length, up to the positive constant 2/τ_T. Positive sandwich = the noise vector is lengthening
at this instant; negative = it is shrinking.** (Geometric reading: qᵀ(Aq) is the component of
the velocity Aq along the state itself — the radial part, the only part that changes length; a
velocity component perpendicular to q rotates it without stretching, and indeed contributes
qᵀq̇ = 0.) By linearity the total rate splits into the two forces' contributions:

$$
\frac{d}{dt}\,\|q\|^2 \;=\; \frac{2}{\tau_T}\,\Big[ -\,\pi_T\,\big(q^\top S_T\, q\big) \;+\; |\pi_{ST}|\,\big(q^\top N_S\, q\big) \Big] :
$$

the first bracketed term is the **shrink rate supplied by the pull**, the second the **growth
rate supplied by the push**. The lemma is a race between them.

### Step 4 — the pull's shrink rate has a floor

First, the pull's contribution is genuinely a shrink, never a growth: for any vector v the
sandwich gives

$$
v^\top S_T\, v \;=\; v^\top M_T^\top M_T\, v \;=\; \|M_T\, v\|^2 \;\ge\; 0 .
$$

**Through the master identity this says: the term −π_T qᵀS_T q always contributes a negative (or
zero) amount to the length derivative — the teacher's own dynamics never inflates any state.**
The contribution is zero exactly on the manifold (M_T v = 0): memories feel no restoring force —
the bowl's bottom is flat.

S_T is symmetric, so the spectral theorem applies: an orthonormal eigenbasis in which manifold
directions carry eigenvalue 0 and off-manifold directions carry positive eigenvalues σ_j², whose
minimum is the **spectral gap** σ_min² — *the softest spring off the manifold*. Expand q in this
basis with coefficients a_j. **Step 2 now does its job:** q is perpendicular to every manifold
vector, so its coefficients on all zero-eigenvalue modes vanish — every unit of q's length sits
on a spring. The quadratic form is a sum over positive eigenvalues only (cross terms die by
orthonormality):

$$
q^\top S_T\, q \;=\; \sum_{j:\,\sigma_j^2 > 0} \sigma_j^2\, a_j^2 \;\ge\; \sigma_{\min}^2 \sum_j a_j^2 \;=\; \sigma_{\min}^2\, \|q\|^2 ,
$$

the last equality being Parseval. **Translated through the master identity: the pull's
contribution to the length derivative is a shrink of at least π_T σ_min² per unit of squared
length. The defense has a floor — and a pure-noise state cannot hide any of its length in the
force-free flat bottom.** (Had q contained any memory component, part of its weight would sit on
zero-eigenvalue modes and this bound would fail; this is exactly where the hypothesis of Step 2
is used — and the only place.)

### Step 5 — the push's growth rate has a ceiling

The novelty dials of N_S satisfy, rewriting Lemma 1's formula by adding and subtracting π_TS in
the numerator:

$$
n_k \;=\; \frac{\pi_S\, \mu_k}{\pi_{TS} + \pi_S\, \mu_k} \;=\; 1 \;-\; \frac{\pi_{TS}}{\pi_{TS} + \pi_S\, \mu_k} \;<\; 1 ,
$$

since the subtracted fraction is strictly positive (π_TS > 0, denominator positive). *n_k is the
fraction of a component the student fails to copy, and no failure exceeds 100% — even total
ignorance (μ_k → ∞) saturates the dial strictly below 1.*

Expand the **same** q in N_S's own orthonormal eigenbasis — the student's frame U of Lemma 1,
generally **misaligned** with the teacher's frame of Step 4. That is allowed: we are merely
re-expressing the same vector in a second orthonormal basis. With coefficients b_k = u_kᵀq:

$$
q^\top N_S\, q \;=\; \sum_k n_k\, b_k^2 \;<\; \sum_k b_k^2 \;=\; \|q\|^2 ,
$$

Parseval again, now in the student's frame. **Translated through the master identity: the push's
contribution to the length derivative is a growth of strictly less than |π_ST| per unit of
squared length — no matter how novel the direction. The attack has a ceiling.**

### Step 6 — floor versus ceiling: the guard, and exponential death

Why the proof could not be per-mode, and why it does not need to be: Lemma 1 decoupled the
dynamics because a *single* symmetric operator was in play. Here two symmetric operators live in
two different eigenbases, and they share a frame only if they commute — which nothing guarantees
(the student's learned directions need not align with the teacher's memories, especially
mid-transfer). But both Step 4 and Step 5 stated their bounds against ‖q‖², and **a vector's
length is the same number in every orthonormal frame** — the frame-independent currency both
rates were converted into. So the bounds add inside the master identity:

$$
\frac{d}{dt}\,\|q\|^2 \;=\; \frac{2}{\tau_T}\, q^\top A\, q \;\le\; \frac{2}{\tau_T}\,\big(-\,\pi_T\, \sigma_{\min}^2 + |\pi_{ST}|\big)\, \|q\|^2 .
$$

This is worst-defense versus best-attack: the softest spring against a fully saturated novelty
dial. If the floor beats the ceiling, alignment between the two frames is irrelevant — the sign
is settled for every direction of the danger zone at once. The bracket is negative exactly under
the **guard**

$$
|\pi_{ST}| \;<\; \pi_T\, \sigma_{\min}^2 ,
$$

and then, writing γ for the positive margin,

$$
\gamma \;=\; \pi_T\, \sigma_{\min}^2 - |\pi_{ST}| \qquad\Longrightarrow\qquad \frac{d}{dt}\,\|q\|^2 \;\le\; -\,\frac{2\gamma}{\tau_T}\,\|q\|^2 \;<\; 0 .
$$

**By the master identity this is the conclusion itself: every pure-noise state has strictly
negative length derivative.** Because the bound is uniform — the same γ for *every* nonzero
off-manifold vector — it holds at every instant along the trajectory while the state remains
off-manifold. Divide by ‖q‖² and recognize the derivative of a logarithm (Grönwall):

$$
\frac{d}{dt}\,\log\|q\|^2 \;\le\; -\,\frac{2\gamma}{\tau_T} \qquad\Longrightarrow\qquad \|q(t)\|^2 \;\le\; \|q(0)\|^2\; e^{-2\gamma t/\tau_T} .
$$

Exponential death of pure noise, decay rate growing with the distance below the guard. ∎

### Step 7 — scope: what the sandwich cannot see

The hypothesis q = Q_T q was used exactly once — in Step 4, to empty the zero-eigenvalue modes —
so the certified statement covers states of pure noise. For a **mixed** state the noise can be
**replenished** from the memory part through the cross-block Q_T A P_T = |π_ST| Q_T N_S P_T (see
the cross-block section above): the residual leakage — together with the leash and the noise ξ,
both set aside in Step 0 — is measured numerically, not proved zero.

**In one line:** the derivative of the noise's length equals the sandwich qᵀAq (Step 3); the
pull makes that sandwich negative at rate at least π_T σ_min² — softest spring, and pure noise
cannot hide in the flat bottom (Step 4); the push makes it positive at rate strictly below
|π_ST| — novelty is a fraction, capped under 100% (Step 5); under the guard the floor beats the
ceiling in every direction at once, so pure noise dies exponentially (Step 6).
