# Prioritized memory transfer — model and analysis (final)

> **⚠ STATUS 2026-07-27 — THIS DOCUMENT IS NOW OUT OF DATE WITH THE PAPER'S FRAMING.**
> The mathematics below is unchanged and still correct; the **naming and ordering** are not.
> A rename/renumber pass is pending. Until it is done, read this document with these substitutions:
>
> 1. **`π_ST` → `κ`** ("adversarial coupling", signed, plain, no subscript; derivations use `|κ|`).
>    It is **never** called a precision: a precision weights an error term in a free energy the agent
>    *minimizes*; κ weights one the teacher *maximizes*, so it is not an inverse variance.
> 2. **The system is "adversarially coupled predictive coding"**, not a predictive-coding network.
>    Components descend free energies; the coupled system does not. No global VFE claim.
> 3. **Order is saddle-first.** The paper's order is:
>    Prop 1 (setup) → **Theorem 1 = the min-max** (this document's §8 Proposition 2, promoted to
>    first) → **Corollary 1** = the push is ascent on the student's surprise (this document's §4
>    Theorem 1, now downstream) → **Lemma 1** (§5, unchanged) → **Corollary 2** = the surprise
>    identity (this document's §6 Corollary 1) → prioritization remark (§7).
>    **Proposition 2 no longer exists as a number.**
>    *Justification (checked 2026-07-27):* §8 needs only `∇F_T` + the **sign identity** of §4 Step 1
>    (valid at any `x_S`) and explicitly does NOT use A1a/A1b. The adiabatic elimination, the
>    uniqueness argument and the envelope step enter only at **Corollary 1** (§4 Steps 0 and 2). So
>    the min-max is the result with the fewest assumptions, and saddle-first is the logically minimal
>    ordering, not a rhetorical one.
> 4. **Operating point is now `κ = −π_TS`** (the exact zero-sum point) as the default, for all
>    simulations. Containment is therefore NOT obtained by shrinking κ any more: `π_T` is raised
>    above `|κ|` by a hand-chosen, numerically verified factor. The note at the end of the Results
>    list below (about keeping |π_ST| at half the stability threshold) is superseded.
> 5. **`Φ_w = F_T − w·F_S^eq` (`w = |κ|/π_TS`): CHECKED 2026-07-27, and it is NOT a minimax.**
>    True for the **teacher block alone under adiabatic elimination** (= §8 Remark 3 with the
>    coefficient left general). It is **false** for the joint system: §8's (⇒) direction proves that
>    for `κ ≠ −π_TS` *no* C² potential generates the descent–ascent flow with these fixed block
>    mobilities, and the circulation `|κ+π_TS|√d` measures precisely that non-integrability.
>    Never write `Φ_w` as a "weighted minimax"; it would contradict §8.
>
> Rationale and full detail: [../outline.md](../outline.md) "Math placement" and
> [../writing_tracker.md](../writing_tracker.md) decisions D6–D9.

This document is self-contained: it defines the model, states every assumption, and proves every
formal result intended for the paper, in logical order. Results are stated at the level of detail
meant for checking, not for teaching. Empirical claims (completion of the full learning loop,
merging, continual learning, the measured size of the cross-block leakage) are outside its scope
and are listed at the end.

**Results.**

- Proposition 1 — student inference and learning are (projected) gradient descent on F_S.
- Theorem 1 — the student's push on the teacher is exact gradient ascent on the student's settled
  free energy F_S^eq.
- Lemma 1 — adiabatic elimination of the student produces the novelty operator N_S.
- Corollary 1 — the landscape: F_S^eq(x_T) = (π_TS/2) x_Tᵀ N_S x_T.
- Remark (prioritization) — the ascent decouples per eigendirection: learned frozen, novel
  amplified in proportion to novelty; self-limiting.
- Proposition 2 — the sleep dynamics is a minimax flow on the single potential Φ = F_T − F_S if
  and only if π_ST = −π_TS.
- (No containment theorem. Manifold containment is an operating choice plus a measurement: the
  reversed precision is kept conservatively small — the code default sets |π_ST| to half the
  stability threshold — and the teacher's stay on its memory manifold during sleep is verified
  numerically. An archived sufficient-condition lemma lives in
  additional_proofs_not_in_paper.md; it is deliberately not part of the paper.)

---

## 1. The model

Two populations of d neurons each: a **teacher** T (weights frozen, higher in the hierarchy) and a
**student** S (weights plastic, initialized empty). States x_T, x_S ∈ ℝ^d. Recurrent weights
W_T, W_S ∈ ℝ^{d×d}, both with zero diagonal (no autapses); W_T is frozen and stores the memories,
W_S is plastic. Define the mismatch operators

$$
M_T = I - W_T, \qquad M_S = I - W_S ,
$$

and the errors (three separate error populations)

$$
\varepsilon_T = M_T\, x_T, \qquad \varepsilon_S = M_S\, x_S, \qquad \varepsilon_{TS} = x_S - x_T .
$$

ε_T is the teacher's self-prediction error, ε_S the student's, and ε_TS the interface mismatch:
the teacher, being higher, supplies the top-down prediction x_T of the student's state, and the
student returns x_S; both interface synapses are the identity.

**Precisions.** π_T, π_S > 0 (self-precisions), π_TS > 0 (student's interface precision), and
π_ST **signed** (teacher's interface precision). The sign of π_ST is the phase:

- **wake** π_ST > 0 — ordinary precision; both populations minimize the interface error (recall);
- **sleep** π_ST < 0 — reversed precision; the teacher is pushed *away* from the student's
  prediction (replay). All results below concern sleep.

**Free energies.**

$$
F_S(x_S, x_T;\, W_S) = \frac{\pi_{TS}}{2}\,\|x_S - x_T\|^2 + \frac{\pi_S}{2}\,\|M_S\, x_S\|^2 ,
\qquad
F_T(x_T) = \frac{\pi_T}{2}\,\|M_T\, x_T\|^2 .
$$

**Status of these definitions — what generates what.** The model is the three dynamical
equations below; the free energies are *descriptors*, and deliberately partial ones. Three of
the four forces in the dynamics are exact gradients of the two bounded quadratics above: the
student's state and weight forces descend F_S (Proposition 1), and the teacher's self-pull
descends F_T. The fourth — the teacher's interface coupling π_ST ε_TS — is a free-energy force
only in wake: for π_ST > 0 it completes an ordinary bounded VFE for the teacher,
F_T + (π_ST/2)‖ε_TS‖², and the whole network is then standard predictive coding, every
population descending its own free energy. The sleep flip π_ST < 0 makes that completed
quantity unbounded below (an inverted parabola along the interface), so in sleep no legitimate
free energy owns the term: **the reversed-precision push is postulated, not derived from VFE**
— it is the model's one deliberate departure from the framework. F_S and F_T are precisely the
energies that remain well-defined in both phases; that is the (explicit) bookkeeping choice
behind their definitions. The results then locate the outlier exactly: Theorem 1 shows the push
is gradient *ascent* on the student's settled F_S — an anti-VFE force directed at the other
population's energy, a curiosity mechanism rather than an inference one — and Proposition 2
shows that at π_ST = −π_TS, and only there, it is recovered inside the single potential
Φ = F_T − F_S as the negative half of a zero-sum saddle. The resemblance of that structure to
active inference is a resonance, not a derivation, and is treated as speculative throughout.

**Dynamics.** Three coupled processes:

$$
\tau_T\, \dot x_T = -\,\pi_T\, M_T^\top \varepsilon_T \;+\; \pi_{ST}\, \varepsilon_{TS} \;+\; \xi ,
\qquad \text{then renormalize } \|x_T\| = r_0 ,
$$

$$
\tau_S\, \dot x_S = -\,\pi_{TS}\, \varepsilon_{TS} \;-\; \pi_S\, M_S^\top \varepsilon_S ,
$$

$$
\dot W_S = \eta\, \pi_S\, \varepsilon_S\, x_S^\top , \qquad \text{diagonal re-zeroed after each step} .
$$

The displayed noisy equation is the **discrete simulation rule**: at each step ξ is drawn i.i.d.
from 𝒩(0, σ_ξ² I), σ_ξ small. (A formal continuous-time treatment would require an SDE; none is
needed here, since every proof concerns the deterministic limit — assumption A3.) r₀ is a fixed
norm (the amplitude leash on the teacher only).

**Operators derived from the weights.** The self-error operators

$$
S_T = M_T^\top M_T, \qquad S_S = M_S^\top M_S
$$

are symmetric and positive semidefinite (for any v, vᵀ S v = ‖M v‖² ≥ 0), and

$$
\ker S_T = \ker M_T, \qquad \ker S_S = \ker M_S
$$

(S x = 0 implies xᵀS x = ‖M x‖² = 0, hence M x = 0; the converse is immediate).

**Memories.** The stored patterns m_1, …, m_P are linearly independent and satisfy M_T m_p = 0.
This alone gives only the inclusion span{m_p} ⊆ ker M_T; assumption (A5) below asserts equality,
and the teacher's **memory manifold** is defined as ker M_T. The zero diagonal caps the capacity:
if the patterns spanned all of ℝ^d, then W_T x = x for every x, i.e. W_T = I, contradicting
diag W_T = 0 — so necessarily P ≤ d − 1. Let U_T be an orthonormal basis of ker M_T, and

$$
P_T = U_T U_T^\top, \qquad Q_T = I - P_T
$$

the orthogonal projectors onto the manifold and its complement. A direction u is **learned** by
the student when M_S u = 0. The zero diagonal also constrains what is exactly learnable: a single
unit direction u admits a zero-diagonal W with W u = u if and only if u is not supported on a
single coordinate (for u = e_i, row i would require Σ_{j≠i} W_ij · 0 = 1, impossible);
simultaneous exact representability of a whole collection or subspace requires further
compatibility conditions. Every "exactly learned" below carries this caveat.

---

## 2. Assumptions

- **(A1) Timescale separation:** τ_S ≪ τ_T ≪ 1/η. It is used in two distinct ways.
  **(A1a) Fast student state:** on the timescale of the teacher's motion, x_S sits at its
  equilibrium for the current x_T (adiabatic elimination: set ẋ_S = 0 and solve; the equilibrium
  is unique because its defining linear system has a positive-definite matrix — Theorem 1,
  Step 0).
  **(A1b) Frozen student weights:** on the same timescale W_S is constant, so all operators built
  from it (S_S, and later N_S) are fixed.
- **(A2) Sleep:** π_ST < 0 (and π_T, π_S, π_TS > 0). We write |π_ST| = −π_ST.
- **(A3) Deterministic analysis:** the noise ξ is set to zero in every proof. Its role in the
  model is to seed components of x_T that are exactly zero (see the caveats of Theorem 1).
- **(A4) The renormalization** ‖x_T‖ = r₀ is set aside in the main statements; where it matters
  it is treated explicitly, idealized as the tangential projection

$$
P_\perp = I - \frac{x_T x_T^\top}{\|x_T\|^2}
$$

  applied to the teacher's drift (continuous-time limit of "step, then renormalize").
- **(A5) Exact and exhaustive storage:** ker M_T = span{m_1, …, m_P} — the stored patterns are
  the *only* zero-error directions of the teacher. (M_T m_p = 0 alone gives just the inclusion;
  A5 excludes additional zero-error directions. With the zero diagonal this entails P ≤ d − 1,
  Section 1.)

**Operating regime (not hypotheses of any proof below):** π_TS > π_S, and |π_ST| kept well
below the teacher's self-precision so that the self-pull dominates off the manifold (the code
default sets |π_ST| to half the stability threshold; that the teacher indeed stays on its
memory manifold throughout sleep is verified numerically — Section 9). On the first: for the
initially blank student, W_S = 0 gives S_S = I, so π_TS > π_S is exactly the mode-wise
input-dominance condition at initialization; later in learning the mode-wise comparison is
π_TS > π_S μ_k, and no uniform claim is made without controlling the spectrum of S_S. Its
reading — the student trusts its input over its prior, preventing confabulation — is an
interpretation, not a proved statement. The proofs of Propositions 1–2, Theorem 1, Lemma 1 and
Corollary 1 hold with or without these.

---

## 3. Proposition 1 — student inference and learning descend F_S

**Proposition 1.** The student's state dynamics is exact gradient descent on F_S,

$$
\tau_S\, \dot x_S = -\,\nabla_{x_S} F_S ,
$$

and the student's weight update is projected gradient descent on F_S,

$$
\dot W_S = -\,\eta\, P_0\big(\nabla_{W_S} F_S\big) ,
$$

where P_0 zeroes the diagonal. Consequently, **for fixed teacher state x_T**, F_S is
non-increasing under either process and under both combined.

**Proof.** *State gradient.* For a fixed matrix A, expanding ½‖A(x+δ)‖² = ½‖Ax‖² + xᵀAᵀA δ +
½‖Aδ‖² identifies

$$
\nabla_x\, \tfrac12 \|A x\|^2 = A^\top A\, x .
$$

Applying this to the two terms of F_S (with x_T held fixed): the interface term gives
π_TS (x_S − x_T) = π_TS ε_TS (case A = I applied to x_S − x_T), the self term gives
π_S M_Sᵀ M_S x_S = π_S M_Sᵀ ε_S. Hence

$$
\nabla_{x_S} F_S = \pi_{TS}\, \varepsilon_{TS} + \pi_S\, M_S^\top \varepsilon_S ,
$$

and the model's state equation is exactly −(this).

*Weight gradient.* Only the self term depends on W_S. Perturbing W_S → W_S + Δ sends
ε_S → ε_S − Δ x_S, so

$$
\frac{\pi_S}{2}\|\varepsilon_S - \Delta x_S\|^2 = \frac{\pi_S}{2}\|\varepsilon_S\|^2 - \pi_S\, \varepsilon_S^\top \Delta\, x_S + O(\|\Delta\|^2) .
$$

Since aᵀΔ b = ⟨a bᵀ, Δ⟩ in the Frobenius inner product ⟨X, Y⟩ = tr(XᵀY), the linear term
identifies

$$
\nabla_{W_S} F_S = -\,\pi_S\, \varepsilon_S\, x_S^\top .
$$

The model's update η π_S ε_S x_Sᵀ with the diagonal re-zeroed is therefore −η P_0(∇_{W_S} F_S),
where P_0(X) = X − diag(X) is the orthogonal projection onto the subspace of zero-diagonal
matrices (diagonal and off-diagonal matrices are Frobenius-orthogonal).

*Descent.* With x_T held fixed, along both student processes simultaneously (using P_0 = P_0²
and self-adjointness of P_0 for the weight term),

$$
\left.\frac{dF_S}{dt}\right|_{x_T\ \mathrm{fixed}} = -\,\frac{1}{\tau_S}\,\big\|\nabla_{x_S} F_S\big\|^2 \;-\; \eta\, \big\|P_0\big(\nabla_{W_S} F_S\big)\big\|_F^2 \ \le\ 0 . \qquad \blacksquare
$$

*This is a monotonicity statement at fixed x_T only.* Along the full coupled dynamics, F_S gains
the additional term ∇_{x_T}F_Sᵀ ẋ_T — which in sleep is positive along the push: that is
precisely Theorem 1. F_S is a Lyapunov function for the student's own processes, not for the
coupled teacher–student flow.

*(Equivalently: the diagonal entries are not parameters of the model — there are no autapses —
so learning is plain gradient descent on the off-diagonal entries; P_0 is how that update looks
written as a full matrix.)*

---

## 4. Theorem 1 — the student's push is gradient ascent on the student's surprise

The teacher's deterministic sleep drift splits into two forces:

$$
\tau_T\, \dot x_T = \underbrace{-\,\pi_T\, M_T^\top \varepsilon_T}_{\text{self-pull}} \;+\; \underbrace{\pi_{ST}\, \varepsilon_{TS}}_{\text{push } u} .
$$

The self-pull is the teacher holding itself near its own manifold (that it wins off-manifold is
an operating-regime matter — |π_ST| kept well below π_T — verified numerically, Section 9). The
theorem concerns the push u = π_ST ε_TS.

**Theorem 1** *(assumptions A1, A2; unprojected flow — the leash is treated in Remark 1).*
Let x_S\*(x_T) be the student's settled state (unique — Step 0), and define the student's
settled free energy

$$
F_S^{\mathrm{eq}}(x_T) \;=\; F_S\big(x_S^{*}(x_T),\, x_T\big) .
$$

Then the push is exact gradient ascent on F_S^eq:

$$
\tau_T\, \dot x_T \big|_{\text{push}} \;=\; u \;=\; \frac{|\pi_{ST}|}{\pi_{TS}}\, \nabla_{x_T} F_S^{\mathrm{eq}}(x_T)
$$

at every point x_T. The identity is exact **within the adiabatically reduced model**; the
reduction (A1a) is itself the idealization.

**Proof.**

*Step 0 — the settled state is unique, so F_S^eq is well defined.* The settled state is the
equilibrium of the student's dynamics, ∇_{x_S}F_S = 0 (Proposition 1), i.e. the linear system

$$
\big(\pi_{TS}\, I + \pi_S\, S_S\big)\, x_S \;=\; \pi_{TS}\, x_T .
$$

Its matrix is positive definite — vᵀ(π_TS I + π_S S_S) v = π_TS ‖v‖² + π_S ‖M_S v‖² ≥
π_TS ‖v‖² > 0 for v ≠ 0 — hence invertible: the settled state x_S\*(x_T) exists, is unique, and
is linear (so smooth) in x_T. By (A1a) the student sits there.

*Step 1 — the sign identity (valid at any x_S, settled or not).* Only the interface term of F_S
contains x_T, so

$$
\nabla_{x_T} F_S \;=\; \pi_{TS}\,(x_T - x_S) \;=\; -\,\pi_{TS}\, \varepsilon_{TS} .
$$

Since π_ST < 0,

$$
u \;=\; \pi_{ST}\, \varepsilon_{TS} \;=\; \frac{-\pi_{ST}}{\pi_{TS}}\,\big(-\pi_{TS}\, \varepsilon_{TS}\big) \;=\; \frac{|\pi_{ST}|}{\pi_{TS}}\, \nabla_{x_T} F_S \Big|_{x_S} .
$$

Teacher and student feel the same interface energy ½π_TS‖x_S − x_T‖² through opposite-signed
precisions, so the teacher's drive is the student's gradient in the x_T-slot, rescaled by a
positive constant. (For π_ST > 0 — wake — the same line gives *descent*: both populations then
jointly minimize, and there is no transfer drive.)

*Step 2 — chain rule; the cross term dies by definition.* F_S^eq(x_T) = F_S(x_S\*(x_T), x_T)
depends on x_T twice — directly, and through the settled state it drags along. Chain rule on
both dependencies:

$$
\nabla_{x_T} F_S^{\mathrm{eq}} \;=\; \Big(\frac{d x_S^{*}}{d x_T}\Big)^{\!\top} \underbrace{\nabla_{x_S} F_S\big|_{x_S^{*}}}_{=\,0} \;+\; \nabla_{x_T} F_S\big|_{x_S^{*}} \;=\; \nabla_{x_T} F_S\big|_{x_S^{*}} .
$$

The first term vanishes **because ∇_{x_S}F_S = 0 is the very equation that defines the settled
state** (Step 0): however the settled state moves as the teacher moves, its motion is multiplied
by zero. Evaluating Step 1 at x_S = x_S\*(x_T) and substituting:

$$
u \;=\; \frac{|\pi_{ST}|}{\pi_{TS}}\, \nabla_{x_T} F_S^{\mathrm{eq}}(x_T) . \qquad \blacksquare
$$

**Remarks.**

1. *With the leash (A4).* Under the projected push ẋ_T = P⊥ u / τ_T, since P⊥ is a symmetric
   idempotent projection,

$$
\frac{d F_S^{\mathrm{eq}}}{dt} = \nabla F_S^{\mathrm{eq}\,\top}\, \dot x_T = \frac{|\pi_{ST}|}{\pi_{TS}\,\tau_T}\, \big\|P_\perp \nabla F_S^{\mathrm{eq}}\big\|^2 \ \ge\ 0 :
$$

   the ascent survives the renormalization as monotone ascent of F_S^eq restricted to the
   sphere. This concerns the push component alone: the self-pull's contribution to dF_S^eq/dt is
   proportional to −x_Tᵀ N_S S_T x_T and has no fixed sign unless S_T and N_S commute.
2. *Locality.* The push is the local gradient — it climbs from the current state; it is not a
   pointer at the global maximum of F_S^eq.
3. *Seeding.* The push rescales components of x_T (see the prioritization remark); it cannot
   create a component that is exactly zero. In the full model the noise ξ supplies the seed.
4. *Without (A1a)* the identity of Step 1 still holds at every instant: the push performs
   instantaneous partial-gradient ascent on F_S in the x_T-slot, whatever the student's state.
5. *The variational reading (not used by the proof).* The quadratic form of Step 0 also shows
   F_S is strictly convex in x_S, so the settled state is its unique minimum:
   F_S^eq(x_T) = min over x_S of F_S(x_S, x_T) — the free energy after inference has run, which
   is what the paper's variational language refers to. (Step 2 at a stationary point is the
   classical envelope argument; only the chain rule was used.)

---

## 5. Lemma 1 — eliminating the student produces the novelty operator

**Lemma 1** *(assumption A1a; only the student's parameters enter — the sign of π_ST plays no
role).* The student's settled state at teacher state x_T, and the resulting interface error, are

$$
x_S^{*} = (I - N_S)\, x_T , \qquad \varepsilon_{TS} = -\,N_S\, x_T ,
$$

with the **novelty operator** N_S defined spectrally: in an orthonormal eigenbasis U of S_S
(S_S = U diag(μ_1, …, μ_d) Uᵀ, μ_k ≥ 0),

$$
N_S \;=\; U\, \mathrm{diag}\big(n(\mu_k)\big)\, U^\top , \qquad n(\mu) = \frac{\pi_S\, \mu}{\pi_{TS} + \pi_S\, \mu} \ \in\ [0, 1) ,
$$

with n(0) = 0 and n strictly increasing; equivalently N_S = π_S S_S (π_TS I + π_S S_S)⁻¹, the
closed form computed in the code. In particular N_S is symmetric positive semidefinite with
‖N_S‖ < 1, and ker N_S = ker S_S = ker M_S: N_S vanishes exactly on the learned directions.

**Proof.**

*Step 1 — separate into modes (dynamics, not energy).* S_S = M_SᵀM_S is symmetric positive
semidefinite (vᵀ S_S v = ‖M_S v‖² ≥ 0), so it has an orthonormal eigenbasis U as above. Pass to
the coordinates

$$
c = U^\top x_T , \qquad s = U^\top x_S .
$$

Rotate the student's state equation itself — τ_S ẋ_S = −π_TS(x_S − x_T) − π_S S_S x_S — into the
frame (U is constant under A1b, and Uᵀ S_S = D_μ Uᵀ): it falls apart into **d uncoupled scalar
equations**,

$$
\tau_S\, \dot s_k \;=\; -\,\pi_{TS}\,\big(s_k - c_k\big) \;-\; \pi_S\, \mu_k\, s_k , \qquad k = 1, \dots, d :
$$

per mode, an interface spring pulling s_k toward the teacher's coordinate against a leak of
strength π_S μ_k toward the student's own (empty) prediction. No free energy is invoked anywhere
in this proof.

*Step 2 — settle each mode.* Set ṡ_k = 0 and divide by the strictly positive number
π_TS + π_S μ_k (π_TS > 0, μ_k ≥ 0 — a scalar division; no invertibility argument is needed):

$$
\pi_{TS}\,(s_k - c_k) + \pi_S\, \mu_k\, s_k \;=\; 0 \qquad\Longrightarrow\qquad s_k^{*} \;=\; \frac{\pi_{TS}}{\pi_{TS} + \pi_S\, \mu_k}\, c_k \;=\; \big(1 - n_k\big)\, c_k ,
$$

writing n_k = n(μ_k); the last equality is 1 − n(μ) = π_TS/(π_TS + π_S μ), one line from the
definition of n. The per-mode solution is therefore the pair

$$
s_k^{*} \;=\; (1 - n_k)\, c_k , \qquad s_k^{*} - c_k \;=\; -\,n_k\, c_k :
$$

each mode is copied up to the attenuation factor 1 − n_k, and the fraction n_k is left behind as
interface error.

*Step 3 — reassemble the modes: the novelty operator emerges.* Collect the d attenuation dials
into one diagonal matrix, D_n = diag(n_1, …, n_d). The d scalar solutions of Step 2 then read,
in vector form,

$$
s^{*} \;=\; (I - D_n)\, c , \qquad s^{*} - c \;=\; -\,D_n\, c .
$$

Rotate back to the original frame (x_S\* = U s\*, c = Uᵀ x_T, and UUᵀ = I):

$$
x_S^{*} \;=\; U\,(I - D_n)\,U^\top x_T \;=\; \big(I - U D_n U^\top\big)\, x_T , \qquad
\varepsilon_{TS} \;=\; U\,(s^{*} - c) \;=\; -\,U D_n U^\top\, x_T .
$$

The operator that appears in both — the same rotation with the novelty dials in the middle —

$$
N_S \;=\; U\, D_n\, U^\top \;=\; U\, \mathrm{diag}\big(n(\mu_k)\big)\, U^\top
$$

is the **novelty operator**: applied to a teacher state, it reads the state's coordinates on the
student's axes, scales coordinate k by that direction's novelty n_k, and rotates back. With it,

$$
x_S^{*} \;=\; (I - N_S)\, x_T , \qquad \varepsilon_{TS} \;=\; -\,N_S\, x_T ,
$$

as claimed: the settled state is an **attenuated copy** of the teacher's (I − N_S is a shrinkage
operator, not an orthogonal projector), and the interface error is exactly the novelty-weighted
part of the teacher's state — the object Corollary 1 turns into the landscape.

*Step 4 — properties, read off the dials.* Symmetric and PSD (orthogonal U, dials n_k ≥ 0);
‖N_S‖ = n(μ_max) < 1 since n(μ) ∈ [0, 1); n(0) = 0 and n′(μ) = π_S π_TS/(π_TS + π_S μ)² > 0; and
n(μ) = 0 ⟺ μ = 0 gives

$$
\ker N_S = \ker S_S = \ker M_S :
$$

on a learned direction the student copies the teacher's component exactly (subject to the
representability caveat of Section 1). Finally, the closed form:
π_S S_S (π_TS I + π_S S_S)⁻¹ acts in the same frame with dials π_S μ_k/(π_TS + π_S μ_k)
= n(μ_k), so it equals N_S. ∎

---

## 6. Corollary 1 — the landscape is the novelty-weighted quadratic

**Corollary 1** *(of Lemma 1).* The landscape climbed in Theorem 1 has the closed form

$$
F_S^{\mathrm{eq}}(x_T) \;=\; \frac{\pi_{TS}}{2}\, x_T^\top N_S\, x_T , \qquad \nabla_{x_T} F_S^{\mathrm{eq}} = \pi_{TS}\, N_S\, x_T .
$$

The student's settled surprise about the teacher's state is the novelty-weighted energy of that
state: a quadratic that is exactly flat along learned directions and curved along novel ones.

**Proof.** Work in the modes of Lemma 1's proof. A rotation preserves norms, so the free energy
separates into per-mode one-variable energies — this corollary, being an energy statement, is the
one place that fact is needed:

$$
F_S \;=\; \sum_{k=1}^{d} f_k(s_k,\, c_k) , \qquad f_k(s,\, c) \;=\; \frac{\pi_{TS}}{2}\,(s - c)^2 \;+\; \frac{\pi_S\, \mu_k}{2}\, s^2 .
$$

By Lemma 1, Step 2 the settled mode is s_k\* = (1 − n_k) c_k, and by its Step 3 the quadratic form
of N_S is x_TᵀN_S x_T = Σ_k n_k c_k². Substitute s_k\* into each f_k, using two scalar facts read
off the definition of n_k: 1 − n_k = π_TS/(π_TS + π_S μ_k) and π_S μ_k (1 − n_k) = π_TS n_k. Then

$$
f_k(s_k^{*},\, c_k) \;=\; \underbrace{\frac{\pi_{TS}}{2}\, n_k^2\, c_k^2}_{\text{interface}} \;+\; \underbrace{\frac{\pi_S\, \mu_k}{2}\,(1 - n_k)^2\, c_k^2}_{\text{self}} \;=\; \frac{\pi_{TS}}{2}\,\big[\, n_k^2 + n_k\,(1 - n_k) \,\big]\, c_k^2 \;=\; \frac{\pi_{TS}}{2}\, n_k\, c_k^2 .
$$

Summing the modes:

$$
F_S^{\mathrm{eq}}(x_T) \;=\; \frac{\pi_{TS}}{2}\, \sum_k n_k\, c_k^2 \;=\; \frac{\pi_{TS}}{2}\, x_T^\top N_S\, x_T .
$$

The gradient of this quadratic (N_S symmetric and, by A1b, constant) is π_TS N_S x_T — and it
agrees with Theorem 1, Step 2, which gives ∇F_S^eq = π_TS(x_T − x_S\*) = π_TS N_S x_T by
Lemma 1: the two routes coincide. ∎

*Reading, per mode:* the interface term carries n_k² (the squared fraction of the component the
student failed to copy), the self term n_k(1 − n_k) (the student's own error on the part it did
copy); however the equilibrium splits the cost, the total is exactly the novelty n_k.

---

## 7. Remark — prioritization, axis by axis

*(Downstream of Theorem 1 and Corollary 1; this is what the experiments measure.)*

Under the push alone, by Theorem 1 and Corollary 1,

$$
\tau_T\, \dot x_T \big|_{\text{push}} = \frac{|\pi_{ST}|}{\pi_{TS}}\,\nabla F_S^{\mathrm{eq}} = |\pi_{ST}|\, N_S\, x_T .
$$

(Directly, without the landscape: substituting the per-mode error of Lemma 1, Step 2 into the push
u = π_ST ε_TS gives the same per-mode law — the energetic and the dynamical readings coincide,
which is Corollary 1's consistency content.)

By (A1b) the decomposition N_S = U D_n Uᵀ of Lemma 1, Step 3 is constant in time. In the
coordinates c = Uᵀ x_T the dynamics is diagonal,

$$
\tau_T\, \dot c_k = |\pi_{ST}|\, n_k\, c_k , \qquad\text{hence}\qquad c_k(t) = c_k(0)\, \exp\!\Big(\frac{|\pi_{ST}|\, n_k}{\tau_T}\, t\Big) :
$$

**learned directions (n_k = 0) are exactly frozen; novel directions grow exponentially at a rate
proportional to their novelty.**

*With the leash (A4).* The projected flow ẋ_T = P⊥(|π_ST| N_S x_T)/τ_T gives, in the same
coordinates, ċ_k = (|π_ST|/τ_T)(n_k − n̄) c_k with n̄ = x_Tᵀ N_S x_T / r₀², so the *ratios* obey
exactly the unconstrained law:

$$
\frac{d}{dt}\, \log\Big|\frac{c_k}{c_l}\Big| \;=\; \frac{|\pi_{ST}|}{\tau_T}\,\big(n_k - n_l\big) .
$$

Renormalization rescales all components equally, so prioritization is a statement about ratios
and is unaffected: among the directions the state occupies, the most novel one comes to dominate
the teacher's heading (power iteration). Two scope notes. If the largest occupied novelty
eigenvalue is degenerate, the state converges to the corresponding eigen*space*, not to a single
direction. And the decoupling concerns the push alone: the full drift adds the self-pull
−π_T S_T x_T, which mixes the novelty eigenvectors unless S_T and N_S commute.

*Self-limiting (conditional).* From n(μ) ≤ (π_S/π_TS) μ and μ_k = ‖M_S u_k‖² (unit eigenvector
u_k):

$$
|\pi_{ST}|\, n_k \ \le\ \frac{|\pi_{ST}|\, \pi_S}{\pi_{TS}}\, \|M_S\, u_k\|^2 .
$$

Hence *if* learning drives the residual ‖M_S u‖ → 0 along a direction, the push there vanishes
at least quadratically in the residual. That the coupled dynamics actually achieves this — on
every direction, including the moving eigendirections — is demonstrated numerically, not proved
(Section 9). Self-termination of the transfer (each visited direction gets learned, its n_k
drops, the landscape flattens under the teacher) is the mechanism this bound exhibits.

---

## 8. Proposition 2 — the exact free-energy saddle

**Proposition 2** *(the deterministic equations, A3; all three variables evolve — neither A1a
nor A1b is used; the renormalization is discussed in the remarks).* Define the single potential

$$
\Phi(x_T,\, x_S,\, W_S) \;=\; F_T(x_T) \;-\; F_S(x_S,\, x_T;\, W_S) .
$$

The deterministic sleep dynamics is the **minimax flow of Φ** — the teacher descends it, the
student ascends it, each with its own mobility:

$$
\tau_T\, \dot x_T = -\,\nabla_{x_T} \Phi , \qquad \tau_S\, \dot x_S = +\,\nabla_{x_S} \Phi , \qquad \dot W_S = +\,\eta\, P_0\big(\nabla_{W_S} \Phi\big)
$$

**if and only if π_ST = −π_TS.** Moreover, when π_ST ≠ −π_TS, no C² potential generates these
state equations as the displayed Euclidean descent–ascent flow with these fixed block
mobilities. (The qualification matters: under a different metric or a state-dependent mobility
the integrability condition changes.)

**Proof.** *(⇐, block by block.)* The two student blocks hold for every value of π_ST, because
F_T does not depend on (x_S, W_S), so ∇_{x_S}Φ = −∇_{x_S}F_S and ∇_{W_S}Φ = −∇_{W_S}F_S; by
Proposition 1 the model's student updates are exactly +∇_{x_S}Φ/τ_S and +η P_0(∇_{W_S}Φ).

Teacher block: using ∇F_T = π_T S_T x_T (the same rule as Proposition 1, A = M_T) and the sign
identity ∇_{x_T}F_S = −π_TS ε_TS (Theorem 1, Step 1),

$$
-\,\nabla_{x_T} \Phi \;=\; -\,\pi_T\, S_T\, x_T \;-\; \pi_{TS}\, \varepsilon_{TS} ,
$$

while the model's teacher drift is −π_T S_T x_T + π_ST ε_TS. The two agree for all states iff
π_ST = −π_TS.

*(⇒, no other potential in this pattern.)* Suppose some C² function Ψ satisfied
τ_T ẋ_T = −∇_{x_T}Ψ and τ_S ẋ_S = +∇_{x_S}Ψ. Then the field (−τ_T ẋ_T, +τ_S ẋ_S) is the
gradient of Ψ, and **the Jacobian of a gradient field is symmetric** (it is the Hessian of Ψ —
equality of mixed partials). Its two cross-blocks, read off the model's equations, are

$$
\frac{\partial\,(-\tau_T\, \dot x_T)}{\partial x_S} \;=\; -\,\pi_{ST}\, I , \qquad \frac{\partial\,(+\tau_S\, \dot x_S)}{\partial x_T} \;=\; \pi_{TS}\, I ,
$$

and symmetry requires them to be transposes of each other: −π_ST I = (π_TS I)ᵀ, i.e.
π_ST = −π_TS. ∎

**Remarks.**

1. *Reading.* At π_ST = −π_TS the teacher's reversed precision exactly mirrors the student's
   interface precision — a zero-sum coupling. The teacher descends Φ = F_T − F_S (descends its
   own surprise, ascends the student's); the student, descending F_S by Proposition 1, ascends Φ:
   a genuine minimax on one potential.
2. *With the leash.* The renormalization is a constraint, not generated by Φ: with it, the
   teacher block becomes the projected flow τ_T ẋ_T = −P⊥ ∇_{x_T}Φ, which still monotonically
   decreases Φ along the teacher's own move (dΦ/dt|_T = −‖P⊥∇_{x_T}Φ‖²/τ_T ≤ 0) — the saddle is
   exact as a projected saddle on the sphere.
3. *Consistency with Theorem 1.* Eliminating the fast student (A1a), the teacher block reads
   τ_T ẋ_T = −∇(F_T − F_S^eq): the same condition |π_ST| = π_TS is the point where Theorem 1's
   ascent coefficient equals 1, and the potential becomes Φ^eq = F_T − F_S^eq. (Theorem 1,
   Step 2 transports the gradient through the elimination.)
4. *Off the condition.* For π_ST ≠ −π_TS the flow fails the mixed-partials integrability
   condition (it has a non-gradient part); the transfer mechanism of Theorem 1 does not require
   the condition — only the single-potential interpretation does.

---

## 9. What is *not* proved here

Stated for honesty; all are empirical results or measured quantities in the paper.

- **Completion of the full coupled loop** (state and weights both evolving): that the system
  visits every unlearned direction, that learning drives the residuals ‖M_S u‖ → 0 there (used
  only *conditionally* in the self-limiting remark, Section 7), and that the process terminates.
  The analysis freezes W_S (Theorem 1) or treats single processes (Propositions 1–2); the
  coupled convergence is demonstrated numerically.
- **Manifold containment**: an operating choice plus a measurement, with no theorem claimed —
  the reversed precision is kept well below the teacher's self-precision (the code default sets
  |π_ST| to half the stability threshold), and the teacher's off-manifold occupancy during
  sleep, with the cross-coupling, the leash, and the noise all active, stays negligible
  (`manifold_leakage`, `offmanifold_growth`, `terminal_occupancy`).
- **Merging (interleaved rehearsal) and continual learning**: figure-level results built on the
  same engine; no theorem is claimed.
- **Discrete-episode replay**: the linear model transfers a subspace (directions, not named
  patterns); individuating episodes requires an added nonlinearity, outside the present analysis.
