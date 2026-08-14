# Theorem 1 — the intuitive, no-step-skipped derivation

**What this is.** The geometric, child-story version of the theory chain — *the student's push is
gradient ascent on the student's surprise* — told so no step is jumped and every step has a picture.
Companion to the algebraic version in [detailed_proofs.md](detailed_proofs.md); **same structure**:
the theorem first (a sign identity + the envelope theorem — no novelty operator), then Lemma 1
(d scalar tug-of-wars; the meter and the prioritization schedule are born there), then Corollary 1
(the bridge where the energy and dynamics stories meet), then the saddle (Proposition 2).

**Language.** A matrix's eigenvector is a **special direction**; its eigenvalue a **stretch factor** — a
machine sends its special direction to a scaled copy of itself (e.g. N_S u_k = n_k u_k below).

**Two words, kept apart.** **Novelty** belongs to the student's *weights*: it says which directions the
student cannot yet predict (the meter N_S below — born in Lemma 1's tug-of-wars, needed for the
schedule and the landscape, never for the theorem). **Surprise** belongs to a *state*: it is the free energy the student (F_S) or the teacher
(F_T) feels right now. The theorem climbs the **student's surprise**; the teacher's own surprise F_T
only appears as the self-pull force, never as the thing being climbed.

---

## Theorem 1 — the story is a rope and a valley

### Step 1 — the rope (the sign identity)

Teacher and student are tied by **one shared rope**: the interface energy

$$
\frac{\pi_{TS}}{2}\,\|x_S - x_T\|^2
$$

— one number, the same for both, measuring how stretched the rope is. The student descends its free
energy (Prop 1), so the student always pulls to **shorten** the rope. The teacher holds the *same*
rope through its own interface precision π_ST — and in sleep π_ST is **negative**, so the teacher
pulls to **lengthen** it. Same energy, opposite signs: whatever direction the student's end of the
rope points, the teacher's drive is that same direction, flipped and rescaled. Algebraically, in
one line: only the interface term of F_S contains x_T, so ∇_{x_T}F_S = −π_TS ε_TS, and the push is

$$
u \;=\; \pi_{ST}\,\varepsilon_{TS} \;=\; \frac{|\pi_{ST}|}{\pi_{TS}}\, \nabla_{x_T} F_S \Big|_{x_S} .
$$

*Picture:* a tug-of-war where one side has reversed instructions. Nothing was derived — the
adversarial pairing is **built into the sign of π_ST**, and that sign is the entire wake/sleep
switch (π_ST > 0: both shorten the rope, ordinary recall, no transfer; π_ST < 0: the teacher is
driven toward whatever the student cannot yet match). This already holds at every instant, for
*any* student state, settled or not.

### Step 2 — the valley (the envelope theorem)

Step 1's slope is a *partial* slope: it still mentions where the student happens to be. But the fast
student always sits at the **bottom of its valley** — the settled state x_S\*(x_T), the unique
minimum of F_S in x_S. Define the landscape over the teacher's position alone:

$$
F_S^{\mathrm{eq}}(x_T) \;=\; \min_{x_S} F_S(x_S,\, x_T)
$$

— the student's surprise *after it has done its best*. Now the classical **envelope theorem** (one
picture): as the teacher moves, the settled student gets dragged along the valley floor — but
**motion along a valley floor is free at first order** (the slope there is zero; that is what
"bottom" means). So dragging the student contributes nothing, and the slope of the landscape is just
Step 1's partial slope, frozen at the bottom:

$$
\nabla_{x_T} F_S^{\mathrm{eq}} \;=\; -\,\pi_{TS}\,\varepsilon_{TS}\big|_{\text{settled}} .
$$

*(Why "envelope": one curve of F_S over x_T per frozen x_S; the landscape traces the lower envelope
of the family, tangent — same value, same slope — to whichever curve it touches.)* Combine with
Step 1 at the settled state, and the rope drops out of sight:

$$
\boxed{\ \tau_T\, \dot x_T \big|_{\text{push}} \;=\; \frac{|\pi_{ST}|}{\pi_{TS}}\, \nabla_{x_T} F_S^{\mathrm{eq}}(x_T) \ } .
$$

**That is the theorem.** Wherever the teacher stands, the student's push points exactly up the slope
of the student's own surprise landscape — exact gradient **ascent**, at every point, toward maximum
student surprise locally (a gradient points up the slope the fastest — standard fact, one clause in
the paper). *Note what was not needed:* no novelty operator, no eigen-anything — a sign and a
valley. The meter only enters when we ask the follow-up: what does the landscape **look like**?

---

## Lemma 1 — d little tug-of-wars, and the meter is born

*(The dynamics-first derivation — the default in detailed_proofs.md. No inverses, no operator
algebra: one scalar story per ruler.)*

Rotate everything into the student's rulers u_k (the eigen-directions of S_S; frozen weights ⇒
frozen rulers). On each ruler, the student's coordinate s_k feels exactly **two springs**:

- an **interface spring** of stiffness π_TS pulling s_k toward the teacher's coordinate c_k
  ("copy what you see"), and
- a **leak** of strength π_S μ_k pulling s_k toward 0 ("stay at what your own weights predict" —
  and along a novel ruler the weights predict *nothing*; μ_k = ‖M_S u_k‖² is how badly).

No ruler talks to any other — the whole system is d independent one-dimensional tug-of-wars. Each
settles where the springs balance:

$$
s_k^{*} = (1 - n_k)\, c_k , \qquad e_k = s_k^{*} - c_k = -\,n_k\, c_k , \qquad n_k = \frac{\pi_S\,\mu_k}{\pi_{TS} + \pi_S\,\mu_k} .
$$

*Picture:* the student **copies a fraction and leaves a fraction**. On a learned ruler (μ_k = 0)
it copies everything — no error. On a hopeless ruler (μ_k huge) it barely copies — almost the whole
coordinate is left as error. The number n_k ∈ [0, 1) is that left-behind fraction: the ruler's
**novelty dial**, zero iff learned, rising and saturating with the student's own failure.

**The teacher's push, ruler by ruler — prioritization falls out here.** The push on ruler k is
π_ST · e_k, and both signs are negative (reversed precision × backward error), so they cancel:

$$
\tau_T\, \dot c_k = |\pi_{ST}|\; n_k\; c_k \qquad\Longrightarrow\qquad c_k(t) = c_k(0)\, e^{|\pi_{ST}| n_k t/\tau_T} .
$$

*Picture:* each coordinate of the teacher's state grows at a rate equal to **its own novelty
dial**. Learned rulers: exactly frozen. Novel rulers: exponential growth, the most novel occupied
ruler eventually dominating the heading. And the dial obeys n(μ) ≤ (π_S/π_TS)μ, so as the student
learns a ruler the push there dies **quadratically** — the steering switches itself off, ruler by
ruler. That ordering is the **prioritized** of the title, derived with no free energy anywhere.

**The meter, born at the end.** Stack the d leftover fractions back into one vector:
ε_TS = −Σ_k n_k c_k u_k = −(U D_n Uᵀ)x_T. The matrix that appears — rotate in, apply the dials,
rotate back — is *named* the **novelty meter**:

$$
N_S = U\, D_n\, U^\top , \qquad \varepsilon_{TS} = -\,N_S\, x_T , \qquad x_S^{*} = (I - N_S)\, x_T .
$$

Point it along any direction and it reports how badly the student's weights fail there. It was
never assumed — it is the name of the reassembly.

---

## Corollary 1 — the bridge: the two stories meet

We now have **two independent stories about the same flow**, sharing no steps: Theorem 1's
*energy* story (the push climbs the student's surprise landscape — a rope and a valley, no meter
anywhere) and Lemma 1's *dynamics* story (d tug-of-wars, dials, exponential schedule — no energy
anywhere). Corollary 1 is where they meet: computing the settled free energy dial by dial,

$$
F_S^{\mathrm{eq}}(x_T) = \frac{\pi_{TS}}{2}\, x_T^\top N_S\, x_T
$$

— **the landscape Theorem 1 climbs IS the meter's total reading on the state**: a bowl, exactly
flat along learned rulers (walking there costs nothing, surprises no one) and curved along novel
ones (the more novel, the steeper). Per ruler the interface pays n², the student's own error pays
n(1−n), and they always recombine to exactly n — however the tug-of-war splits the bill, the total
is the novelty. And the bowl's slope is π_TS N_S x_T, which times (|π_ST|/π_TS) is exactly Lemma 1's
reassembled push: the energy story and the dynamics story are **one fact** (checked in code to
~1e-15). *Picture:* the valley floor of Theorem 1, seen from above, has the meter's dials as its
curvatures.

---

## Proposition 2 — the saddle in one look

Both forces on the teacher are now gradients: the self-pull is −∇F_T (the teacher descending its
*own* surprise), and the push is (|π_ST|/π_TS)∇F_S^eq (the teacher ascending the *student's*). So
the whole sleep flow reads

$$
\tau_T\, \dot x_T = -\,\nabla F_T \;+\; \frac{|\pi_{ST}|}{\pi_{TS}}\,\nabla F_S^{\mathrm{eq}} ,
$$

which is downhill on the single seesaw potential Φ = F_T − F_S^eq exactly when the ascent's
coefficient is 1 — that is, **|π_ST| = π_TS**, i.e. π_ST = −π_TS: *climb the student's surprise
exactly as strongly as you measured it.* Meanwhile the student, descending F_S, ascends Φ — a
minimax on one potential.

---

## Full sentence

Teacher and student hold one rope with opposite-signed precisions, so the student's push on the
teacher is the student's own slope, flipped (the sign identity); with the student settled at the
bottom of its valley, the envelope theorem turns that into exact gradient ascent on the fixed
landscape F_S^eq — the theorem, a sign plus a valley, no meter needed. Independently, each of the
student's rulers is a little tug-of-war that settles by copying a fraction and leaving a fraction —
and the left-behind fractions, fed back as the push, give the schedule: learned rulers frozen,
novel rulers amplified in proportion to their novelty, each push extinguishing quadratically as the
student learns (Lemma 1 — the meter N_S born as the name of the reassembly, the *prioritized* of
the title). The two stories meet in the bridge: the landscape is the novelty-weighted bowl
(π_TS/2)x_TᵀN_Sx_T, flat where learned, curved where novel (Corollary 1). And the full flow is one
seesaw, Φ = F_T − F_S^eq, exactly at π_ST = −π_TS. Never a "teacher's surprise": the teacher
descends its own F_T while being steered up the student's F_S.

*Deferred to future work (the fun part): which of the novel directions ultimately dominate once the
teacher's own memory geometry is folded in. Keeping the teacher near its memory is an
operating-regime matter (reversed precision kept well below the teacher's self-precision), shown
numerically.*
