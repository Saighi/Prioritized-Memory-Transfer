# Theorem 1 — the intuitive, no-step-skipped derivation

**What this is.** The geometric, child-story version of Theorem 1 — *the student steers the teacher up
the **student's** surprise* — told so no step is jumped and every step has a picture. Companion to the
terse algebraic version in [detailed_proofs.md](detailed_proofs.md); same result, but here we *feel* it.

**Language.** A matrix's eigenvector is a **special direction**; its eigenvalue a **stretch factor** — a
machine sends its special direction to a scaled copy of itself (e.g. N_S u_k = n_k u_k below).

**Two words, kept apart.** **Novelty** belongs to the student's *weights*: it says which directions the
student cannot yet predict (the meter N_S and its numbers n_k below). **Surprise** belongs to a *state*:
it is the free energy the student (F_S) or the teacher (F_T) feels right now. The theorem climbs the
**student's surprise**; the teacher's own surprise F_T only appears as the self-pull force, never as the
thing being climbed.

---

## The cast

- Full space: one axis per neuron. The teacher's state x_T is a vector there.
- **N_S**: the student's **novelty meter** (from Lemma 1). Its special directions are the student's
  rulers u_k, each simply scaled by its novelty number:

$$
N_S\, u_k = n_k\, u_k, \qquad n_k \in [0,1),
$$

  with n_k = 0 for a learned direction, n_k large for a novel one. So N_S is a **meter**: point it along
  any direction and it reports how badly the student's weights fail to predict it. The meter is built
  from the weights alone — no state has been mentioned yet.
- **F_S^eq**: the student's **surprise** about the teacher's state — what F_S is worth once the fast
  student has settled while the teacher sits at x_T. Corollary 1 (in detailed_proofs.md) computes it:

$$
F_S^{\mathrm{eq}}(x_T) = \frac{\pi_{TS}}{2}\, x_T^\top N_S\, x_T .
$$

  *Picture:* the meter's total reading on the whole state — novelty (the spectrum) priced out on where
  the state actually sits (the score). Not a new quantity invented for the theorem: it is the model's
  own F_S with the student settled.

---

## Step 1 — the student's push, straight from Lemma 1

The teacher in sleep is pushed by two forces; substituting the fast-student mismatch ε_TS = −N_S x_T
(Lemma 1) and using the reversed precision (π_ST < 0, so −π_ST = |π_ST|):

$$
\tau_T\, \dot x_T = \underbrace{-\pi_T\, S_T\, x_T}_{\text{teacher holds its own memory}} \; + \; \underbrace{|\pi_{ST}|\, N_S\, x_T}_{\text{the student's push } u} .
$$

The student's push is just the second term, read off with no extra work:

$$
u = |\pi_{ST}|\, N_S\, x_T .
$$

*Picture:* the push is the teacher's own state x_T sent through the student's novelty meter. (The first
force is the teacher pulling itself back toward its own memories — that is the teacher descending its
*own* surprise F_T; the guard, Lemma 2, checks it wins off-manifold. Not part of this story.)

---

## Step 2 — the push amplifies novel directions, ignores learned ones

Describe the teacher's state in the student's rulers: x_T = Σ_k c_k u_k, where c_k = u_kᵀ x_T is how much
of x_T points along ruler u_k. Watch how the push changes one such component. Under the push alone
(ẋ_T = u/τ_T):

$$
\dot c_k = u_k^\top \dot x_T = \frac{1}{\tau_T}\, u_k^\top u = \frac{|\pi_{ST}|}{\tau_T}\, u_k^\top N_S\, x_T .
$$

The meter reads n_k along u_k (u_kᵀ N_S = n_k u_kᵀ), and u_kᵀ x_T = c_k, so

$$
\boxed{\ \dot c_k = \frac{|\pi_{ST}|}{\tau_T}\, n_k\, c_k\ } .
$$

*(This "each ruler minds its own business" is proved in full in the detailed version — Theorem 1
Step 2 there writes N_S = U D_n Uᵀ (rotate into the rulers' frame, scale each coordinate by its
novelty dial, rotate back — Lemma 1, Step 3b), rotates the whole equation of motion into that frame
where it becomes ċ = (|π_ST|/τ_T) D_n c — a diagonal system, whose off-diagonal zeros ARE the
no-mixing — and solves each axis to c_k(t) = c_k(0)e^{|π_ST| n_k t/τ_T}.)*

*Picture:* each component of the teacher's state grows at a rate equal to **its own novelty**. A learned
direction (n_k = 0) doesn't budge; a novel direction grows, and the more novel it is, the faster.
So the teacher's heading tips toward the directions the student can't yet predict, never toward learned
ones. **That is already the theorem** — the student steers the teacher toward novel places, from nothing
but the push and "the meter reads n_k along u_k."

---

## Step 3 — the push points *up the student's surprise the fastest* (the "maximum surprise" reading)

A second, global way to see the same thing. The teacher's state has a single score attached: the
**student's surprise** about it,

$$
F_S^{\mathrm{eq}}(x_T) = \frac{\pi_{TS}}{2}\, x_T^\top N_S\, x_T = \frac{\pi_{TS}}{2} \sum_k n_k\, c_k^2 .
$$

*Picture:* the score adds up "how much of the state sits on ruler u_k" (that's c_k²) times "how novel
u_k is" (n_k). Zero if the state lies entirely on learned rulers; bigger the more novel content it has.
And by Corollary 1 this score is literally what the settled student *feels* — its free energy, the
model's own measure of surprise.

Now — **what is the gradient of a scalar like F_S^eq?** By definition (the (G1) template in
detailed_proofs.md, "The atoms"), it is the unique vector g with
F(x + δ) = F(x) + gᵀδ + O(‖δ‖²) — nudge, expand, and whatever multiplies δ linearly *is* the
gradient. And it is the vector of steepest increase: the direction you'd step to raise the score
fastest (Cauchy–Schwarz, spelled out in the detailed proof's expository block). Nudging x_T by a small δ and
reading the linear part:

$$
F_S^{\mathrm{eq}}(x_T + \delta) = F_S^{\mathrm{eq}}(x_T) + \big(\pi_{TS}\, N_S x_T\big)^\top \delta + O(\|\delta\|^2) \quad\Longrightarrow\quad \nabla F_S^{\mathrm{eq}} = \pi_{TS}\, N_S\, x_T ,
$$

where the two equal middle terms of the quadratic expansion (equal because N_S is symmetric) combined
into one — the ½ cancelling their factor of 2. Compare with the push:

$$
u = |\pi_{ST}|\, N_S\, x_T = \frac{|\pi_{ST}|}{\pi_{TS}}\, \nabla F_S^{\mathrm{eq}} .
$$

**The push points exactly along the gradient of the student's surprise** (a positive number times a
gradient still points the same way). And the gradient is the steepest-ascent direction: for any unit
step e the change of the score is ∇F_S^eq·e ≤ ‖∇F_S^eq‖ (Cauchy–Schwarz), with equality only when e
points along the gradient. So of every direction the teacher could go, the student's push is the one
that makes **the student** most surprised, fastest.

*Honest caveat:* this is the steepest climb **from where the teacher currently is** (a local slope), not
an arrow at the globally most-surprising direction; and a ruler the state doesn't occupy (c_k = 0) gets
no push (ċ_k = 0) — noise seeds those in the full model.

---

## Step 4 — it climbs, and it switches itself off

Under the push, the student's surprise never goes down:

$$
\frac{dF_S^{\mathrm{eq}}}{dt} = \nabla F_S^{\mathrm{eq}} \cdot \dot x_T = \frac{\pi_{TS}\,|\pi_{ST}|}{\tau_T}\, \|N_S x_T\|^2 \ \ge\ 0 ,
$$

zero only when N_S x_T = 0 (nothing left that surprises the student). And it self-limits: as the student
learns a direction (Proposition 1 drives its self-error ‖M_S u_k‖ → 0), its novelty n_k → 0, so the push
there (rate |π_ST| n_k) fades and switches off exactly when the direction is learned. *Picture:* the
teacher climbs a landscape that its own climbing flattens — every place it drags the student to gets
learned, and the hill sinks beneath it.

---

## Step 5 — bonus: the saddle condition in one look

Both forces in Step 1 are now gradients: the self-pull is −∇F_T (the teacher descending its *own*
surprise), and the push is (|π_ST|/π_TS)∇F_S^eq (the teacher ascending the *student's*). So the whole
sleep flow reads

$$
\tau_T\, \dot x_T = -\,\nabla F_T \;+\; \frac{|\pi_{ST}|}{\pi_{TS}}\,\nabla F_S^{\mathrm{eq}} ,
$$

which is downhill on the single seesaw potential Φ = F_T − F_S^eq exactly when the ascent's coefficient
is 1 — that is, **|π_ST| = π_TS**, i.e. π_ST = −π_TS. That is where Proposition 3's exact-saddle
condition comes from: *climb the student's surprise exactly as strongly as you measured it.*

---

## Full sentence

The student's push on the teacher is the teacher's state passed through the student's novelty meter. It
grows each direction of the teacher's state in proportion to how novel it is to the student (learned →
nothing, novel → grows), which is the same as saying the push points up the **student's surprise** the
fastest — a local steepest ascent of the student's own free energy, not of any "teacher's surprise."
And it turns itself off on each direction the moment the student has learned it.

*Deferred to future work (the fun part): which of the novel directions ultimately dominate once the
teacher's own memory geometry is folded in. Keeping the teacher near its memory is the guard's job
(Lemma 2), shown numerically.*
