# The novelty operator, one direction at a time

**What this is.** A standalone trace of the *per-orthogonal-direction* derivation of the novelty
operator: rotate once into the student's own axes, then never touch a matrix again until the very
last step. Every intermediate object is a scalar, every equation is one-dimensional, and the
operator `N_S` appears only at the end, as the *name* of the reassembled result rather than as an
object assumed up front.

**Why keep this version.** It buys the whole of Lemma 1 (and the prioritization schedule that comes
with it) at a very low price in machinery. Nothing here uses a matrix inverse, a commutation
argument, positive-definiteness, or a free energy. The complete toolkit is: the spectral theorem
once, reading a coordinate by dotting with a unit vector, and dividing by a positive number.

**Scope of the derivation** (three standing simplifications, each stated once):

1. the teacher's self-pull is dropped. It is *exactly* zero on the teacher's own memory manifold
   (there `M_T x_T = 0`, hence `S_T x_T = M_T^T (M_T x_T) = 0`), and the guard keeps the state near
   that manifold, so what remains to analyze is the student's push;
2. the noise is dropped (the analysis is deterministic). Its role is to *seed* coordinates that are
   exactly zero, which the push alone can only rescale;
3. the student's weights are frozen on this timescale, so the axes found in Step 1 are fixed while
   the states move.

---

## Step 1. One rotation, then forget that matrices exist

The student's self-error operator `S_S = M_S^T M_S` is symmetric, so the spectral theorem provides
an orthonormal basis of the whole space made of its eigenvectors,

$$
S_S\, u_k = \mu_k\, u_k , \qquad u_k^\top u_l = \delta_{kl} , \qquad k = 1, \dots, d .
$$

Call these the student's **rulers**. Because they form a basis of the entire space, every vector has
a unique set of coordinates on them, read off by dotting:

$$
c_k = u_k^\top x_T \quad (\text{teacher}), \qquad s_k = u_k^\top x_S \quad (\text{student}) .
$$

**What the eigenvalue means.** Sandwiching `S_S` between a unit ruler and itself,

$$
\mu_k = u_k^\top S_S\, u_k = u_k^\top M_S^\top M_S\, u_k = \|M_S\, u_k\|^2 \ \ge\ 0 .
$$

So `μ_k` is the **squared self-prediction error of the student along its own ruler `u_k`**: zero
exactly when the student can predict that direction perfectly (`M_S u_k = 0`, a *learned* ruler),
and large when its weights say nothing useful there. This is the only place the sandwich is needed,
and the only nonnegativity fact required later.

---

## Step 2. The student's dynamics becomes `d` independent scalar equations

The student's state equation is

$$
\tau_S\, \dot x_S = -\,\pi_{TS}\,\big(x_S - x_T\big) \;-\; \pi_S\, S_S\, x_S .
$$

Dot it with one ruler `u_k`. On the left, the rulers are constant in time (frozen weights), so the
derivative passes through the dotting. On the right, `u_k^\top S_S = \mu_k\, u_k^\top` because `u_k`
is an eigenvector of the symmetric `S_S`. Every term collapses onto that ruler's coordinate:

$$
\boxed{\ \tau_S\, \dot s_k \;=\; -\,\pi_{TS}\,\big(s_k - c_k\big) \;-\; \pi_S\, \mu_k\, s_k \ } , \qquad k = 1, \dots, d .
$$

The equation for `s_k` contains `s_k` and `c_k` and nothing else. The `d`-dimensional coupled system
has become `d` one-dimensional systems that never speak to each other.

**Reading.** Each ruler carries its own little tug-of-war between two springs:

- an **interface spring** of stiffness `π_TS` pulling the student's coordinate toward the teacher's
  coordinate `c_k` (copy what you are shown), and
- a **leak** of strength `π_S μ_k` pulling it back toward `0` (trust your own model, which along a
  novel ruler predicts nothing at all).

The leak strength is exactly the raw self-error of Step 1. A learned ruler has no leak.

---

## Step 3. Settle one ruler

Fast student: set `ṡ_k = 0`, gather the `s_k` terms, and divide by their coefficient.

$$
\pi_{TS}\,\big(s_k - c_k\big) + \pi_S\, \mu_k\, s_k = 0
\qquad\Longrightarrow\qquad
\big(\pi_{TS} + \pi_S\, \mu_k\big)\, s_k = \pi_{TS}\, c_k .
$$

The coefficient `π_TS + π_S μ_k` is a strictly positive **number** (`π_TS > 0`, `π_S > 0`,
`μ_k ≥ 0`), so the division is legitimate with no invertibility argument anywhere:

$$
s_k^{*} \;=\; \frac{\pi_{TS}}{\pi_{TS} + \pi_S\, \mu_k}\; c_k \;=\; \big(1 - n_k\big)\, c_k ,
\qquad\text{with}\qquad
n_k \;=\; \frac{\pi_S\, \mu_k}{\pi_{TS} + \pi_S\, \mu_k} .
$$

The leftover interface error on that ruler follows by putting `c_k` over the same denominator:

$$
e_k \;=\; s_k^{*} - c_k \;=\; \frac{\pi_{TS} - \big(\pi_{TS} + \pi_S \mu_k\big)}{\pi_{TS} + \pi_S \mu_k}\, c_k \;=\; -\,n_k\, c_k .
$$

**Reading: copied plus missing equals whole.** On each ruler the student copies the fraction
`1 − n_k` of the teacher's coordinate and leaves the fraction `n_k` behind as error. Two limiting
checks, both immediate: on a learned ruler (`μ_k = 0`) it copies everything and the error is exactly
zero; on a hopeless ruler (`μ_k` very large) it copies almost nothing and nearly the whole
coordinate survives as error.

---

## Step 4. The dial `n(μ)`: how a raw error becomes a normalized novelty

Everything specific to this model is now contained in one scalar function of one scalar variable,

$$
n(\mu) \;=\; \frac{\pi_S\, \mu}{\pi_{TS} + \pi_S\, \mu} , \qquad \mu \ge 0 ,
$$

with `π_S, π_TS > 0` fixed. Four facts, one line each.

**(a) The floor.** `n(0) = 0`. The denominator is `π_TS > 0`, so this is an honest zero, not an
indeterminate form. A perfectly predicted ruler produces no error at all.

**(b) The ceiling, approached but never reached.** Dividing numerator and denominator by `μ` gives
`n(μ) → 1` as `μ → ∞`. Yet for every *finite* `μ` the denominator strictly exceeds the numerator
(because `π_TS > 0`), so

$$
n(\mu) \in [0, 1) .
$$

**(c) Strictly increasing.** Rewrite the dial by adding and subtracting `π_TS` in the numerator:

$$
n(\mu) \;=\; 1 \;-\; \frac{\pi_{TS}}{\pi_{TS} + \pi_S\, \mu} .
$$

As `μ` grows the subtracted fraction shrinks, so `n` rises. (Equivalently, by the quotient rule,
`n'(μ) = π_S π_TS / (π_TS + π_S μ)² > 0`.) Note that this rewriting also exhibits the copied
fraction `1 − n(μ) = π_TS / (π_TS + π_S μ)` directly.

**(d) The linear bound near zero.** From `π_TS + π_S μ ≥ π_TS`, take reciprocals (which reverses the
inequality, both sides being positive) and multiply by `π_S μ ≥ 0`:

$$
n(\mu) \;\le\; \frac{\pi_S}{\pi_{TS}}\; \mu , \qquad \text{with equality only at } \mu = 0 .
$$

Since `n'(0) = π_S/π_TS`, this line is the tangent to `n` at the origin, and the saturating curve
lies below its tangent thereafter.

### The comparison: raw error versus normalized dial

| | raw self-error `μ_k` | normalized dial `n_k = n(μ_k)` |
|---|---|---|
| definition | `‖M_S u_k‖²`, how badly the student's own weights predict ruler `u_k` | the fraction of the teacher's coordinate left unmatched once the student has settled |
| range | `[0, ∞)`, unbounded | `[0, 1)`, bounded |
| dimension | a squared activity | a pure fraction, dimensionless |
| learned ruler | `0` | `0` |
| hopeless ruler | grows without bound | saturates strictly below `1` |
| near zero | `μ` | `≈ (π_S/π_TS)·μ`, proportional |
| eigenvectors | the rulers `u_k` | the same rulers `u_k`, unchanged |

The map `μ ↦ n(μ)` is therefore a **saturating renormalization of the same spectrum**: it keeps the
student's axes exactly as they are and only re-scales the numbers attached to them, compressing an
unbounded raw error into a bounded fraction. The compression is not cosmetic, it is behavioral: the
student always retains the option of abandoning a hopeless ruler entirely (settling at `s_k ≈ 0` and
paying only the interface penalty), so the error it can be forced to carry on one ruler cannot exceed
the whole coordinate, whatever the raw `μ`. Proportional for small errors, capped for large ones, is
exactly the profile of that option.

*A note on the third quantity in the neighbourhood.* The pair above is entirely weight-level, so in
the terminology of the paper both are *novelty* (`μ` raw, `n` normalized). The state-level
**surprise** is the free energy, and its settled value per ruler turns out to be `(π_TS/2) n_k c_k²`,
the dial priced against how much of the state sits on that ruler. That computation belongs to
Corollary 1 and is not needed anywhere below.

---

## Step 5. The teacher's motion, ruler by ruler: prioritization

The student's push on the teacher is `u = π_ST ε_TS` (read straight off the teacher's equation, with
the self-pull dropped as declared in the scope). Its coordinate on ruler `k` uses Step 3's error and
the fact that in sleep `π_ST` is negative, `π_ST = -|π_ST|`:

$$
\tau_T\, \dot c_k \;=\; \pi_{ST}\, e_k \;=\; \big(-|\pi_{ST}|\big)\cdot\big(-\,n_k\, c_k\big) \;=\; |\pi_{ST}|\; n_k\; c_k .
$$

**The two minus signs cancel.** Reversed precision multiplied by a backward-pointing error gives
forward amplification. This single cancellation is the entire mechanism of the model, and here it is
visible on one line of scalar algebra.

Each ruler now obeys the elementary growth law `ċ = λ c`, so

$$
c_k(t) \;=\; c_k(0)\; \exp\!\Big(\frac{|\pi_{ST}|\, n_k}{\tau_T}\; t\Big) .
$$

Four readings, one look each:

1. **Learned rulers are frozen exactly.** `n_k = 0` gives `e⁰ = 1`, so `c_k(t) = c_k(0)` for all
   time. Consolidated content is not merely spared, it is untouched.
2. **Rate ordered by novelty.** Between two rulers, the one with the larger dial grows strictly
   faster, at every instant.
3. **The most novel occupied ruler wins.** The ratio of two coordinates obeys
   `c_k(t)/c_l(t) = (c_k(0)/c_l(0)) e^{(n_k - n_l)|π_ST| t/τ_T}`, so however small its share at the
   start, the largest-dial ruler comes to dominate the teacher's heading. The word *occupied*
   matters: a coordinate that is exactly zero stays exactly zero, since the push rescales and never
   seeds. That is the job of the noise dropped in the scope.
4. **The drive extinguishes itself, quadratically.** Combining the rate with the linear bound (d)
   and `μ_k = ‖M_S u_k‖²`,

$$
|\pi_{ST}|\, n_k \;\le\; \frac{|\pi_{ST}|\, \pi_S}{\pi_{TS}}\; \|M_S\, u_k\|^2 ,
$$

   so as learning shrinks the residual along a ruler, the push there dies like the *square* of that
   residual. Halve the remaining error and the drive falls to a quarter.

This is the **prioritized** of the paper's title, obtained from the dynamics alone. No free energy
has been mentioned anywhere in Steps 1 to 5.

---

## Step 6. Reassemble: the operator appears, and only now

Every vector is the sum of its coordinates times the rulers. Stack the `d` scalar results back up.

**The interface error.**

$$
\varepsilon_{TS} \;=\; \sum_k e_k\, u_k \;=\; -\sum_k n_k\, c_k\, u_k .
$$

Read that sum as a product: scaling coordinate `k` by `n_k` is what a diagonal matrix does, and
recombining the rulers weighted by the result is what `U` does. With
`U = [u_1 | ... | u_d]` and `D_n = diag(n_1, ..., n_d)`, and using `c = U^T x_T`,

$$
\varepsilon_{TS} \;=\; -\,U\, D_n\, c \;=\; -\,\big(U\, D_n\, U^\top\big)\, x_T .
$$

Give that matrix a name:

$$
\boxed{\ N_S \;:=\; U\, D_n\, U^\top , \qquad \varepsilon_{TS} \;=\; -\,N_S\, x_T \ } .
$$

**The settled state**, stacked the same way from `s_k^* = (1 - n_k) c_k`:

$$
x_S^{*} \;=\; U\,(I - D_n)\,U^\top x_T \;=\; \big(I - N_S\big)\, x_T ,
$$

an attenuated copy of the teacher's state, not a projection. **The teacher's law**, stacked from
Step 5:

$$
\tau_T\, \dot x_T \big|_{\text{push}} \;=\; |\pi_{ST}|\; N_S\, x_T .
$$

### Properties, all free by construction

Nothing below requires a computation beyond what is already on the page.

- **Symmetric.** `(U D_n U^T)^T = U D_n^T U^T = U D_n U^T`, since a diagonal matrix is its own
  transpose.
- **Eigenpairs.** `N_S u_k = U D_n (U^T u_k) = U (n_k e_k) = n_k u_k`. The operator has the student's
  rulers as its axes and the dials as its eigenvalues, because that is how it was assembled.
- **Positive semidefinite, with norm below one.** Its eigenvalues are the `n_k ∈ [0, 1)` of Step 4.
- **It annihilates learned directions.** `μ_k = 0` gives `n_k = 0`, hence `N_S u_k = 0` and
  `ε_TS = 0`. A teacher pointing along a learned direction produces no drive whatsoever.
- **Same axes as `S_S`, renormalized eigenvalues.** There is nothing to prove: `U` was taken from
  `S_S` and each dial was computed from that ruler's own `μ_k`.

### Agreement with the closed form

The compact expression used elsewhere (and computed in code) is
`N_S = π_S S_S (π_TS I + π_S S_S)^{-1}`. Apply it to a single ruler, factor by factor, each acting as
a number on an eigenvector: `S_S` acts as `μ_k`, the shifted matrix as `π_TS + π_S μ_k`, and its
inverse as division by that positive number. Hence

$$
\pi_S\, S_S\,\big(\pi_{TS} I + \pi_S S_S\big)^{-1} u_k \;=\; \frac{\pi_S\, \mu_k}{\pi_{TS} + \pi_S\, \mu_k}\; u_k \;=\; n_k\, u_k .
$$

Two matrices that agree on every vector of a basis are equal, so the closed form is the same
operator. Note the direction of the logic: the closed form is *identified* at the end, it is never
needed to obtain anything.

---

## What the derivation cost

| ingredient | where | used for |
|---|---|---|
| spectral theorem (symmetric matrix, orthonormal eigenbasis) | Step 1 | the rulers, once |
| coordinate read-off by dotting with a unit vector | Steps 1, 2 | passing to scalars and back |
| one sandwich, `v^T M^T M v = ‖M v‖²` | Step 1 | `μ_k ≥ 0` and its meaning |
| division by a positive number | Step 3 | settling each ruler |
| one-variable algebra and one limit | Step 4 | the shape of the dial |
| the solution of `ċ = λ c` | Step 5 | the schedule |
| stacking coordinates back into a vector | Step 6 | naming the operator |

Not used at any point: matrix inversion, existence or uniqueness arguments for inverses,
commutation of operators, projections, and free energy.

---

## In one paragraph

Rotate once into the student's own axes and the coupled system falls apart into `d` independent
one-dimensional tug-of-wars, each between an interface spring of stiffness `π_TS` and a leak of
strength `π_S μ_k`, where `μ_k` is the student's squared self-error along that axis. Settling one
tug-of-war is a division by a positive number, and it shows the student copying the fraction
`1 − n_k` of the teacher's coordinate and leaving the fraction `n_k = π_S μ_k / (π_TS + π_S μ_k)`
behind as error. That dial is the raw error renormalized onto `[0, 1)`, same axes, squashed numbers,
proportional for small errors and saturating for large ones because abandoning a hopeless axis is
always an option. Feeding the leftover back to the teacher through the reversed precision cancels
two minus signs and yields, per axis, `τ_T ċ_k = |π_ST| n_k c_k`: learned axes exactly frozen, novel
axes growing in proportion to their novelty, the most novel occupied axis eventually dominating, and
every drive extinguishing quadratically as its axis is learned. Only at the end do the `d` scalar
statements get stacked back into vectors, and the matrix that appears in the stacking is what we
call the novelty operator.
