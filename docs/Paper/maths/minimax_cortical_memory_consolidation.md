# A free-energy minimax principle for cortical memory consolidation

## Worst-case free-energy and reconstruction guarantees in a two-network model

### Overview and central claim

This note derives a two-network memory-transfer architecture from one normative requirement:
minimize the largest settled free energy of the complete network over the states currently
supported by a hippocampal teacher. The derivation is self-contained for the linear,
timescale-separated model and states explicitly which parts remain approximations in the simulated
network.

Let the complete network free energy be

$$
F(x_T,x_S;W_T,W_S)
=
F_T(x_T;W_T)+F_S(x_S,x_T;W_S),
$$

where $F_T$ is the teacher's recurrent free energy and $F_S$ contains the interface mismatch and
the student's recurrent free energy. Valid teacher states lie on the fixed-energy memory sphere
$\mathbb S_T\subseteq\ker M_T$. Therefore $F_T=0$ throughout the optimization domain, and the
network free energy restricted to that domain is exactly $F_S$.

The fundamental object is the **largest settled free energy of the complete network over everything
the hippocampal teacher currently represents**:

$$
\boxed{
\mathcal{F}_{\max}(W_S;W_T)
=
\max_{x_T\in\mathbb S_T}
\min_{x_S}
F(x_T,x_S;W_T,W_S).
}
$$

The complete idealized computation is

$$
\boxed{
\min_{W_S}
\max_{x_T\in\mathbb S_T}
\min_{x_S}
F(x_T,x_S;W_T,W_S).
}
$$

The three nested operations map directly onto the three timescales of the network:

1. the fast student state minimizes $F$ and evaluates the settled network free energy for a
   candidate teacher state;
2. the teacher state maximizes that settled free energy over its memory manifold and exposes the
   current cortical blind spot;
3. the slow student weights descend the resulting worst-case free-energy envelope.

Under the explicit assumptions below, the proof establishes:

- $\mathcal{F}_{\max}$ is a uniform certificate on cortical reconstruction error and cortical
  self-inconsistency for every valid teacher state;
- any Lipschitz downstream readout inherits a uniform functional-error bound;
- teacher ascent on the settled network free energy requires a negative teacher-side interface
  coupling;
- once a generic worst-case direction has been selected, the local plasticity rule is exact
  gradient descent on $\mathcal{F}_{\max}$;
- among sufficiently small synaptic changes of the same amplitude, that negative-gradient direction
  gives the greatest possible first-order decrease of the bound;
- $\mathcal{F}_{\max}=0$ exactly when the student stores every teacher-memory direction.

The result is local and timescale-separated. It is not a claim of globally optimal learning time,
global convergence for arbitrary nonlinear networks, or exact minimax behavior when inference,
selection and plasticity occur on comparable timescales.

---

## 1. Network objects and assumptions

There are two linear recurrent populations:

- a frozen **teacher** $T$, interpreted as a hippocampal source of currently stored content;
- a plastic **student** $S$, interpreted as a cortical recipient.

Their states are $x_T,x_S\in\mathbb R^d$ and their recurrent weights are $W_T,W_S$. Define

$$
M_T=I-W_T,
\qquad
M_S=I-W_S,
$$

and the errors

$$
\varepsilon_T=M_Tx_T,
\qquad
\varepsilon_S=M_Sx_S,
\qquad
\varepsilon_{TS}=x_S-x_T.
$$

With positive precisions $\pi_T,\pi_{TS},\pi_S$, define the teacher, recipient-side and complete
network free energies by

$$
F_T(x_T;W_T)
=
\frac{\pi_T}{2}\|\varepsilon_T\|^2,
$$

$$
F_S(x_S,x_T;W_S)
=
\frac{\pi_{TS}}2\|\varepsilon_{TS}\|^2
+
\frac{\pi_S}{2}\|\varepsilon_S\|^2,
$$

and

$$
\boxed{
F(x_T,x_S;W_T,W_S)
=
F_T(x_T;W_T)+F_S(x_S,x_T;W_S).
}
$$

This $F$ is the variational free energy of the complete two-network hierarchy. The teacher weights
$W_T$ remain fixed; consolidation changes $W_S$ so that the hierarchy can attain low free energy
uniformly over the content supported by the teacher.

The proof uses the following assumptions.

**A1 — Linear memory criterion.** A state is stored by a network when its recurrent prediction
error is zero. The teacher memory space is therefore

$$
\mathcal U_T=\ker M_T.
$$

**A2 — Exhaustive teacher memory space.** The span of the named teacher patterns is exactly
$\mathcal U_T$. Named patterns may be correlated or redundant. Their effective number of
independent directions is

$$
r=\dim\mathcal U_T.
$$

**A3 — Equal replay-energy budget.** Candidate teacher states obey

$$
\|x_T\|=1.
$$

This prevents the maximization from increasing a quadratic free energy merely by increasing
activity amplitude. It makes the comparison one between directions at equal state energy.

**A4 — Timescale separation.** The state and learning timescales satisfy

$$
\tau_S\ll\tau_T\ll1/\eta.
$$

For almost-fixed $x_T$ and $W_S$, the student first settles. For almost-fixed $W_S$, the teacher
then searches the settled landscape. The weights move only after those two faster operations have
approximately equilibrated.

**A5 — Valid teacher search.** The mathematical maximization is restricted to valid teacher-memory
states. A hard implementation projects onto $\mathcal U_T$ and normalizes activity. The simulated
network implements the restriction softly through teacher recurrent correction plus normalization.
That soft implementation is accurate only when off-manifold leakage remains small.

**A6 — Simplified plasticity geometry.** The exact proof first treats unconstrained $W_S$. With a
no-autapse constraint, the zero-diagonal projection is included explicitly and gives the steepest
descent available inside the feasible weight subspace. Finite-step and finite-timescale effects are
not part of the infinitesimal theorem.

**A7 — Generic worst-case direction for the smooth steepest-descent statement.** The largest
eigenvalue of the restricted discrepancy operator is assumed simple. Because the cost is even in
$x_T$, the two maximizers $x_T^*$ and $-x_T^*$ remain; they produce the same weight derivatives.
Exact multidirectional ties are explicitly excluded from the theorem in Section 14.

Together, A1–A7 are sufficient for the hard-constrained local minimax result proved below.

---

## 2. The teacher memory sphere

Define the teacher self-error matrix

$$
S_T=M_T^\top M_T.
$$

For every vector $v$,

$$
v^\top S_Tv
=
v^\top M_T^\top M_Tv
=
\|M_Tv\|^2
\ge0.
$$

Therefore $S_T$ is symmetric and positive semidefinite.

It also has the same kernel as $M_T$. If $M_Tv=0$, then clearly $S_Tv=0$. Conversely, if
$S_Tv=0$, then

$$
0=v^\top S_Tv=\|M_Tv\|^2,
$$

which implies $M_Tv=0$. Hence

$$
\ker S_T=\ker M_T=\mathcal U_T.
$$

### The spectral theorem used here

The **spectral theorem** says that every real symmetric matrix has an orthonormal basis of
eigenvectors. Since $S_T$ is symmetric, its eigenvectors with eigenvalue zero form an orthonormal
basis of its kernel. Collect those $r$ vectors as the columns of

$$
U_T\in\mathbb R^{d\times r},
\qquad
U_T^\top U_T=I_r.
$$

The matrix $U_T$ is not a new memory system. It is only a convenient set of orthogonal coordinates
for the content already stored by the teacher. This is why correlated or redundant named patterns
do not cause a problem: the proof works with the independent directions in their span.

Every unit-norm teacher-memory state can now be written as

$$
x_T=U_Ta,
\qquad
\|a\|=1.
$$

The set of candidates is the teacher memory sphere

$$
\boxed{
\mathbb S_T
=
\{x_T\in\ker M_T:\|x_T\|=1\}.
}
$$

For every $x_T\in\mathbb S_T$,

$$
\varepsilon_T=M_Tx_T=0,
\qquad
F_T(x_T;W_T)=0,
$$

and consequently

$$
\boxed{
F(x_T,x_S;W_T,W_S)
=
F_S(x_S,x_T;W_S)
\qquad
\text{for every }x_T\in\mathbb S_T.
}
$$

Thus the teacher memory sphere is simultaneously the search domain and the zero-free-energy
manifold of the teacher. Restricting the complete network to this manifold leaves only the free
energy that remains because the student has not yet reconstructed and recurrently supported the
teacher state.

The sphere is closed and bounded in a finite-dimensional space, and is therefore compact. That
elementary fact will guarantee that the continuous settled free energy actually attains a maximum
on it.

---

## 3. Why the comparison is state-based, not weight-based

A quantity such as

$$
\|W_T-W_S\|_F
$$

does not measure functional memory transfer. Different recurrent matrices can support the same
memory space, and similar matrices need not support the same states.

The relevant directed question is instead:

> For every zero-free-energy state supported by the teacher, how much free energy must remain in the
> complete hierarchy after the student has made its best possible reconstruction?

This question is deliberately asymmetric. We want all teacher content to become available in the
student, but we do not want to penalize the student for memories it already stores in addition to
the teacher's current content. Reconstruction and recurrent support enter as consequences of making
the whole hierarchy low in free energy on the teacher's memory manifold.

---

## 4. The settled network free energy for one teacher state

Fix $x_T\in\mathbb S_T$ and consider a candidate cortical state $x_S$. It must meet two demands:

1. **Reconstruct the teacher state:** $x_S$ should be close to $x_T$.
2. **Be cortically supportable:** $M_Sx_S$ should be close to zero.

For this candidate pair of states, the complete network free energy is

$$
\boxed{
F(x_T,x_S;W_T,W_S)
=
\frac{\pi_T}{2}\|M_Tx_T\|^2
+
\frac{\pi_{TS}}2\|x_S-x_T\|^2
+
\frac{\pi_S}2\|M_Sx_S\|^2,
}
$$

where $\pi_T,\pi_{TS},\pi_S>0$. Because $x_T\in\mathbb S_T$, the teacher term is exactly zero.
The restricted network free energy is therefore

$$
F(x_T,x_S;W_T,W_S)
=
F_S(x_S,x_T;W_S)
=
\frac{\pi_{TS}}2\|x_S-x_T\|^2
+
\frac{\pi_S}2\|M_Sx_S\|^2.
$$

The first term measures reconstruction mismatch. The second measures the failure of the student
state to satisfy its own recurrent memory equation. These are the only contributions that can
remain in the whole-network free energy on the teacher manifold: the student must either disagree
with the teacher or occupy a state it cannot maintain autonomously.

The state should be evaluated after the student has made its best response. Define the **settled
network free energy for one teacher state**

$$
\boxed{
q(x_T;W_S)
=
\min_{x_S}F(x_T,x_S;W_T,W_S).
}
$$

Equivalently, because $F_T=0$ on $\mathbb S_T$,

$$
q(x_T;W_S)
=
\min_{x_S}F_S(x_S,x_T;W_S).
$$

This is not an arbitrary loss attached to a transient state. It is the smallest free energy the
complete hierarchy can attain for the specified teacher-supported state after the student has
settled. Its value is also the student's irreducible joint reconstruction-and-support deficit for
that state.

### When is the settled network free energy zero?

All terms of $F$ are nonnegative. Therefore $q\ge0$.

If the student stores $x_T$, choose $x_S=x_T$. Then

$$
x_S-x_T=0,
\qquad
M_Sx_S=M_Sx_T=0,
$$

so $q=0$.

Conversely, suppose $q=0$. Section 5 proves that the minimum is attained at a unique state $x_S^*$.
At that minimizer, a sum of two nonnegative terms is zero. Each term must therefore be zero:

$$
x_S^*=x_T,
\qquad
M_Sx_S^*=0.
$$

Thus $M_Sx_T=0$. We have proved

$$
\boxed{
q(x_T;W_S)=0
\quad\Longleftrightarrow\quad
x_T\in\ker M_S.
}
$$

Thus zero settled free energy on a teacher-supported state is equivalent to cortical storage of
that state. The same quantity is simultaneously a network-level free energy and a directed measure
of how far the teacher-supported state remains from cortical storage.

---

## 5. The student minimization, one direction at a time

For a fixed teacher state $x_T$, the inner minimization asks which cortical state minimizes the
complete network free energy after the student has settled. The teacher term is constant with
respect to $x_S$ and is zero on $\mathbb S_T$, so this minimization is exactly the minimization of
$F_S$. Instead of solving a matrix equation, rotate once into the student's own orthogonal
directions. In those coordinates the full problem separates into $d$ independent one-dimensional
minimizations.

Introduce the student self-error operator

$$
S_S=M_S^\top M_S.
$$

It is symmetric. The spectral theorem therefore provides an orthonormal basis
$u_1,\ldots,u_d$ and nonnegative numbers $\mu_1,\ldots,\mu_d$ satisfying

$$
S_Su_k=\mu_ku_k,
\qquad
u_k^\top u_\ell=\delta_{k\ell}.
$$

The eigenvalues are nonnegative because

$$
\mu_k
=
u_k^\top S_Su_k
=
u_k^\top M_S^\top M_Su_k
=
\|M_Su_k\|^2
\ge0.
$$

Thus $\mu_k$ is the student's squared recurrent self-error along the unit direction $u_k$.

### Express both states on the student's directions

Because the $u_k$ form an orthonormal basis of the whole state space, write

$$
x_T=\sum_k c_ku_k,
\qquad
c_k=u_k^\top x_T,
$$

and write an arbitrary candidate student state as

$$
x_S=\sum_k s_ku_k,
\qquad
s_k=u_k^\top x_S.
$$

The numbers $c_k$ and $s_k$ are simply the teacher and student coordinates along direction $u_k$.
Orthonormality gives

$$
\|x_S-x_T\|^2
=
\sum_k(s_k-c_k)^2.
$$

For the recurrent term,

$$
\begin{aligned}
\|M_Sx_S\|^2
&=
x_S^\top S_Sx_S
\\
&=
\sum_k\mu_ks_k^2.
\end{aligned}
$$

Substituting these two expressions into $F_S$ gives

$$
\boxed{
F_S(x_S,x_T;W_S)
=
\sum_k
\left[
\frac{\pi_{TS}}2(s_k-c_k)^2
+
\frac{\pi_S\mu_k}{2}s_k^2
\right].
}
$$

Define the scalar cost carried by direction $k$ as

$$
f_k(s_k;c_k)
:=
\frac{\pi_{TS}}2(s_k-c_k)^2
+
\frac{\pi_S\mu_k}{2}s_k^2.
$$

The full cost is the sum $F_S=\sum_k f_k$. More importantly, $f_k$ contains only $s_k$ and $c_k$:
no student coordinate appears in the equation of another direction. The matrix minimization has
become $d$ independent scalar minimizations.

### Settle one direction

Differentiate the scalar cost with respect to its scalar student coordinate:

$$
\frac{df_k}{ds_k}
=
\pi_{TS}(s_k-c_k)
+
\pi_S\mu_ks_k.
$$

Setting this derivative to zero gives

$$
(\pi_{TS}+\pi_S\mu_k)s_k
=
\pi_{TS}c_k.
$$

The coefficient $\pi_{TS}+\pi_S\mu_k$ is a strictly positive number because
$\pi_{TS},\pi_S>0$ and $\mu_k\ge0$. Ordinary scalar division therefore gives one and only one
stationary value:

$$
\boxed{
s_k^*
=
\frac{\pi_{TS}}{\pi_{TS}+\pi_S\mu_k}c_k.
}
$$

We still need to show that this stationary value is the global minimum, rather than merely a
stationary point. Take any scalar displacement $h_k$ and substitute $s_k=s_k^*+h_k$. Expanding and
using the stationary equation gives

$$
f_k(s_k^*+h_k;c_k)
=
f_k(s_k^*;c_k)
+
\frac{\pi_{TS}+\pi_S\mu_k}{2}h_k^2.
$$

The added term is strictly positive whenever $h_k\ne0$. Therefore $s_k^*$ is the unique global
minimum of the cost on direction $k$.

### Reassemble the unique best response

The complete student state is minimized exactly when every scalar cost $f_k$ is minimized. Hence

$$
\boxed{
x_S^*(x_T)
=
\sum_k
\frac{\pi_{TS}}{\pi_{TS}+\pi_S\mu_k}
c_ku_k.
}
$$

This state exists because every coefficient has been constructed explicitly. It is unique because
any different state must differ in at least one coordinate, and that coordinate then has strictly
larger scalar cost. At the best response, $df_k/ds_k=0$ for every member of a complete orthonormal
basis, so the full derivative with respect to $x_S$ is zero:

$$
\left.\frac{\partial F_S}{\partial x_S}\right|_{x_S^*}=0.
$$

The inner minimization is therefore well defined and single valued, without requiring a matrix
inverse or a separate positive-definite Hessian argument. The next section names and reassembles
the scalar fractions that appear in this solution.

---

## 6. The novelty operator and the closed form of settled free energy

Section 5 found the settled student coordinate

$$
s_k^*
=
\frac{\pi_{TS}}{\pi_{TS}+\pi_S\mu_k}c_k.
$$

Define the scalar novelty function

$$
n(\mu)
=
\frac{\pi_S\mu}{\pi_{TS}+\pi_S\mu},
$$

and abbreviate its value on mode $k$ by

$$
\boxed{
n_k:=n(\mu_k)
=
\frac{\pi_S\mu_k}{\pi_{TS}+\pi_S\mu_k}.
}
$$

Thus $n_k$ is not an additional independent quantity: it is the value of the same scalar function
on eigenvalue $\mu_k$. Since

$$
1-n_k
=
\frac{\pi_{TS}}{\pi_{TS}+\pi_S\mu_k},
$$

the settled coordinate is

$$
s_k^*=(1-n_k)c_k,
$$

and the teacher--student mismatch in that mode is

$$
s_k^*-c_k=-n_kc_k.
$$

The meaning is direct: along direction $u_k$, the student copies the fraction $1-n_k$ of the
teacher coordinate and leaves the fraction $n_k$ unresolved.

Because $\mu_k\ge0$ and $\pi_{TS},\pi_S>0$,

$$
0\le n_k<1.
$$

Moreover, $n_k=0$ exactly when $\mu_k=0$, meaning exactly when $M_Su_k=0$. The scalar function is
strictly increasing because

$$
n'(\mu)
=
\frac{\pi_S\pi_{TS}}{(\pi_{TS}+\pi_S\mu)^2}
>0.
$$

Thus a direction with greater recurrent self-error receives a larger unresolved fraction, while the
fraction remains bounded below one.

### Reassemble the scalar novelty fractions

Collect the student eigenvectors and novelty values into

$$
U=[u_1\mid\cdots\mid u_d],
\qquad
D_n=\operatorname{diag}(n_1,\ldots,n_d).
$$

Because the $u_k$ form an orthonormal basis,

$$
U^\top U=UU^\top=I.
$$

Now define the novelty operator by reassembling the directionwise scaling rule:

$$
\boxed{
N_S:=UD_nU^\top.
}
$$

This definition comes after the scalar calculation: $N_S$ is simply the matrix that multiplies the
coordinate on $u_k$ by $n_k$. Indeed,

$$
N_Su_k=n_ku_k.
$$

Reassembling $s_k^*=(1-n_k)c_k$ gives

$$
\boxed{
x_S^*
=
U(I-D_n)U^\top x_T
=
(I-N_S)x_T.
}
$$

Similarly, reassembling $s_k^*-c_k=-n_kc_k$ gives

$$
\boxed{
\varepsilon_{TS}^*
=
x_S^*-x_T
=
-N_Sx_T.
}
$$

The vector $N_Sx_T$ is therefore the part of the candidate teacher state left unresolved after the
student has made its best possible response.

Several properties are immediate from the construction:

1. $N_S$ is symmetric because $D_n$ is diagonal.
2. Its eigenvectors are the student directions $u_k$, with eigenvalues $n_k$.
3. It is positive semidefinite because every $n_k\ge0$.
4. All its eigenvalues lie in $[0,1)$.
5. It annihilates exactly the student directions with zero recurrent self-error.

If an eigenvalue $\mu$ of $S_S$ is repeated, its entire eigenspace receives the same scalar value
$n(\mu)$. Therefore the reassembled operator does not depend on which orthonormal basis is chosen
inside a repeated eigenspace.

### The settled network free energy, one direction at a time

On the teacher memory sphere, $F_T=0$ and $F=F_S$. We can therefore substitute $x_S^*$ into the
two remaining terms of the complete network free energy. Orthogonality gives

$$
\|x_S^*-x_T\|^2
=
\sum_k n_k^2c_k^2.
$$

Moreover,

$$
\|M_Sx_S^*\|^2
=
x_S^{*\top}S_Sx_S^*
=
\sum_k\mu_k(1-n_k)^2c_k^2.
$$

Therefore the contribution of mode $k$ to the settled network free energy is

$$
\frac{\pi_{TS}}2n_k^2c_k^2
+
\frac{\pi_S\mu_k}{2}(1-n_k)^2c_k^2.
$$

From the definition of $n_k$,

$$
\pi_S\mu_k(1-n_k)=\pi_{TS}n_k.
$$

Using this relation, the two contributions in mode $k$ reduce to

$$
\frac{\pi_{TS}}2n_kc_k^2.
$$

Summing over the orthonormal modes and using $x_T^\top N_Sx_T=\sum_k n_kc_k^2$ gives the closed form

$$
\boxed{
q(x_T;W_S)
=
\frac{\pi_{TS}}2x_T^\top N_Sx_T.
}
$$

This formula is the promised elimination of the fast student variable: the complete network's
settled free energy $q$ now depends on the teacher state and student weights only through the
novelty operator $N_S$.

### Agreement with the compact matrix formula

The derivation above did not assume or manipulate a matrix inverse. The compact expression used
elsewhere follows afterwards from the same scalar construction. Since

$$
S_S=U\operatorname{diag}(\mu_1,\ldots,\mu_d)U^\top,
$$

we have

$$
\pi_{TS}I+\pi_SS_S
=
U\operatorname{diag}(\pi_{TS}+\pi_S\mu_1,\ldots,
\pi_{TS}+\pi_S\mu_d)U^\top.
$$

Every diagonal number is strictly positive. Scalar reciprocation therefore gives

$$
(\pi_{TS}I+\pi_SS_S)^{-1}
=
U\operatorname{diag}\!\left(
\frac1{\pi_{TS}+\pi_S\mu_1},\ldots,
\frac1{\pi_{TS}+\pi_S\mu_d}
\right)U^\top.
$$

Applying $\pi_SS_S(\pi_{TS}I+\pi_SS_S)^{-1}$ to $u_k$ multiplies it by

$$
\frac{\pi_S\mu_k}{\pi_{TS}+\pi_S\mu_k}
=
n_k.
$$

This compact matrix and the spectrally constructed $N_S$ agree on every vector of the basis, so

$$
\boxed{
N_S
=
\pi_SS_S(\pi_{TS}I+\pi_SS_S)^{-1}.
}
$$

The inverse and the shared eigenvectors are therefore consequences of the directionwise solution,
not prerequisites for it.

The directions $u_k$ were used only to solve the student's response in the full state space. They
need not themselves be valid teacher-memory directions. Section 7 first restricts the reassembled
operator $N_S$ to the teacher memory space and only then finds the worst valid teacher direction.

---

## 7. The central object: largest settled network free energy

Define

$$
\boxed{
\mathcal{F}_{\max}(W_S;W_T)
=
\max_{x_T\in\mathbb S_T}q(x_T;W_S)
=
\max_{x_T\in\mathbb S_T}
\min_{x_S}
F(x_T,x_S;W_T,W_S).
}
$$

The notation emphasizes that this is a free-energy envelope: for each teacher-supported state the
student first attains its best response, and the maximum then retains the largest settled free
energy rather than averaging over teacher states. Since the teacher's own term vanishes on
$\mathbb S_T$, this whole-network quantity is numerically equal to the largest unresolved cortical
deficit. Its network-level free-energy meaning is primary; its reconstruction and functional-error
bounds are consequences derived below.

The maximum exists. The set $\mathbb S_T$ is compact and $q$ is continuous in $x_T$, so the
**extreme-value theorem** applies: a continuous real-valued function on a compact set attains both
its minimum and its maximum.

Write

$$
x_T=U_Ta,
\qquad
\|a\|=1,
$$

where $U_T\in\mathbb R^{d\times r}$ has orthonormal columns spanning the teacher memory space and
$a\in\mathbb R^r$ contains the coordinates of the teacher state in that basis. Because
$U_T^\top U_T=I_r$, the condition $\|a\|=1$ is equivalent to $\|x_T\|=1$.

### Restricting novelty to teacher-supported directions

The operator $N_S\in\mathbb R^{d\times d}$ measures unresolved content in the full network state
space. However, the maximization is not over every vector in $\mathbb R^d$; it is restricted to
teacher-supported states in $\mathcal U_T$. We therefore define

$$
\boxed{
A_S=U_T^\top N_SU_T\in\mathbb R^{r\times r}.
}
$$

The three factors have distinct roles:

1. $U_T$ maps teacher coordinates $a\in\mathbb R^r$ to the full state $x_T=U_Ta\in\mathbb R^d$;
2. $N_S$ applies the student's novelty operator in the full state space;
3. $U_T^\top$ projects the result back onto teacher-memory coordinates.

Thus $A_Sa=U_T^\top N_SU_Ta$ gives the teacher-coordinate representation of the unresolved content
generated by the teacher direction $a$. In matrix dimensions, the calculation is

$$
(r\times d)(d\times d)(d\times r)=r\times r.
$$

This compression is useful because only the action of $N_S$ on teacher-supported directions can
affect the maximum. It replaces a $d\times d$ operator by an $r\times r$ operator when the teacher
stores only an $r$-dimensional subspace. Most importantly, it preserves the scalar novelty score:

$$
x_T^\top N_Sx_T
=
(U_Ta)^\top N_S(U_Ta)
=
a^\top U_T^\top N_SU_Ta
=
a^\top A_Sa.
$$

Consequently,

$$
q(U_Ta;W_S)
=
\frac{\pi_{TS}}{2}a^\top A_Sa.
$$

The matrix $A_S$ is symmetric because $N_S$ is symmetric:

$$
A_S^\top
=
U_T^\top N_S^\top U_T
=
A_S.
$$

It is also positive semidefinite because, for every $a$,

$$
a^\top A_Sa=(U_Ta)^\top N_S(U_Ta)\ge0.
$$

### What $\lambda_{\max}(A_S)$ means

An eigenvector $v\ne0$ of $A_S$ satisfies

$$
A_Sv=\lambda v,
$$

where $\lambda$ is its eigenvalue. Because $A_S$ is real and symmetric, all its eigenvalues are
real and it has an orthonormal eigenbasis. The notation

$$
\lambda_{\max}(A_S)
$$

means the largest eigenvalue of $A_S$. Since $A_S$ is positive semidefinite, its eigenvalues are
nonnegative. Each eigenvalue quantifies the novelty score along its associated teacher-memory
direction; $\lambda_{\max}(A_S)$ is the score of the most unresolved such direction before
multiplication by $\pi_{TS}/2$.

### The Rayleigh–Ritz theorem used here

For a real symmetric matrix $A$, the **Rayleigh–Ritz theorem** states

$$
\max_{\|a\|=1}a^\top Aa=\lambda_{\max}(A).
$$

Here are the steps. Let $v_1,\ldots,v_r$ be an orthonormal eigenbasis of $A$, with corresponding
eigenvalues $\lambda_1,\ldots,\lambda_r$. Expand an arbitrary unit vector as

$$
a=\sum_{k=1}^r\alpha_kv_k.
$$

Orthonormality and $\|a\|=1$ imply

$$
\sum_{k=1}^r\alpha_k^2=1.
$$

Using $Av_k=\lambda_kv_k$ gives

$$
a^\top Aa
=
\sum_{k=1}^r\lambda_k\alpha_k^2.
$$

The nonnegative numbers $\alpha_k^2$ sum to one, so this expression is a weighted average of the
eigenvalues. It therefore satisfies

$$
\sum_{k=1}^r\lambda_k\alpha_k^2
\le
\lambda_{\max}(A)\sum_{k=1}^r\alpha_k^2
=
\lambda_{\max}(A).
$$

The upper bound is attained by choosing $a$ to be a unit eigenvector whose eigenvalue is
$\lambda_{\max}(A)$. More generally, equality holds exactly when $a$ lies entirely in the top
eigenspace.

Applying the theorem to $A_S$ now gives

$$
\begin{aligned}
\mathcal{F}_{\max}(W_S;W_T)
&=
\max_{\|a\|=1}q(U_Ta;W_S)
\\
&=
\frac{\pi_{TS}}2
\max_{\|a\|=1}a^\top A_Sa
\\
&=
\frac{\pi_{TS}}2\lambda_{\max}(A_S).
\end{aligned}
$$

Therefore

$$
\boxed{
\mathcal{F}_{\max}(W_S;W_T)
=
\frac{\pi_{TS}}{2}\lambda_{\max}(A_S).
}
$$

If

$$
E_{\max}
=
\ker\!\left(A_S-\lambda_{\max}(A_S)I_r\right)
$$

denotes the top eigenspace of $A_S$, the complete maximizer set is

$$
\boxed{
\operatorname*{arg\,max}_{x_T\in\mathbb S_T}q(x_T;W_S)
=
\{U_Ta:a\in E_{\max},\ \|a\|=1\}.
}
$$

This statement includes exact degeneracies correctly: when several directions share the largest
eigenvalue, every normalized mixture within their common eigenspace is also a maximizer.

---

## 8. The induced distance-like interpretation

The primary object $\mathcal{F}_{\max}$ is a worst-case settled free energy. Because its teacher
term vanishes and its remaining terms measure mismatch and recurrent inconsistency, it also induces
a functional distance-like quantity between the two memory systems. This induced quantity is not a
mathematical metric.

- It is directed from teacher to student.
- It is not generally symmetric under exchanging the networks.
- It need not satisfy a triangle inequality.
- It does not penalize extra student memories.

The last property is intentional. If the student contains the complete teacher memory space plus
additional old memories, transfer is complete for the present teacher.

There is nevertheless a precise connection to a one-sided distance between memory spaces. Consider
the limit $\pi_S\to\infty$, which imposes an increasingly strong penalty on cortical
self-inconsistency. In an eigenmode of $S_S$,

$$
n(\mu)
=
\frac{\pi_S\mu}{\pi_{TS}+\pi_S\mu}
\longrightarrow
\begin{cases}
0,&\mu=0,\\
1,&\mu>0.
\end{cases}
$$

Since $\ker S_S=\ker M_S$, the operator $N_S$ converges to the orthogonal projector onto
$(\ker M_S)^\perp$. Therefore

$$
q(x_T;W_S)
\longrightarrow
\frac{\pi_{TS}}2
\operatorname{dist}(x_T,\ker M_S)^2,
$$

and

$$
\mathcal{F}_{\max}
\longrightarrow
\frac{\pi_{TS}}2
\max_{x_T\in\mathbb S_T}
\operatorname{dist}(x_T,\ker M_S)^2.
$$

Thus the proposed object is a softened, directed, worst-case distance from the teacher memory
sphere to the student memory space. The finite-$\pi_S$ form is preferable for the dynamics because
it permits a graded tradeoff between matching the teacher and being recurrently supportable.

---

## 9. Uniform reconstruction and self-consistency bounds

The term **bound** has a precise meaning here.

At the settled student state,

$$
q(x_T;W_S)
=
\frac{\pi_{TS}}2\|x_S^*(x_T)-x_T\|^2
+
\frac{\pi_S}{2}\|M_Sx_S^*(x_T)\|^2.
$$

Both terms are nonnegative. Therefore each term is no larger than their sum:

$$
\frac{\pi_{TS}}2\|x_S^*(x_T)-x_T\|^2
\le q(x_T;W_S),
$$

and

$$
\frac{\pi_S}{2}\|M_Sx_S^*(x_T)\|^2
\le q(x_T;W_S).
$$

By definition of the maximum,

$$
q(x_T;W_S)\le\mathcal{F}_{\max}
$$

for every $x_T\in\mathbb S_T$. Combining the inequalities and taking square roots gives

$$
\boxed{
\|x_S^*(x_T)-x_T\|
\le
\sqrt{\frac{2\mathcal{F}_{\max}}{\pi_{TS}}}
\qquad
\text{for every }x_T\in\mathbb S_T.
}
$$

It also gives

$$
\boxed{
\|M_Sx_S^*(x_T)\|
\le
\sqrt{\frac{2\mathcal{F}_{\max}}{\pi_S}}
\qquad
\text{for every }x_T\in\mathbb S_T.
}
$$

These are **uniform** guarantees: the right-hand sides do not depend on which valid teacher state
is later queried.

The first inequality bounds representational reconstruction error. The second bounds cortical
self-inconsistency, meaning the amount of recurrent error remaining in the reconstruction. The
quantity $\mathcal{F}_{\max}$ jointly certifies both properties.

The inequalities are upper bounds, not identities. Because $q$ contains both error terms, the
reconstruction-only bound can be conservative. Its value is that it holds simultaneously for every
teacher-supported state without knowing which one will matter later.

---

## 10. From representational error to functional error: Lipschitz readouts

The previous section directly bounds state-space reconstruction. To claim a bound on downstream
function, one must specify how downstream systems read those states.

Let

$$
h:\mathbb R^d\to\mathbb R^m
$$

be a readout shared by the corresponding teacher and student representations. The output might be a
motor command, a prediction, a vector of decision variables, semantic features, or logits used for
classification.

### Definition of an $L$-Lipschitz readout

The function $h$ is **$L$-Lipschitz** on the relevant state region if

$$
\boxed{
\|h(u)-h(v)\|
\le
L\|u-v\|
}
$$

for every pair of relevant states $u,v$.

The definition says that changing the representation by an amount $\delta$ cannot change the
readout by more than $L\delta$. The constant $L$ is a worst-case sensitivity or gain. A small $L$
means a stable readout; a large $L$ means that the readout can amplify small representational
differences.

Apply the definition to

$$
u=x_S^*(x_T),
\qquad
v=x_T.
$$

The reconstruction bound immediately implies

$$
\boxed{
\|h(x_S^*(x_T))-h(x_T)\|
\le
L\sqrt{\frac{2\mathcal{F}_{\max}}{\pi_{TS}}}
\qquad
\text{for every }x_T\in\mathbb S_T.
}
$$

This is the promised worst-case functional-error certificate. Minimizing $\mathcal{F}_{\max}$ tightens
the guarantee for every $L$-Lipschitz function of the reconstructed state.

### Linear readouts

For a linear readout

$$
h(x)=Rx,
$$

the induced Euclidean operator norm is

$$
\|R\|_2
=
\max_{v\ne0}\frac{\|Rv\|}{\|v\|}.
$$

By this definition,

$$
\|Ru-Rv\|
=
\|R(u-v)\|
\le
\|R\|_2\|u-v\|.
$$

Thus $R$ is $\|R\|_2$-Lipschitz and

$$
\boxed{
\|Rx_S^*(x_T)-Rx_T\|
\le
\|R\|_2
\sqrt{\frac{2\mathcal{F}_{\max}}{\pi_{TS}}}.
}
$$

### Why a complex readout can still be Lipschitz

Lipschitz does **not** mean linear, simple, or biologically unrealistic. It only excludes an
unbounded output jump in response to an arbitrarily small change of the relevant representation.

There are several common reasons a complex readout has a finite Lipschitz constant.

#### 1. Smooth readouts on the relevant bounded region

Suppose $h$ is continuously differentiable on a convex region containing every line segment
between the relevant $x_T$ and $x_S^*(x_T)$. Let $J_h(x)$ be its Jacobian. If

$$
\|J_h(x)\|_2\le L
$$

throughout that region, then the fundamental theorem of calculus along the segment from $v$ to $u$
gives

$$
h(u)-h(v)
=
\int_0^1J_h(v+t(u-v))(u-v)\,dt.
$$

Taking norms and using the Jacobian bound gives

$$
\|h(u)-h(v)\|
\le
\int_0^1L\|u-v\|\,dt
=
L\|u-v\|.
$$

The readout may contain many interacting nonlinear computations. Only its maximal sensitivity on
the states actually visited matters.

In particular, take the compact convex hull of the relevant teacher states and their settled
student reconstructions. If $J_h$ is continuous on a neighborhood of that compact convex set, then
$\|J_h(x)\|_2$ attains a finite maximum there. Hence a smooth complex readout is automatically
Lipschitz on the bounded region needed by the proof. It need not be globally Lipschitz over all of
$\mathbb R^d$.

#### 2. Multilayer neural readouts

Consider a feedforward readout

$$
h(x)
=
R_L\sigma_{L-1}
\bigl(R_{L-1}\sigma_{L-2}(\cdots\sigma_1(R_1x))\bigr).
$$

If activation $\sigma_\ell$ is $L_\ell$-Lipschitz and each matrix has finite operator norm, repeated
application of the Lipschitz inequality gives the valid bound

$$
L_h
\le
\|R_L\|_2
\prod_{\ell=1}^{L-1}
L_\ell\|R_\ell\|_2.
$$

ReLU, for example, is $1$-Lipschitz in Euclidean norm. Deep composition can make the bound large or
loose, but not conceptually invalid. A large $L_h$ correctly records that an unstable downstream
readout demands a smaller cortical reconstruction error to guarantee the same output accuracy.

#### 3. Complex dynamics over a finite time horizon

A downstream recurrent or dynamical system can also define a readout: initialize it from $x$ and
observe its output after a fixed time. For example, suppose its dynamics are

$$
\dot y=f(y)
$$

and $f$ is $K$-Lipschitz on the relevant trajectories. The **Grönwall inequality** says that two
solutions starting from $y_1(0)$ and $y_2(0)$ obey

$$
\|y_1(t)-y_2(t)\|
\le
e^{Kt}\|y_1(0)-y_2(0)\|.
$$

The intuition is that a difference can grow at a rate no larger than $K$ times its current size;
integrating that worst-case growth gives the exponential factor. If the final output map is
$L_{\mathrm{out}}$-Lipschitz, the complete finite-time readout is therefore at most
$L_{\mathrm{out}}e^{Kt}$-Lipschitz. The constant may grow with the observation horizon, which
correctly records amplification by unstable downstream dynamics.

### Discrete choices and classification margins

A hard categorical decision such as $\operatorname*{arg\,max}_j h_j(x)$ is discontinuous at a
decision boundary and is not itself globally Lipschitz. The continuous score or logit vector $h$
can nevertheless be Lipschitz.

Suppose class $c$ has teacher-state margin

$$
\gamma(x_T)
=
h_c(x_T)-\max_{j\ne c}h_j(x_T)>0.
$$

Each logit can change by at most $\|h(x_S^*)-h(x_T)\|$, so the winning logit and a competitor can
approach one another by at most twice that amount. The cortical reconstruction is therefore
guaranteed to preserve the class whenever

$$
\boxed{
2L\sqrt{\frac{2\mathcal{F}_{\max}}{\pi_{TS}}}
<
\gamma(x_T).
}
$$

To guarantee the label for every teacher state, the right-hand side can be replaced by the smallest
margin over $\mathbb S_T$, when that minimum is positive.

### What the Lipschitz result does and does not say

The result says that a small uniform representational error controls every downstream readout whose
sensitivity is bounded on the relevant states. It does not say that all imaginable functions have a
small $L$, nor that an arbitrarily discontinuous decision is protected without a margin.

The present model also assumes corresponding teacher and student states share a common interface
coordinate system. If hippocampal and cortical codes live in different spaces, the interface should
contain a learned map $B$ and the reconstruction term should compare $x_S$ with $Bx_T$. The same
argument then bounds $h_S(x_S^*)-h_S(Bx_T)$, or bounds teacher-versus-student function when their
readouts satisfy an explicit alignment relation. A functional bound always requires such a bridge;
mere equality of latent dimensions is not enough biologically.

---

## 11. The teacher finds the state of maximal settled free energy

Hold the student weights $W_S$ fixed and suppose that, for each candidate teacher state $x_T$, the
student has already settled to its best response $x_S^*(x_T)$. The teacher then sees the complete
network's settled free energy

$$
q(x_T;W_S)
=
F(x_T,x_S^*(x_T);W_T,W_S)
=
F_S(x_S^*(x_T),x_T;W_S)
$$

and must solve

$$
\max_{x_T\in\mathbb S_T}q(x_T;W_S),
\qquad
\mathbb S_T=\{x_T\in\mathcal U_T:\|x_T\|=1\}.
$$

Thus the teacher state is the optimization variable, whereas $W_S$ is treated as constant during
the search. Section 7 solved this problem algebraically: its maximizers are the teacher-memory
directions associated with the largest eigenvalue of $A_S=U_T^\top N_SU_T$. The present section
asks a different question: can continuous teacher-state dynamics find those directions using the
current network errors rather than explicitly constructing and diagonalizing $A_S$?

The answer is yes for the idealized, hard-constrained, timescale-separated system. The argument has
three parts:

1. the settled interface error gives the derivative of the network free-energy envelope $q$ with
   respect to the teacher state;
2. tangent projection converts that derivative direction into ascent on the valid teacher-memory sphere;
3. in teacher coordinates, the resulting dynamics are continuous-time power iteration.

### The envelope identity for the teacher derivative

Recall

$$
q(x_T;W_S)
=
F(x_T,x_S^*(x_T);W_T,W_S).
$$

Changing $x_T$ affects this expression directly through $F_T$ and the mismatch term, and indirectly
because the best response $x_S^*(x_T)$ changes. Differentiating with respect to $x_T$ while holding
$W_S$ and $W_T$ fixed, the ordinary chain rule gives

$$
\frac{\partial q}{\partial x_T}
=
\left(\frac{dx_S^*}{dx_T}\right)^\top
\underbrace{\left.\frac{\partial F}{\partial x_S}\right|_{x_S^*}}_{0}
+
\left.\frac{\partial F}{\partial x_T}\right|_{x_S^*}.
$$

The first term vanishes because $x_S^*$ is the minimizer and therefore satisfies
$\partial F/\partial x_S=0$. This elementary use of the chain rule is often called the
**envelope theorem**:
when differentiating an optimized value, the derivative through the optimizing argument disappears
at an interior stationary optimum. The theorem does not say that $x_S^*$ is independent of $x_T$;
it says that its first-order contribution is multiplied by a zero derivative at the settled optimum.

The direct teacher derivative is

$$
\frac{\partial F}{\partial x_T}
=
\pi_TM_T^\top M_Tx_T
+
\pi_{TS}(x_T-x_S).
$$

On $\mathbb S_T$, $M_Tx_T=0$, so the teacher contribution and its gradient both vanish. Therefore

$$
\boxed{
\frac{\partial q}{\partial x_T}
=
\pi_{TS}(x_T-x_S^*)
=
-\pi_{TS}\varepsilon_{TS}^*.
}
$$

Using $\varepsilon_{TS}^*=-N_Sx_T$, the same result can be written as

$$
\frac{\partial q}{\partial x_T}=\pi_{TS}N_Sx_T.
$$

This agrees with differentiating the closed form
$q=(\pi_{TS}/2)x_T^\top N_Sx_T$, because $N_S$ is symmetric. The identity is important
computationally: the teacher does not need to represent $N_S$, form $A_S$, or differentiate through
the student's settling trajectory. Once the student has settled, the locally available mismatch
$\varepsilon_{TS}^*$ points exactly opposite to the derivative direction that increases the settled
network free energy.

### Why the derivative direction must be projected

The derivative vector $\partial q/\partial x_T$ lies in the full state space. Following it
directly could move $x_T$ outside the teacher memory space or change its norm. Valid search
directions must instead lie in the tangent space

$$
T_{x_T}\mathbb S_T
=
\{z\in\mathcal U_T:x_T^\top z=0\}.
$$

The condition $z\in\mathcal U_T$ keeps the motion within teacher-supported content, and
$x_T^\top z=0$ removes radial motion, thereby preserving $\|x_T\|=1$ to first order.

Because the columns of $U_T$ are orthonormal, $U_TU_T^\top$ is the orthogonal projector onto
$\mathcal U_T$. Because $x_T$ is a unit vector in that subspace, $x_Tx_T^\top$ is the projector
onto its radial direction. Their difference

$$
\boxed{
P_{T,x}
=
U_TU_T^\top
-
x_Tx_T^\top
}
$$

is therefore the orthogonal projector onto $T_{x_T}\mathbb S_T$. In particular,

$$
P_{T,x}x_T=0,
\qquad
x_T^\top P_{T,x}z=0
$$

for every $z$, and the range of $P_{T,x}$ lies inside $\mathcal U_T$.

### Projected ascent

Let $\tau_T>0$ be the teacher-state time constant and let $\mu_T>0$ be its search mobility. The
ideal hard-constrained search is

$$
\boxed{
\tau_T\dot x_T\big|_{\mathrm{search}}
=
\mu_TP_{T,x}\frac{\partial q}{\partial x_T}.
}
$$

The positive sign makes this ascent rather than descent. If the initial state belongs to
$\mathbb S_T$, the projected velocity remains tangent to the sphere, so the trajectory stays in
the valid search set. Moreover,

$$
\begin{aligned}
\frac{dq}{dt}
&=
\left(\frac{\partial q}{\partial x_T}\right)^\top\dot x_T
\\
&=
\frac{\mu_T}{\tau_T}
\left(\frac{\partial q}{\partial x_T}\right)^\top
P_{T,x}\frac{\partial q}{\partial x_T}
\\
&=
\frac{\mu_T}{\tau_T}
\left\|P_{T,x}\frac{\partial q}{\partial x_T}\right\|^2
\ge0.
\end{aligned}
$$

The last equality uses the fact that $P_{T,x}$ is an orthogonal projector. Thus the settled network
free energy cannot decrease along the ideal teacher search. Equality holds exactly when the tangent
derivative vanishes. This establishes monotone ascent, but it does not yet prove that every
stationary point is a global maximum.

### Teacher-coordinate form

Write $x_T=U_Ta$ with $\|a\|=1$. Since

$$
x_Tx_T^\top=U_Taa^\top U_T^\top,
$$

the tangent projector becomes

$$
P_{T,x}=U_T(I_r-aa^\top)U_T^\top.
$$

Using $\partial q/\partial x_T=\pi_{TS}N_SU_Ta$, multiplying the projected dynamics by $U_T^\top$,
and using
$A_S=U_T^\top N_SU_T$ gives

$$
\begin{aligned}
\tau_T\dot a
&=
\mu_T\pi_{TS}(I_r-aa^\top)A_Sa
\\
&=
\mu_T\pi_{TS}
\left[A_S-(a^\top A_Sa)I_r\right]a.
\end{aligned}
$$

This expression also makes norm preservation explicit:

$$
\frac{d}{dt}\|a\|^2
=
2a^\top\dot a
=
0.
$$

The subtraction of $(a^\top A_Sa)a$ removes the radial component of $A_Sa$ while retaining the
component that rotates $a$ toward directions with larger novelty.

### Modal dynamics and continuous-time power iteration

Let $v_k$ be an orthonormal eigenbasis of $A_S$,

$$
A_Sv_k=\nu_kv_k,
\qquad
\nu_1\ge\nu_2\ge\cdots\ge\nu_r\ge0,
$$

and expand

$$
a=\sum_kc_kv_k,
\qquad
\sum_kc_k^2=1.
$$

The current Rayleigh quotient is

$$
\bar\nu
:=
a^\top A_Sa
=
\sum_j\nu_jc_j^2.
$$

Projecting the coordinate dynamics onto $v_k$ yields

$$
\boxed{
\tau_T\dot c_k
=
\mu_T\pi_{TS}(\nu_k-\bar\nu)c_k.
}
$$

A component grows when its eigenvalue is above the current weighted average $\bar\nu$ and shrinks
when its eigenvalue is below that average. More directly, for two nonzero components,

$$
\frac{d}{dt}\log\left|\frac{c_k}{c_j}\right|
=
\frac{\mu_T\pi_{TS}}{\tau_T}(\nu_k-\nu_j).
$$

Hence components with larger eigenvalues grow exponentially relative to components with smaller
eigenvalues. This is the normalized continuous-time analogue of power iteration.

If the largest eigenvalue is simple and $c_1(0)\ne0$, all lower-eigenvalue components vanish
relative to $c_1$, and

$$
a(t)\longrightarrow
\operatorname{sign}(c_1(0))v_1.
$$

Consequently, $x_T(t)=U_Ta(t)$ approaches one of the two unit teacher states with maximal settled
network free energy. If the largest eigenvalue is repeated and the initial state has a nonzero
projection onto its eigenspace, the dynamics approach a unit vector in that top eigenspace rather
than selecting a unique direction.

Every eigenvector is a stationary point of the projected dynamics, not only the top eigenvector.
A lower eigenvector is unstable to any perturbation along a higher-eigenvalue direction, but an
exactly absent component remains absent in the deterministic equations. Therefore the convergence
claim requires a nonzero initial projection onto the top eigenspace. Noise can supply such a
component in a simulation, but that is an additional implementation feature rather than part of
the deterministic theorem.

### Why the teacher coupling must be negative

Let $\kappa$ denote the signed teacher-side interface coupling. This is called `pi_ST` in the
current code, but it is better interpreted as a coupling gain rather than a probabilistic
precision. With the convention

$$
\varepsilon_{TS}=x_S-x_T,
$$

write the raw interface contribution to the teacher dynamics as

$$
\tau_T\dot x_T\big|_{\mathrm{interface}}
=
\kappa\varepsilon_{TS}.
$$

At the settled student state,

$$
\frac{\partial q}{\partial x_T}=-\pi_{TS}\varepsilon_{TS}^*.
$$

Thus $\varepsilon_{TS}^*$ itself points toward decreasing settled network free energy, whereas
$-\varepsilon_{TS}^*$ points toward increasing it. To make the interface drive equal to the
desired unprojected ascent drive $\mu_T\,\partial q/\partial x_T$, we require

$$
\kappa\varepsilon_{TS}^*
=
-\mu_T\pi_{TS}\varepsilon_{TS}^*.
$$

For nonzero mismatch this gives

$$
\boxed{
\kappa=-\mu_T\pi_{TS}<0.
}
$$

The negative sign is therefore determined by the fact that the teacher maximizes the complete
network's settled free energy on its memory manifold. A positive coupling would move the teacher
toward the settled student response and would descend, rather than ascend, the mismatch
contribution.

The raw interface term supplies the ascent direction. Teacher recurrence must keep the activity
within $\mathcal U_T$, and normalization or radial projection must remove the component that changes
its norm. Together, these constraint mechanisms turn the raw drive into the projected ascent
$P_{T,x}\,\partial q/\partial x_T$ analyzed above.

The magnitude $|\kappa|$ sets a search gain and is not fixed by the sign argument. It remains
limited by stability and timescale-separation requirements. The special numerical equality
$\kappa=-\pi_{TS}$ corresponds to choosing $\mu_T=1$; it is not required by the worst-case
objective.

### What this section establishes

For fixed $W_S$, exactly settled student responses, and an exactly enforced teacher-memory sphere,
the analysis establishes that:

1. the settled interface mismatch provides the exact teacher derivative through the envelope
   identity;
2. projected teacher dynamics keep the state valid and increase $q$ monotonically;
3. with a nonzero initial top-eigenspace component, the dynamics approach the maximizer set
   identified in Section 7;
4. the teacher-side interface coupling must be negative under the convention
   $\varepsilon_{TS}=x_S-x_T$.

### What this section does not establish

The argument does not show that:

1. every initial state reaches a top direction—an exactly missing eigendirection remains missing
   without noise or another perturbation;
2. the maximizer is unique when the largest eigenvalue is degenerate;
3. the softly constrained simulated teacher exactly follows the hard projected flow;
4. ascent remains monotone when student inference, teacher search, and weight plasticity occur on
   comparable timescales;
5. the selected direction must be a named episodic pattern—it may be any linear combination in the
   teacher memory space;
6. teacher ascent itself reduces $\mathcal{F}_{\max}$—it exposes the current worst direction, while
   the later student plasticity step is what reduces the worst-case free-energy envelope.

---

## 12. The student state implements the inner minimization

For each student coordinate $(x_S)_i$, let the dynamics move opposite to its ordinary derivative:

$$
\tau_S\dot{(x_S)_i}
=
-\frac{\partial F}{\partial (x_S)_i}.
$$

Because $F_T$ is independent of $x_S$, direct differentiation of the complete network free energy
with respect to coordinate $(x_S)_i$ gives

$$
\frac{\partial F}{\partial (x_S)_i}
=
\pi_{TS}(\varepsilon_{TS})_i
+
\pi_S\bigl[M_S^\top\varepsilon_S\bigr]_i.
$$

Substitution gives

$$
\tau_S\dot{(x_S)_i}
=
-\pi_{TS}(\varepsilon_{TS})_i
-
\pi_S\bigl[M_S^\top\varepsilon_S\bigr]_i.
$$

Collecting all $d$ coordinate equations into one vector equation gives

$$
\boxed{
\tau_S\dot x_S
=
-\pi_{TS}\varepsilon_{TS}
-
\pi_SM_S^\top\varepsilon_S.
}
$$

The interface term pulls the student toward the teacher state. The recurrent transpose term pulls
the student toward states supported by its current memory. The transpose is not separately
postulated: it follows from differentiating the squared recurrent error.

Because $F$ is strictly convex in $x_S$, this state dynamics has one equilibrium for fixed
$x_T,W_S$, and that equilibrium is the global minimizer used to define $q$.

---

## 13. The weight derivative for one exposed teacher state

We first differentiate with respect to one scalar weight rather than introducing a matrix gradient.
Let

$$
w_{ij}:=(W_S)_{ij}.
$$

The student recurrent error has components

$$
(\varepsilon_S)_k
=
(x_S)_k
-
\sum_\ell (W_S)_{k\ell}(x_S)_\ell.
$$

Holding $x_S$ fixed while taking the partial derivative gives

$$
\frac{\partial(\varepsilon_S)_k}{\partial w_{ij}}
=
-\delta_{ki}(x_S)_j,
$$

where $\delta_{ki}=1$ if $k=i$ and $0$ otherwise. The teacher and mismatch terms of $F$ do not
depend directly on $w_{ij}$. Therefore

$$
\begin{aligned}
\frac{\partial F}{\partial w_{ij}}
&=
\pi_S\sum_k
(\varepsilon_S)_k
\frac{\partial(\varepsilon_S)_k}{\partial w_{ij}}
\\
&=
-\pi_S(\varepsilon_S)_i(x_S)_j.
\end{aligned}
$$

Now hold the teacher state fixed and recall

$$
q(x_T;W_S)
=
F\!\left(x_T,x_S^*(x_T,W_S);W_T,W_S\right).
$$

Changing $w_{ij}$ changes the optimized value directly and also changes the settled state
$x_S^*$. The ordinary chain rule gives

$$
\frac{\partial q}{\partial w_{ij}}
=
\left.
\left(\frac{\partial F}{\partial x_S}\right)^\top
\right|_{x_S^*}
\frac{\partial x_S^*}{\partial w_{ij}}
+
\left.\frac{\partial F}{\partial w_{ij}}\right|_{x_S^*}.
$$

The first term vanishes because the settled state is the unconstrained minimizer:

$$
\left.\frac{\partial F}{\partial x_S}\right|_{x_S^*}=0.
$$

This is the first envelope step. It leaves only the direct weight derivative:

$$
\boxed{
\frac{\partial q}{\partial w_{ij}}
=
\left.\frac{\partial F}{\partial w_{ij}}\right|_{x_S^*}
=
-\pi_S(\varepsilon_S^*)_i(x_S^*)_j.
}
$$

The local plasticity rule for the same synapse is

$$
\boxed{
\dot w_{ij}
=
\eta\pi_S(\varepsilon_S^*)_i(x_S^*)_j
=
-\eta\frac{\partial q}{\partial w_{ij}}.
}
$$

Thus every weight moves opposite to its derivative of the settled network free energy for the
currently expressed teacher state. Collecting the scalar plasticity equations gives the familiar
matrix rule

$$
\boxed{
\dot W_S
=
\eta\pi_S\varepsilon_S^*x_S^{*\top}.
}
$$

The remaining question is why, after teacher selection, this is descent on the maximum
$\mathcal{F}_{\max}$ rather than only on one arbitrarily chosen $q$.

---

## 14. Two envelope steps and descent of the maximum

This section derives the weight derivative of the nested optimized value directly from the ordinary
chain rule. Assume that the largest eigenvalue of $A_S$ is simple. The maximizing teacher line is
then unique, although its two antipodal representatives $x_T^*$ and $-x_T^*$ remain. Their sign
ambiguity is handled explicitly below.

### Work with one scalar weight

Choose one arbitrary student weight

$$
w:=(W_S)_{ij},
$$

and hold all other weights fixed. Proving the derivative identity for every such $w$ determines the
derivative with respect to the complete matrix $W_S$.

For fixed $x_T$ and $w$, write the unique settled student response as

$$
x_S^*=x_S^*(x_T,w).
$$

The settled network free energy is

$$
q(x_T,w)
=
F\!\left(x_T,x_S^*(x_T,w);W_T,w\right).
$$

### First envelope step: eliminate the optimized student state

Differentiate $q$ with respect to $w$ while holding $x_T$ fixed. The complete chain rule is

$$
\frac{\partial q(x_T,w)}{\partial w}
=
\left.
\left(\frac{\partial F}{\partial x_S}\right)^\top
\right|_{x_S^*}
\frac{\partial x_S^*(x_T,w)}{\partial w}
+
\left.\frac{\partial F}{\partial w}\right|_{x_S^*}.
$$

Because $x_S^*$ is the unconstrained minimizer,

$$
\left.\frac{\partial F}{\partial x_S}\right|_{x_S^*}=0.
$$

The indirect term through the movement of $x_S^*$ therefore vanishes:

$$
\boxed{
\frac{\partial q(x_T,w)}{\partial w}
=
\left.\frac{\partial F}{\partial w}\right|_{x_S^*}.
}
$$

Although $x_S^*$ depends on $w$, that dependence makes no first-order contribution to the optimized
value because it is multiplied by the zero derivative with respect to the optimized variable.

We will also need the derivative with respect to $x_T$. The same chain rule gives

$$
\frac{\partial q}{\partial x_T}
=
\left(\frac{\partial x_S^*}{\partial x_T}\right)^\top
\left.\frac{\partial F}{\partial x_S}\right|_{x_S^*}
+
\left.\frac{\partial F}{\partial x_T}\right|_{x_S^*}.
$$

The first term again vanishes, so

$$
\boxed{
\frac{\partial q}{\partial x_T}
=
\left.\frac{\partial F}{\partial x_T}\right|_{x_S^*}.
}
$$

### Second envelope step: eliminate the optimized teacher state

Because the top eigenvalue is simple, one can locally choose one sign of the maximizing teacher
state continuously and differentiably as the weight changes. Denote that local choice by
$x_T^*(w)$. Then

$$
\mathcal{F}_{\max}(w)
=
q(x_T^*(w),w).
$$

Differentiate using the ordinary chain rule:

$$
\frac{d\mathcal{F}_{\max}(w)}{dw}
=
\left.
\left(\frac{\partial q}{\partial x_T}\right)^\top
\right|_{x_T^*(w)}
\frac{dx_T^*(w)}{dw}
+
\left.\frac{\partial q}{\partial w}\right|_{x_T^*(w)}.
$$

The teacher constraint set $\mathbb S_T$ is fixed because $W_T$ is frozen. Thus $x_T^*(w)$
remains in the teacher memory space and remains unit norm as $w$ changes. Differentiating the norm
constraint gives

$$
\frac{d}{dw}\left(x_T^{*\top}x_T^*\right)
=
2x_T^{*\top}\frac{dx_T^*}{dw}
=
0.
$$

Therefore $dx_T^*/dw$ is tangent to the teacher-memory sphere. Because $x_T^*$ is a constrained
maximum of $q$, $q$ has zero first-order change along every allowed tangent motion at that point. In
particular,

$$
\boxed{
\left.
\left(\frac{\partial q}{\partial x_T}\right)^\top
\right|_{x_T^*(w)}
\frac{dx_T^*(w)}{dw}
=
0.
}
$$

This statement does not require the full derivative $\partial q/\partial x_T$ to be zero. At a
constrained maximum on a sphere, the derivative can have a radial component. Only its first-order
effect along allowed tangent directions must vanish.

The second envelope identity is therefore

$$
\boxed{
\frac{d\mathcal{F}_{\max}(w)}{dw}
=
\left.\frac{\partial q}{\partial w}\right|_{x_T^*(w)}.
}
$$

### The two envelope steps in one complete chain rule

Writing out the complete nesting gives

$$
\boxed{
\mathcal{F}_{\max}(w)
=
F\!\left(
x_T^*(w),
x_S^*(x_T^*(w),w);
W_T,w
\right).
}
$$

The settled student state depends on $w$ both directly and indirectly through the maximizing
teacher state:

$$
\frac{dx_S^*}{dw}
=
\frac{\partial x_S^*}{\partial x_T}
\frac{dx_T^*}{dw}
+
\frac{\partial x_S^*}{\partial w}.
$$

The complete chain rule is consequently

$$
\begin{aligned}
\frac{d\mathcal{F}_{\max}}{dw}
&=
\left(\frac{\partial F}{\partial x_S}\right)^\top
\left(
\frac{\partial x_S^*}{\partial x_T}
\frac{dx_T^*}{dw}
+
\frac{\partial x_S^*}{\partial w}
\right)
\\
&\quad+
\left(\frac{\partial F}{\partial x_T}\right)^\top
\frac{dx_T^*}{dw}
+
\frac{\partial F}{\partial w}.
\end{aligned}
$$

All derivatives of $F$ in this expression are evaluated at

$$
x_T=x_T^*(w),
\qquad
x_S=x_S^*(x_T^*(w),w).
$$

The first indirect term vanishes because

$$
\left.\frac{\partial F}{\partial x_S}\right|_{x_S^*}=0.
$$

For the second indirect term, the first envelope identity established

$$
\left.\frac{\partial F}{\partial x_T}\right|_{x_S^*}
=
\frac{\partial q}{\partial x_T}.
$$

The second envelope step then gives

$$
\left(\frac{\partial F}{\partial x_T}\right)^\top
\frac{dx_T^*}{dw}
=
\left(\frac{\partial q}{\partial x_T}\right)^\top
\frac{dx_T^*}{dw}
=
0.
$$

The two indirect terms thus vanish for different reasons:

1. $x_S^*$ is an unconstrained minimum, so $\partial F/\partial x_S=0$;
2. $x_T^*$ is a constrained maximum, so $q$ has zero first-order change along the tangent movement
   $dx_T^*/dw$.

Only the direct weight derivative remains:

$$
\boxed{
\frac{d\mathcal{F}_{\max}}{dw}
=
\left.\frac{\partial q}{\partial w}\right|_{x_T^*}
=
\left.\frac{\partial F}{\partial w}\right|_{x_T^*,x_S^*}.
}
$$

### Exact treatment of the antipodal pair

The complete network free energy has the exact symmetry

$$
F(-x_T,-x_S;W_T,W_S)=F(x_T,x_S;W_T,W_S).
$$

Because the student minimizer is unique, this implies

$$
x_S^*(-x_T,W_S)=-x_S^*(x_T,W_S).
$$

Consequently,

$$
q(-x_T,W_S)=q(x_T,W_S).
$$

When the largest eigenvalue is simple, the only maximizing states are the antipodal pair

$$
x_{T,+}^*(w),
\qquad
x_{T,-}^*(w)=-x_{T,+}^*(w).
$$

Because $q(-x_T,w)=q(x_T,w)$ for every $w$, differentiating this identity with $x_T$ held fixed
gives

$$
\frac{\partial q}{\partial w}(-x_T,w)
=
\frac{\partial q}{\partial w}(x_T,w).
$$

The two maximizing representatives therefore give exactly the same derivative:

$$
\boxed{
\frac{d\mathcal{F}_{\max}}{dw}
=
\frac{\partial q}{\partial w}(x_{T,+}^*,w)
=
\frac{\partial q}{\partial w}(x_{T,-}^*,w).
}
$$

Thus the unavoidable sign ambiguity creates no ambiguity in the plasticity rule.

### Assemble the scalar derivatives into the weight gradient

Section 13 showed that, for $w_{ij}=(W_S)_{ij}$,

$$
\frac{\partial F}{\partial w_{ij}}
=
-\pi_S(\varepsilon_S)_i(x_S)_j.
$$

The double-envelope identity therefore gives

$$
\boxed{
\frac{\partial\mathcal{F}_{\max}}{\partial (W_S)_{ij}}
=
-\pi_S(\varepsilon_S^*)_i(x_S^*)_j.
}
$$

Only now do we assemble these $d^2$ scalar derivatives into the matrix gradient, defined entry by
entry by

$$
\bigl[\nabla_{W_S}\mathcal{F}_{\max}\bigr]_{ij}
:=
\frac{\partial\mathcal{F}_{\max}}{\partial (W_S)_{ij}}.
$$

Hence

$$
\boxed{
\nabla_{W_S}\mathcal{F}_{\max}
=
-\pi_S\varepsilon_S^*x_S^{*\top}.
}
$$

The plasticity rule is therefore

$$
\boxed{
\dot W_S
=
\eta\pi_S\varepsilon_S^*x_S^{*\top}
=
-\eta\nabla_{W_S}\mathcal{F}_{\max}.
}
$$

This is gradient descent on the largest settled network free energy. In scalar form, every synapse
obeys

$$
\dot w_{ij}
=
-\eta\frac{\partial\mathcal{F}_{\max}}{\partial w_{ij}}.
$$

Along the slow idealized weight dynamics,

$$
\begin{aligned}
\frac{d\mathcal{F}_{\max}}{dt}
&=
\sum_{i,j}
\frac{\partial\mathcal{F}_{\max}}{\partial w_{ij}}
\dot w_{ij}
\\
&=
-\eta\sum_{i,j}
\left(
\frac{\partial\mathcal{F}_{\max}}{\partial w_{ij}}
\right)^2
\le0.
\end{aligned}
$$

Thus the largest settled network free energy cannot increase to first order in this generic,
equilibrated regime. It decreases strictly whenever at least one weight derivative is nonzero.

### Limitation: genuine multidirectional ties are excluded

The proof above assumes that the largest eigenvalue of $A_S$ is simple. It handles the antipodal
pair exactly, because the two signs give the same weight derivatives.

The proof does not cover a genuine multidirectional tie, meaning that the largest eigenvalue has
multiplicity greater than one. Distinct directions in the top eigenspace can then give different
weight derivatives, the selected maximizing state may fail to be a single differentiable function
of $w$, and $\mathcal{F}_{\max}$ may fail to have an ordinary gradient. No claim is made here that an
arbitrarily selected tied direction immediately descends the numerical maximum.

Small generic perturbations or noise typically split an exact eigenvalue degeneracy, but this
observation is not used as part of the theorem. The theorem is the simple-largest-eigenvalue result
proved above.

---

## 15. In what sense is the decrease "as fast as possible"?

The statement compares small plasticity events of the same amplitude. To avoid mixing the words
"infinitesimal" and "fixed amplitude," separate the size of an event from its direction. Write

$$
\Delta(W_S)_{ij}=\rho V_{ij},
$$

where $\rho>0$ is a common small amplitude and $V$ is a normalized direction satisfying

$$
\sum_{i,j}V_{ij}^2=1.
$$

Let

$$
d_{ij}
:=
\frac{\partial\mathcal{F}_{\max}}{\partial(W_S)_{ij}}.
$$

For small $\rho$, the first-order expansion is

$$
\mathcal{F}_{\max}(W_S+\rho V)
=
\mathcal{F}_{\max}(W_S)
+
\rho\sum_{i,j}d_{ij}V_{ij}
+
o(\rho).
$$

All candidate events use the same small $\rho$, so their first-order effects differ only through
$\sum_{i,j}d_{ij}V_{ij}$. The Cauchy–Schwarz inequality gives

$$
\sum_{i,j}d_{ij}V_{ij}
\ge
-\sqrt{\sum_{i,j}d_{ij}^2}
\sqrt{\sum_{i,j}V_{ij}^2}
=
-\sqrt{\sum_{i,j}d_{ij}^2}.
$$

Equality is attained by

$$
\boxed{
V_{ij}
=
-\frac{d_{ij}}{\sqrt{\sum_{k,\ell}d_{k\ell}^2}}
}
$$

when at least one $d_{ij}$ is nonzero. Thus the optimal direction places every weight change
opposite to its derivative, with the relative magnitudes determined by the derivative sizes. This
is exactly the negative-gradient direction assembled at the end of Section 14.

The network plasticity rule points in this direction. The learning rate sets its overall magnitude.

The precise defensible statement is therefore:

> Once fast cortical inference and hippocampal worst-case selection have equilibrated, and away
> from an exact multidirectional tie, among all sufficiently small plasticity events of the same
> amplitude, cortical plasticity produces the largest first-order decrease of the worst-case
> settled network free energy.

This does **not** prove:

- minimum wall-clock consolidation time among all learning algorithms;
- global optimality over finite weight changes;
- convergence to a global minimum when the weight objective is nonconvex;
- optimality when the three timescales are not separated.

It is nevertheless an exact local result and requires no empirical assumption relating discrepancy
magnitude to a separately postulated transfer rate.

### The no-autapse constraint

Let

$$
\mathcal C
=
\{W\in\mathbb R^{d\times d}:\operatorname{diag}(W)=0\}
$$

be the feasible linear subspace. Its orthogonal projector simply sets the diagonal of a matrix to
zero. Equivalently, restrict the comparison above to directions satisfying

$$
V_{ii}=0
$$

for every $i$. The same componentwise Cauchy–Schwarz argument shows that the best allowed direction
uses the negative derivatives of all off-diagonal weights and zero on the diagonal. After the scalar
derivatives are assembled, this update can be written as

$$
\dot W_S
=
-\eta P_{\mathcal C}\nabla_{W_S}\mathcal{F}_{\max}.
$$

The projection can reduce the available descent magnitude and can couple idealized modes, but the
update is still the direction with the largest allowed first-order decrease for a fixed small
plasticity amplitude.

---

## 16. The teacher recurrent term enforces the search domain

The hard minimax problem restricts the teacher to $\mathbb S_T$. The network implements the memory
part of that restriction softly with the teacher self-energy

$$
F_T(x_T)
=
\frac{\pi_T}{2}\|M_Tx_T\|^2.
$$

Its derivative vector is

$$
\frac{\partial F_T}{\partial x_T}
=
\pi_TM_T^\top M_Tx_T
=
\pi_TM_T^\top\varepsilon_T.
$$

Moving opposite to this derivative gives the teacher self-correction term

$$
-\frac{\partial F_T}{\partial x_T}
=
-\pi_TM_T^\top\varepsilon_T.
$$

Combining teacher self-correction and adversarial search gives

$$
\boxed{
\tau_T\dot x_T
=
-\pi_TM_T^\top\varepsilon_T
+
\kappa\varepsilon_{TS},
\qquad
\kappa<0,
}
$$

followed by fixed-norm projection or renormalization.

At the settled student state,

$$
\kappa\varepsilon_{TS}^*
=
|\kappa|N_Sx_T.
$$

Inside $\ker M_T$, this term searches for the teacher-supported state with the largest settled
network free energy. Outside the teacher memory space, it could instead amplify arbitrary
surprising activity. The recurrent correction must therefore
dominate normal to the memory space, and off-manifold occupancy must be checked in the full
dynamics. In the exact hard-constrained theorem this issue is removed by projection; in the actual
soft implementation it is an operating-regime condition rather than an identity.

The roles are distinct:

- teacher recurrence says **which states count as valid source memories**;
- normalization says **compare them at equal activity energy**;
- negative coupling says **move toward the valid state with largest settled network free energy**.

---

## 17. Why the timescale order matches the minimax nesting

The central object is

$$
\min_{W_S}
\max_{x_T\in\mathbb S_T}
\min_{x_S}
F(x_T,x_S;W_T,W_S).
$$

It contains three nested computations:

$$
\underbrace{\min_{x_S}}_{\text{settle network free energy}}
\quad\longrightarrow\quad
\underbrace{\max_{x_T\in\mathbb S_T}}_{\text{find its largest manifold value}}
\quad\longrightarrow\quad
\underbrace{\min_{W_S}}_{\text{reduce the worst-case envelope}}.
$$

The dynamical order

$$
\tau_S\ll\tau_T\ll1/\eta
$$

implements this nesting.

- If the student has not settled, the observed free energy includes transient inference error
  rather than the minimum attainable for that teacher state.
- If the teacher has not searched the settled landscape, the weight update need not address the
  current maximum.
- If the weights move too quickly, the landscape changes while the teacher is still evaluating it.

The hierarchy minimizes $\mathcal{F}_{\max}$ on the **slow learning timescale**. On the faster
selection timescale, the teacher intentionally ascends the settled free energy $q$ in order to
locate the manifold maximum that the weights will then reduce. The minimax computation therefore
contains descent in $x_S$, constrained ascent in $x_T$, and descent of the resulting envelope in
$W_S$.

---

## 18. Completion and self-extinction

Because $A_S$ is positive semidefinite,

$$
\mathcal{F}_{\max}
=
\frac{\pi_{TS}}{2}\lambda_{\max}(A_S)
\ge0.
$$

If every teacher direction is stored by the student, then $q=0$ on $\mathbb S_T$ and
$\mathcal{F}_{\max}=0$.

Conversely, suppose $\mathcal{F}_{\max}=0$. Since every $q\ge0$ and every $q\le\mathcal{F}_{\max}$, we have

$$
q(x_T;W_S)=0
$$

for every $x_T\in\mathbb S_T$. Section 4 then implies that every such $x_T$ lies in
$\ker M_S$. By homogeneity, the entire teacher memory space lies in the student memory space:

$$
\boxed{
\mathcal{F}_{\max}=0
\quad\Longleftrightarrow\quad
\mathcal U_T\subseteq\ker M_S.
}
$$

At completion,

$$
x_S^*=x_T,
\qquad
\varepsilon_{TS}^*=0,
\qquad
\varepsilon_S^*=0
$$

for every valid teacher state. Both the adversarial interface drive and the student plasticity
signal vanish. Selection and learning therefore extinguish themselves when the largest settled
network free energy reaches its minimum value of zero.

---

## 19. The complete derivation in one chain

Start from

$$
\boxed{
\min_{W_S}
\max_{\substack{x_T\in\ker M_T\\\|x_T\|=1}}
\min_{x_S}
\left[
\frac{\pi_T}{2}\|M_Tx_T\|^2
+
\frac{\pi_{TS}}2\|x_S-x_T\|^2
+
\frac{\pi_S}{2}\|M_Sx_S\|^2
\right].
}
$$

Then:

1. The bracketed quantity is the complete network free energy $F=F_T+F_S$.
2. $\ker M_T$ defines the teacher's valid memory content and makes $F_T=0$ throughout the
   maximization domain.
3. Fixed norm makes candidate states comparable at equal activity energy.
4. The inner minimization finds the smallest network free energy attainable for one valid teacher
   state, equivalently the student's best supported reconstruction.
5. In the student's orthonormal eigenbasis, the inner problem separates into scalar costs, each
   with one unique minimum; reassembling them gives the unique settled student state.
6. The settled network free energy is

   $$
   q(x_T;W_S)
   =
   \frac{\pi_{TS}}2x_T^\top N_Sx_T.
   $$

7. The largest settled network free energy over teacher content is

   $$
   \mathcal{F}_{\max}
   =
   \frac{\pi_{TS}}{2}
   \lambda_{\max}(U_T^\top N_SU_T).
   $$

8. $\mathcal{F}_{\max}$ uniformly bounds both settled cortical mismatch and cortical recurrent error.
9. Every $L$-Lipschitz downstream readout inherits a bound of size

   $$
   L\sqrt{\frac{2\mathcal{F}_{\max}}{\pi_{TS}}}.
   $$

10. The teacher envelope derivative is

   $$
   \frac{\partial q}{\partial x_T}=-\pi_{TS}\varepsilon_{TS}^*.
   $$

11. Teacher maximization therefore requires

    $$
    \kappa<0.
    $$

12. Student state descent gives

    $$
    \tau_S\dot x_S
    =
    -\pi_{TS}\varepsilon_{TS}
    -\pi_SM_S^\top\varepsilon_S.
    $$

13. For every scalar weight $w_{ij}=(W_S)_{ij}$, the first envelope step gives

    $$
    \frac{\partial q}{\partial w_{ij}}
    =
    -\pi_S(\varepsilon_S^*)_i(x_S^*)_j.
    $$

14. At a simple top eigenvalue, the second envelope step gives

    $$
    \frac{\partial\mathcal{F}_{\max}}{\partial w_{ij}}
    =
    \frac{\partial q}{\partial w_{ij}}(x_T^*;W_S).
    $$

15. The antipodal maximizers $x_T^*$ and $-x_T^*$ give exactly the same scalar weight derivatives.
16. Assembling the scalar derivatives gives

    $$
    \nabla_{W_S}\mathcal{F}_{\max}
    =
    -\pi_S\varepsilon_S^*x_S^{*\top},
    $$

    so the local plasticity rule is negative-gradient descent on the worst-case free-energy
    envelope.
17. A componentwise Cauchy–Schwarz argument proves that, among sufficiently small plasticity events
    of the same amplitude, this direction gives the greatest first-order decrease.
18. Teacher recurrent correction, normalization and timescale separation implement the constraint
    and nesting approximately in the full network.

The resulting architecture is

$$
\boxed{
\tau_T\dot x_T
=
-\pi_TM_T^\top\varepsilon_T
+
\kappa\varepsilon_{TS},
\qquad
\kappa<0,
}
$$

$$
\boxed{
\tau_S\dot x_S
=
-\pi_{TS}\varepsilon_{TS}
-
\pi_SM_S^\top\varepsilon_S,
}
$$

$$
\boxed{
\dot W_S
=
\eta\pi_S\varepsilon_Sx_S^\top,
}
$$

with zero-diagonal projection when autapses are excluded.

---

## 20. Why minimize the largest settled free energy while hippocampus stores the information?

The computational purpose of consolidation is not merely to prevent information from existing
nowhere. It is to remove dependence on a temporary source.

If hippocampal storage were permanent, unlimited, costless to query, immune to interference and
always available to every downstream computation, there would be little reason to transfer its
content. The motivation for systems consolidation begins precisely because these conditions are not
assumed.

### 20.1 Hippocampal dependence is a functional liability

As long as some content cannot be reconstructed and sustained cortically, successful use remains
conditional on several events:

- the hippocampal trace must still be preserved;
- the hippocampus must be available at retrieval;
- hippocampo-cortical communication must succeed;
- later encoding must not have made the trace inaccessible;
- the relevant cortical systems must wait for or coordinate with the temporary store.

Cortical acquisition removes these dependencies. The largest settled network free energy therefore
measures not whether information exists somewhere, but how large the remaining dependence can be
in the worst case. Its reconstruction interpretation follows because only the interface and student
self-energy terms remain on the teacher's zero-free-energy manifold.

### 20.2 The largest settled free energy is the remaining bottleneck

Suppose most components of a memory are cortically available but one component is not. Average
error may already be small, yet a future computation requiring that missing component remains
hippocampus-dependent. Complete independence is achieved only when every relevant component can be
reconstructed.

The maximum is therefore a natural progress variable:

$$
\text{the least-transferred component sets the remaining dependency.}
$$

Reducing the largest settled free energy first is a form of weakest-link protection rather than a
claim that all biological goals are literally minimax.

### 20.3 Future queries are unknown

If the brain knew exactly which feature would matter later, it could preferentially consolidate
that feature. Usually the future query is uncertain. An average objective can achieve good mean
performance while leaving a large blind spot. The worst-case objective instead provides the
uniform guarantee

> Whatever hippocampally represented component is needed later, its settled cortical
> reconstruction error is bounded.

The guarantee becomes functionally meaningful through the Lipschitz result: whatever downstream
readout is later applied, if its sensitivity is bounded, its output error is bounded too.

### 20.4 The deadline for transfer is uncertain

The system generally does not know when a memory will next be needed, when competing encoding will
arrive, when the source trace will weaken, or when hippocampal access will be unavailable. Rapid
local tightening of the largest bound is a robust response to that uncertain deadline. The proof
does not claim globally minimum elapsed consolidation time; it shows the strongest infinitesimal
tightening available under the model's local synaptic-change geometry.

### 20.5 A fast store must remain available for new encoding

A fast-learning source is useful because it can capture new experiences quickly. Retaining every
detail indefinitely can create capacity pressure and interference with later encoding. Transferring
content to a slower distributed substrate allows the fast store to continue serving future
experience. The teacher-supported state with the largest settled network free energy is the
component that still prevents the corresponding content from becoming independent of that source.

This is a functional motivation, not a claim that the present linear model contains an explicit
finite hippocampal-capacity variable. Adding such turnover would be a separate mechanistic
extension.

### 20.6 Cortical storage enables autonomous integration and use

Information being present in hippocampus does not imply that it is already embedded in the cortical
substrates that combine it with existing knowledge, support distributed predictions, or drive
behavior without hippocampal supervision. Cortical reconstruction and recurrent support place the
content in the recipient's own dynamical memory system. The self-consistency term in $q$ is
important here: merely copying a transient cortical activity pattern is not sufficient if cortical
recurrence cannot sustain it.

### 20.7 The objective has a local circuit implementation

The brain need not explicitly calculate $\mathcal{F}_{\max}$, enumerate all memories, or compare a table
of error values.

In the model:

1. the student reconstruction cancels teacher activity that is already shared;
2. poorly reconstructed teacher-supported activity leaves a larger interface residual;
3. negative feedback converts that residual into ascent within the teacher memory set;
4. recurrent competition and normalization make the largest unresolved direction dominate;
5. cortical plasticity reduces the exposed settled free energy;
6. as that direction becomes supported, its residual disappears and another direction can dominate.

The global-looking maximum is therefore realized by local dynamics, not by a homunculus selecting
which memory to replay.

### 20.8 Why the teacher-memory restriction is load-bearing

Pure worst-case optimization over all neural states would prioritize arbitrary noise, unreachable
patterns or adversarial outliers. The maximization is meaningful only over states the teacher
actually supports. Teacher recurrent correction is therefore part of the normative interpretation,
not only a stability device.

For a nonlinear or attractor-based teacher, the natural generalization is

$$
\mathcal{F}_{\max}
=
\max_{x_T\in\mathcal M_T}q(x_T;W_S),
$$

where $\mathcal M_T$ is the reachable set of valid teacher memories rather than the complete linear
sphere. Salience, confidence or expected future relevance could later enter through

$$
\mathcal{F}_{\max,w}
=
\max_{x_T\in\mathcal M_T}w(x_T)q(x_T;W_S).
$$

The unweighted linear sphere is the simplest case in which every valid, equal-energy teacher
direction is treated as potentially important.

---

## 21. Exact claims, assumptions and limitations

### Proved analytically in the hard-constrained, timescale-separated linear model

1. On the teacher memory sphere, $F_T=0$ and the complete network free energy satisfies $F=F_S$.
2. The student has a unique settled best response for every teacher state.
3. $q(x_T;W_S)=\min_{x_S}F$ is the complete network's settled free energy for one
   teacher-supported state, and it is zero exactly when the student stores that state.
4. The largest settled network free energy is

   $$
   \mathcal{F}_{\max}
   =
   \max_{x_T\in\mathbb S_T}\min_{x_S}F
   =
   \frac{\pi_{TS}}{2}
   \lambda_{\max}(U_T^\top N_SU_T).
   $$

5. $\mathcal{F}_{\max}$ uniformly bounds settled cortical reconstruction error and recurrent
   self-inconsistency.
6. An $L$-Lipschitz readout inherits the corresponding uniform functional-error bound.
7. Teacher ascent on the settled network free energy requires a negative interface coupling
   $\kappa<0$.
8. Student state dynamics implement the inner free-energy minimization.
9. Student plasticity descends the selected settled free energy.
10. At a simple top discrepancy eigenvalue, student plasticity is exact gradient descent on
   $\mathcal{F}_{\max}$.
11. Among sufficiently small plasticity events of the same amplitude, its direction gives the
    largest possible first-order decrease of $\mathcal{F}_{\max}$.
12. With a zero-diagonal constraint, the projected update is the steepest feasible first-order
    descent direction.
13. $\mathcal{F}_{\max}=0$ exactly when all teacher memory content is stored by the student.

### Assumptions or operating-regime requirements

1. Student inference settles before teacher selection, and teacher selection settles before
   appreciable weight change.
2. Fixed activity is an adequate equal-resource comparison across teacher states.
3. The teacher's soft recurrent constraint keeps the simulated state close to the hard memory
   domain used by the theorem.
4. The differentiable weight-descent theorem assumes a simple largest discrepancy eigenvalue and
   excludes exact multidirectional ties. Small generic perturbations often split such degeneracies,
   but that observation is not used in the theorem.
5. A downstream functional interpretation requires an explicit shared or aligned readout with
   bounded sensitivity on the relevant states.

### Claims deliberately not made

1. That every variable descends $F$. Student inference minimizes $F$ over $x_S$, teacher selection
   maximizes its settled value over $x_T$, and student plasticity minimizes the resulting envelope
   over $W_S$.
2. Global minimization time among all possible learning controllers.
3. Global convergence for arbitrary initial weights or nonlinear networks.
4. Exact monotone decrease of $\mathcal{F}_{\max}$ during the fast teacher search phase.
5. Immediate strict decrease of $\mathcal{F}_{\max}$ from one arbitrary update at an exact
   multidirectional tie.
6. That the linear teacher sphere represents discrete named episodes; it represents their stored
   subspace.
7. That every downstream function is Lipschitz with a small constant, or that a discontinuous
   decision is protected without a margin.
8. That the brain explicitly represents the scalar $\mathcal{F}_{\max}$. The claim is that local circuit
   dynamics implement the corresponding nested optimization.

The central result can be summarized as follows:

> Consolidation minimizes the largest settled free energy of the complete network over everything
> represented on the teacher's zero-free-energy memory manifold. Fast cortical inference minimizes
> network free energy for each candidate state, negative cortical-to-hippocampal coupling drives the
> teacher toward the manifold state where that settled free energy is largest, and cortical
> plasticity follows the locally steepest feasible direction for reducing the resulting envelope.
> Because the teacher contribution vanishes on the manifold, the same quantity uniformly controls
> cortical reconstruction, recurrent self-consistency and every sufficiently stable downstream
> readout.
