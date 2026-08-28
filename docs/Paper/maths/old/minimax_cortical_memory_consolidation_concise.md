# A concise proof of prioritized memory transfer

This note proves two local results for the linear, timescale-separated model. First, settled
teacher selection followed by recipient plasticity gives a gradient step on the largest settled
network free energy over the teacher memory manifold. Second, once the common component of two
correlated memories has been stored, selecting their unresolved residual reduces this worst-case
deficit faster than replaying either named memory.

## 1. Model, objective, and assumptions

Let the teacher and recipient states be \(x_T,x_S\in\mathbb R^d\), with recurrent weights
\(W_T,W_S\in\mathbb R^{d\times d}\). Write

$$
M_T=I-W_T,
\qquad
M_S=I-W_S,
$$

and define the recurrent and interface errors

$$
\varepsilon_T=M_Tx_T,
\qquad
\varepsilon_S=M_Sx_S,
\qquad
\varepsilon_{TS}=x_T-x_S.
$$

For positive coefficients \(\pi_T,\pi_{TS},\pi_S\), the complete network free energy is

$$
F(x_T,x_S;W_T,W_S)
=
\frac{\pi_T}{2}\|M_Tx_T\|^2
+
\frac{\pi_{TS}}2\|x_T-x_S\|^2
+
\frac{\pi_S}{2}\|M_Sx_S\|^2.
\tag{1}
$$

The teacher weights are fixed. Its memory space and equal-activity memory sphere are

$$
\mathcal U_T=\ker M_T,
\qquad
\mathbb S_T=\{x_T\in\mathcal U_T:\|x_T\|=1\}.
\tag{2}
$$

The named teacher memories are assumed to span \(\mathcal U_T\). Thus the search domain contains
exactly the states supported by the teacher, including their linear combinations. Since
\(M_Tx_T=0\) on \(\mathbb S_T\), the teacher term in \(F\) vanishes throughout the search.
The remaining value is still the free energy of the complete network evaluated on that domain.
It measures the unavoidable cost left in the recipient and the interface after the recipient has
made its best response. No comparison between \(W_T\) and \(W_S\) is required, since different
recurrent matrices may support the same memory states.

For a fixed teacher state, define the settled deficit

$$
q(x_T;W_S)
=
\min_{x_S}F(x_T,x_S;W_T,W_S).
\tag{3}
$$

In this note, *reconstruction deficit* means \(q\): the recipient's smallest joint reconstruction
and recurrent-consistency cost after its state has settled. It does not mean the mismatch term
\(\|x_T-x_S\|^2\) alone. The worst settled deficit over the teacher memory sphere is

$$
\boxed{
\mathcal D_{\max}(W_S;W_T)
=
\max_{x_T\in\mathbb S_T}q(x_T;W_S)
=
\max_{x_T\in\mathbb S_T}\min_{x_S}F(x_T,x_S;W_T,W_S).
}
\tag{4}
$$

The corresponding consolidation objective is \(\min_{W_S}\mathcal D_{\max}\). We use the following
standing assumptions.

1. The state equations are linear, all three coefficients in Eq. (1) are positive, and \(W_T\)
   remains fixed during consolidation.
2. A network stores a state when its recurrent error vanishes. The named teacher memories span
   exactly \(\mathcal U_T=\ker M_T\).
3. Candidate teacher states have unit norm. This prevents the maximization from raising a
   quadratic cost by increasing activity amplitude instead of changing direction.
4. The mathematical teacher search is restricted exactly to \(\mathbb S_T\). The recurrent
   teacher correction and renormalization used by the simulated network are a soft implementation
   of this constraint, not part of the exact result.

The dynamical result also assumes the timescale order

$$
\tau_S\ll\tau_T\ll1/\eta.
\tag{5}
$$

For nearly fixed \(x_T,W_S\), the recipient state first reaches its unique equilibrium. For nearly
fixed \(W_S\), the teacher then reaches a maximum of the settled deficit. Only then does one small
plastic event change \(W_S\). The proof concerns this settle, select, update limit. It first treats
all entries of \(W_S\) as independent parameters. The no-autapse case follows by projecting the
gradient onto the zero-diagonal weight subspace. The additional regularity assumptions needed for
teacher convergence and differentiation of the maximum are stated with the theorem.

## 2. Lemma: settled recipient and worst-case deficit

**Lemma 1.** Let

$$
S_S=M_S^\top M_S,
\qquad
N_S=\pi_SS_S(\pi_{TS}I+\pi_SS_S)^{-1}.
\tag{6}
$$

For every \(x_T\in\mathbb S_T\), the unique minimizing recipient state is

$$
\boxed{x_S^*(x_T)=(I-N_S)x_T.}
\tag{7}
$$

The settled deficit is

$$
\boxed{q(x_T;W_S)=\frac{\pi_{TS}}2x_T^\top N_Sx_T,}
\tag{8}
$$

and

$$
q(x_T;W_S)=0
\quad\Longleftrightarrow\quad
M_Sx_T=0.
\tag{9}
$$

Let \(U_T\in\mathbb R^{d\times r}\) have orthonormal columns spanning \(\mathcal U_T\), and define

$$
A_S=U_T^\top N_SU_T.
\tag{10}
$$

Then

$$
\boxed{
\mathcal D_{\max}(W_S;W_T)
=
\frac{\pi_{TS}}2\lambda_{\max}(A_S).
}
\tag{11}
$$

**Proof.** On \(\mathbb S_T\), minimizing \(F\) over \(x_S\) is equivalent to minimizing

$$
F_S(x_S,x_T;W_S)
=
\frac{\pi_{TS}}2\|x_T-x_S\|^2
+
\frac{\pi_S}{2}\|M_Sx_S\|^2.
\tag{12}
$$

Its derivative and Hessian with respect to \(x_S\) are

$$
\nabla_{x_S}F_S
=
\pi_{TS}(x_S-x_T)+\pi_SS_Sx_S,
\qquad
H_S=\pi_{TS}I+\pi_SS_S.
\tag{13}
$$

For every nonzero \(y\),

$$
y^\top H_Sy
=
\pi_{TS}\|y\|^2+\pi_S\|M_Sy\|^2
>0.
$$

Hence \(F_S\) is strictly convex in \(x_S\) and has one global minimizer. Solving
\(\nabla_{x_S}F_S=0\) gives

$$
x_S^*
=
\pi_{TS}H_S^{-1}x_T
=
\left[I-\pi_SS_SH_S^{-1}\right]x_T
=
(I-N_S)x_T,
$$

which proves Eq. (7).

Expanding Eq. (12) as a quadratic in \(x_S\), then substituting
\(x_S^*=\pi_{TS}H_S^{-1}x_T\), gives

$$
\begin{aligned}
q(x_T;W_S)
&=
\frac{\pi_{TS}}2x_T^\top x_T
-
\frac{\pi_{TS}^2}{2}x_T^\top H_S^{-1}x_T
\\
&=
\frac{\pi_{TS}}2x_T^\top
\left[I-\pi_{TS}H_S^{-1}\right]x_T
\\
&=
\frac{\pi_{TS}}2x_T^\top N_Sx_T.
\end{aligned}
$$

To read this quadratic, diagonalize the symmetric positive semidefinite matrix \(S_S\). If
\(S_Su_k=\mu_ku_k\), then \(N_S\) has the same orthonormal eigenvectors and eigenvalues

$$
n(\mu_k)
=
\frac{\pi_S\mu_k}{\pi_{TS}+\pi_S\mu_k}
\in[0,1).
\tag{14}
$$

Thus \(N_S\) is symmetric and positive semidefinite, and

$$
\ker N_S=\ker S_S=\ker M_S.
$$

Equation (7) also gives the settled interface error directly:

$$
\varepsilon_{TS}^*=x_T-x_S^*=N_Sx_T.
$$

In a recipient eigenmode with squared recurrent error \(\mu_k\), the scalar \(n(\mu_k)\) is the
fraction of the teacher coordinate left unresolved after recipient settling. It is zero exactly
on a stored direction and increases monotonically with \(\mu_k\). Equation (8) sums those
directionwise deficits with their activity coefficients. Since a positive semidefinite quadratic
vanishes exactly on the kernel of its matrix, Eq. (9) follows.

Finally, every \(x_T\in\mathbb S_T\) has the form \(x_T=U_Ta\) with \(\|a\|=1\). Therefore

$$
q(U_Ta;W_S)
=
\frac{\pi_{TS}}2a^\top A_Sa.
$$

The matrix \(A_S\) is symmetric and positive semidefinite. Rayleigh-Ritz gives

$$
\max_{\|a\|=1}a^\top A_Sa=\lambda_{\max}(A_S),
$$

which proves Eq. (11). \(\square\)

The maximum in Eq. (4) is therefore attained. This also follows directly because \(q\) is
continuous and \(\mathbb S_T\) is compact. If the leading eigenvalue of \(A_S\) is simple, its unit
eigenvector \(v_1\) determines the two maximizers \(x_T^*=U_Tv_1\) and \(-x_T^*\). If it is
repeated, every unit vector in the leading eigenspace is a maximizer. The theorem uses the simple
case because the maximum is then differentiable with respect to the recipient weights.

The same object controls both terms in the settled recipient cost. For every
\(x_T\in\mathbb S_T\), nonnegativity of the two terms in Eq. (12) gives

$$
\|x_T-x_S^*(x_T)\|
\leq
\sqrt{\frac{2\mathcal D_{\max}}{\pi_{TS}}},
\qquad
\|M_Sx_S^*(x_T)\|
\leq
\sqrt{\frac{2\mathcal D_{\max}}{\pi_S}}.
$$

Thus reducing \(\mathcal D_{\max}\) reduces one uniform upper bound on reconstruction mismatch and
recipient recurrent inconsistency over the complete teacher memory sphere. This is the precise
sense in which the maximal settled network free energy is also a worst-case reconstruction
deficit.

## 3. Theorem: greedy descent of maximal settled free energy

Let

$$
P_{T,x}=U_TU_T^\top-x_Tx_T^\top
\tag{15}
$$

be the orthogonal projector onto the tangent space of \(\mathbb S_T\) at \(x_T\). Consider the
idealized state dynamics

$$
\tau_S\dot x_S=-\nabla_{x_S}F,
\qquad
\tau_T\dot x_T=\mu_TP_{T,x}\nabla_{x_T}q,
\qquad
\mu_T>0,
\tag{16}
$$

followed, after both states have settled, by the local recipient update

$$
\Delta W_S=\eta\pi_S\varepsilon_S^*x_S^{*\top}.
\tag{17}
$$

Here the stars denote the recipient equilibrium at the selected teacher state.

**Theorem 1.** Assume Eq. (5), exact projection onto \(\mathbb S_T\), and a nonzero initial
component along the leading eigendirection of \(A_S\). Assume also that the largest eigenvalue of
\(A_S\) is simple. Then the recipient state implements the inner minimization in Eq. (4), the
teacher state converges to the maximizing antipodal pair, and the plastic event in Eq. (17) is

$$
\boxed{
\Delta W_S=-\eta\nabla_{W_S}\mathcal D_{\max}.
}
\tag{18}
$$

Consequently,

$$
\boxed{
\mathcal D_{\max}(W_S+\Delta W_S)
=
\mathcal D_{\max}(W_S)
-
\eta\|\nabla_{W_S}\mathcal D_{\max}\|_F^2
+o(\eta).
}
\tag{19}
$$

The architecture therefore performs a greedy gradient step on the maximal settled network free
energy. Its direction gives the largest first-order decrease among weight changes with the same
Frobenius norm.

**Proof.** We follow the three timescales in Eq. (5).

*Recipient settling.* For fixed \(x_T,W_S\), Eqs. (13) and (16) give

$$
\tau_S\dot x_S=-H_S(x_S-x_S^*).
$$

Since \(H_S\) is positive definite, every eigenmode decays exponentially. Hence \(x_S\) converges
to the unique minimizer in Lemma 1 and implements \(\min_{x_S}F\).

More explicitly,

$$
x_S(t)-x_S^*
=
\exp\!\left(-\frac{H_St}{\tau_S}\right)
\bigl(x_S(0)-x_S^*\bigr).
$$

All eigenvalues of the matrix exponential decay because the eigenvalues of \(H_S\) are strictly
positive. This is why the inner minimum can be treated as completed before teacher selection in
the separated-timescale limit.

*Teacher selection.* The minimized value depends on \(x_T\) both directly and through
\(x_S^*(x_T)\). The chain rule gives

$$
\nabla_{x_T}q
=
\left(\frac{\partial x_S^*}{\partial x_T}\right)^\top
\underbrace{\nabla_{x_S}F\big|_{x_S^*}}_{0}
+
\nabla_{x_T}F\big|_{x_S^*}.
$$

The indirect term vanishes at the recipient minimum. On \(\mathbb S_T\), the teacher recurrent
term and its derivative are zero, so

$$
\boxed{
\nabla_{x_T}q
=
\pi_{TS}(x_T-x_S^*)
=
\pi_{TS}N_Sx_T.
}
\tag{20}
$$

This is the first envelope step. Along the projected teacher flow,

$$
\frac{dq}{dt}
=
\frac{\mu_T}{\tau_T}
\left\|P_{T,x}\nabla_{x_T}q\right\|^2
\geq0.
\tag{21}
$$

The projection preserves both membership in \(\mathcal U_T\) and unit norm. To identify the
limit, write \(x_T=U_Ta\), with \(\|a\|=1\). Equations (15), (16), and (20) give

$$
\tau_T\dot a
=
\mu_T\pi_{TS}(I-aa^\top)A_Sa.
\tag{22}
$$

Let \(A_Sv_k=\nu_kv_k\), where \(\nu_1>\nu_2\geq\cdots\), and write
\(a=\sum_kc_kv_k\). For nonzero components,

$$
\tau_T\dot c_k
=
\mu_T\pi_{TS}(\nu_k-\bar\nu)c_k,
\qquad
\bar\nu=a^\top A_Sa=\sum_j\nu_jc_j^2.
$$

Consequently,

$$
\frac{d}{dt}\log\left|\frac{c_k}{c_j}\right|
=
\frac{\mu_T\pi_{TS}}{\tau_T}(\nu_k-\nu_j).
\tag{23}
$$

Thus the leading component grows relative to every lower component. If \(c_1(0)\neq0\), then
\(a\) converges to \(\operatorname{sign}(c_1(0))v_1\). These two signs give the maximizing pair
identified by Lemma 1. Equation (23) also explains the initialization condition. A component that
is exactly zero stays zero under the deterministic flow. Noise can supply a missing component in a
simulation, but noise is not needed for the theorem once \(c_1(0)\neq0\).

The local interface form of the teacher search is

$$
\tau_T\dot x_T\big|_{\mathrm{interface}}
=
\kappa P_{T,x}(x_T-x_S^*).
$$

Comparison with Eqs. (16) and (20) fixes its gain:

$$
\boxed{\kappa=\mu_T\pi_{TS}>0.}
\tag{24}
$$

*Recipient plasticity.* First hold \(x_T\) fixed. A perturbation \(dW_S\) changes the recurrent
error by \(d\varepsilon_S=-dW_Sx_S\). At the settled state,

$$
dF
=
-\pi_S\varepsilon_S^{*\top}dW_Sx_S^*
=
\left\langle
-\pi_S\varepsilon_S^*x_S^{*\top},dW_S
\right\rangle_F.
$$

The derivative through \(x_S^*(W_S)\) again vanishes because \(\nabla_{x_S}F=0\). Therefore

$$
\nabla_{W_S}q(x_T;W_S)
=
-\pi_S\varepsilon_S^*x_S^{*\top}.
\tag{25}
$$

Now let \(x_T^*(W_S)\) be a maximizing teacher state. The simple leading eigenvalue makes its line
locally smooth. Differentiating \(\mathcal D_{\max}=q(x_T^*(W_S);W_S)\), the derivative through
\(x_T^*\) vanishes because the change in a constrained maximizer is tangent to \(\mathbb S_T\),
where the first-order derivative of \(q\) is zero. This is the second envelope step. The two
representatives \(x_T^*\) and \(-x_T^*\) cause no ambiguity: linearity gives
\(x_S^*(-x_T^*)=-x_S^*(x_T^*)\) and
\(\varepsilon_S^*(-x_T^*)=-\varepsilon_S^*(x_T^*)\), so their outer products in Eq. (25) agree.

To make the second step explicit, take one scalar weight \(w=(W_S)_{ij}\). The chain rule gives

$$
\frac{d\mathcal D_{\max}}{dw}
=
\left.\nabla_{x_T}q^\top\right|_{x_T^*}
\frac{dx_T^*}{dw}
+
\left.\frac{\partial q}{\partial w}\right|_{x_T^*}.
$$

The teacher constraint set does not change with \(w\) because \(W_T\) is fixed. Therefore
\(dx_T^*/dw\) is tangent to \(\mathbb S_T\). At the constrained maximum, the first term is zero.
Applying Eq. (25) to every weight leaves only the direct derivative and yields

$$
\boxed{
\nabla_{W_S}\mathcal D_{\max}
=
-\pi_S\varepsilon_S^*x_S^{*\top}.
}
\tag{26}
$$

Equations (17) and (26) prove Eq. (18). A first-order expansion then gives Eq. (19).
The remainder is \(o(\eta)\) because the simple-eigenvalue assumption makes
\(\mathcal D_{\max}\) differentiable in a neighborhood of the current weights. Equation (19) is a
statement about one sufficiently small event. It does not require the full weight objective to be
convex.

For completeness, take any perturbation \(\Delta W=\rho V\) with \(\|V\|_F=1\). Its first-order
effect is

$$
\mathcal D_{\max}(W_S+\rho V)-\mathcal D_{\max}(W_S)
=
\rho\langle\nabla\mathcal D_{\max},V\rangle_F+o(\rho).
$$

Cauchy-Schwarz gives

$$
\langle\nabla\mathcal D_{\max},V\rangle_F
\geq
-\|\nabla\mathcal D_{\max}\|_F,
$$

with equality only in the negative-gradient direction when the gradient is nonzero. This proves
the greedy first-order claim. \(\square\)

**No-autapse constraint.** If the feasible weights satisfy \(\operatorname{diag}(W_S)=0\), let
\(P_0\) zero the diagonal of a matrix. The feasible update is

$$
\Delta W_S=-\eta P_0\nabla_{W_S}\mathcal D_{\max}.
$$

It decreases the objective by
\(-\eta\|P_0\nabla_{W_S}\mathcal D_{\max}\|_F^2+o(\eta)\) and is the steepest first-order
direction within the zero-diagonal subspace. The projection can change the exact rate comparison
in the next corollary.

The equal-norm statement concerns directions in weight space. The actual event in Eq. (17) has
norm \(\eta\|\nabla\mathcal D_{\max}\|_F\), set by the local errors. Among all events with that
same norm, no other direction produces a larger first-order reduction. The correlation corollary
below asks a different experimental question. It holds the learning rate, event duration, and
teacher-state norm fixed while changing which teacher state drives the same plasticity rule.

## 4. Corollary: correlation-dependent local advantage

**Corollary 1.** Let two correlated unit teacher memories be

$$
m_+=cu+sv,
\qquad
m_-=cu-sv,
\tag{27}
$$

where \(u,v\) are orthonormal and

$$
c^2=\frac{1+\chi}{2},
\qquad
s^2=\frac{1-\chi}{2},
\qquad
0\leq\chi<1.
\tag{28}
$$

Then \(m_+^\top m_-=\chi\). Assume that these memories span \(\mathcal U_T\), the recipient stores
their common component but not their residual,

$$
M_Su=0,
\qquad
M_Sv\neq0,
\tag{29}
$$

and the unconstrained, exactly settled, infinitesimal regime of Theorem 1 holds. Worst-deficit
selection replays \(v\) or \(-v\). For equal-duration plastic events with the same learning rate
and unit teacher activity,

$$
\boxed{
\frac{-\dot{\mathcal D}_{\max}^{\mathrm{priority}}}
{-\dot{\mathcal D}_{\max}^{\mathrm{random}}}
=
\frac{2}{1-\chi}.
}
\tag{30}
$$

Here random replay selects either named memory with equal probability.

**Proof.** Since \(\ker N_S=\ker M_S\), Eq. (29) gives \(N_Su=0\). Symmetry of \(N_S\) then gives
\(u^\top N_Sv=0\). In the teacher basis \((u,v)\),

$$
A_S
=
\begin{pmatrix}
0&0\\
0&v^\top N_Sv
\end{pmatrix}.
$$

The lower-right entry is positive because \(v\notin\ker N_S\). Hence the unique maximizing line is
\(\{v,-v\}\).

Define the settled response and recurrent error for residual replay by

$$
z=x_S^*(v)=(I-N_S)v,
\qquad
e=M_Sz.
\tag{31}
$$

The vector \(e\) is nonzero. Otherwise the recipient stationarity equation would reduce to
\(\pi_{TS}(z-v)=0\), which would imply \(M_Sv=0\) and contradict Eq. (29).

The settled map \(I-N_S\) is linear and symmetric. It fixes \(u\), so

$$
u^\top z
=
u^\top(I-N_S)v
=
\bigl((I-N_S)u\bigr)^\top v
=
u^\top v
=0.
\tag{32}
$$

Residual replay produces

$$
\dot W_v=\eta\pi_Sez^\top.
\tag{33}
$$

For the named memories, linearity and Eq. (29) give

$$
x_S^*(m_\pm)=cu\pm sz,
\qquad
\varepsilon_S^*(m_\pm)=\pm se.
$$

Their plasticity events are therefore

$$
\boxed{
\dot W_\pm
=
\eta\pi_S
\left(s^2ez^\top\pm sc\,eu^\top\right).
}
\tag{34}
$$

At the maximizing residual direction, Theorem 1 gives

$$
\nabla_{W_S}\mathcal D_{\max}=-\pi_Sez^\top.
$$

The second term in Eq. (34) is orthogonal to this gradient because

$$
\langle ez^\top,eu^\top\rangle_F
=
\|e\|^2z^\top u
=0.
$$

It follows for either named memory that

$$
\dot{\mathcal D}_{\max}^{\mathrm{named}}
=
s^2\dot{\mathcal D}_{\max}^{\mathrm{priority}},
\tag{35}
$$

where

$$
\dot{\mathcal D}_{\max}^{\mathrm{priority}}
=
-\eta\|\nabla_{W_S}\mathcal D_{\max}\|_F^2<0.
$$

Both named memories have the same first-order effect, so uniform random selection has that effect
in expectation. Substituting \(s^2=(1-\chi)/2\) into Eq. (35) proves Eq. (30). \(\square\)

At \(\chi=0\), prioritized residual replay has twice the instantaneous reduction rate of named
replay in this partially learned state. As \(\chi\) approaches one, the residual coefficient in
each named memory approaches zero, while \(v\) remains a normalized teacher-supported direction.
The ratio therefore diverges. The case \(\chi=1\) is excluded because the two named memories then
coincide and no longer span the residual direction.

A useful endpoint check follows from Lemma 1. Since \(q\geq0\),

$$
\mathcal D_{\max}=0
\quad\Longleftrightarrow\quad
\mathcal U_T\subseteq\ker M_S.
$$

At this endpoint, every valid teacher state has \(x_S^*=x_T\),
\(\varepsilon_{TS}^*=0\), and \(\varepsilon_S^*=0\). Both the teacher search signal in Eq. (20)
and the plasticity signal in Eq. (17) vanish. This characterizes complete transfer, but it does
not prove that an arbitrary finite-step trajectory reaches the endpoint.

## 5. Scope of the result

The theorem establishes a local, greedy optimization result. It does not prove convergence of the
complete weight trajectory, minimum consolidation time, or a finite-step speed ratio. Equation
(30) compares instantaneous derivatives for equal-duration events with the same learning rate and
unit teacher activity. It does not compare updates after normalizing their Frobenius norms.

The exact derivation also requires linear dynamics, complete settling of both states before each
plastic event, a hard teacher-manifold constraint, and a simple leading eigenvalue. With an exact
multidirectional tie, \(\mathcal D_{\max}\) need not have an ordinary gradient. Finite timescale
separation and soft recurrent enforcement of the teacher manifold approximate the nested
optimization rather than reproduce it exactly. Nonlinear networks and finite plasticity events
require separate analysis. The no-autapse projection preserves steepest feasible descent but not
the exact factor in Eq. (30).
