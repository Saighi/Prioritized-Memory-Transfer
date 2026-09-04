# Prioritized memory transfer as a nested optimization

This note proves the local optimization claim in the same order as the network computation. The
fast student state first reaches its unique equilibrium. The teacher state then reaches the
largest settled deficit on its memory sphere. Once both states have converged, one student
plasticity event is a gradient step on that maximum. A final corollary compares this prioritized
event with random clamping of two correlated named memories.

## 1. Model and maximal settled cost

Let \(x_T,x_S\in\mathbb R^d\) be the teacher and student states, and let
\(W_T,W_S\in\mathbb R^{d\times d}\) be their recurrent weights. Define

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

The teacher weights remain fixed during consolidation. Its memory space and unit memory sphere
are

$$
\mathcal U_T=\ker M_T,
\qquad
\mathbb S_T
=
\{x_T\in\mathcal U_T:\|x_T\|=1\}.
\tag{2}
$$

The named teacher memories are assumed to span \(\mathcal U_T\). The unit-norm restriction makes
the teacher optimization a comparison between activity directions rather than amplitudes.
Because \(M_Tx_T=0\) on \(\mathbb S_T\), the teacher term in Eq. (1) vanishes on the optimization
domain.

For one teacher state, define the settled deficit

$$
q(x_T;W_S)
=
\min_{x_S}F(x_T,x_S;W_T,W_S).
\tag{3}
$$

The word deficit refers to this minimized joint cost. It contains the teacher-student mismatch
and the student recurrent inconsistency. It is not the mismatch term alone. The largest settled
deficit over the teacher memory sphere is

$$
\boxed{
\mathcal D_{\max}(W_S;W_T)
=
\max_{x_T\in\mathbb S_T}q(x_T;W_S)
=
\max_{x_T\in\mathbb S_T}\min_{x_S}
F(x_T,x_S;W_T,W_S).
}
\tag{4}
$$

The result below concerns the effect of one student-plasticity event on \(\mathcal D_{\max}\). It
does not assert that the weight dynamics attain a minimum of this quantity. The proof assumes
linear dynamics, positive coefficients, a hard teacher-memory constraint, and the timescale separation

$$
\tau_S\ll\tau_T\ll1/\eta.
\tag{5}
$$

Thus the student state settles while \(x_T,W_S\) are nearly fixed, the teacher state then
settles while \(W_S\) is nearly fixed, and plasticity acts last.

## 2. Lemma 1: student equilibrium

**Lemma 1.** Fix \(x_T\in\mathbb S_T\) and \(W_S\). Define

$$
S_S=M_S^\top M_S,
\qquad
H_S=\pi_{TS}I+\pi_SS_S,
\tag{6}
$$

Define the settled student response operator and its complementary residual operator:

$$
B_S
=
\pi_{TS}H_S^{-1},
\qquad
N_S
=
I-B_S
=
\pi_SS_SH_S^{-1}.
\tag{7}
$$

The student free energy has a unique minimizer

$$
\boxed{
x_S^*(x_T)
=
B_Sx_T
=
(I-N_S)x_T.
}
\tag{8}
$$

The student gradient flow converges to this minimizer:

$$
\tau_S\dot x_S=-\nabla_{x_S}F
\quad\Longrightarrow\quad
x_S(t)\longrightarrow x_S^*(x_T).
\tag{9}
$$

At equilibrium,

$$
\boxed{
q(x_T;W_S)
=
\frac{\pi_{TS}}2x_T^\top N_Sx_T,
}
\tag{10}
$$

and

$$
q(x_T;W_S)=0
\quad\Longleftrightarrow\quad
M_Sx_T=0.
\tag{11}
$$

**Proof.** On \(\mathbb S_T\), the teacher term is zero, so the dependence on \(x_S\) is

$$
F_S(x_S,x_T;W_S)
=
\frac{\pi_{TS}}2\|x_T-x_S\|^2
+
\frac{\pi_S}{2}\|M_Sx_S\|^2.
\tag{12}
$$

Its gradient and Hessian are

$$
\nabla_{x_S}F_S
=
\pi_{TS}(x_S-x_T)+\pi_SS_Sx_S,
\qquad
\nabla_{x_S}^2F_S=H_S.
\tag{13}
$$

For every nonzero \(y\),

$$
y^\top H_Sy
=
\pi_{TS}\|y\|^2+\pi_S\|M_Sy\|^2
>0.
$$

Thus \(H_S\) is positive definite. The function \(F_S\) is strictly convex in \(x_S\) and has a
unique global minimizer. Solving \(\nabla_{x_S}F_S=0\) gives

$$
H_Sx_S^*=\pi_{TS}x_T,
\qquad
x_S^*=\pi_{TS}H_S^{-1}x_T=B_Sx_T.
$$

The response operator \(B_S\) maps the teacher state to the settled student state. Define the part
left unmatched by this response as its complement:

$$
N_S=I-B_S.
$$

Then

$$
x_T-x_S^*
=
(I-B_S)x_T
=
N_Sx_T,
$$

so \(N_S\) maps the teacher state to the settled interface residual. Finally, using
\(H_S=\pi_{TS}I+\pi_SS_S\),

$$
N_S
=
I-\pi_{TS}H_S^{-1}
=
(H_S-\pi_{TS}I)H_S^{-1}
=
\pi_SS_SH_S^{-1},
$$

which verifies Eq. (7) and proves Eq. (8).

The student dynamics are

$$
\tau_S\dot x_S
=
-H_Sx_S+\pi_{TS}x_T
=
-H_S(x_S-x_S^*).
$$

Therefore

$$
x_S(t)-x_S^*
=
\exp\!\left(-\frac{H_St}{\tau_S}\right)
\bigl(x_S(0)-x_S^*\bigr).
$$

Every eigenvalue of \(H_S\) is positive, so every mode in the matrix exponential decays. This
proves Eq. (9).

Expanding Eq. (12) as a quadratic in \(x_S\) gives

$$
F_S
=
\frac{\pi_{TS}}2x_T^\top x_T
-\pi_{TS}x_T^\top x_S
+\frac12x_S^\top H_Sx_S.
$$

At the minimum, \(H_Sx_S^*=\pi_{TS}x_T\). Therefore

$$
\begin{aligned}
q(x_T;W_S)
&=
\frac{\pi_{TS}}2x_T^\top x_T
-\pi_{TS}x_T^\top x_S^*
+\frac{\pi_{TS}}2x_S^{*\top}x_T
\\
&=
\frac{\pi_{TS}}2x_T^\top(x_T-x_S^*)
\\
&=
\frac{\pi_{TS}}2x_T^\top(I-B_S)x_T
\\
&=
\frac{\pi_{TS}}2x_T^\top N_Sx_T.
\end{aligned}
$$

This calculation first identifies \(B_Sx_T=x_S^*\) as the settled student response and then
identifies \(N_Sx_T=(I-B_S)x_T=x_T-x_S^*\) as the part left unmatched.

To determine the kernel of \(N_S\), diagonalize the symmetric positive semidefinite matrix
\(S_S\). If \(S_Su_k=\mu_ku_k\), then \(N_S\) has the same orthonormal eigenvectors and the
eigenvalues

$$
n(\mu_k)
=
\frac{\pi_S\mu_k}{\pi_{TS}+\pi_S\mu_k}
\in[0,1).
\tag{14}
$$

Hence \(N_S\) is symmetric and positive semidefinite, with

$$
\ker N_S=\ker S_S=\ker M_S.
$$

Equation (10) is a positive semidefinite quadratic, so it vanishes exactly when
\(x_T\in\ker N_S\). This proves Eq. (11). \(\square\)

Lemma 1 completes the first operation in Eq. (4). It also gives the settled interface error

$$
\varepsilon_{TS}^*
=
x_T-x_S^*
=
N_Sx_T.
\tag{15}
$$

## 3. Lemma 2: teacher maximum

Let \(U_T\in\mathbb R^{d\times r}\) have orthonormal columns spanning \(\mathcal U_T\), and define
the novelty operator restricted to the teacher memory space:

$$
A_S=U_T^\top N_SU_T.
\tag{16}
$$

**Lemma 2.** Fix \(W_S\) and suppose the student remains at the equilibrium from Lemma 1. Then

$$
\boxed{
\mathcal D_{\max}(W_S;W_T)
=
\frac{\pi_{TS}}2\lambda_{\max}(A_S).
}
\tag{17}
$$

The maximizing teacher states are the unit states whose coordinates belong to the leading
eigenspace of \(A_S\). If its leading eigenvalue is simple, with unit eigenvector \(v_1\), then

$$
\boxed{
x_T^*=\pm U_Tv_1.
}
\tag{18}
$$

Consider the projected teacher dynamics

$$
\tau_T\dot x_T
=
\mu_TP_{T,x}\nabla_{x_T}q,
\qquad
\mu_T>0,
\tag{19}
$$

where

$$
P_{T,x}=U_TU_T^\top-x_Tx_T^\top
\tag{20}
$$

projects onto the tangent space of \(\mathbb S_T\). If the leading eigenvalue is simple and the
initial state has a nonzero component along \(U_Tv_1\), then the teacher dynamics converge to one
of the two states in Eq. (18).

**Proof.** Every \(x_T\in\mathbb S_T\) can be written uniquely as

$$
x_T=U_Ta,
\qquad
\|a\|=1.
$$

The equivalence of the two norm constraints follows from \(U_T^\top U_T=I_r\). Lemma 1 gives

$$
q(U_Ta;W_S)
=
\frac{\pi_{TS}}2
a^\top U_T^\top N_SU_Ta
=
\frac{\pi_{TS}}2a^\top A_Sa.
\tag{21}
$$

The matrix \(A_S\) is symmetric because

$$
A_S^\top
=
U_T^\top N_S^\top U_T
=
A_S.
$$

Let \(v_1,\ldots,v_r\) be an orthonormal eigenbasis of \(A_S\), with

$$
A_Sv_k=\nu_kv_k,
\qquad
\nu_1\geq\nu_2\geq\cdots\geq\nu_r.
$$

Write a unit teacher coordinate as \(a=\sum_kc_kv_k\), where \(\sum_kc_k^2=1\). Then

$$
a^\top A_Sa
=
\sum_k\nu_kc_k^2
\leq
\nu_1\sum_kc_k^2
=
\nu_1.
\tag{22}
$$

Equality holds exactly when \(a\) lies in the leading eigenspace. This proves Eq. (17) and the
general statement about the maximizers. If \(\nu_1>\nu_2\), the leading eigenspace is the line
spanned by \(v_1\), which gives Eq. (18).

It remains to show that the teacher dynamics reach this static maximum. With \(W_S\) fixed,
\(N_S\) does not depend on \(x_T\). Lemma 1 gives the explicit settled deficit

$$
q(x_T;W_S)
=
\frac{\pi_{TS}}2x_T^\top N_Sx_T.
$$

Under a perturbation \(dx_T\),

$$
\begin{aligned}
dq
&=
\frac{\pi_{TS}}2
d\!\left(x_T^\top N_Sx_T\right)
\\
&=
\frac{\pi_{TS}}2
\left[
(dx_T)^\top N_Sx_T
+
x_T^\top N_S\,dx_T
\right]
\\
&=
\frac{\pi_{TS}}2
\left[
x_T^\top N_S^\top dx_T
+
x_T^\top N_S\,dx_T
\right]
\\
&=
\pi_{TS}x_T^\top N_S\,dx_T
\\
&=
\left(\pi_{TS}N_Sx_T\right)^\top dx_T.
\end{aligned}
$$

The third equality rewrites the first scalar term, and the fourth uses the symmetry
\(N_S^\top=N_S\). By the defining identity \(dq=(\nabla_{x_T}q)^\top dx_T\),

$$
\boxed{
\nabla_{x_T}q
=
\pi_{TS}N_Sx_T
=
\pi_{TS}(x_T-x_S^*).
}
\tag{23}
$$

The final equality uses the settled interface error from Eq. (15). Thus the local interface error
is the gradient of the explicit settled deficit, up to the positive factor \(\pi_{TS}\).

Along Eq. (19),

$$
\frac{dq}{dt}
=
\frac{\mu_T}{\tau_T}
\left\|P_{T,x}\nabla_{x_T}q\right\|^2
\geq0.
\tag{24}
$$

The projection preserves membership in \(\mathcal U_T\) and unit norm. Writing \(x_T=U_Ta\) turns
the projected flow into

$$
\tau_T\dot a
=
\mu_T\pi_{TS}(I-aa^\top)A_Sa.
\tag{25}
$$

Expand \(a=\sum_kc_kv_k\) and define

$$
\bar\nu
=
a^\top A_Sa
=
\sum_j\nu_jc_j^2.
$$

Equation (25) gives

$$
\tau_T\dot c_k
=
\mu_T\pi_{TS}(\nu_k-\bar\nu)c_k.
$$

For two nonzero components,

$$
\frac{d}{dt}
\log\left|\frac{c_k}{c_j}\right|
=
\frac{\mu_T\pi_{TS}}{\tau_T}(\nu_k-\nu_j).
\tag{26}
$$

If \(\nu_1>\nu_2\) and \(c_1(0)\neq0\), every lower component decays relative to \(c_1\). Unit norm
then implies

$$
a(t)\longrightarrow\operatorname{sign}(c_1(0))v_1,
$$

which proves convergence to Eq. (18). An exactly absent leading component remains absent under
the deterministic dynamics, which is why the initialization condition is required.

Finally, write the projected teacher interface drive as

$$
\tau_T\dot x_T\big|_{\mathrm{interface}}
=
\kappa P_{T,x}(x_T-x_S^*).
$$

Comparison with Eqs. (19) and (23) gives

$$
\boxed{
\kappa=\mu_T\pi_{TS}>0.
}
\tag{27}
$$

Thus the local interface error supplies the ascent direction that selects the largest settled
deficit. \(\square\)

Lemma 2 completes the second operation in Eq. (4). It identifies the maximum algebraically and
shows that the projected teacher dynamics reach it under the stated spectral and initialization
conditions.

## 4. Theorem 1: one local plasticity step on the maximum

**Theorem 1.** Assume the timescale separation in Eq. (5). Let the student and teacher states
have reached the equilibria established by Lemmas 1 and 2:

$$
x_T=x_T^*,
\qquad
x_S=x_S^*(x_T^*).
\tag{28}
$$

Assume the leading eigenvalue of \(A_S\) is simple. Then

$$
\boxed{
\nabla_{W_S}\mathcal D_{\max}
=
-\pi_S\varepsilon_S^*x_S^{*\top}.
}
\tag{29}
$$

Consequently, one local student plasticity event

$$
\boxed{
\Delta W_S
=
\eta\pi_S\varepsilon_S^*x_S^{*\top}
=
-\eta\nabla_{W_S}\mathcal D_{\max}
}
\tag{30}
$$

is a gradient-descent step on the largest settled network free energy. For sufficiently small
\(\eta\),

$$
\boxed{
\mathcal D_{\max}(W_S+\Delta W_S)
=
\mathcal D_{\max}(W_S)
-
\eta\|\nabla_{W_S}\mathcal D_{\max}\|_F^2
+
o(\eta).
}
\tag{31}
$$

Since the gradient in Eq. (29) is defined with respect to the Frobenius inner product, its
negative is the steepest first-order direction for a fixed infinitesimal weight-change norm.

**Proof.** First fix the teacher state. Only the student recurrent term in Eq. (1) depends
directly on \(W_S\). Under a perturbation \(dW_S\),

$$
d\varepsilon_S=-dW_Sx_S.
$$

At the settled student state,

$$
\begin{aligned}
dF
&=
\pi_S\varepsilon_S^{*\top}d\varepsilon_S
\\
&=
-\pi_S\varepsilon_S^{*\top}dW_Sx_S^*
\\
&=
\left\langle
-\pi_S\varepsilon_S^*x_S^{*\top},
dW_S
\right\rangle_F.
\end{aligned}
$$

The settled value \(q\) also depends on \(W_S\) through \(x_S^*(W_S)\). Differentiating this
dependence gives

$$
dq
=
\underbrace{\nabla_{x_S}F^\top\big|_{x_S^*}dx_S^*}_{0}
+
\left.dF\right|_{x_S=x_S^*}.
$$

The first term vanishes because \(x_S^*\) is the student minimum from Lemma 1. Hence

$$
\nabla_{W_S}q(x_T;W_S)
=
-\pi_S\varepsilon_S^*x_S^{*\top}.
\tag{32}
$$

Now include the dependence of the maximizing teacher state on \(W_S\):

$$
\mathcal D_{\max}(W_S)
=
q(x_T^*(W_S);W_S).
$$

For one scalar weight \(w=(W_S)_{ij}\),

$$
\frac{d\mathcal D_{\max}}{dw}
=
\left.\nabla_{x_T}q^\top\right|_{x_T^*}
\frac{dx_T^*}{dw}
+
\left.\frac{\partial q}{\partial w}\right|_{x_T^*}.
\tag{33}
$$

The teacher constraint set is fixed because \(W_T\) does not change. Therefore
\(dx_T^*/dw\) is tangent to \(\mathbb S_T\). At the constrained maximum from Lemma 2, the
directional derivative of \(q\) along every allowed tangent direction is zero. The first term in
Eq. (33) therefore vanishes.

The simple leading eigenvalue makes the maximizing line locally smooth. Its two representatives
cause no ambiguity. Linearity gives

$$
x_S^*(-x_T^*)=-x_S^*(x_T^*),
\qquad
\varepsilon_S^*(-x_T^*)=-\varepsilon_S^*(x_T^*),
$$

so the outer product in Eq. (32) is the same at both signs. Applying Eq. (32) to every weight
proves Eq. (29), and the local plasticity rule gives Eq. (30).

Finally, differentiability of \(\mathcal D_{\max}\) at a simple leading eigenvalue gives

$$
\mathcal D_{\max}(W_S+\Delta W_S)
=
\mathcal D_{\max}(W_S)
+
\left\langle
\nabla_{W_S}\mathcal D_{\max},
\Delta W_S
\right\rangle_F
+
o(\eta).
$$

Substitution of Eq. (30) proves Eq. (31). \(\square\)

If autapses are forbidden, let \(P_0\) set the diagonal of a matrix to zero. The feasible update is

$$
\Delta W_S
=
-\eta P_0\nabla_{W_S}\mathcal D_{\max}.
\tag{34}
$$

It is the steepest first-order direction within the zero-diagonal weight subspace. The projection
can change the exact rate comparison below, so Corollary 1 treats unconstrained weights.

## 5. Corollary 1: advantage over random clamping

Consider two correlated unit teacher memories

$$
m_+=cu+sv,
\qquad
m_-=cu-sv,
\tag{35}
$$

where \(u,v\) are orthonormal and

$$
c^2=\frac{1+\chi}{2},
\qquad
s^2=\frac{1-\chi}{2},
\qquad
0\leq\chi<1.
\tag{36}
$$

Then \(m_+^\top m_-=\chi\). The common component is \(u\), while

$$
v=\frac{m_+-m_-}{\sqrt{2(1-\chi)}}
$$

is the normalized residual direction.

Define one random-clamping event as follows. Choose \(m_+\) or \(m_-\) with equal probability,
clamp \(x_T\) to that unit memory, let \(x_S\) settle as in Lemma 1, and apply the same local
plasticity rule for the same duration and learning rate as in a prioritized event.

**Corollary 1.** Suppose the two memories in Eq. (35) span \(\mathcal U_T\), and suppose the
student already stores the common component but not the residual:

$$
M_Su=0,
\qquad
M_Sv\neq0.
\tag{37}
$$

Under unconstrained, exactly settled, infinitesimal plasticity, Lemma 2 selects \(v\) or \(-v\).
The positive instantaneous reduction rates satisfy

$$
\boxed{
\frac{-\dot{\mathcal D}_{\max}^{\mathrm{priority}}}
{-\mathbb E[
\dot{\mathcal D}_{\max}^{\mathrm{random}}
]}
=
\frac{2}{1-\chi}.
}
\tag{38}
$$

Thus a prioritized residual event decreases the worst settled deficit faster than random clamping
of either named memory under the matched event protocol.

**Proof.** Lemma 1 gives

$$
\ker N_S=\ker M_S.
$$

Hence \(N_Su=0\). Since \(N_S\) is symmetric, \(u^\top N_Sv=0\). In the teacher basis
\((u,v)\), the restricted novelty operator is

$$
A_S
=
\begin{pmatrix}
0&0\\
0&v^\top N_Sv
\end{pmatrix}.
$$

The lower-right entry is positive because \(v\notin\ker N_S\). The leading eigendirection is
therefore \(v\), so Lemma 2 gives the prioritized teacher state \(x_T^*=\pm v\).

Define the settled student response and recurrent error for \(v\):

$$
z=x_S^*(v)=B_Sv=(I-N_S)v,
\qquad
e=M_Sz.
\tag{39}
$$

The error \(e\) is nonzero. Otherwise the student stationarity equation would imply \(z=v\),
which would contradict \(M_Sv\neq0\).

The response operator \(B_S=I-N_S\) is linear and symmetric. It fixes \(u\), so

$$
u^\top z
=
u^\top B_Sv
=
\bigl(B_Su\bigr)^\top v
=
u^\top v
=0.
\tag{40}
$$

The plasticity rate for residual replay is

$$
\dot W_v
=
\eta\pi_Sez^\top.
\tag{41}
$$

For either named memory, linearity and Eq. (37) give

$$
x_S^*(m_\pm)=cu\pm sz,
\qquad
\varepsilon_S^*(m_\pm)=\pm se.
$$

Therefore

$$
\boxed{
\dot W_\pm
=
\eta\pi_S
\left(
s^2ez^\top
\pm
sc\,eu^\top
\right).
}
\tag{42}
$$

At the selected residual direction, Theorem 1 gives

$$
\nabla_{W_S}\mathcal D_{\max}
=
-\pi_Sez^\top.
\tag{43}
$$

The second term in Eq. (42) makes no first-order contribution to \(\mathcal D_{\max}\), since

$$
\left\langle ez^\top,eu^\top\right\rangle_F
=
\|e\|^2z^\top u
=0.
$$

It follows separately for \(m_+\) and \(m_-\) that

$$
\dot{\mathcal D}_{\max}^{\mathrm{named}}
=
s^2
\dot{\mathcal D}_{\max}^{\mathrm{priority}},
\tag{44}
$$

where both derivatives are negative. Since the two named memories give the same derivative,
uniform random clamping gives the same value in expectation:

$$
\mathbb E[
\dot{\mathcal D}_{\max}^{\mathrm{random}}
]
=
s^2
\dot{\mathcal D}_{\max}^{\mathrm{priority}}.
$$

Substituting \(s^2=(1-\chi)/2\) proves Eq. (38). \(\square\)

The comparison fixes the teacher-state norm, learning rate, and event duration. It is not an
equal-weight-change-norm comparison. At \(\chi=0\), the local rate ratio is \(2\). As
\(\chi\to1\), each named memory contains a vanishing residual coefficient and the ratio diverges.
The case \(\chi=1\) is excluded because the two memories then coincide and no longer span \(v\).

## 6. Completion and scope

Lemma 1 also gives the endpoint identity

$$
\mathcal D_{\max}=0
\quad\Longleftrightarrow\quad
\mathcal U_T\subseteq\ker M_S.
\tag{45}
$$

At this endpoint, \(x_S^*=B_Sx_T=x_T\), \(\varepsilon_{TS}^*=0\), and
\(\varepsilon_S^*=0\) for every valid teacher state. The teacher search drive and the student
plasticity signal both vanish. This characterizes complete transfer but does not prove that an
arbitrary weight trajectory reaches it.

The results are local and timescale-separated. They assume linear dynamics, exact student and
teacher settling, a hard teacher-memory constraint, and a simple leading eigenvalue for the
ordinary weight gradient. An exact multidirectional tie can make \(\mathcal D_{\max}\)
nondifferentiable. Finite timescale separation, soft manifold enforcement, nonlinear activity,
and finite plasticity events require separate analysis. The factor in Eq. (38) also assumes that
the common component is stored exactly and that the weight update is not projected onto a
no-autapse constraint.
