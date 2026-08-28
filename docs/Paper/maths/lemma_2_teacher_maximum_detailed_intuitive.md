# Lemma 2: detailed and intuitive proof of the teacher maximum

This note expands Lemma 2 of
[the stepwise minimax proof](minimax_cortical_memory_consolidation_stepwise.md). It starts from the
recipient result proved in
[the detailed Lemma 1 note](lemma_1_recipient_equilibrium_detailed_intuitive.md), identifies the
worst represented teacher-memory direction, constructs the tangent projection needed to respect
the teacher constraints, and proves convergence of the projected teacher flow.

## Statement

Let \(U_T\in\mathbb R^{d\times r}\) have orthonormal columns spanning the teacher memory space

$$
\mathcal U_T=\ker M_T.
$$

Define the unit teacher memory sphere and the restricted novelty operator by

$$
\mathbb S_T
=
\{x_T\in\mathcal U_T:\|x_T\|=1\},
\qquad
A_S=U_T^\top N_SU_T.
\tag{L2.1}
$$

For fixed recipient weights \(W_S\),

$$
\boxed{
\mathcal D_{\max}(W_S;W_T)
=
\max_{x_T\in\mathbb S_T}q(x_T;W_S)
=
\frac{\pi_{TS}}2\lambda_{\max}(A_S).
}
\tag{L2.2}
$$

The maximizing states are the unit teacher states whose coordinates lie in the leading eigenspace
of \(A_S\). If its leading eigenvalue is simple, with unit eigenvector \(v_1\), then

$$
\boxed{x_T^*=\pm U_Tv_1.}
\tag{L2.3}
$$

Consider the projected teacher dynamics

$$
\tau_T\dot x_T
=
\mu_TP_{T,x}\nabla_{x_T}q,
\qquad
\mu_T>0,
\tag{L2.4}
$$

where

$$
P_{T,x}=U_TU_T^\top-x_Tx_T^\top.
\tag{L2.5}
$$

If the leading eigenvalue is simple and the initial state has a nonzero component along \(U_Tv_1\),
then the teacher state converges to one of the two maximizers in Eq. (L2.3).

## 1. Inputs from recipient settling

Lemma 1 supplies

$$
x_S^*(x_T)=(I-N_S)x_T,
\tag{L2.6}
$$

$$
x_T-x_S^*=N_Sx_T,
\tag{L2.7}
$$

and

$$
q(x_T;W_S)
=
\min_{x_S}F(x_T,x_S;W_T,W_S)
=
\frac{\pi_{TS}}2x_T^\top N_Sx_T.
\tag{L2.8}
$$

The novelty operator has the properties

$$
N_S=N_S^\top,
\qquad
N_S\succeq0,
\qquad
\ker N_S=\ker M_S.
\tag{L2.9}
$$

Equation (L2.8) is the free energy left after the recipient has made its best response to the
teacher state. Lemma 2 asks which unit teacher-memory direction makes this settled deficit largest.

The unit-norm constraint is necessary. Without it,

$$
q(\alpha x_T;W_S)=\alpha^2q(x_T;W_S),
$$

so any direction with positive deficit could produce an arbitrarily large value merely by
increasing its amplitude. The sphere makes the optimization compare directions at equal norm.

## 2. Coordinates inside the teacher memory space

Because the columns of \(U_T\) form an orthonormal basis of \(\mathcal U_T\), every teacher memory
state has a unique representation

$$
x_T=U_Ta,
\qquad
a\in\mathbb R^r.
\tag{L2.10}
$$

The coordinate vector is \(a=U_T^\top x_T\). Its norm satisfies

$$
\begin{aligned}
\|x_T\|^2
&=
(U_Ta)^\top U_Ta\\
&=
a^\top U_T^\top U_Ta\\
&=
a^\top a\\
&=
\|a\|^2.
\end{aligned}
\tag{L2.11}
$$

Thus

$$
x_T\in\mathbb S_T
\quad\Longleftrightarrow\quad
x_T=U_Ta
\text{ with }\|a\|=1.
\tag{L2.12}
$$

The constrained set in \(\mathbb R^d\) has become the ordinary unit sphere in
\(\mathbb R^r\).

Substitute \(x_T=U_Ta\) into Eq. (L2.8):

$$
\begin{aligned}
q(U_Ta;W_S)
&=
\frac{\pi_{TS}}2(U_Ta)^\top N_S(U_Ta)\\
&=
\frac{\pi_{TS}}2a^\top U_T^\top N_SU_Ta.
\end{aligned}
\tag{L2.13}
$$

Define

$$
\boxed{A_S=U_T^\top N_SU_T.}
\tag{L2.14}
$$

Then

$$
\boxed{
q(U_Ta;W_S)
=
\frac{\pi_{TS}}2a^\top A_Sa.
}
\tag{L2.15}
$$

The matrix \(N_S\) measures novelty in the full state space. The teacher cannot search the full
state space. It can only express directions in \(\mathcal U_T\). The matrix \(A_S\) is the part of
\(N_S\) that can be measured through teacher-memory states.

The leading eigenvector of \(N_S\) need not belong to \(\mathcal U_T\). That is why the teacher
optimization uses \(A_S\), rather than maximizing \(N_S\) over all directions in
\(\mathbb R^d\).

## 3. Why \(A_S\) admits an eigenvalue maximization

The matrix \(A_S\) is symmetric:

$$
\begin{aligned}
A_S^\top
&=
(U_T^\top N_SU_T)^\top\\
&=
U_T^\top N_S^\top U_T\\
&=
U_T^\top N_SU_T\\
&=
A_S.
\end{aligned}
\tag{L2.16}
$$

It is also positive semidefinite. For every \(a\),

$$
a^\top A_Sa
=
(U_Ta)^\top N_S(U_Ta)
\geq0.
\tag{L2.17}
$$

The spectral theorem therefore gives an orthonormal eigenbasis
\(v_1,\ldots,v_r\) and nonnegative eigenvalues

$$
A_Sv_k=\nu_kv_k,
\qquad
\nu_1\geq\nu_2\geq\cdots\geq\nu_r\geq0.
\tag{L2.18}
$$

## 4. The static maximum

Expand any unit teacher coordinate in this eigenbasis:

$$
a=\sum_{k=1}^rc_kv_k,
\qquad
\sum_{k=1}^rc_k^2=1.
\tag{L2.19}
$$

Then

$$
\begin{aligned}
a^\top A_Sa
&=
\left(\sum_jc_jv_j\right)^\top
A_S
\left(\sum_kc_kv_k\right)\\
&=
\sum_{j,k}c_jc_k\nu_kv_j^\top v_k\\
&=
\sum_k\nu_kc_k^2.
\end{aligned}
\tag{L2.20}
$$

All cross terms vanish because the eigenvectors are orthonormal. The remaining expression is a
weighted average of the eigenvalues. The weights \(c_k^2\) are nonnegative and sum to one.
Therefore

$$
\begin{aligned}
a^\top A_Sa
&=
\sum_k\nu_kc_k^2\\
&\leq
\sum_k\nu_1c_k^2\\
&=
\nu_1.
\end{aligned}
\tag{L2.21}
$$

The upper bound is attained by \(a=v_1\), so

$$
\max_{\|a\|=1}a^\top A_Sa
=
\nu_1
=
\lambda_{\max}(A_S).
\tag{L2.22}
$$

Substitution into Eq. (L2.15) proves

$$
\boxed{
\mathcal D_{\max}(W_S;W_T)
=
\frac{\pi_{TS}}2\lambda_{\max}(A_S).
}
\tag{L2.23}
$$

This is the Rayleigh-Ritz theorem written out in the form needed here.

### Simple and repeated leading eigenvalues

If \(\nu_1>\nu_2\), equality in Eq. (L2.21) requires

$$
c_2=\cdots=c_r=0.
$$

Unit norm then gives \(c_1=\pm1\). Hence

$$
\boxed{x_T^*=\pm U_Tv_1.}
\tag{L2.24}
$$

The two signs have the same deficit because a quadratic form is even:

$$
q(-x_T;W_S)=q(x_T;W_S).
\tag{L2.25}
$$

Suppose instead that the leading eigenvalue has multiplicity \(m\):

$$
\nu_1=\cdots=\nu_m>\nu_{m+1}.
$$

Every unit linear combination

$$
a=\sum_{k=1}^mc_kv_k,
\qquad
\sum_{k=1}^mc_k^2=1,
\tag{L2.26}
$$

satisfies

$$
a^\top A_Sa
=
\nu_1\sum_{k=1}^mc_k^2
=
\nu_1.
\tag{L2.27}
$$

The maximizers form the unit sphere inside the leading eigenspace. A one-dimensional leading
eigenspace gives the two points \(\pm v_1\). A two-dimensional leading eigenspace gives a circle.
Higher multiplicities give the corresponding higher-dimensional unit spheres.

## 5. The teacher gradient

There are two useful derivations of the teacher gradient.

### Direct differentiation

From Eq. (L2.8),

$$
q(x_T;W_S)
=
\frac{\pi_{TS}}2x_T^\top N_Sx_T.
$$

For a symmetric matrix \(B\),

$$
\nabla_x\left(\frac12x^\top Bx\right)=Bx.
$$

Since \(N_S=N_S^\top\),

$$
\boxed{
\nabla_{x_T}q
=
\pi_{TS}N_Sx_T.
}
\tag{L2.28}
$$

Equation (L2.7) then gives

$$
\boxed{
\nabla_{x_T}q
=
\pi_{TS}(x_T-x_S^*).
}
\tag{L2.29}
$$

Thus the settled interface error is the gradient of the settled deficit, up to the positive
factor \(\pi_{TS}\).

### Envelope-theorem derivation

The same result follows without inserting the explicit quadratic formula for \(q\). Write

$$
q(x_T;W_S)
=
F\bigl(x_T,x_S^*(x_T);W_T,W_S\bigr).
\tag{L2.30}
$$

The chain rule gives

$$
\nabla_{x_T}q
=
\left(
\frac{\partial x_S^*}{\partial x_T}
\right)^\top
\nabla_{x_S}F\big|_{x_S^*}
+
\nabla_{x_T}F\big|_{x_S^*}.
\tag{L2.31}
$$

Recipient settling gives

$$
\nabla_{x_S}F\big|_{x_S^*}=0.
\tag{L2.32}
$$

The indirect term in Eq. (L2.31) vanishes. Although \(x_S^*\) moves when \(x_T\) moves, that
motion has no first-order effect on the minimized value because the recipient is at a stationary
point.

The direct derivative of the full free energy is

$$
\nabla_{x_T}F
=
\pi_TM_T^\top M_Tx_T
+
\pi_{TS}(x_T-x_S).
\tag{L2.33}
$$

Every allowed teacher state lies in \(\ker M_T\), so

$$
M_Tx_T=0
\quad\Longrightarrow\quad
M_T^\top M_Tx_T=0.
\tag{L2.34}
$$

At recipient equilibrium,

$$
\nabla_{x_T}q
=
\pi_{TS}(x_T-x_S^*),
\tag{L2.35}
$$

which agrees with Eq. (L2.29).

The envelope form will be reused when the optimized deficit is differentiated with respect to
\(W_S\).

## 6. Legal teacher velocities and the tangent space

The teacher must satisfy both

$$
x_T\in\mathcal U_T
\qquad\text{and}\qquad
\|x_T\|=1.
\tag{L2.36}
$$

An instantaneous velocity \(z\) preserves the first constraint when

$$
z\in\mathcal U_T.
\tag{L2.37}
$$

To find the condition imposed by unit norm, differentiate:

$$
\frac{d}{dt}\|x_T\|^2
=
2x_T^\top\dot x_T.
\tag{L2.38}
$$

The norm is preserved when \(x_T^\top\dot x_T=0\). Therefore the tangent space at a valid state
\(x_T\) is

$$
\boxed{
T_{x_T}\mathbb S_T
=
\{z\in\mathcal U_T:x_T^\top z=0\}.
}
\tag{L2.39}
$$

A legal velocity must lie in the teacher memory space and be perpendicular to the current state.

## 7. Constructing the tangent projector

Let

$$
P_U=U_TU_T^\top.
\tag{L2.40}
$$

Because \(U_T\) has orthonormal columns, \(P_U\) is the orthogonal projector onto
\(\mathcal U_T\). Given an arbitrary direction \(g\), the vector \(P_Ug\) is its component inside
the teacher memory space.

This component may still point partly along \(x_T\). Since \(\|x_T\|=1\), its radial component is

$$
x_Tx_T^\top g.
\tag{L2.41}
$$

Subtracting this part gives

$$
\boxed{
P_{T,x}g
=
\left(U_TU_T^\top-x_Tx_T^\top\right)g.
}
\tag{L2.42}
$$

Hence

$$
\boxed{
P_{T,x}=U_TU_T^\top-x_Tx_T^\top.
}
\tag{L2.43}
$$

Both terms in Eq. (L2.42) lie in \(\mathcal U_T\), so the result lies in that subspace. It is also
perpendicular to \(x_T\):

$$
\begin{aligned}
x_T^\top P_{T,x}g
&=
x_T^\top U_TU_T^\top g
-
x_T^\top x_Tx_T^\top g\\
&=
x_T^\top g-x_T^\top g\\
&=
0.
\end{aligned}
\tag{L2.44}
$$

The simplification uses \(U_TU_T^\top x_T=x_T\) and \(x_T^\top x_T=1\).

The matrix is symmetric. It is also idempotent:

$$
\begin{aligned}
P_{T,x}^2
&=
(P_U-x_Tx_T^\top)^2\\
&=
P_U-P_Ux_Tx_T^\top-x_Tx_T^\top P_U
+x_Tx_T^\top x_Tx_T^\top\\
&=
P_U-x_Tx_T^\top-x_Tx_T^\top+x_Tx_T^\top\\
&=
P_{T,x}.
\end{aligned}
\tag{L2.45}
$$

A symmetric idempotent matrix is an orthogonal projector. Equations (L2.44) and (L2.45) confirm
that \(P_{T,x}\) projects onto the tangent space in Eq. (L2.39).

### Why both terms are needed

Using only \(U_TU_T^\top\) would keep the state in the teacher memory space but could change its
norm. Using only \(I-x_Tx_T^\top\) would preserve the norm but could move the state outside the
teacher memory space. Their combination enforces both constraints.

## 8. Constraint preservation under the flow

Consider

$$
\tau_T\dot x_T
=
\mu_TP_{T,x}\nabla_{x_T}q.
\tag{L2.46}
$$

The velocity belongs to \(\mathcal U_T\). A trajectory that starts in this linear subspace stays in
it.

The squared norm obeys

$$
\begin{aligned}
\frac{d}{dt}\|x_T\|^2
&=
2x_T^\top\dot x_T\\
&=
\frac{2\mu_T}{\tau_T}
x_T^\top P_{T,x}\nabla_{x_T}q\\
&=
0.
\end{aligned}
\tag{L2.47}
$$

A trajectory that starts at unit norm remains at unit norm. Thus the projected flow stays on
\(\mathbb S_T\).

## 9. The correct stationarity condition

At an unconstrained optimum, the full gradient vanishes. At a constrained optimum on
\(\mathbb S_T\), the correct condition is

$$
\boxed{
P_{T,x}\nabla_{x_T}q=0.
}
\tag{L2.48}
$$

The full gradient need not be zero. If the norm constraint were removed, the quadratic deficit
could still increase by scaling \(x_T\). In teacher coordinates, the gradient at an eigenvector is
radial:

$$
\nabla_aq
=
\pi_{TS}A_Sa,
\qquad
\nabla_aq\big|_{a=v_k}
=
\pi_{TS}\nu_kv_k.
\tag{L2.49}
$$

In the full state space, the gradient can also have a component perpendicular to
\(\mathcal U_T\). The projector removes both the radial component and the component outside the
teacher memory space.

To verify stationarity at \(x_T=U_Tv_k\), note that

$$
\begin{aligned}
U_TU_T^\top N_SU_Tv_k
&=
U_TA_Sv_k\\
&=
\nu_kU_Tv_k,
\end{aligned}
\tag{L2.50}
$$

while

$$
\begin{aligned}
x_Tx_T^\top N_Sx_T
&=
x_T(v_k^\top A_Sv_k)\\
&=
\nu_kx_T.
\end{aligned}
\tag{L2.51}
$$

The two terms cancel:

$$
P_{T,x}N_Sx_T=0.
\tag{L2.52}
$$

Every eigenvector is therefore a stationary point of the projected flow, not only the leading
one. Monotonic ascent alone does not prove convergence to the global maximum.

## 10. Why the projected flow increases the deficit

Let \(g=\nabla_{x_T}q\). Along Eq. (L2.46),

$$
\begin{aligned}
\frac{dq}{dt}
&=
g^\top\dot x_T\\
&=
\frac{\mu_T}{\tau_T}g^\top P_{T,x}g.
\end{aligned}
\tag{L2.53}
$$

Since \(P_{T,x}=P_{T,x}^\top=P_{T,x}^2\),

$$
g^\top P_{T,x}g
=
g^\top P_{T,x}^\top P_{T,x}g
=
\|P_{T,x}g\|^2.
\tag{L2.54}
$$

Therefore

$$
\boxed{
\frac{dq}{dt}
=
\frac{\mu_T}{\tau_T}
\|P_{T,x}\nabla_{x_T}q\|^2
\geq0.
}
\tag{L2.55}
$$

The settled deficit never decreases during teacher settling. It increases strictly unless the
tangent gradient is zero.

## 11. Converting the flow to teacher coordinates

Write

$$
x_T=U_Ta,
\qquad
\|a\|=1.
$$

Since \(x_T\) remains in \(\mathcal U_T\),

$$
\dot x_T=U_T\dot a.
\tag{L2.56}
$$

Substitute Eq. (L2.28) into the projected flow and multiply by \(U_T^\top\):

$$
\begin{aligned}
\tau_T\dot a
&=
U_T^\top\tau_T\dot x_T\\
&=
\mu_T\pi_{TS}
U_T^\top
\left(
U_TU_T^\top-U_Taa^\top U_T^\top
\right)
N_SU_Ta\\
&=
\mu_T\pi_{TS}
(I-aa^\top)A_Sa.
\end{aligned}
\tag{L2.57}
$$

The projected teacher dynamics in memory coordinates are

$$
\boxed{
\tau_T\dot a
=
\mu_T\pi_{TS}(I-aa^\top)A_Sa.
}
\tag{L2.58}
$$

The raw novelty push is \(A_Sa\). The factor \(I-aa^\top\) removes the part that would change
\(\|a\|\).

## 12. Dynamics of each novelty component

Expand

$$
a=\sum_kc_kv_k,
\qquad
A_Sa=\sum_k\nu_kc_kv_k.
\tag{L2.59}
$$

Define the current Rayleigh quotient

$$
\bar\nu
:=
a^\top A_Sa
=
\sum_j\nu_jc_j^2.
\tag{L2.60}
$$

The radial term is

$$
aa^\top A_Sa
=
a(a^\top A_Sa)
=
\bar\nu a.
\tag{L2.61}
$$

Equation (L2.58) becomes

$$
\begin{aligned}
\tau_T\dot a
&=
\mu_T\pi_{TS}(A_Sa-\bar\nu a)\\
&=
\mu_T\pi_{TS}
\sum_k(\nu_k-\bar\nu)c_kv_k.
\end{aligned}
\tag{L2.62}
$$

Matching coefficients gives

$$
\boxed{
\tau_T\dot c_k
=
\mu_T\pi_{TS}(\nu_k-\bar\nu)c_k.
}
\tag{L2.63}
$$

This equation gives the selection mechanism directly. A component with novelty above the current
weighted average grows. A component below the average shrinks. The norm stays fixed, so growth
means an increasing share of the teacher state rather than increasing total amplitude.

The factor \(c_k\) is equally important. If \(c_k=0\), then \(\dot c_k=0\). In the eigenbasis of
\(A_S\), the deterministic novelty drive grades components already present in the teacher state.
It does not generate an exactly absent orthogonal component.

In an arbitrary teacher-memory basis, \(A_S\) may mix coordinates. The independent-component
interpretation applies in its eigenbasis.

## 13. Ratio proof of convergence

The average \(\bar\nu\) changes as the teacher state changes. Comparing two coefficients removes
it. For nonzero \(c_k\) and \(c_j\),

$$
\begin{aligned}
\frac{d}{dt}\log\left|\frac{c_k}{c_j}\right|
&=
\frac{\dot c_k}{c_k}-\frac{\dot c_j}{c_j}\\
&=
\frac{\mu_T\pi_{TS}}{\tau_T}
\left[
(\nu_k-\bar\nu)-(\nu_j-\bar\nu)
\right]\\
&=
\frac{\mu_T\pi_{TS}}{\tau_T}
(\nu_k-\nu_j).
\end{aligned}
\tag{L2.64}
$$

Integration gives

$$
\boxed{
\left|\frac{c_k(t)}{c_j(t)}\right|
=
\left|\frac{c_k(0)}{c_j(0)}\right|
\exp\left[
\frac{\mu_T\pi_{TS}}{\tau_T}
(\nu_k-\nu_j)t
\right].
}
\tag{L2.65}
$$

Assume that the leading eigenvalue is simple:

$$
\nu_1>\nu_2.
\tag{L2.66}
$$

Compare each lower component with \(c_1\):

$$
\left|\frac{c_k(t)}{c_1(t)}\right|
=
\left|\frac{c_k(0)}{c_1(0)}\right|
\exp\left[
-\frac{\mu_T\pi_{TS}}{\tau_T}
(\nu_1-\nu_k)t
\right].
\tag{L2.67}
$$

If \(c_1(0)\neq0\), then

$$
\frac{c_k(t)}{c_1(t)}\longrightarrow0
\qquad
\text{for every }k>1.
\tag{L2.68}
$$

The unit-norm constraint gives

$$
\sum_kc_k(t)^2=1.
\tag{L2.69}
$$

It follows that

$$
|c_1(t)|\longrightarrow1,
\qquad
c_k(t)\longrightarrow0
\quad(k>1).
\tag{L2.70}
$$

The sign of \(c_1\) cannot change. Equation (L2.63) has the form

$$
\dot c_1=h(t)c_1,
$$

whose solution is

$$
c_1(t)
=
c_1(0)
\exp\left[
\frac{\mu_T\pi_{TS}}{\tau_T}
\int_0^t(\nu_1-\bar\nu(s))\,ds
\right].
\tag{L2.71}
$$

The exponential factor is positive. Therefore

$$
\operatorname{sign}c_1(t)=\operatorname{sign}c_1(0).
\tag{L2.72}
$$

Combining Eqs. (L2.68) through (L2.72),

$$
\boxed{
a(t)
\longrightarrow
\operatorname{sign}(c_1(0))v_1,
}
\tag{L2.73}
$$

and

$$
\boxed{
x_T(t)
\longrightarrow
\operatorname{sign}(c_1(0))U_Tv_1.
}
\tag{L2.74}
$$

This proves convergence to one of the two global maximizers.

## 14. Why the initialization condition is necessary

If \(c_1(0)=0\), Eq. (L2.63) gives

$$
\dot c_1(0)=0.
$$

More strongly, the differential equation preserves \(c_1(t)=0\) for all time. The deterministic
flow cannot create a missing leading component. It then selects the largest-eigenvalue direction
present in the initial state, which may be a lower eigenvector.

This explains why every eigenvector is stationary. If the teacher sits exactly on \(v_k\), the
novelty operator scales that existing direction by \(\nu_k\). The resulting push is radial, and the
tangent projector removes it. No other eigencomponent appears.

## 15. Noise, spectral gap, and practical selection

A continuously distributed random initialization has

$$
c_1(0)=v_1^\top a(0)\neq0
$$

with probability one. Small initialization noise therefore supplies the leading component needed
by the deterministic theorem.

Equation (L2.67) shows how quickly it takes over. The relative decay rate of component \(k\) is

$$
\frac{\mu_T\pi_{TS}}{\tau_T}(\nu_1-\nu_k).
\tag{L2.75}
$$

Selection slows when the leading component starts extremely small or when the spectral gap
\(\nu_1-\nu_2\) is small.

If noise acts only during initialization and the system then settles deterministically, the proof
applies directly. If noise continues during settling, the state generally fluctuates near the
leading direction rather than converging to it exactly. The deterministic flow remains the drift
part of that stochastic dynamics.

Noise must respect the teacher constraints. One may project it tangentially,

$$
P_{T,x}\xi(t),
$$

or project the perturbed state back into \(\mathcal U_T\) and renormalize it. Unprojected noise can
move the teacher outside its memory sphere.

## 16. Matching the local interface drive

Write the projected teacher interface drive as

$$
\tau_T\dot x_T\big|_{\mathrm{interface}}
=
\kappa P_{T,x}(x_T-x_S^*).
\tag{L2.76}
$$

Projected gradient ascent gives

$$
\begin{aligned}
\tau_T\dot x_T
&=
\mu_TP_{T,x}\nabla_{x_T}q\\
&=
\mu_T\pi_{TS}
P_{T,x}(x_T-x_S^*),
\end{aligned}
\tag{L2.77}
$$

where Eq. (L2.29) supplied the second line. Comparing Eqs. (L2.76) and (L2.77) yields

$$
\boxed{
\kappa=\mu_T\pi_{TS}>0.
}
\tag{L2.78}
$$

The local settled interface error supplies the direction of teacher ascent. The positive scalar
\(\kappa\) sets its speed.

## 17. Results passed to Theorem 1

Lemma 2 supplies the following facts:

$$
\mathcal D_{\max}(W_S;W_T)
=
\frac{\pi_{TS}}2\lambda_{\max}
\left(U_T^\top N_SU_T\right),
\tag{L2.79}
$$

$$
x_T^*=\pm U_Tv_1
\quad\text{when the leading eigenvalue is simple},
\tag{L2.80}
$$

$$
P_{T,x_T^*}\nabla_{x_T}q=0,
\tag{L2.81}
$$

and

$$
x_T(t)\longrightarrow
\operatorname{sign}(c_1(0))U_Tv_1
\quad\text{when }c_1(0)\neq0.
\tag{L2.82}
$$

Equation (L2.81) is the constrained first-order condition. The full gradient need not vanish.
Only its component along legal tangent directions vanishes. Theorem 1 uses this fact when it
differentiates the optimized deficit with respect to \(W_S\).
