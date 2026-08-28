# Lemma 1: detailed and intuitive proof of student equilibrium

This note expands Lemma 1 of
[the stepwise minimax proof](minimax_cortical_memory_consolidation_stepwise.md). It derives the
student equilibrium, proves convergence of the student dynamics, and computes the free energy
left after settling. The argument also explains why the response operator \(B_S\) produces the
settled student state and why \(N_S=I-B_S\) measures what the student leaves unexplained.

## Statement

Fix a teacher state \(x_T\in\mathbb S_T\) and student weights \(W_S\). Define

$$
M_S=I-W_S,
\qquad
S_S=M_S^\top M_S,
\qquad
H_S=\pi_{TS}I+\pi_SS_S,
\tag{L1.1}
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
\tag{L1.2}
$$

For \(\pi_{TS}>0\) and \(\pi_S>0\), the student free energy has the unique minimizer

$$
\boxed{
x_S^*(x_T)
=
B_Sx_T
=
(I-N_S)x_T.
}
\tag{L1.3}
$$

The gradient flow

$$
\tau_S\dot x_S=-\nabla_{x_S}F
\tag{L1.4}
$$

converges to \(x_S^*(x_T)\) from every initial state. At equilibrium,

$$
\boxed{
q(x_T;W_S)
:=
\min_{x_S}F
=
\frac{\pi_{TS}}2x_T^\top N_Sx_T,
}
\tag{L1.5}
$$

and

$$
\boxed{
q(x_T;W_S)=0
\quad\Longleftrightarrow\quad
M_Sx_T=0.
}
\tag{L1.6}
$$

## 1. Isolating the student problem

The full network free energy is

$$
F(x_T,x_S;W_T,W_S)
=
\frac{\pi_T}{2}\|M_Tx_T\|^2
+
\frac{\pi_{TS}}2\|x_T-x_S\|^2
+
\frac{\pi_S}{2}\|M_Sx_S\|^2.
\tag{L1.7}
$$

During student settling, \(x_T\), \(W_T\), and \(W_S\) are fixed. The teacher recurrent term
does not depend on \(x_S\), so it cannot affect the minimizing student state. It also vanishes
because \(x_T\in\mathbb S_T\subseteq\ker M_T\). The relevant function is therefore

$$
F_S(x_S;x_T,W_S)
=
\frac{\pi_{TS}}2\|x_T-x_S\|^2
+
\frac{\pi_S}{2}\|M_Sx_S\|^2.
\tag{L1.8}
$$

The two terms impose competing requirements. The interface term favors \(x_S=x_T\). The recurrent
term favors \(M_Sx_S=0\), which means that \(x_S\) lies in the student memory space. The
equilibrium is the unique compromise between these requirements.

## 2. Computing the student gradient

For the interface term,

$$
\begin{aligned}
\frac12\|x_T-x_S\|^2
&=
\frac12(x_T-x_S)^\top(x_T-x_S)\\
&=
\frac12x_T^\top x_T-x_T^\top x_S+\frac12x_S^\top x_S.
\end{aligned}
$$

Differentiation with respect to \(x_S\) gives

$$
\nabla_{x_S}
\left[
\frac{\pi_{TS}}2\|x_T-x_S\|^2
\right]
=
\pi_{TS}(x_S-x_T).
\tag{L1.9}
$$

The sign has the expected meaning. Its negative points from \(x_S\) toward \(x_T\).

For the recurrent term,

$$
\frac12\|M_Sx_S\|^2
=
\frac12x_S^\top M_S^\top M_Sx_S
=
\frac12x_S^\top S_Sx_S.
$$

The matrix \(S_S=M_S^\top M_S\) is symmetric. The quadratic-form rule

$$
\nabla_x\left(\frac12x^\top Ax\right)=Ax
$$

therefore gives

$$
\nabla_{x_S}
\left[
\frac{\pi_S}{2}\|M_Sx_S\|^2
\right]
=
\pi_SS_Sx_S.
\tag{L1.10}
$$

Combining Eqs. (L1.9) and (L1.10),

$$
\begin{aligned}
\nabla_{x_S}F_S
&=
\pi_{TS}(x_S-x_T)+\pi_SS_Sx_S\\
&=
(\pi_{TS}I+\pi_SS_S)x_S-\pi_{TS}x_T\\
&=
H_Sx_S-\pi_{TS}x_T.
\end{aligned}
\tag{L1.11}
$$

The Hessian is consequently

$$
\nabla_{x_S}^2F_S=H_S.
\tag{L1.12}
$$

## 3. Why the minimum exists and is unique

For every nonzero \(y\),

$$
\begin{aligned}
y^\top H_Sy
&=
\pi_{TS}y^\top y+\pi_Sy^\top M_S^\top M_Sy\\
&=
\pi_{TS}\|y\|^2+\pi_S\|M_Sy\|^2\\
&>0.
\end{aligned}
\tag{L1.13}
$$

The final inequality follows from \(\pi_{TS}>0\) and \(y\neq0\). Hence \(H_S\) is positive
definite.

Positive definiteness supplies the needed conclusions:

1. \(H_S\) is invertible.
2. \(F_S\) is strictly convex in \(x_S\).
3. A stationary point of \(F_S\) is its unique global minimum.

The interface term is what makes the Hessian strictly positive. The matrix \(S_S\) may have a
nontrivial kernel, but adding \(\pi_{TS}I\) raises every Hessian eigenvalue by the positive amount
\(\pi_{TS}\).

## 4. Solving for the equilibrium and introducing \(B_S\) and \(N_S\)

Set the gradient in Eq. (L1.11) to zero:

$$
H_Sx_S^*-\pi_{TS}x_T=0.
$$

Since \(H_S\) is invertible,

$$
\boxed{
x_S^*=\pi_{TS}H_S^{-1}x_T.
}
\tag{L1.14}
$$

The matrix multiplying \(x_T\) is the settled student response operator. Define

$$
B_S:=\pi_{TS}H_S^{-1}.
$$

Then Eq. (L1.14) reads

$$
\boxed{
x_S^*=B_Sx_T.
}
\tag{L1.15}
$$

The state left unmatched after the student settles is

$$
x_T-x_S^*
=
(I-B_S)x_T.
$$

This motivates the residual operator

$$
N_S:=I-B_S.
$$

Therefore

$$
\boxed{
x_T-x_S^*=N_Sx_T.
}
\tag{L1.16}
$$

Equivalently,

$$
\boxed{
x_S^*=B_Sx_T=(I-N_S)x_T.
}
\tag{L1.17}
$$

Thus \(B_S\) first maps the teacher state to the settled student response. The complementary
operator \(N_S=I-B_S\) then maps the teacher state to the part left as interface error.

To recover the closed form for \(N_S\), use \(H_S=\pi_{TS}I+\pi_SS_S\):

$$
\begin{aligned}
N_S
&=
I-\pi_{TS}H_S^{-1}\\
&=
(H_S-\pi_{TS}I)H_S^{-1}\\
&=
\pi_SS_SH_S^{-1}.
\end{aligned}
$$

## 5. Spectral meaning of \(N_S\)

The spectral theorem gives an orthonormal eigenbasis \(u_1,\ldots,u_d\) for the symmetric positive
semidefinite matrix \(S_S\):

$$
S_Su_k=\mu_ku_k,
\qquad
\mu_k\geq0.
\tag{L1.18}
$$

For a unit eigenvector,

$$
\mu_k
=
u_k^\top S_Su_k
=
\|M_Su_k\|^2.
\tag{L1.19}
$$

The number \(\mu_k\) is the squared recurrent inconsistency along \(u_k\). It is zero exactly when
the student stores that direction.

The same vector is an eigenvector of \(H_S\):

$$
H_Su_k
=
(\pi_{TS}+\pi_S\mu_k)u_k.
\tag{L1.20}
$$

Therefore

$$
H_S^{-1}u_k
=
\frac{1}{\pi_{TS}+\pi_S\mu_k}u_k,
\tag{L1.21}
$$

so the response operator has eigenvalue

$$
\boxed{
B_Su_k
=
\underbrace{
\frac{\pi_{TS}}{\pi_{TS}+\pi_S\mu_k}
}_{b(\mu_k)}
u_k.
}
\tag{L1.22}
$$

The residual operator is the complement \(N_S=I-B_S\), so

$$
\boxed{
N_Su_k
=
n(\mu_k)u_k,
\qquad
n(\mu_k)
=
1-b(\mu_k)
=
\frac{\pi_S\mu_k}{\pi_{TS}+\pi_S\mu_k}.
}
\tag{L1.22a}
$$

Because \(\pi_{TS},\pi_S>0\) and \(\mu_k\geq0\),

$$
0\leq n(\mu_k)<1.
\tag{L1.23}
$$

The endpoints have direct interpretations:

$$
n(0)=0,
\qquad
\lim_{\mu\to\infty}n(\mu)=1.
$$

A stored direction has no settled interface error. A direction with large recurrent
inconsistency is copied only weakly and leaves most of its teacher amplitude as interface error.

To see the split explicitly, let the teacher state have one component \(c_ku_k\). Equations
(L1.15) and (L1.22) give

$$
x_S^*
=
b(\mu_k)c_ku_k
=
\bigl(1-n(\mu_k)\bigr)c_ku_k
=
\frac{\pi_{TS}}{\pi_{TS}+\pi_S\mu_k}c_ku_k,
\tag{L1.24}
$$

and

$$
x_T-x_S^*
=
n(\mu_k)c_ku_k
=
\frac{\pi_S\mu_k}{\pi_{TS}+\pi_S\mu_k}c_ku_k.
\tag{L1.25}
$$

The teacher coefficient divides into a copied fraction and a leftover fraction:

$$
c_k
=
\bigl(1-n(\mu_k)\bigr)c_k+n(\mu_k)c_k.
\tag{L1.26}
$$

This calculation also proves that \(N_S\) is symmetric and positive semidefinite. In matrix form,

$$
N_S
=
U\operatorname{diag}
\left(
\frac{\pi_S\mu_k}{\pi_{TS}+\pi_S\mu_k}
\right)U^\top,
\tag{L1.27}
$$

where the columns of \(U\) are the eigenvectors \(u_k\). Although a product of symmetric matrices
need not be symmetric in general, \(S_S\) and \(H_S^{-1}\) share this eigenbasis and commute.

## 6. Convergence of the student dynamics

The student follows gradient descent:

$$
\tau_S\dot x_S
=
-\nabla_{x_S}F_S
=
-H_Sx_S+\pi_{TS}x_T.
\tag{L1.28}
$$

The equilibrium satisfies

$$
H_Sx_S^*=\pi_{TS}x_T.
$$

Subtract this identity from Eq. (L1.28):

$$
\tau_S\dot x_S=-H_S(x_S-x_S^*).
\tag{L1.29}
$$

Define the displacement from equilibrium,

$$
r(t)=x_S(t)-x_S^*.
$$

The teacher state and the weights remain fixed during student settling, so \(x_S^*\) is
constant on this timescale. Hence

$$
\tau_S\dot r=-H_Sr.
\tag{L1.30}
$$

The solution of this linear differential equation is

$$
r(t)
=
\exp\left(-\frac{H_St}{\tau_S}\right)r(0).
\tag{L1.31}
$$

Diagonalizing \(H_S\) makes the decay explicit. Its eigenvalue along \(u_k\) is

$$
h_k=\pi_{TS}+\pi_S\mu_k>0.
$$

If \(r(0)=\sum_kr_k(0)u_k\), then

$$
r(t)
=
\sum_k
r_k(0)
\exp\left(
-\frac{\pi_{TS}+\pi_S\mu_k}{\tau_S}t
\right)u_k.
\tag{L1.32}
$$

Every component decays exponentially. Since \(h_k\geq\pi_{TS}\),

$$
\|x_S(t)-x_S^*\|
\leq
\exp\left(-\frac{\pi_{TS}}{\tau_S}t\right)
\|x_S(0)-x_S^*\|.
\tag{L1.33}
$$

Therefore

$$
\boxed{
x_S(t)\longrightarrow x_S^*
\quad\text{for every initial }x_S(0).
}
\tag{L1.34}
$$

The interface term supplies a restoring force even along directions in \(\ker M_S\). This is why
the student equilibrium remains unique when the student stores a multidimensional memory
subspace.

## 7. Computing the free energy after settling

Expand Eq. (L1.8) as a quadratic in \(x_S\):

$$
\begin{aligned}
F_S
&=
\frac{\pi_{TS}}2
\left(
x_T^\top x_T-2x_T^\top x_S+x_S^\top x_S
\right)
+
\frac{\pi_S}{2}x_S^\top S_Sx_S\\
&=
\frac{\pi_{TS}}2x_T^\top x_T
-\pi_{TS}x_T^\top x_S
+
\frac12x_S^\top H_Sx_S.
\end{aligned}
\tag{L1.35}
$$

At the minimum,

$$
H_Sx_S^*=\pi_{TS}x_T.
\tag{L1.36}
$$

Use this identity in the last term:

$$
x_S^{*\top}H_Sx_S^*
=
\pi_{TS}x_S^{*\top}x_T
=
\pi_{TS}x_T^\top x_S^*.
\tag{L1.37}
$$

Substitute \(x_S^*\) into Eq. (L1.35):

$$
\begin{aligned}
q(x_T;W_S)
&=
\frac{\pi_{TS}}2x_T^\top x_T
-\pi_{TS}x_T^\top x_S^*
+
\frac{\pi_{TS}}2x_T^\top x_S^*\\
&=
\frac{\pi_{TS}}2x_T^\top(x_T-x_S^*)\\
&=
\frac{\pi_{TS}}2x_T^\top(I-B_S)x_T\\
&=
\frac{\pi_{TS}}2x_T^\top N_Sx_T.
\end{aligned}
\tag{L1.38}
$$

The last two lines use \(x_S^*=B_Sx_T\) followed by \(N_S=I-B_S\). Thus

$$
\boxed{
q(x_T;W_S)
=
\frac{\pi_{TS}}2x_T^\top N_Sx_T.
}
\tag{L1.39}
$$

This formula may look surprising because the original energy contains two squared errors, while
the settled result contains one quadratic form in \(N_S\). A calculation along one eigendirection
shows how the two costs combine.

For \(x_T=c_ku_k\), write \(n_k=n(\mu_k)\). The interface cost is

$$
\frac{\pi_{TS}}2n_k^2c_k^2.
$$

The recurrent cost is

$$
\frac{\pi_S}{2}
\mu_k(1-n_k)^2c_k^2.
$$

Since

$$
n_k=\frac{\pi_S\mu_k}{\pi_{TS}+\pi_S\mu_k},
\qquad
1-n_k=\frac{\pi_{TS}}{\pi_{TS}+\pi_S\mu_k},
$$

their sum is

$$
\begin{aligned}
&\frac12
\left[
\pi_{TS}n_k^2+\pi_S\mu_k(1-n_k)^2
\right]c_k^2\\
&\qquad=
\frac{\pi_{TS}}2n_kc_k^2.
\end{aligned}
\tag{L1.40}
$$

The interface and recurrent penalties divide the cost differently as their relative precisions
change, but their minimized sum is exactly proportional to the leftover fraction \(n_k\).

For a general teacher state \(x_T=\sum_kc_ku_k\), orthogonality gives

$$
q(x_T;W_S)
=
\frac{\pi_{TS}}2\sum_kn(\mu_k)c_k^2,
\tag{L1.41}
$$

which is the eigenbasis form of Eq. (L1.39).

## 8. When the settled deficit is zero

Equation (L1.27) shows that \(N_S\) is positive semidefinite. Therefore

$$
x_T^\top N_Sx_T=0
$$

exactly when \(x_T\) contains components only in eigenvectors whose novelty eigenvalue is zero.
From Eq. (L1.22a),

$$
n(\mu_k)=0
\quad\Longleftrightarrow\quad
\mu_k=0.
$$

Hence

$$
\ker N_S=\ker S_S.
\tag{L1.42}
$$

It remains to connect \(S_S\) to the student memory operator \(M_S\). For every \(y\),

$$
y^\top S_Sy
=
y^\top M_S^\top M_Sy
=
\|M_Sy\|^2.
\tag{L1.43}
$$

Thus

$$
S_Sy=0
\quad\Longleftrightarrow\quad
M_Sy=0,
$$

and

$$
\boxed{
\ker N_S=\ker S_S=\ker M_S.
}
\tag{L1.44}
$$

Finally, combine Eqs. (L1.39) and (L1.44):

$$
\boxed{
q(x_T;W_S)=0
\quad\Longleftrightarrow\quad
x_T\in\ker N_S
\quad\Longleftrightarrow\quad
M_Sx_T=0.
}
\tag{L1.45}
$$

The settled deficit vanishes exactly when the teacher direction is already a fixed direction of
the student weights. In that case \(N_Sx_T=0\), so

$$
x_S^*=x_T,
\qquad
x_T-x_S^*=0,
\qquad
M_Sx_S^*=0.
$$

## 9. Results passed to Lemma 2

Lemma 2 needs four outputs from the student calculation:

$$
x_S^*=B_Sx_T=(I-N_S)x_T,
\tag{L1.46}
$$

$$
x_T-x_S^*=N_Sx_T,
\tag{L1.47}
$$

$$
q(x_T;W_S)
=
\frac{\pi_{TS}}2x_T^\top N_Sx_T,
\tag{L1.48}
$$

and

$$
N_S=N_S^\top\succeq0,
\qquad
\ker N_S=\ker M_S.
\tag{L1.49}
$$

These identities reduce the teacher problem to maximizing a symmetric positive-semidefinite
quadratic form over the unit sphere inside the teacher memory space.
