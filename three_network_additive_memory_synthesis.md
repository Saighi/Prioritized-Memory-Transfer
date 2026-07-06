# Additive Memory Synthesis from Two Frozen Predictive-Coding Networks

**Model specification and analysis. No code in this document — architecture, equations, reduced dynamics, capacity, stability, convergence target, and simulation predictions.**

---

## 1. What the model is

We consider three recurrent predictive-coding associative-memory networks:

- `T₁`: a frozen teacher containing memory subspace `𝒰₁`;
- `T₂`: a frozen teacher containing memory subspace `𝒰₂`;
- `S`: a plastic synthesis network that must learn everything represented by either teacher.

The two teachers do **not** send two separate errors to the synthesis network. Their activities are first added into one combined prediction:

```
y = α₁ x₁ + α₂ x₂
```

The synthesis network compares its state with this combined prediction:

```
ε_Σ = x_S - y
```

This single common error is then sent back to both teachers.

The intended behavior is:

1. small decorrelated noise explores both frozen teacher memory spaces;
2. the common error extracts the part of their combined activity that `S` cannot yet predict;
3. in sleep, reversed teacher-side precision amplifies that unexplained structure;
4. `S` learns it;
5. once a direction is learned, its novelty disappears;
6. the dynamics automatically shift toward whichever teacher directions remain unexplained;
7. transfer ends when `S` has learned the complete combined subspace.

Because the networks are linear, the model transfers a **subspace**, not individually identifiable episodic memories.

The target is therefore not the set-theoretic union `𝒰₁ ∪ 𝒰₂`, which is generally not a subspace. The target is the **subspace sum**:

```
𝒰_Σ = 𝒰₁ + 𝒰₂
     = span(𝒰₁ ∪ 𝒰₂)
```

---

## 2. Networks, notation, and wiring

All three populations initially use the same neural coordinate system and have dimension `d`.

| Symbol | Meaning |
|---|---|
| `x₁ ∈ ℝᵈ` | state of frozen teacher `T₁` |
| `x₂ ∈ ℝᵈ` | state of frozen teacher `T₂` |
| `x_S ∈ ℝᵈ` | state of plastic synthesis network `S` |
| `W₁, W₂` | frozen recurrent weights of the two teachers |
| `W_S` | plastic recurrent weights of the synthesis network |
| `M₁ = I - W₁` | mismatch operator of teacher `T₁` |
| `M₂ = I - W₂` | mismatch operator of teacher `T₂` |
| `M_S = I - W_S` | mismatch operator of the synthesis network |
| `S₁ = M₁ᵀM₁` | teacher-1 surprise operator |
| `S₂ = M₂ᵀM₂` | teacher-2 surprise operator |
| `S_S = M_SᵀM_S` | synthesis surprise operator |
| `α₁, α₂ > 0` | additive coupling weights |
| `π₁, π₂ > 0` | teacher self-precisions |
| `π_I > 0` | synthesis input/interface precision |
| `π_S > 0` | synthesis recurrent precision |
| `ρ` | signed teacher-side interface precision |
| `τ₁, τ₂, τ_S` | state time constants |
| `η` | synthesis learning rate |
| `ξ₁, ξ₂` | small decorrelated teacher noise |
| `r₁, r₂` | fixed teacher-state norms |

The teacher memory spaces are

```
𝒰₁ = ker M₁
𝒰₂ = ker M₂
```

The wiring is

```
                 frozen teacher T₁
                    state x₁
                       │
                      α₁
                       │
                       ├──────────────┐
                       │              │
                       │      y = α₁x₁ + α₂x₂
                       │              │
                       ├──────────────┘
                       │
                      α₂
                       │
                    state x₂
                 frozen teacher T₂

                            │
                            ▼

                    ε_Σ = x_S - y

                            │
                            ▼

                plastic synthesis network S
                    state x_S, weights W_S
```

The same error `ε_Σ` is back-projected to both teachers. In the identity-coordinate case, the back-projection weights are simply `α₁` and `α₂`.

---

## 3. Why the architecture must use one common additive error

A tempting alternative is to give the synthesis network two separate interface errors:

```
ε₁S = x_S - x₁
ε₂S = x_S - x₂
```

The corresponding interface energy is

```
F_sep
=
(π₁S / 2) ‖x_S - x₁‖²
+
(π₂S / 2) ‖x_S - x₂‖²
```

Define the precision-weighted teacher average

```
x̄ = (π₁S x₁ + π₂S x₂) / (π₁S + π₂S)
```

Then the energy decomposes as

```
F_sep
=
[(π₁S + π₂S) / 2] ‖x_S - x̄‖²
+
[π₁S π₂S / (2(π₁S + π₂S))] ‖x₁ - x₂‖²
```

The first term makes `x_S` approach a storage-filtered average of `x₁` and `x₂`.

The second term is different: it directly penalizes disagreement between the two teachers. It remains nonzero whenever `x₁ ≠ x₂`, even if `S` has perfectly learned both teacher memory spaces.

Therefore, with separate errors:

- the two interface errors do not generally vanish after complete transfer;
- teacher disagreement is confused with student novelty;
- there is no clean common termination signal;
- the teachers can continue being pushed simply because they are expressing different valid memories.

The additive architecture instead defines

```
y   = α₁x₁ + α₂x₂
ε_Σ = x_S - y
```

and uses the single interface energy

```
F_Σ,int = (π_I / 2) ‖x_S - (α₁x₁ + α₂x₂)‖²
```

This energy can become exactly zero when `S` can represent the complete combined teacher space.

---

## 4. Error populations

The model contains four distinct error populations.

### Teacher self-errors

```
ε₁ = M₁x₁
ε₂ = M₂x₂
```

These errors measure whether each teacher is consistent with its own frozen recurrent memory.

On the teacher manifolds:

```
x₁ ∈ 𝒰₁  ⟹  ε₁ = 0
x₂ ∈ 𝒰₂  ⟹  ε₂ = 0
```

### Common additive interface error

```
ε_Σ = x_S - α₁x₁ - α₂x₂
```

This measures which part of the combined teacher state is not currently explained by `S`.

### Synthesis self-error

```
ε_S = M_Sx_S
```

This measures whether the current synthesis state is supported by the recurrent model stored in `W_S`.

The key novelty configuration is

```
ε₁ ≈ 0
ε₂ ≈ 0
ε_Σ ≠ 0
```

Both teachers are expressing valid stored structure, but the synthesis network cannot yet explain their combination.

---

## 5. Full dynamics

After each teacher-state update, renormalize its state separately:

```
‖x₁‖ = r₁
‖x₂‖ = r₂
```

After each synthesis weight update, set the diagonal of `W_S` back to zero.

### Teacher dynamics

For `k ∈ {1,2}`:

```
τ_k · dx_k/dt
=
-π_k M_kᵀ ε_k
+ρ α_k ε_Σ
+ξ_k
```

Equivalently,

```
τ₁ · dx₁/dt = -π₁S₁x₁ + ρ α₁ε_Σ + ξ₁
τ₂ · dx₂/dt = -π₂S₂x₂ + ρ α₂ε_Σ + ξ₂
```

The first term damps activity outside each teacher's own memory space.

The second term is the common interface drive.

The sign of `ρ` sets the phase:

```
ρ > 0   wake:  teachers reduce the common mismatch
ρ < 0   sleep: teachers increase the common mismatch
```

In sleep, because `ε_Σ = x_S - y`, the negative value of `ρ` pushes each teacher away from what the synthesis network currently predicts.

### Synthesis-state dynamics

```
τ_S · dx_S/dt
=
-π_I ε_Σ
-π_S M_Sᵀ ε_S
```

Equivalently,

```
τ_S · dx_S/dt
=
π_I(y - x_S)
-π_S S_Sx_S
```

The first term pulls `x_S` toward the combined teacher prediction. The second term applies the current synthesis recurrent prior.

### Synthesis learning

```
dW_S/dt
=
η π_S P₀(ε_S x_Sᵀ)
```

`P₀` removes the diagonal, enforcing the no-autapse constraint.

This is projected gradient descent on the synthesis free energy with respect to `W_S`.

### Required timescale separation

```
τ_S ≪ τ₁, τ₂ ≪ 1/η
```

Interpretation:

- `x_S` rapidly infers the current additive teacher state;
- the teachers explore on an intermediate timescale;
- synthesis learning is slow and accumulates the explored structure.

---

## 6. Check: the one-teacher limit recovers the revised two-network model

Remove `T₂`, set `α₁ = 1`, and rename `x₁ = x_T`.

Then

```
y   = x_T
ε_Σ = x_S - x_T = ε_TS
```

The equations become

```
τ_T · dx_T/dt
=
-π_T M_Tᵀ ε_T
+ρ ε_TS
+ξ
```

```
τ_S · dx_S/dt
=
-π_I ε_TS
-π_S M_Sᵀ ε_S
```

With the identification

```
ρ   = π_ST
π_I = π_TS
```

these are exactly the revised two-network equations.

The revised negative-precision convention is also equivalent to the older positive-gain convention.

If the older mismatch was

```
ε_old = x_T - x_S = -ε_TS
```

and the older novelty gain was `κ > 0`, choose

```
π_ST = -κ
```

Then

```
π_ST ε_TS
=
(-κ)(-ε_old)
=
κ ε_old
```

Thus the revised formulation changes the interpretation and sign convention, not the actual sleep dynamics.

---

## 7. Synthesis free energy and the sleep/wake interpretation

The synthesis free energy is

```
F_S
=
(π_I / 2) ‖ε_Σ‖²
+
(π_S / 2) ‖ε_S‖²
```

Its state gradient is

```
-∇_xS F_S
=
-π_I ε_Σ
-π_S M_Sᵀε_S
```

which exactly gives the synthesis-state equation.

Its weight gradient gives

```
dW_S/dt
=
-η P₀(∇_WS F_S)
=
η π_S P₀(ε_Sx_Sᵀ)
```

So synthesis perception is exact gradient descent, and synthesis learning is projected gradient descent.

### Wake

When `ρ > 0`, both teachers reduce the common interface error. All three states move toward agreement. This is ordinary joint inference or recall.

### Sleep

When `ρ < 0`, the teacher-side precision is reversed. `S` still minimizes `F_S`, but the teachers are driven to increase the common mismatch while their recurrent self-errors keep them near valid memory structure.

This is the exploration phase.

### Exact saddle condition

Define the teacher self-energies

```
F₁ = (π₁ / 2) ‖M₁x₁‖²
F₂ = (π₂ / 2) ‖M₂x₂‖²
```

The candidate sleep saddle is

```
Φ = F₁ + F₂ - F_S
```

The synthesis equations are generated by ascent on `Φ`, equivalently descent on `F_S`.

The teacher equations are generated by descent on the same `Φ` only when

```
ρ = -π_I
```

This is the exact three-network zero-sum condition.

With the two teacher norm constraints, the saddle is projected onto the product of the two teacher spheres.

When `ρ ≠ -π_I`, the novelty-transfer mechanism still works, but the full three-network flow is not generated by one scalar potential.

---

## 8. Fast-synthesis reduction

Assume

```
τ_S ≪ τ₁, τ₂
```

Freeze `x₁`, `x₂`, and `W_S` momentarily and set

```
dx_S/dt = 0
```

The synthesis equilibrium satisfies

```
π_I(y - x_S) = π_S S_Sx_S
```

Therefore

```
x_S* = π_I(π_I I + π_S S_S)⁻¹ y
```

Define the synthesis completion operator

```
K_S = π_I(π_I I + π_S S_S)⁻¹
```

and the synthesis novelty operator

```
N_S
=
I - K_S
=
π_S S_S(π_I I + π_S S_S)⁻¹
```

Then

```
x_S* = K_Sy
```

and

```
ε_Σ*
=
x_S* - y
=
-N_Sy
```

The common interface error is therefore the part of the additive teacher state that `S` cannot explain.

---

## 9. Direction-wise meaning of the matrix inverse

Let `v_j` be an eigenvector of `S_S`:

```
S_Sv_j = μ_jv_j
```

Define the population-level projections

```
ŷ_j   = v_jᵀy
x̂_S,j = v_jᵀx_S
ε̂_Σ,j = v_jᵀε_Σ
```

These are coordinates of the whole population activity along eigen-direction `v_j`. They are **not** activities of neuron `j`.

At synthesis equilibrium:

```
x̂_S,j*
=
[π_I / (π_I + π_Sμ_j)] ŷ_j
```

and

```
ε̂_Σ,j*
=
-[π_Sμ_j / (π_I + π_Sμ_j)] ŷ_j
```

Define the novelty gain

```
n(μ)
=
π_Sμ / (π_I + π_Sμ)
```

Then

```
ε̂_Σ,j* = -n(μ_j)ŷ_j
```

Interpretation:

- learned direction: `μ_j = 0`, so `n(μ_j) = 0`;
- unlearned direction: `μ_j > 0`, so a residual remains;
- partly learned direction: intermediate novelty gain;
- more poorly learned directions produce larger novelty gain.

The inverse does not insert arbitrary stored activity. It filters the current teacher mixture according to what `S` has stored.

---

## 10. Reduced teacher dynamics

Substitute

```
ε_Σ* = -N_Sy
```

into the teacher equations:

```
τ_k · dx_k/dt
=
-π_kS_kx_k
-ρ α_kN_S(α₁x₁ + α₂x₂)
+ξ_k
```

In sleep, define

```
β = -ρ > 0
```

Then

```
τ_k · dx_k/dt
=
-π_kS_kx_k
+β α_kN_Sy
+ξ_k
```

The reduced sleep dynamics are therefore:

- damping toward each teacher's own memory space;
- positive growth along combined directions that `S` has not learned;
- small noise that breaks symmetry and cancellation.

### Stacked form

Define

```
X = [ x₁ ]
    [ x₂ ]
```

```
C = [ α₁I   α₂I ]
```

so that

```
y = CX
```

Define the block operators

```
D = [ π₁S₁    0   ]
    [   0    π₂S₂ ]
```

```
T = [ τ₁I   0   ]
    [  0   τ₂I  ]
```

and

```
Ξ = [ ξ₁ ]
    [ ξ₂ ]
```

Then the reduced sleep dynamics are

```
T · dX/dt
=
[-D + βCᵀN_SC]X
+Ξ
```

The two load-bearing operators are visible:

```
-D          teacher-specific manifold confinement
+βCᵀN_SC    amplification of unexplained combined structure
```

---

## 11. Direction-wise growth analysis

Consider a direction `v` that lies in one or both teacher memory spaces and is approximately an eigen-direction of `S_S`.

Write

```
a₁ = vᵀx₁
a₂ = vᵀx₂
```

The additive amplitude is

```
y_v = vᵀy = α₁a₁ + α₂a₂
```

Along a valid memory direction for teacher `k`, its self-damping term vanishes. In sleep:

```
τ_k · da_k/dt
=
β α_k n(μ)y_v
+noise
```

If both teachers contain `v` and have equal time constant `τ_T`, then

```
τ_T · dy_v/dt
=
βn(μ)(α₁² + α₂²)y_v
+α₁ξ₁,v + α₂ξ₂,v
```

### Unlearned direction

If

```
μ > 0
```

then

```
n(μ) > 0
```

and any small nonzero additive amplitude `y_v` is amplified.

### Learned direction

If

```
μ = 0
```

then

```
n(μ) = 0
```

and the deterministic novelty drive disappears.

### Exact cancellation

Suppose

```
y_v = α₁a₁ + α₂a₂ = 0
```

The deterministic novelty drive is momentarily zero. But decorrelated teacher noise contributes

```
α₁ξ₁,v + α₂ξ₂,v
```

which has nonzero variance.

Once the noise creates a small nonzero `y_v`, the positive novelty gain amplifies it.

Therefore exact anti-phase cancellation is not expected to be attracting along an unlearned direction.

Without noise, exact cancellation can remain invariant by symmetry. Small noise is therefore load-bearing.

---

## 12. Why one teacher should not permanently monopolize transfer

Because `N_S` is linear,

```
ε_Σ*
=
-N_Sy
=
-α₁N_Sx₁
-α₂N_Sx₂
```

Suppose teacher `T₁` initially contributes more strongly. `S` then learns directions from `𝒰₁` faster.

For a teacher-1 direction `u₁`, learning gives

```
N_Su₁ → 0
```

Its contribution to the common novelty then vanishes:

```
α₁N_Sx₁ → 0
```

If teacher `T₂` still contains unexplained directions, the common error becomes approximately

```
ε_Σ* ≈ -α₂N_Sx₂
```

The relative drive therefore shifts toward teacher `T₂`.

This is **novelty homeostasis**:

> The more strongly a source has already been sampled and learned, the less future novelty drive it receives.

This does not guarantee equal transfer speed. A very small `α₂`, very weak noise in `T₂`, or poor geometric visibility can make teacher 2 much slower.

But a teacher that has already been learned cannot continue dominating merely because its raw activity stays large.

---

## 13. Target space and capacity

Let

```
r₁ = dim 𝒰₁
r₂ = dim 𝒰₂
```

The synthesis target is

```
𝒰_Σ = 𝒰₁ + 𝒰₂
```

Its dimension is

```
r_Σ
=
dim 𝒰_Σ
=
r₁ + r₂ - dim(𝒰₁ ∩ 𝒰₂)
```

Shared directions are counted only once.

For a `d`-dimensional zero-diagonal linear covPCN, the nominal generic capacity condition is

```
r_Σ ≤ d - 1
```

If both teachers are strictly less than half full,

```
r₁ < d/2
r₂ < d/2
```

then, because the ranks are integers,

```
r₁ + r₂ ≤ d - 1
```

and therefore

```
r_Σ ≤ r₁ + r₂ ≤ d - 1
```

So the complete combined subspace fits even in the worst case where the two teacher spaces do not overlap.

The relevant quantity is effective rank, not the number of named memory patterns.

---

## 14. Why additive mixtures can reveal the complete combined space

The synthesis network observes states of the form

```
y = α₁u₁ + α₂u₂
```

with

```
u₁ ∈ 𝒰₁
u₂ ∈ 𝒰₂
```

If `α₁` and `α₂` are nonzero and the two teacher states vary independently, then

```
span{y over time} = 𝒰₁ + 𝒰₂
```

The synthesis network does not need to observe each teacher in isolation. It only needs sufficiently varied mixtures.

Define the additive covariance

```
Σ_y = 𝔼[yyᵀ]
```

Expanding:

```
Σ_y
=
α₁²Σ₁
+α₂²Σ₂
+α₁α₂𝔼[x₁x₂ᵀ + x₂x₁ᵀ]
```

If the teacher fluctuations are approximately independent and centered, the cross term is small:

```
Σ_y ≈ α₁²Σ₁ + α₂²Σ₂
```

If each teacher explores all directions in its own memory space,

```
Σ₁ ⪰ c₁P₁  on 𝒰₁
Σ₂ ⪰ c₂P₂  on 𝒰₂
```

then the additive covariance is positive on the complete sum space:

```
U_ΣᵀΣ_yU_Σ ≻ 0
```

where `U_Σ` is an orthonormal basis of `𝒰_Σ`.

This is the persistent-excitation condition needed for full transfer.

---

## 15. Full-transfer state and self-termination

Let `U_Σ` be an orthonormal basis of the complete target space.

Define the transfer set

```
𝒯_Σ
=
{ W : diag(W)=0 and (I-W)U_Σ=0 }
```

Define the transfer deficit

```
E_Σ(W_S)
=
‖M_SU_Σ‖_F²
```

Then

```
E_Σ = 0
```

if and only if `S` nulls every direction in `𝒰_Σ`.

Full transfer is a set of possible weight matrices, not one unique matrix.

If

```
W_S ∈ 𝒯_Σ
```

then every possible additive teacher state satisfies

```
y ∈ 𝒰_Σ
M_Sy = 0
N_Sy = 0
```

Therefore

```
x_S* = y
ε_Σ  = 0
ε_S  = 0
```

Consequently:

- synthesis learning stops;
- reversed-precision teacher forcing stops;
- each teacher only undergoes weak noise-driven motion near its own manifold.

The common-error architecture therefore self-terminates cleanly after complete transfer.

---

## 16. Idealized convergence argument

A complete proof for the finite-timescale noisy model is not claimed here. But the idealized descent argument is direct.

Assume:

- exact fast synthesis inference;
- exact confinement of all training states to `𝒰_Σ`;
- infinitely slow learning compared with state mixing;
- persistent excitation on `𝒰_Σ`;
- bounded synthesis weights;
- nonempty transfer set `𝒯_Σ`.

Choose any reference matrix

```
W_* ∈ 𝒯_Σ
```

and define

```
Δ = W_S - W_*
```

Because both matrices have zero diagonal, `Δ` has zero diagonal.

On the target space:

```
M_*U_Σ = 0
```

For `x_S ∈ 𝒰_Σ`:

```
ε_S
=
M_Sx_S
=
-Δx_S
```

The averaged learning rule is

```
dW_S/dt
=
-ηπ_S P₀(ΔΣ_S)
```

where

```
Σ_S = 𝔼[x_Sx_Sᵀ]
```

Use the reference-distance function

```
D = (1/2) ‖Δ‖_F²
```

Then

```
dD/dt
=
-ηπ_S Tr(ΔᵀΔΣ_S)
```

If the synthesis states persistently excite the full target space,

```
Σ_S ⪰ cP_Σ
```

with `c > 0`, then

```
dD/dt
≤
-ηπ_S c ‖ΔU_Σ‖_F²
```

But

```
ΔU_Σ = -M_SU_Σ
```

so

```
dD/dt ≤ -ηπ_S c E_Σ
```

Thus the transfer deficit must approach zero in the idealized system.

### Real-model qualification

In the actual model, `x_S` and the teacher states remain near, but not exactly on, their target manifolds during partial learning.

The averaged update therefore contains a leakage bias. The realistic target is a residual bound of the form

```
lim sup 𝔼[E_Σ]
≲
C_τ · (τ_S / min(τ₁,τ₂))
+
C_leak · off-manifold covariance
+
C_η · ητ_mix
+
C_dt · dt^a
```

Exact transfer is recovered only in the joint idealized limit where these perturbations vanish.

---

## 17. Stability and teacher-manifold leakage

Let `P₁` and `P₂` project onto the two teacher memory spaces.

Define the product-space projector

```
P = [ P₁   0  ]
    [  0   P₂ ]
```

and

```
Q = I - P
```

The reduced sleep operator is

```
G = -D + βCᵀN_SC
```

### Normal stability guard

A natural operator guard is

```
Q(D - βCᵀN_SC)Q ≻ 0
```

This requires the teacher self-damping to beat novelty amplification in every direction normal to the product teacher manifold.

Because

```
0 ⪯ N_S ≺ I
```

one conservative sufficient condition is

```
β‖C‖²
<
min(π₁σ₁,min², π₂σ₂,min²)
```

where `σ_k,min²` is the smallest nonzero eigenvalue of `S_k`.

### Important caveat

The normal guard does **not** imply exact manifold invariance.

The cross block

```
QCᵀN_SCP
```

can tilt teacher activity slightly away from the product manifold during partial learning.

The honest claim is therefore:

> Below the structure guard, each teacher should remain concentrated near its own memory space, but small off-manifold leakage is generally expected during transfer.

The simulations must measure this leakage rather than assume it is zero.

### Why separate norm constraints are required

Without the constraints

```
‖x₁‖ = r₁
‖x₂‖ = r₂
```

positive novelty growth can make teacher amplitudes diverge.

Renormalization converts radial growth into rotation toward the currently most novel combined directions and keeps the stochastic process bounded.

---

## 18. General coordinate maps

The teachers do not need to use the same neural coordinate system as `S`.

Let

```
C₁ : ℝ^d1 → ℝ^dS
C₂ : ℝ^d2 → ℝ^dS
```

map teacher states into synthesis coordinates.

The additive prediction becomes

```
y = α₁C₁x₁ + α₂C₂x₂
```

and

```
ε_Σ = x_S - y
```

The correct teacher feedback is the transpose mapping:

```
τ_k · dx_k/dt
=
-π_kS_kx_k
+ρ α_kC_kᵀε_Σ
+ξ_k
```

The target becomes

```
𝒰_Σ
=
span(C₁𝒰₁, C₂𝒰₂)
```

All previous formulas hold with

```
C = [ α₁C₁   α₂C₂ ]
```

---

## 19. What the simulations should show

### 19.1 Complete combined-subspace transfer

Construct `U_Σ` from the frozen teacher subspaces and track

```
E_Σ(t) = ‖M_S(t)U_Σ‖_F²
```

Expected result:

```
E_Σ(t) → a small perturbation-dependent floor
```

The floor should shrink as:

- timescale separation improves;
- numerical step size decreases;
- learning becomes slower;
- teacher manifold leakage decreases.

Also track the source-specific deficits

```
E₁(t) = ‖M_S(t)U₁‖_F²
E₂(t) = ‖M_S(t)U₂‖_F²
```

Both should approach zero when capacity is sufficient.

### 19.2 Restricted novelty spectrum

Track the eigenvalues of

```
U_ΣᵀN_S(t)U_Σ
```

Expected result:

- all eigenvalues decrease toward zero;
- the independent number of decaying modes is `r_Σ`;
- shared teacher directions appear only once;
- decay may look staircase-like under strong timescale separation;
- exact square steps are not guaranteed in the linear model.

### 19.3 Automatic source balancing

Track

```
J₁(t) = ‖α₁N_Sx₁‖²
J₂(t) = ‖α₂N_Sx₂‖²
```

Expected result:

- the initially dominant source may transfer first;
- its novelty contribution then decreases;
- relative novelty shifts toward the still-unlearned source;
- both contributions approach zero after full transfer.

Also track the cross term

```
J₁₂(t) = 2α₁α₂ x₁ᵀN_S²x₂
```

because the total novelty is not always the simple sum `J₁ + J₂`.

### 19.4 Overlap is learned once

Use teacher spaces with a prescribed intersection.

Expected result:

```
r_Σ = r₁ + r₂ - dim(𝒰₁ ∩ 𝒰₂)
```

The shared directions should not require duplicate capacity or duplicate novelty modes.

### 19.5 Mixture exploration

Estimate

```
Σ_y = time average of yyᵀ
```

and track

```
λ_min(U_ΣᵀΣ_yU_Σ)
```

Expected result:

- positive minimum eigenvalue with decorrelated teacher noise;
- smaller value with strongly correlated noise;
- zero or near-zero value when the mixtures fail to explore part of `𝒰_Σ`.

### 19.6 Cancellation escape

Initialize a shared direction with

```
α₁a₁ + α₂a₂ = 0
```

Expected result:

- with no noise, cancellation can remain invariant;
- with small decorrelated noise, cancellation is broken;
- the resulting nonzero additive component is amplified if still novel.

### 19.7 Teacher manifold concentration

Let

```
Q₁ = I - P₁
Q₂ = I - P₂
```

Track

```
L₁(t) = ‖Q₁x₁(t)‖²
L₂(t) = ‖Q₂x₂(t)‖²
```

Expected result:

- both remain small below the structure guard;
- they are generally not exactly zero during partial learning;
- they increase sharply when `β` is too large.

Also track

```
L_cross(t) = ‖QCᵀN_SCP‖
```

This can remain nonzero during transfer even while actual state leakage stays small.

### 19.8 Common-error termination

Track

```
‖ε_Σ‖²
‖ε_S‖²
‖dW_S/dt‖_F
```

Expected result: all approach a small noise- and discretization-dependent floor after full transfer.

In the separate-error control, the individual interface errors should remain nonzero whenever `x₁ ≠ x₂`.

---

## 20. Recommended simulation conditions

### Teacher geometries

Use several controlled cases:

1. **Orthogonal spaces:** `𝒰₁ ⟂ 𝒰₂`.
2. **Partly overlapping spaces:** prescribed intersection dimension.
3. **Oblique spaces:** no exact overlap but small principal angles.
4. **Correlated memory sets:** named pattern count larger than effective rank.
5. **Capacity edge:** `r_Σ = d - 1`.
6. **Over capacity:** `r_Σ ≥ d`.

### Initial conditions

- train `W₁` and `W₂`, then freeze them;
- initialize `W_S = 0`;
- initialize each teacher near its own manifold;
- use independent small Gaussian noise `ξ₁` and `ξ₂`;
- normalize the two teacher states separately;
- begin with pure sleep, `ρ < 0`.

### Later wake/sleep experiment

After the pure-sleep mechanism works, alternate:

```
wake:  ρ > 0
sleep: ρ < 0
```

Wake should produce agreement and recall. Sleep should produce novelty-driven synthesis.

---

## 21. Required ablations

### A. Separate errors instead of one common additive error

Prediction:

- persistent teacher-disagreement residual;
- continued forcing after apparent transfer;
- no clean common termination.

### B. No teacher noise

Prediction:

- exact cancellation states can remain invariant;
- just-learned directions can become absorbing saddles;
- transfer may stall except for numerical roundoff.

### C. Perfectly correlated teacher noise

Prediction:

- reduced mixture diversity;
- smaller minimum covariance eigenvalue;
- slower or incomplete exploration of `𝒰_Σ`.

### D. Reversed precision above the structure guard

Prediction:

- large teacher off-manifold leakage;
- learning of non-memory directions;
- failure of the target novelty spectrum to vanish cleanly.

### E. Wake sign throughout

Set

```
ρ > 0
```

Prediction:

- states move toward agreement;
- novelty is not amplified;
- autonomous transfer is weak or absent.

### F. Remove norm constraints

Prediction: teacher amplitudes diverge along unlearned directions.

### G. Synthesis network over capacity

Prediction:

```
E_Σ(t)
```

reaches a nonzero structural floor even in otherwise ideal conditions.

### H. Strong source imbalance

Set

```
α₂ ≪ α₁
```

Prediction:

- `T₁` transfers first;
- `T₂` should still transfer after `T₁` becomes predictable;
- transfer of `T₂` becomes increasingly slow as `α₂ → 0`;
- at `α₂ = 0`, teacher 2 is invisible and cannot transfer.

### I. Slow synthesis inference

Increase

```
τ_S / min(τ₁,τ₂)
```

Prediction:

- delayed novelty filtering;
- weaker source balancing;
- blurred or oscillatory dynamics;
- poorer agreement with the fast-synthesis reduction.

---

## 22. Success criteria

The additive three-network architecture succeeds if all of the following hold.

### Complete acquisition

```
E_Σ(t) → small floor
E₁(t)  → small floor
E₂(t)  → small floor
```

### Correct novelty extinction

```
eigenvalues of U_ΣᵀN_SU_Σ → 0
```

### Source balancing

A source that becomes predictable loses its novelty contribution, allowing the remaining source to dominate the drive.

### Correct capacity law

Transfer succeeds when

```
r_Σ ≤ d - 1
```

and fails structurally when the target exceeds synthesis capacity.

### Overlap efficiency

Shared directions are learned once, so the required capacity follows `r_Σ`, not `r₁ + r₂`.

### Cancellation escape

Decorrelated small noise breaks anti-phase cancellation and exposes unlearned shared directions.

### Manifold selectivity

Teacher states remain concentrated near their respective frozen memory spaces below the structure guard.

### Self-termination

After full transfer:

```
ε_Σ ≈ 0
ε_S ≈ 0
dW_S/dt ≈ 0
```

The separate-error architecture should fail this termination test.

---

## 23. Main interpretation

The additive architecture is a direct multi-teacher extension of the revised two-network memory-transfer dynamics.

Its essential operation is

```
two frozen memory spaces
        ↓
additive combined prediction y
        ↓
common synthesis novelty N_Sy
        ↓
reversed-precision exploration in the teachers
        ↓
slow recurrent learning in S
        ↓
novelty extinction and automatic source rebalancing
```

The central claim is:

> With sufficient capacity, small decorrelated teacher noise, appropriate timescale separation, and a teacher-side structure guard, one plastic synthesis network should learn the complete subspace sum of two simultaneously active frozen teacher networks in a single continuous run.

The strongest new feature is not merely that `S` can store both spaces. It is that the same novelty operator that prioritizes unlearned directions within one teacher should also redistribute exploration between multiple teachers: as one source becomes predictable, its drive disappears and the remaining unexplained source takes over.
