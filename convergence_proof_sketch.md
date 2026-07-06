# Convergence of Prioritized Memory Transfer — Proof Roadmap

**Status.** Roadmap of *ingredients*, not a proof. It assembles what a convergence theorem needs,
flags which pieces are standard and which are genuinely open, and records the measurements that
constrain the statement.

Companion docs: spec `two_population_memory_transfer_model.md`; analysis `analysis_stress_test.md`
(esp. §3 transfer mechanism, §4 saddle, §5 guard, §8 growth-rate proof, §9 subspace); geometry
`notebooks/06_eigenspace_geometry.py`.

Difficulty tags: **[easy]** · **[std]** standard-but-technical · **[crux]** load-bearing · **[hard]** out of scope.

---

## 0. Measured facts that constrain the theorem (read first)

From direct simulation (`d=3,12,24`, orthonormal patterns, covPCN, the model's own dynamics):

| quantity | meaning | finding |
|---|---|---|
| `L_inv = ‖Q_T S_S U_T‖_F` | does `S_S` preserve the manifold? | **violated mid-transfer** (0.3–0.9 of `‖S_S U_T‖`); `→0` only at convergence |
| `𝔼‖Q_T x_T‖²` | off-manifold leakage of the *data* | **small** (~1.4% of norm²) but **nonzero** |
| `λ_max(S_T)` vs `σ²_min` | stiffest vs weakest off-manifold mode | **differ** for `d>3` (ratio ~5); equal only at `d=3` |

Consequences for the roadmap: (i) the literal invariance condition `[P_T,S_S]=0` is **false** during
learning, so the clean theorem must *impose* an exact manifold constraint rather than assume
invariance; (ii) the *data* staying near-manifold (small leakage) is the weaker, true, and
measurable condition; (iii) the discretization bound must use `λ_max`, not `σ²_min`.

---

## 1. Deficit and target — a *set*, not a point

Transfer deficit (the strongest single choice):
```
E(W_S) = ‖ M_S U_T ‖_F²  = Tr( U_Tᵀ S_S U_T )  ≥ 0,   E = 0 ⟺ S nulls T's manifold.
```
Full transfer is **not** a unique `W_S`: any zero-diagonal `W` with `W U_T = U_T` (equivalently
`M_W U_T = 0`) has `E = 0`. The target is the **transfer set**
```
𝒯 = { W : diag W = 0,  M_W U_T = 0 } = zero-set of E,
```
and the theorem must say `𝒯` is globally attracting — not that some unique matrix is a stable
equilibrium. (Convergence of `E` also does **not** imply rank-by-rank jumps; restricted-novelty
eigenvalues stay positive until the limit. The "effective-rank staircase" is a finite-resolution
empirical description, not a corollary — see §10.)

---

## 2. The fast `x_T` process is a *reversible gradient* diffusion (a gift)

Freeze `W_S`. After adiabatic elimination of `x_S`, the reduced sleep operator
`G(W_S) = −π_T S_T − π_ST N_S(W_S)` is **symmetric** (`S_T`, `N_S` both symmetric; with the reversed precision `π_ST<0` the `−π_ST N_S = +|π_ST| N_S` block grows). The renormalized
deterministic flow
```
ẋ_T = P_{x_T^⊥} G x_T
```
is **gradient ascent of the Rayleigh potential** `φ_W(x) = ½ xᵀ G x` on the sphere. With isotropic
tangent noise the fast process is therefore **reversible**, with an explicit **Bingham invariant
measure**
```
μ_W(dx) ∝ exp( c · xᵀ G(W) x ) d vol(x).
```
This is far better than a generic non-reversible Hörmander argument: it gives existence,
uniqueness, **full support**, an explicit density, and direct covariance/concentration bounds.
*(Caveat, consistent with the saddle reading: the full `(x_T,x_S,W_S)` sleep flow is non-potential
when `π_ST≠−π_TS`; it is only the **adiabatically reduced, frozen-`W_S`** `x_T` flow that is a gradient
diffusion.)* **[std]**

---

## 3. The proof spine — reference-weight distance

Do **not** claim the averaged learning flow is gradient descent on an averaged free energy. It is a
**semi-gradient** flow: the per-sample update is `−η P_0(∇_{W_S} F_S)`, but after averaging over
`μ_{W_S}` — which itself depends on `W_S` — the field is *not* `−∇ 𝔼_{μ_W}[F_S]` (a
`∂μ_W/∂W` term is dropped). So derive descent **directly from the averaged update**.

Adiabatic `x_S = K_S x_T`, `K_S = π_TS(π_TS I + π_S S_S)⁻¹`. Let `Δ = W_S − W_T` (zero-diagonal,
since both are). Using `M_S = M_T − Δ`:
```
ε_S = M_S x_S = −Δ x_S + M_T x_S.
```
Averaged learning (`Σ_S = 𝔼_{μ_W}[x_S x_Sᵀ]`):
```
Ẇ_S = η π_S P_0( 𝔼[ε_S x_Sᵀ] ) = −η π_S P_0(Δ Σ_S) + η π_S P_0(M_T Σ_S).
```
With `D = ½‖Δ‖_F²` and `Δ` zero-diagonal (so `⟨Δ, P_0(X)⟩ = ⟨Δ, X⟩`):
```
Ḋ = −η π_S Tr(Δᵀ Δ Σ_S)  +  η π_S ⟨Δ, P_0(M_T Σ_S)⟩
   ≤ −η π_S c · E          +  η π_S ‖Δ‖_F · ‖M_T Σ_S‖_F
        └ clean descent ┘     └─── leakage bias ───┘
```
using `Δ U_T = −M_S U_T` (so `Tr(ΔᵀΔ P_T)=E`) and a **persistent-excitation** bound
`Σ_S ⪰ c P_T` on `U_T`. The bias term is `‖M_T Σ_S‖` = the **off-manifold covariance of `x_S`**.

**This single inequality is the spine.** Its two faces give the two theorems below.

---

## 4. Two theorems (the recommended organization)

### Theorem 1 — idealized exact transfer **[the rigorous core]**
Assume: `τ_S = 0`; `x_T` (hence `x_S`) constrained **exactly** to the memory sphere
`𝕊(U_T)` (so `M_T x_S = 0` and the **bias vanishes**); bounded weights; uniform persistent
excitation `Σ_S ⪰ c P_T`; learning infinitely slow vs fast mixing. Then `Ḋ ≤ −η π_S c E`, so
`D` is non-increasing, `∫₀^∞ E dt < ∞`, and Barbalat (or LaSalle on `{Ḋ=0}={E=0}`) gives
```
E(t) → 0,   W_S → 𝒯  (the transfer set).
```
Short and genuinely rigorous, *because the exact constraint kills the bias* — we impose the
manifold confinement rather than hoping invariance holds (it doesn't, §0).

### Theorem 2 — robustness of transfer **[perturbative]**
Relax to the real model: finite `τ_S`, finite normal stiffness, imperfect manifold confinement,
discrete integration, constant `η`. Target a residual bound of the form
```
limsup_{t→∞} 𝔼[E(t)]  ≲  C_τ·(τ_S/τ_T)  +  C_leak·(off-manifold covariance)
                          +  C_η·η·τ_mix  +  C_dt·dt^α,
```
with exact transfer recovered in the joint idealized limit. The off-manifold-covariance term is the
`‖M_T Σ_S‖` bias of §3; bounding it is the heart of §5.

---

## 5. The central open conditions

**5.1 Persistent excitation `Σ_S ⪰ c P_T`, uniform along the trajectory. [crux]**
Supplied by the Bingham measure (§2): on the *unlearned* subspace `G` is near-degenerate, so `μ_W`
spreads across it (the noise fills it). The work is making `c>0` **uniform in `W_S`** along the
bounded learning path, including the partial-learning regime.

**5.2 Off-manifold leakage is small — the structure guard alone does NOT give it. [crux]**
The guard controls the normal–normal block `Q(π_T S_T − |π_ST| N_S)Q ≻ 0`, but **not the cross-block
`Q N_S P`**. If `Q N_S P ≠ 0`, the dominant eigenvector of `G` tilts off-manifold even below the
scalar bound (a 2×2 example makes this explicit). What you can prove is **concentration, not
confinement**: with a normal gap `Q(π_T S_T − |π_ST| N_S)Q ⪰ δ Q`,
```
𝔼_{μ_W}‖Q x_T‖²  ≲  σ_ξ²/δ  +  π_ST²‖Q N_S P‖²/δ²,
```
i.e. the invariant measure has **full support** (positive density everywhere) but is *concentrated*
near the manifold. *Measured nuance (§0):* the cross-coupling `‖Q N_S P‖` is **large mid-transfer**
(invariance violated) but `→ 0 at convergence` (since `N_S → N_T = f(S_T)`, which is block-diagonal
w.r.t. `(P_T,Q_T)`), while the *data* leakage `𝔼‖Q x_T‖²` stays small (~1.4%). So the cross-coupling
is a **transient obstacle on the path**, not a final-residual contributor; the residual floor is the
noise part `σ_ξ²/δ`.

**5.3 No partial-learning trap — the genuinely open lemma. [crux]**
Theorem 2 needs the leakage bias in §3 to never overwhelm the `−c E` descent **throughout** the
transient (where, per §0, invariance is badly violated). Empirically transfer still completes, so a
controlling argument exists — this is where real work remains.

**Refinement (supported by §0).** The load-bearing condition is **not** `S_S`-invariance
`[P_T,S_S]=0` (measured false, yet transfer succeeds); it is the weaker, measurable
**"`x_S` stays near the manifold" (small `‖M_T Σ_S‖`)**, plus 5.1/5.3. Track
`L_inv(t)=‖Q_T S_S U_T‖_F` *and* `𝔼‖Q x_T‖²` together; the latter is what enters the bias.

---

## 6. Noise / residual story

- **Annealing `η` does not give exact transfer at fixed `σ_ξ`.** It removes the SA *fluctuation*
  but not the *bias*: the zero of the averaged field sits where off-manifold fitting balances
  on-manifold fitting, generally `E>0`. Exact transfer needs the **bias → 0**, i.e. leakage → 0
  (`σ_ξ→0` and/or the exact constraint of Theorem 1).
- **Mixing time grows as noise shrinks.** On the neutral unlearned subspace the stationary measure
  is ~uniform (amplitude-independent); only `τ_mix ∼ 1/σ_ξ²` scales. So the constant-step error
  `∼ η·τ_mix` can get **worse** as `σ_ξ→0` unless `η` shrinks too.
- **Honest joint condition:** `σ_ξ → 0` *with* `η·τ_mix(σ_ξ) → 0` ⟹ exact transfer. There is no
  clean monotone "less noise ⇒ less residual."
- **Why noise at all = saddle escape.** Each just-learned direction is a saddle of the renormalized
  flow; without a kick the deterministic flow stalls there. (Verified: `σ_ξ=0` still transferred in
  finite precision because **round-off** supplies the kick — accidental, fragile, slower. The
  requirement is *a perturbation to escape saddles*; explicit `ξ` is the robust source. Exactly the
  "noise escapes saddle points" phenomenon of non-convex optimization.)

---

## 7. Discretization

- **Euler stability uses the largest eigenvalue:** `dt < 2 τ_T / (π_T λ_max(S_T))`, *not* `σ²_min`
  (the stiffest mode governs). They coincide only at `d=3` (single nonzero eigenvalue); for `d>3`,
  `λ_max/σ²_min ≈ 5` (§0). For the nonlinear sphere update the safe bound is via a Lipschitz/Jacobian
  bound on `x ↦ P_{x^⊥} G x`.
- **Euler–Maruyama error** is strong `O(dt^{1/2})` / weak `O(dt)` — not a generic pathwise `O(dt)`
  shadowing. State Theorem-2's `dt^α` accordingly.

---

## 8. Assumptions — what each is actually for

- **`|π_ST| < π_T·σ²_min` (structure).** Used in 5.2 for the normal gap `δ`; necessary but **not
  sufficient** for confinement (the cross-block escapes it). Keep, but pair with a `Q N_S P` bound.
- **`π_TS > π_S` (precision) is NOT a proof requirement.** The fast Jacobian `−(π_TS I+π_S S_S)/τ_S`
  is negative definite for any `π_TS>0, π_S≥0`, and the linear `K_S` has gains
  `k(μ)=π_TS/(π_TS+π_S μ) ∈(0,1]` — S *filters* the input, it cannot autonomously confabulate a
  separate attractor. So `π_TS>π_S` is a **modeling preference** (S tracks input strongly), not the
  "no-confabulation" condition the proof needs. (Confabulation is a *nonlinear*-extension concern.)
- **`τ_S ≪ τ_T ≪ 1/η`.** Singular perturbation (Tikhonov) for `x_S`; two-timescale averaging
  (Borkar) for `W_S`. The fast Jacobian sign for Tikhonov needs only `π_TS>0`.
- **`σ_ξ>0` + leash.** Excitation (5.1) + a *stationary* fast measure (§2) — the leash is
  load-bearing, not cosmetic (no stationary measure ⇒ nothing to average over).
- **Storage (T spec).** Patterns `=ker M_T` exactly (covPCN, `P≤d−1`), `W_S(0)=0`, zero-diagonal.

---

## 9. On keeping the renormalization
Keep the hard `‖x_T‖→r₀`: it makes the fast process a diffusion on a **compact** manifold with a
**stationary** (Bingham) measure — required by §2 and the averaging; bounded state also gives clean
SA bounds. Cost: tangential-projection bookkeeping (smooth on the sphere — no pathology). Fallback
if the geometry bites: a soft leash `−γ(‖x‖²−r₀²)x` (smooth `ℝ^d` SDE, Has'minskii confinement).

---

## 10. What we deliberately do NOT prove (and how one might)

- **Speed / rate `N(ε)`. [hard]** Spectral gap of the Bingham generator (reversible ⇒ Poincaré/
  Bakry–Émery is available) → non-asymptotic two-timescale SA. Still heavy; separate effort.
- **Prioritization. [std→hard]** *Binary* (pretrain a subset, held-out transfers first): provable,
  easy — pure excitation/support (`μ=0` ⇒ no drive ⇒ frozen). *Graded* ("most-novel-soonest"): a
  monotonicity argument on `g(μ)=|π_ST| n(μ)` (strictly increasing), but **partly not a theorem** —
  among equally-novel directions the order is noise-set (degeneracy / no WTA).
- **Rank-by-rank staircase. [partly heuristic]** The *count* `r` follows from `E→0` + integer
  rank; the *square steps* do not — `E→0` is smooth, eigenvalues stay positive until the limit
  (so "steps" are finite-resolution, sharp only under strong timescale separation; the nonlinear
  extension is what would make them genuinely discrete).
- **Episodic replay. [out of scope]** Needs an individuating nonlinearity (analysis §9) — a different
  model, not a gap in this proof.

---

## 11. Minimal viable proof + risk register

**Attempt first:** Theorem 1 (manifold-constrained, `D=½‖W_S−W_T‖²` spine). It is short, rigorous,
and already captures the mechanism. Then Theorem 2 perturbatively.

**Risks, ranked:** (1) **No partial-learning trap** (5.3) — the leakage bias controlled throughout,
*despite* measured invariance violation — the one place needing a genuinely new argument.
(2) **Uniform excitation** `c>0` along the path (5.1). (3) **Cross-block leakage** `Q N_S P` on the
transient (5.2). (4) Mixing/noise tradeoff (§6). Everything else is assembling known machinery.
