# Prioritized Memory Transfer Between Two Predictive-Coding Associative Memories

**Model specification (for implementation). No code in this document — equations, notation, intended dynamics, what to measure, and how it sits in the literature.**

---

## 1. What the model is, and why we build it

Two associative-memory networks, **T** (teacher) and **S** (student), are coupled across a hierarchy: the teacher sits **above** the student (hippocampus-like, above an association-cortex-like student). T holds a set of stored memories; S starts empty. The goal is **prioritized memory transfer**: S should learn T's memories, *visiting the ones it does not yet have first*, with the prioritization emerging from the dynamics rather than from an external scheduler.

The intended new behavior: during **sleep / replay**, noise stirs T; S cannot predict the memories it has not learned; that **T–S mismatch** drives T's state into the unlearned memory; S learns it; the mismatch vanishes there; and the drive moves on to the remaining novel subspace. Once S has learned everything T holds, the **learning drive self-extinguishes**. This is a statement about the dynamics, not an automatic stopping rule: the simulation runner still executes its configured fixed step budget.

Why this is interesting:
- It is a model of **replay / consolidation** (teacher → student), in the spirit of hippocampus → neocortex transfer, but with **priority = what the student lacks**.
- The substrate is **predictive coding**. We deliberately use the *linear* covariance-learning predictive-coding associative memory (covPCN), whose memories are **not strong intrinsic attractors** — attractor strength is gated by prediction error (surprise). This is the property we exploit: a memory becomes a transient attractor for T *only when the student (S) is surprised by it*. T on its own is flat.

**Scope, stated honestly (see §5, §7).** Because T is linear, its stored memories span a flat *subspace* — every linear combination of patterns is itself a valid memory, so the dynamics individuate *directions*, not named patterns. The base model is therefore best described as **prioritized *subspace* consolidation**: S acquires the unlearned directions of T's memory space, in a novelty-gated order, transferring the **span**. Turning this into discrete, one-episode-at-a-time replay of individual `m_p` requires an added nonlinearity (sparsity / soft-WTA / point attractors), flagged as the natural extension in §12. Everything below holds for the linear (subspace) base model unless noted.

---

## 2. Networks, notation, and wiring

All symbols are defined here and used consistently throughout.

| Symbol | Meaning |
|---|---|
| `d` | number of neurons per population (state dimension) |
| `P` | number of memories stored in T |
| `x_T ∈ ℝ^d` | T's state (value neurons) — the only thing T "is" |
| `x_S ∈ ℝ^d` | S's state (value neurons) |
| `W_T ∈ ℝ^{d×d}` | T's recurrent weights — **frozen**, zero diagonal; encode the stored memories |
| `W_S ∈ ℝ^{d×d}` | S's recurrent weights — **plastic**, zero diagonal, initialized to 0 |
| `M_T = I − W_T` | T's "mismatch operator" (`I` = identity) |
| `M_S = I − W_S` | S's mismatch operator |
| `m_p ∈ ℝ^d`, p = 1…P | the stored memory patterns |
| `π_T > 0` | teacher self-precision (how much T trusts its own recurrent model) |
| `π_S > 0` | student self-precision (prior: how much S trusts its own recurrent model) |
| `π_TS > 0` | student's interface precision (input precision — how much S trusts T's activity) |
| `π_ST` (signed) | teacher's interface precision (epistemic gain). **`< 0` = sleep/replay (a reversed precision); `> 0` = wake.** Its sign *is* the phase — there is no separate gate. |
| `τ_T`, `τ_S` | time constants of T's and S's states |
| `η` | learning rate for `W_S` |
| `ξ` | additive noise on T, white, `ξ ~ 𝒩(0, σ_ξ² I)` |
| `r₀` | fixed norm to which `‖x_T‖` is renormalized (the amplitude leash) |

**Wiring (kept maximally simple and biologically literal):**
- The teacher is **higher** in the predictive hierarchy, so it supplies the **top-down prediction** of the student; the interface error lives at the **lower (student) level**.
- **T → S** is the identity: each S neuron receives `+1` from its corresponding T neuron, so S's top-down prediction is `x_T`.
- **S → T** is the identity: S's ascending synapse delivers `x_S` back to T.
- The interface therefore computes the mismatch `x_S − x_T`.

---

## 3. The three error populations

These are distinct neural populations and must be kept separate (a shared error population breaks the model — see §6/§7).

- **T's self-error:** `ε_T = M_T x_T = x_T − W_T x_T` (general form `+ v_T`; the base model has no external clamp, `v_T = 0`). Measures how far T's state is from being one of its own stored patterns. **Zero on T's memory manifold** (this is what "T is flat" means).
- **Interface / mismatch error:** `ε_TS = x_S − x_T`. The T–S disagreement at the student level. This is the signal that both drives S's learning and drives T's exploration.
- **S's self-error:** `ε_S = M_S x_S = x_S − W_S x_S` (general form `+ v_S`, `v_S = 0` here). How far S's state is from being a pattern S has stored. This is what lets S **complete** a known pattern (and what gates S's learning).

The whole model lives in the gap between `ε_T` and `ε_TS`: at a memory T holds but S lacks, `ε_T ≈ 0` (T is self-coherent) yet `ε_TS ≠ 0` (S is surprised). That co-occurrence is the definition of a "novel memory" and cannot be detected by a single shared error unit.

---

## 4. Dynamics

Three coupled processes, integrated in continuous time (small step). After each T update, renormalize `‖x_T‖ → r₀`. After each `W_S` update, re-zero its diagonal.

**T's state (the environment, driven by the interface + noise):**
```
τ_T · dx_T/dt = − π_T M_Tᵀ ε_T  +  π_ST · ε_TS  +  ξ          (then renormalize ‖x_T‖ = r₀)
```
- `−π_T M_Tᵀ ε_T` = T's intrinsic relaxation onto its manifold (damps anything off the stored-pattern subspace).
- `+ π_ST·ε_TS` = the interface drive, with `π_ST` the teacher's **signed** interface precision; `ξ` = noise, essential because it seeds (symmetry-breaks) the direction T grows into.

**S's state (perception — fast):**
```
τ_S · dx_S/dt = − π_TS · ε_TS  −  π_S M_Sᵀ ε_S
```
- `−π_TS ε_TS = +π_TS (x_T − x_S)` pulls `x_S` toward the top-down prediction `x_T` (perception).
- `−π_S M_Sᵀ ε_S` is S's recurrent pattern-completion (its prior).

**S's learning (slow):**
```
dW_S/dt = η · π_S · ε_S · x_Sᵀ       (zero the diagonal after each step)
```
- Hebbian on S's self-error and S's activity; automatically prioritized (large error → large update) and self-extinguishing (as a pattern is stored, `ε_S → 0` there and learning on it stops).
- *Qualifier:* zeroing the diagonal after each step (no autapses) makes this **projected** gradient descent on `F_S` (descent restricted to the zero-diagonal weight subspace). It is still a proper descent — `F_S` remains a Lyapunov function for learning — but it is not the unconstrained `F_S` gradient.

**Reversed precision — what the *sign* of `π_ST` means (the sleep/wake mechanism).**
The student always *descends* its free energy (perception, `−π_TS ε_TS`). The phase is set by the **sign of the teacher's interface precision** `π_ST` — there is no separate ±1 gate; we reverse the precision itself:
```
wake  (π_ST > 0):  ordinary precision  — the teacher MINIMIZES the interface error (chases / explains away)
sleep (π_ST < 0):  REVERSED precision  — the teacher MAXIMIZES the interface error (drive-to-disagree)
```
- **Wake** is ordinary joint inference: both populations descend, the student conforms to the teacher's prediction and the teacher conforms to the student's evidence. They *agree*; there is no transfer drive — this is recall, not replay.
- **Sleep / replay** *reverses* the teacher's precision (`π_ST < 0`), turning its error-minimization into error-maximization: T is pushed *away* from the student's prediction `x_S`, toward what S cannot explain. Confined to its memory manifold (by `−π_T M_Tᵀ ε_T` and the renormalization), "away from the student" along the manifold means *into the unlearned stored memory*. **Transfer happens in sleep.**

> **A note on the object we introduce.** A *reversed (negative) precision* is not a standard inverse-covariance — those are positive (semi-)definite by construction. We introduce it deliberately: flipping the *sign* of the precision at the teacher's interface synapse is the single knob that converts perception/recall (error-minimization) into replay/exploration (error-maximization). Keep it as a sign on the precision; do **not** rewrite it as a separate `−1` factor times a positive gain — that hides the conceptual move.

So in sleep, T is the *environment* and S is the *agent* whose ignorance is surfaced: perception (`−π_TS ε_TS`, ordinary precision) lives in S, action (the reversed-precision drive `π_ST ε_TS`, `π_ST<0`) lives on T. This is the active-inference split, and the reason the model seeks novelty in sleep instead of explaining it away. (A third option — `clamp`, holding `x_T` fixed at infinite input precision — is standard PC perception of fixed data, but a clamped T cannot be explored, so we do not use it.)

**Fast-S reduction (the novelty operator `N_S`).** Because `τ_S ≪ τ_T`, S sits at its steady state for the current `x_T`. Writing `S_T = M_TᵀM_T` and `S_S = M_SᵀM_S`, that steady state is `x_S* = π_TS(π_TS I + π_S S_S)⁻¹ x_T`, so the mismatch T actually feels is
```
ε_TS = − N_S x_T ,     N_S = π_S S_S (π_TS I + π_S S_S)⁻¹      (the novelty operator)
```
with eigenvalues `n(μ) = π_S μ /(π_TS + π_S μ) ∈ [0,1)`: `N_S` nulls directions S has learned (`μ=0 → n=0`) and passes directions S lacks (`μ=1 → n=π_S/(π_TS+π_S)`). Substituting `ε_TS = −N_S x_T`, T's slow reduced dynamics are
```
τ_T dx_T/dt = − π_T S_T x_T − π_ST N_S x_T + ξ        (then project to the sphere ‖x_T‖ = r₀)
```
In sleep (`π_ST < 0`) the second term is `+|π_ST| N_S x_T` — a *growth* term along the directions S lacks, which is the replay drive. `N_S` is the cleanest handle on the whole model and is used in the guards and observables below.

**Required parameter relations (these are part of the model, not tuning niceties):**
- **Timescale separation:** `τ_S ≪ τ_T ≪ 1/η`. S's state must catch up fast; T drifts slowly; S's weights are the slow, rate-limiting step (the dwell time at each memory).
- **Structure guard (general operator form).** Stability against noise-chasing in sleep is a question about the *magnitude* of the reversed precision. It requires, on the off-manifold subspace (where `S_T` is invertible), the **operator inequality**
  `|π_ST| N_S ≺ π_T S_T`   (equivalently `|π_ST| < π_T / λ_max(S_T^{−1/2} N_S S_T^{−1/2})`) — damping beats drive in *every* direction at once, not just on average.
  Two readable specializations: when `N_S` and `S_T` share eigenvectors it reduces to the scalar `|π_ST| < π_T·σ²_min·(π_TS+π_S)/π_S` (`σ²_min` = smallest **nonzero** eigenvalue of `S_T`, the spectral gap); and since `0 ⪯ N_S ≺ I` always, the spectrum-independent bound `|π_ST| < π_T·σ²_min` is always safe and is the recommended default. Below the bound, only on-manifold (real-memory) directions are amplified; above it, T starts amplifying noise.
- **Precision guard:** `π_TS > π_S` (input trusted over internal belief).
- **Saddle-exactness option:** set `π_ST = −π_TS` if you want Reading 1 of §9 (the single-functional minimax) to be *exact* rather than structural — the teacher's reversed precision then exactly mirrors the student's (a zero-sum game). Even then it is exact only as a *projected* saddle on the sphere `‖x_T‖=r₀` (the renormalization is not generated by `Φ`). This ties `|π_ST|` to the input precision and adds `π_TS < π_T·σ²_min·(π_TS+π_S)/π_S`. It is optional: the dynamics and the transfer mechanism work for `π_ST ≠ −π_TS` — only the exact-potential interpretation requires it.
- `σ_ξ` small but strictly positive; `r₀` fixed (e.g. unit norm).

**Storing memories in T (recipe, not code):** draw P patterns `m_p` (e.g. random ±1 or Gaussian, normalized); set `W_T` by a covariance / Hebbian rule so that each `m_p` lies in the null space of `M_T` (`M_T m_p ≈ 0`, i.e. the patterns are the flat directions of T's manifold). Then freeze `W_T`. Initialize `W_S = 0`.

---

## 5. Intended behavior (the "completion race") and self-extinguishing drive

The selection mechanism is a **race in time** between two pattern-completers — but note that in the required `τ_S ≪ τ_T` regime, S reaches its steady state at each instant, so the *known-vs-novel* discrimination is effectively **spectral** (set by what `M_S` can null), not by raw speed. The "dynamic" content is the competition *among* novel directions, below.

1. Noise perturbs `x_T`. T's self-term immediately damps off-manifold (noise) components; on-manifold (real-memory) components persist and become correlated, structured activity.
2. S reconstructs that activity:
   - **Known pattern** (S has it): S's recurrent model snaps `x_S` onto `x_T`, `ε_TS → 0`. T gets no sustained drive. The memory is *explained away* — flat, no replay.
   - **Novel pattern** (S lacks it): S can only partially copy `x_T` (a fraction `π_TS/(π_TS+π_S)`); `ε_TS` **persists** as a residual `−[π_S/(π_TS+π_S)]·x_T` along that direction. Multiplied by the reversed precision `π_ST < 0`, this drives `x_T` further into the novel memory — self-sustaining.
3. **Which novel direction goes first (no winner-take-all).** All *equally-unlearned* directions carry the *same* novelty gain `n(1)`, so they are neither ranked by novelty nor singled out by the dynamics: the growth operator `−π_T S_T − π_ST N_S` (with `π_ST<0`, the `−π_ST N_S = +|π_ST| N_S` term grows) is degenerate on the unlearned subspace, and L2 renormalization (a power iteration) converges to that whole subspace, not to one vector. What breaks the tie is the slow learning — whichever direction the noise `ξ` happens to load `x_T` onto when `W_S` starts updating is extinguished first, and it need not be a named pattern `m_p`. (This is why the base model transfers a *subspace*; see §1 and §7.)
4. S slowly learns the selected direction (`W_S` update). As it does, that direction's novelty gain falls as `n_v = π_S(1−w)²/(π_TS+π_S(1−w)²) → 0` (quadratically in the residual weight error `1−w`), so `ε_TS → 0`, the drive releases, and the dynamics move to the next novel direction.
5. **Termination:** when S has learned all P patterns, `ε_S` and `ε_TS ≈ 0` everywhere on T's manifold; there is no positive drive left; `x_T` merely diffuses on the manifold. This quiescence is the "transfer complete" signal — the *correct* terminal state, not a failure.

Note: making S slower (raising `τ_S` toward `τ_T`) should blur the known/novel discrimination. Because the discrimination is spectral in the mandated fast-S regime, the genuinely *dynamic* signature is expected to appear only as `τ_S/τ_T` is pushed up — varying it is the test that distinguishes dynamic selection from a purely spectral one.

---

## 6. The two guards (what keeps it honest)

Two failure modes, each blocked by one of the parameter relations above. State both explicitly when reporting.

- **World-facing failure — chasing noise.** If `|π_ST|` exceeds the spectral-gap bound, T's reversed-precision drive overwhelms its own damping and `x_T` grows along unstructured noise directions (which S is also "surprised" by). Guard: `|π_ST| < π_T·σ²_min·(π_TS+π_S)/π_S` (or the safe default `|π_ST| < π_T·σ²_min`), plus T's manifold confinement. This is the discrete analog of the requirement that the world have *learnable structure*.
- **Belief-facing failure — confabulation ("fake memory from S").** If `π_S ≥ π_TS`, S's recurrent prior dominates and `x_S` is captured by S's own stored patterns instead of tracking `x_T`; `ε_TS` then reflects S's belief drift rather than genuine novelty, and S "transfers" patterns that are not in T. Guard: `π_TS > π_S`.

These two guards are symmetric: one stops T drifting into noise, the other stops S drifting into fantasy — the two ways the loop could drive its error down *without actually transferring a memory*.

---

## 7. What to build, run, and measure

Build T (store P patterns, freeze), initialize S empty, integrate the three equations in the sleep regime with renormalization, noise, and the timescale separation. Then characterize:

**Primary result**
- **Novelty-spectrum staircase.** Let `U_T` be an orthonormal basis of T's memory manifold (`ker M_T`). Track the eigenvalues of the **restricted novelty spectrum** `U_Tᵀ N_S(t) U_T` (or, more coarsely, `F_S` probed at each `m_p`). Expectation: the largest eigenvalue falls in discrete steps to a floor. **The number of steps is the *effective rank* of the memory subspace, not necessarily `P`** — for orthonormal patterns rank `= P` (you may see `P` steps); for correlated patterns it is fewer, and learning one direction lowers novelty on its neighbours. This curve is the main evidence that prioritized transfer is real and not diffusion. `U_Tᵀ N_S U_T` is preferred over per-`m_p` `F_S` because it measures *directions*, immune to the basis ambiguity of the subspace.
- **Transfer order vs novelty:** with **binary** novelty — pre-train S on a subset, then check that the held-out directions transfer first. (Do not over-claim a graded ordering among equally-unlearned directions; per §5 that order is noise-driven, with no WTA.)

**Mechanism checks**
- **Per-direction mismatch:** `‖ε_TS‖` projected onto each unlearned manifold direction over time — should spike then decay, one direction at a time.
- **T's alignment:** cosine similarity of `x_T` with the unlearned manifold (and, when patterns are orthonormal, with each `m_p`) over time — `x_T` should concentrate in the unlearned subspace and release a direction as it is learned. With correlated patterns, expect alignment to a *direction*, not necessarily a single named `m_p`.

**Ablations (each should break the model in a specific, predicted way)**
- `π_ST` above the spectral-gap bound → T chases noise; alignment with stored patterns degrades; `F_S` staircase breaks.
- `π_S ≥ π_TS` → confabulation; `ε_TS` stops reflecting true novelty.
- `τ_S ≈ τ_T` (S not fast) → the completion race blurs; known/novel discrimination weakens. There should be a window of `τ_S/τ_T` where prioritization is sharpest — varying it is the signature that the selection is *dynamic*, not merely spectral.
- **Wake regime** (`π_ST > 0`, an un-reversed precision) → both populations descend; there is no transfer drive, so the staircase stays flat (this is the recall/inference control).
- No renormalization → `x_T` amplitude diverges.
- No noise → no symmetry-breaking; degenerate / stuck at zero.
- **Correlated vs orthonormal patterns (subspace test):** staircase step-count should track the *effective rank* of the memory set — `P` clean steps when orthonormal, **fewer** when correlated. Adding a soft-WTA/nonlinearity should restore one-pattern-per-step (episodic replay).

**Success criteria:** S learns the full memory subspace (`U_Tᵀ N_S U_T → 0`, i.e. all independent directions); the staircase has ≈ effective-rank steps; (binary) transfer order matches novelty; each ablation fails as above.

---

## 8. How it maps to free energy — read this carefully

Two halves of the model map onto free energy with **different degrees of certainty**. This distinction matters; do not flatten it.

**S's generative model.** S treats T's activity as generated by a latent cause `x_S`, with a recurrent Gaussian prior (precision set by `M_S`, weight `π_S`) and a Gaussian likelihood `x_T | x_S ~ 𝒩(x_S, π_TS⁻¹ I)` (the identity loading = the value-1 top-down synapse). Its variational free energy is exactly
```
F_S = (π_TS/2)‖ε_TS‖² + (π_S/2)‖ε_S‖².
```

**Variational free energy (VFE) — the certain / exact mapping.** S's state relaxation (perception) is *exact gradient descent on `F_S`*; the `W_S` update is *projected* gradient descent on `F_S` (the diagonal-zeroing constraint; `F_S` still decreases). This is textbook variational inference; treat the VFE mapping as solid. In the **wake** phase the teacher *also* descends (toward `F_S` through the interface and `F_T` through its self-term), so wake is pure joint VFE minimisation — a convex-ish bowl, ordinary inference.

**Expected free energy (EFE) — the structural / constructed mapping.** The reversed-precision drive `π_ST ε_TS` (`π_ST<0`) on T is read as **S's action** minimizing expected free energy — specifically its epistemic (information-gain) term, with `½‖ε_TS‖²` used as the linear-Gaussian *proxy* for information gain about `W_S`. This mapping is structural/functional, not derived: it requires (i) accepting EFE as the action objective at all (EFE is not uniquely derived from the free-energy principle — see Millidge et al.), and (ii) the proxy, which equals true information gain only in the isotropic-uncertainty limit and is kept honest otherwise by T's manifold (which guarantees the surprise S chases is *learnable*). So: **VFE mapping = exact and derived; EFE mapping = an interpretation that is structurally faithful but rests on a construction plus a proxy.** Report it as such. Reversing the sign of `π_ST` is exactly what turns the VFE bowl (wake) into the epistemic/EFE drive (sleep).

The two guards of §6 are the priors this reading needs: structure (so the info-gain proxy targets learnable directions, not noise) and precision (so perception tracks input rather than belief).

---

## 9. Two readings of the same dynamics — and they are distinct mathematical objects

The same equations admit two formal descriptions. **Keep them distinct; do not claim they are the same object.**

**Reading 1 — minimax / saddle (sleep).** Ask whether the full sleep `(x_T, x_S, W_S)` flow is the gradient flow of a *single* functional with T descending and S ascending. The correct candidate is
```
Φ = F_T − F_S = (π_T/2)‖M_T x_T‖² − (π_TS/2)‖ε_TS‖² − (π_S/2)‖ε_S‖²
```
S ascending `Φ` reproduces S's state and weight equations exactly. **T descending `Φ` reproduces T's sleep equation only when `π_ST = −π_TS`** — the teacher's reversed precision must be the *exact negative* of the student's precision (cross-derivative symmetry; a zero-sum game). So:
- **If `π_ST = −π_TS`:** `Φ = F_T − F_S` is an exact saddle/minimax potential — T minimizes `F_T − F_S`, S minimizes `F_S` — *but only as a flow restricted to the sphere* `‖x_T‖=r₀`: the renormalization adds a radial projection `P_{x_T^⊥}=I−x_T x_Tᵀ/‖x_T‖²` that `Φ` does not generate, so read the saddle as living on the sphere, not in full `ℝ^d`. This is a genuine theorem-grade statement *under that relation and that constraint*. Geometrically `Φ` is a **bowl** along directions S can predict (T is pinned — flat) and a **hill** along novel directions (T slides off — the replay drive); S's learning turns each hill into a bowl.
- **If `π_ST ≠ −π_TS`:** the flow is **not** the gradient flow of any single potential; there is no exact `Φ`. Reading 1 then has the same *structural* (not exact) status as the EFE reading below.

**Reading 2 — single-agent descent on EFE.** S is the only agent. It descends VFE (perception, learning) and, in sleep, EFE (action); T is the environment it acts on. S's behavior is pure descent; the closed T–S sleep loop is a saddle (as every active-inference agent's loop is — the world does not minimize the agent's free energy) **when `π_ST = −π_TS`; otherwise the loop is a coupled non-potential flow that merely behaves saddle-like near the transfer fixed points.**

**Crucial caveat for the writeup and the visualization:** Reading 1 (the minimax on `Φ`) and Reading 2 (descent on EFE) are **two distinct objects**. Even where Reading 1 is exact (`π_ST = −π_TS`), `Φ = F_T − F_S` is a *different* functional from the EFE; they coincide with the `−F_S` mismatch part only through the information-gain proxy of §8. Do not conflate `Φ` with the EFE, and do not present the saddle as if it *were* the active-inference object. And do not present Reading 1 as unconditionally exact: its exactness is conditional on `π_ST = −π_TS`.

---

## 10. The myopic action and its hidden prior (the forward model)

The action `u = π_ST ε_TS` (§4, sleep) is the **myopic** (greedy) form of expected-free-energy action selection: it follows the *instantaneous* gradient of the information-gain proxy. This is the globally information-maximizing action **only when T's controlled landscape is benign** — smooth and trap-free, so local ascent reaches the global optimum without ever passing through low-information regions. In a rugged T (separate deep basins, barriers), the greedy action would stall at the nearest local information peak; the genuinely optimal action could require routing *through already-known territory* to reach a distant unknown region — and the bare gradient does not have that shape.

So the simple derived action silently rests on a geometric prior: that following the mismatch leads, step by step, to the information without needing detours. This is **more than reachability** (a yes/no — is a state attainable under *some* control). It is *reachability conditioned by the action* — the map from what S does to where T ends up — and that map is a rigorous, standard object:

| Informal phrasing | Rigorous object |
|---|---|
| where my action takes T | **controlled transition model** `P(s′ \| s, a)` — a *forward / world model* |
| that model rolled forward under a policy | **policy-conditioned predictive distribution** `Q(o \| π)` |
| the states I can drive T into | **policy-conditioned reachable set** (the support of `Q(o \| π)`) |
| discrete active-inference form | the transition matrix **B**; policies index sequences of it; EFE is summed over a horizon |

The implication: to know where its pushes take T, S must — implicitly or explicitly — hold a **causal forward model of T's controlled dynamics**. Non-myopic exploration — planning a route toward distant novelty — *is* optimization over this rolled-forward model across a horizon. The myopic action drops the horizon and is correct only when smoothness makes a single step sufficient.

**This is the same object as the dark-room weighting (the anticipation prior).** The epistemic value weights information gain by `Q(o \| π)` — S's own predictions of where its actions lead. The dark room is the *poverty* of that model (it predicts nothing informative is reachable, so all gain is weighted to zero); the smoothness that lets the greedy action work here is the *benignity* of the same model. One construct, two faces: named from the anticipation side it is "does S expect anything out there to find"; named from the action side it is "does following the push actually get there." Both are the forward / transition model. This also sharpens §8: the EFE mapping is structural not only because of the information-gain proxy, but because the action is myopic — a one-step reduction that is exact only under this smoothness prior.

**Where this model sits (implementation boundary).** Because T is linear and its memory manifold is a subspace, T's controlled landscape is smooth and trap-free: the myopic gradient genuinely *is* globally information-maximizing, so S carries **no explicit transition model of T and does no planning** — the lookahead collapses into one step. This is precisely why the simple action `u = π_ST ε_TS` is sufficient and why the spec needs no policy/horizon machinery. The boundary to flag for any extension: if T is made nonlinear or rugged (e.g. deep point-attractor memories), the greedy action is no longer correct, and S would need an explicit forward model of T plus a planning horizon — at which point `u = π_ST ε_TS` must be replaced by EFE minimization over rolled-out policies.

---

## 11. Visualization ideas

- **Saddle cross-sections (the key intuition figure):** two small potential plots side by side — a **bowl** (a direction S can predict; `x_T` rests at the bottom; stable, no replay) and a **hill** (a novel direction; `x_T` sits near the top and slides off; the replay drive), with the caption that learning flips each hill into a bowl. Together they show why `Φ` is a saddle. *(Caption honestly: this is the `Φ = F_T − F_S` picture, exact under `π_ST = −π_TS`.)*
- **Novelty-spectrum staircase over time:** the primary result plot — eigenvalues of `U_Tᵀ N_S U_T` falling in steps (one per independent direction transferred; ≈ effective rank, not necessarily `P`).
- **Replay raster / alignment heatmap:** `x_T`-vs-(manifold / `m_p`) cosine over time, showing the unlearned subspace shrinking one direction at a time (a single named memory only when patterns are orthonormal).
- **Optional minimax schematic:** T minimizes / S maximizes `Φ` — *with an explicit annotation* that (a) it holds exactly only for `π_ST = −π_TS`, and (b) the single-agent EFE descent is a separate object (per §9), so the figure does not imply the saddle and the EFE are the same thing.

---

## 12. Literature we build on

- **Predictive-coding associative memory (substrate).** Covariance-learning predictive-coding networks, *PLOS Computational Biology* 2023, DOI `10.1371/journal.pcbi.1010719` (Tang and colleagues). We use the **linear** covPCN variant: its stored patterns form a flat manifold (a hyperplane / line attractor) rather than deep point attractors, so T has **no strong intrinsic attractors** and its memories are surprise-gated. This is the property the whole model exploits.
- **Active inference / free-energy principle (Friston).** VFE for perception and learning; EFE and the epistemic-value (information-gain) decomposition for action (Friston et al. 2015, "Active inference and epistemic value"; Friston et al. 2017, "Active inference: a process theory"). Bogacz 2017 ("A tutorial on the free-energy framework…") for the predictive-coding math.
  - **EFE caveat (cite honestly):** Millidge, Tschantz & Buckley 2021, "Whence the Expected Free Energy?" — EFE is a constructed functional, not uniquely derived; the exploratory drive does not fall out of free-energy minimization by itself. This is why §8 marks the EFE mapping as structural rather than certain.
- **Prioritized replay.** Schaul et al. 2015, "Prioritized Experience Replay" — priority by TD error; here priority is the **student's own prediction error**.
- **Generative replay / continual learning.** Shin et al. 2017, "Continual Learning with Deep Generative Replay" — a teacher replays to a student; here the replay is *prioritized toward the gap*.
- **Complementary learning systems.** McClelland, McNaughton & O'Reilly 1995 — T ≈ fast teacher (hippocampus-like, higher in the hierarchy), S ≈ slow student (neocortex-like, lower).

**One-line positioning:** this is *prioritized **subspace** consolidation where the priority signal is the student's own prediction error* (the novelty operator `N_S`) — equivalently, an active-inference reading in which the student (S) forages, during sleep, for novel *directions* in the teacher's (T's) memory space.

**Natural extension (episodic replay).** The linear base model transfers a subspace. To make it replay discrete memories one at a time, add a mechanism that individuates patterns — a nonlinearity / sparsity / soft-WTA in S, or point-attractor (nonlinear covPCN) memories in T. That upgrade is what turns *subspace consolidation* into *episodic consolidation*; it is also the regime (§10) where T's landscape becomes rugged and S would need an explicit forward model + planning, replacing the myopic `u = π_ST ε_TS`.
