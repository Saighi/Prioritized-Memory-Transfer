# Prioritized Memory Transfer Between Two Predictive-Coding Associative Memories

**Model specification (for implementation). No code in this document — equations, notation, intended dynamics, what to measure, and how it sits in the literature.**

---

## 1. What the model is, and why we build it

Two associative-memory networks, **A** (teacher) and **B** (student), are coupled. A holds a set of stored memories; B starts empty. The goal is **prioritized memory transfer**: B should learn A's memories, *visiting the ones it does not yet have first*, with the prioritization emerging from the dynamics rather than from an external scheduler.

The intended new behavior: noise stirs A; B cannot predict the memories it has not learned; that **A–B mismatch** drives A's state into the unlearned memory; B learns it; the mismatch vanishes there; the system moves to the next-most-novel memory; the process self-terminates once B has learned everything A holds.

Why this is interesting:
- It is a model of **replay / consolidation** (teacher → student), in the spirit of hippocampus → neocortex transfer, but with **priority = what the student lacks**.
- The substrate is **predictive coding**. We deliberately use the *linear* covariance-learning predictive-coding associative memory (covPCN), whose memories are **not strong intrinsic attractors** — attractor strength is gated by prediction error (surprise). This is the property we exploit: a memory becomes a transient attractor for A *only when another network (B) is surprised by it*. A on its own is flat.

---

## 2. Networks, notation, and wiring

All symbols are defined here and used consistently throughout.

| Symbol | Meaning |
|---|---|
| `d` | number of neurons per population (state dimension) |
| `P` | number of memories stored in A |
| `x_A ∈ ℝ^d` | A's state (value neurons) — the only thing A "is" |
| `x_B ∈ ℝ^d` | B's state (value neurons) |
| `W_A ∈ ℝ^{d×d}` | A's recurrent weights — **frozen**, zero diagonal; encode the stored memories |
| `W_B ∈ ℝ^{d×d}` | B's recurrent weights — **plastic**, zero diagonal, initialized to 0 |
| `M_A = I − W_A` | A's "mismatch operator" (`I` = identity) |
| `M_B = I − W_B` | B's mismatch operator |
| `m_p ∈ ℝ^d`, p = 1…P | the stored memory patterns |
| `κ > 0` | epistemic gain (strength of the mismatch drive on A) |
| `π_o > 0` | likelihood / input precision (how much B trusts A's activity) |
| `π_s > 0` | prior precision (how much B trusts its own recurrent model) |
| `τ_A`, `τ_s` | time constants of A's state and B's state |
| `η` | learning rate for `W_B` |
| `ξ` | additive noise on A, white, `ξ ~ 𝒩(0, σ_ξ² I)` |
| `r₀` | fixed norm to which `‖x_A‖` is renormalized (the amplitude leash) |

**Wiring (kept maximally simple and biologically literal):**
- **A → B** is the identity: each B neuron receives `+1` from its corresponding A neuron, so B's input is `x_A`.
- **B → A** is the identity: B's descending synapse delivers `x_B` as B's prediction of A.
- The interface therefore computes the raw mismatch `x_A − x_B`.

---

## 3. The three error populations

These are distinct neural populations and must be kept separate (a shared error population breaks the model — see §6/§7).

- **A's self-error:** `ε_A = M_A x_A = x_A − W_A x_A`. Measures how far A's state is from being one of its own stored patterns. **Zero on A's memory manifold** (this is what "A is flat" means).
- **Interface / mismatch error:** `ε_AB = x_A − x_B`. The A–B disagreement. This is the signal that both drives B's learning and drives A's exploration.
- **B's self-error:** `ε_B = M_B x_B = x_B − W_B x_B`. How far B's state is from being a pattern B has stored. This is what lets B **complete** a known pattern (and what gates B's learning).

The whole model lives in the gap between `ε_A` and `ε_AB`: at a memory A holds but B lacks, `ε_A ≈ 0` (A is self-coherent) yet `ε_AB ≠ 0` (B is surprised). That co-occurrence is the definition of a "novel memory" and cannot be detected by a single shared error unit.

---

## 4. Dynamics

Three coupled processes, integrated in continuous time (small step). After each A update, renormalize `‖x_A‖ → r₀`. After each `W_B` update, re-zero its diagonal.

**A's state (the environment, driven by B's action + noise):**
```
τ_A · dx_A/dt = − M_Aᵀ ε_A  +  κ · ε_AB  +  ξ          (then renormalize ‖x_A‖ = r₀)
```
- `−M_Aᵀ ε_A` = A's intrinsic relaxation onto its manifold (damps anything off the stored-pattern subspace).
- `+κ ε_AB` = the mismatch drive (note the **plus** sign — this is the load-bearing choice; it pushes A *toward* states B cannot predict, not away from them).
- `ξ` = noise; it is essential — it seeds (symmetry-breaks) the direction A grows into.

**B's state (perception — fast):**
```
τ_s · dx_B/dt = π_o · ε_AB  −  π_s · M_Bᵀ ε_B
```
- `+π_o ε_AB` pulls `x_B` toward the input `x_A`.
- `−π_s M_Bᵀ ε_B` is B's recurrent pattern-completion (its prior).

**B's learning (slow):**
```
dW_B/dt = η · π_s · ε_B · x_Bᵀ       (zero the diagonal after each step)
```
- Hebbian on B's self-error and B's activity; automatically prioritized (large error → large update) and self-extinguishing (as a pattern is stored, `ε_B → 0` there and learning on it stops).

**Required parameter relations (these are part of the model, not tuning niceties):**
- **Timescale separation:** `τ_s ≪ τ_A ≪ 1/η`. B's state must catch up fast; A drifts slowly; B's weights are the slow, rate-limiting step (the dwell time at each memory).
- **Structure guard:** `κ < σ²_min`, where `σ²_min` is the smallest **nonzero** eigenvalue of `M_Aᵀ M_A` (the "spectral gap" of A's self-error). Below the gap, only on-manifold (real-memory) directions are amplified; above it, A starts amplifying noise.
- **Precision guard:** `π_o > π_s` (input trusted over internal belief).
- `σ_ξ` small but strictly positive; `r₀` fixed (e.g. unit norm).

**Storing memories in A (recipe, not code):** draw P patterns `m_p` (e.g. random ±1 or Gaussian, normalized); set `W_A` by a covariance / Hebbian rule so that each `m_p` lies in the null space of `M_A` (`M_A m_p ≈ 0`, i.e. the patterns are the flat directions of A's manifold). Then freeze `W_A`. Initialize `W_B = 0`.

---

## 5. Intended behavior (the "completion race") and termination

The selection mechanism is a **race in time** between two pattern-completers:

1. Noise perturbs `x_A`. A's self-term immediately damps off-manifold (noise) components; on-manifold (real-memory) components persist and become correlated, structured activity.
2. B races to reconstruct that activity:
   - **Known pattern** (B has it): B's recurrent model snaps `x_B` onto `x_A`, `ε_AB → 0` fast. A gets no sustained drive. The memory is *explained away* — flat, no replay.
   - **Novel pattern** (B lacks it): B can only passively half-copy `x_A`; `ε_AB` **persists** (steady-state residual along that direction). Times `κ`, this drives `x_A` further into the novel memory — self-sustaining.
3. B slowly learns the novel pattern (`W_B` update). Once it can complete it, `ε_AB → 0`, the drive releases, and the dynamics move to the next-most-novel direction.
4. **Termination:** when B has learned all P patterns, `ε_B` and `ε_AB ≈ 0` everywhere on A's manifold; there is no positive drive left; `x_A` merely diffuses. This quiescence is the "transfer complete" signal — it is the *correct* terminal state, not a failure.

Note: this race is the new content relative to a static formulation. The discrimination depends on B being **faster** than A's drift (the `τ_s ≪ τ_A` condition); making B slower should blur the known/novel discrimination.

---

## 6. The two guards (what keeps it honest)

Two failure modes, each blocked by one of the parameter relations above. State both explicitly when reporting.

- **World-facing failure — chasing noise.** If `κ` exceeds the spectral gap, A's drive overwhelms its own damping and `x_A` grows along unstructured noise directions (which B is also "surprised" by). Guard: `κ < σ²_min`, plus A's manifold confinement. This is the discrete analog of the requirement that the world have *learnable structure*.
- **Belief-facing failure — confabulation ("fake memory from B").** If `π_s ≥ π_o`, B's recurrent prior dominates and `x_B` is captured by B's own stored patterns instead of tracking `x_A`; `ε_AB` then reflects B's belief drift rather than genuine novelty, and B "transfers" patterns that are not in A. Guard: `π_o > π_s`.

These two guards are symmetric: one stops A drifting into noise, the other stops B drifting into fantasy — the two ways the loop could drive its error down *without actually transferring a memory*.

---

## 7. What to build, run, and measure

Build A (store P patterns, freeze), initialize B empty, integrate the three equations with renormalization, noise, and the timescale separation. Then characterize:

**Primary result**
- **`F_B` staircase:** track `F_B` averaged over the P stored patterns as a function of time. Expectation: it falls in ~P discrete steps (one per memory transferred) down to a noise floor. This single curve is the main evidence that prioritized transfer is real and not diffusion.
- **Transfer order vs novelty:** the order in which patterns are learned should track how novel each was to B. (Pre-train B on a subset, then check that the held-out patterns transfer first.)

**Mechanism checks**
- **Per-memory mismatch:** `‖ε_AB‖` projected onto each pattern direction over time — should spike then decay, one memory at a time.
- **A's alignment:** cosine similarity of `x_A` with each `m_p` over time — should lock onto a single novel pattern at a time, then release.

**Ablations (each should break the model in a specific, predicted way)**
- `κ` above the spectral gap → A chases noise; alignment with stored patterns degrades; `F_B` staircase breaks.
- `π_s ≥ π_o` → confabulation; `ε_AB` stops reflecting true novelty.
- `τ_s ≈ τ_A` (B not fast) → the completion race blurs; known/novel discrimination weakens. There should be a window of `τ_s/τ_A` where prioritization is sharpest — varying it is the signature that the selection is *dynamic*, not merely spectral.
- No renormalization → `x_A` amplitude diverges.
- No noise → no symmetry-breaking; degenerate / stuck at zero.

**Success criteria:** B learns all P patterns; `F_B → ` floor; transfer order matches novelty; each ablation fails as above.

---

## 8. How it maps to free energy — read this carefully

Two halves of the model map onto free energy with **different degrees of certainty**. This distinction matters; do not flatten it.

**B's generative model.** B treats A's activity as generated by a latent cause `x_B`, with a recurrent Gaussian prior (precision set by `M_B`, weight `π_s`) and a Gaussian likelihood `x_A | x_B ~ 𝒩(x_B, π_o⁻¹ I)` (the identity loading = the value-1 descending synapse). Its variational free energy is exactly
```
F_B = (π_o/2)‖ε_AB‖² + (π_s/2)‖ε_B‖².
```

**Variational free energy (VFE) — the certain / exact mapping.** B's state relaxation (perception) and `W_B` update (learning) are *exact gradient descent on `F_B`*. This is textbook variational inference; it is derived, with no caveats. Treat this as solid.

**Expected free energy (EFE) — the structural / constructed mapping.** A's drive `κ ε_AB` is read as **B's action** minimizing expected free energy — specifically its epistemic (information-gain) term, with `½‖ε_AB‖²` used as the linear-Gaussian *proxy* for information gain about `W_B`. This mapping is structural/functional, not derived: it requires (i) accepting EFE as the action objective at all (EFE is not uniquely derived from the free-energy principle — see Millidge et al.), and (ii) the proxy, which equals true information gain only in the isotropic-uncertainty limit and is kept honest otherwise by A's manifold (which guarantees the surprise B chases is *learnable*). So: **VFE mapping = exact and derived; EFE mapping = an interpretation that is structurally faithful but rests on a construction plus a proxy.** Report it as such.

The two guards of §6 are the priors this reading needs: structure (so the info-gain proxy targets learnable directions, not noise) and precision (so perception tracks input rather than belief).

---

## 9. Two readings of the same dynamics — and they are distinct mathematical objects

The same equations admit two formal descriptions. **Keep them distinct; do not claim they are the same object.**

**Reading 1 — minimax / saddle.** The full `(x_A, x_B, W_B)` flow is the gradient flow of a single functional, schematically `Φ = F_A − κ·F_AB`, with **A descending Φ and B ascending it**. This is an exact algebraic property of the equations: a saddle / minimax. Geometrically, `Φ` is a **bowl** along directions B can predict (A is pinned — flat) and a **hill** along novel directions (A slides off — the replay drive); B's learning turns each hill back into a bowl. This is the solid, theorem-grade statement about the equations.

**Reading 2 — single-agent descent on EFE.** B is the only agent. It descends VFE (perception, learning) and EFE (action); A is the environment it acts on. B's behavior is pure descent; the closed A–B loop is still a saddle (as every active-inference agent's loop is — the world does not minimize the agent's free energy).

**Crucial caveat for the writeup and the visualization:** these are **two distinct objects, not totally related.** The minimax functional `Φ` is exact and well-defined. The EFE is a *different* functional; it coincides with the `−κ F_AB` term of `Φ` only through the information-gain proxy of §8. The dynamics can be *described* both ways, but "the minimax on `Φ`" and "a descent on EFE" are not the same mathematics — the former is exact, the latter is the constructed interpretation. Do not conflate `Φ` with the EFE, and do not present the saddle as if it *were* the active-inference object.

---

## 10. The myopic action and its hidden prior (the forward model)

The action `u = κ ε_AB` (§4) is the **myopic** (greedy) form of expected-free-energy action selection: it follows the *instantaneous* gradient of the information-gain proxy. This is the globally information-maximizing action **only when A's controlled landscape is benign** — smooth and trap-free, so local ascent reaches the global optimum without ever passing through low-information regions. In a rugged A (separate deep basins, barriers), the greedy action would stall at the nearest local information peak; the genuinely optimal action could require routing *through already-known territory* to reach a distant unknown region — and the bare gradient does not have that shape.

So the simple derived action silently rests on a geometric prior: that following the mismatch leads, step by step, to the information without needing detours. This is **more than reachability** (a yes/no — is a state attainable under *some* control). It is *reachability conditioned by the action* — the map from what B does to where A ends up — and that map is a rigorous, standard object:

| Informal phrasing | Rigorous object |
|---|---|
| where my action takes A | **controlled transition model** `P(s′ \| s, a)` — a *forward / world model* |
| that model rolled forward under a policy | **policy-conditioned predictive distribution** `Q(o \| π)` |
| the states I can drive A into | **policy-conditioned reachable set** (the support of `Q(o \| π)`) |
| discrete active-inference form | the transition matrix **B**; policies index sequences of it; EFE is summed over a horizon |

The implication: to know where its pushes take A, B must — implicitly or explicitly — hold a **causal forward model of A's controlled dynamics**. Non-myopic exploration — planning a route toward distant novelty — *is* optimization over this rolled-forward model across a horizon. The myopic action drops the horizon and is correct only when smoothness makes a single step sufficient.

**This is the same object as the dark-room weighting (the anticipation prior).** The epistemic value weights information gain by `Q(o \| π)` — B's own predictions of where its actions lead. The dark room is the *poverty* of that model (it predicts nothing informative is reachable, so all gain is weighted to zero); the smoothness that lets the greedy action work here is the *benignity* of the same model. One construct, two faces: named from the anticipation side it is "does B expect anything out there to find"; named from the action side it is "does following the push actually get there." Both are the forward / transition model. This also sharpens §8: the EFE mapping is structural not only because of the information-gain proxy, but because the action is myopic — a one-step reduction that is exact only under this smoothness prior.

**Where this model sits (implementation boundary).** Because A is linear and its memory manifold is a subspace, A's controlled landscape is smooth and trap-free: the myopic gradient genuinely *is* globally information-maximizing, so B carries **no explicit transition model of A and does no planning** — the lookahead collapses into one step. This is precisely why the simple action `u = κ ε_AB` is sufficient and why the spec needs no policy/horizon machinery. The boundary to flag for any extension: if A is made nonlinear or rugged (e.g. deep point-attractor memories), the greedy action is no longer correct, and B would need an explicit forward model of A plus a planning horizon — at which point `u = κ ε_AB` must be replaced by EFE minimization over rolled-out policies.

---

## 11. Visualization ideas

- **Saddle cross-sections (the key intuition figure):** two small potential plots side by side — a **bowl** (a direction B can predict; `x_A` rests at the bottom; stable, no replay) and a **hill** (a novel direction; `x_A` sits near the top and slides off; the replay drive), with the caption that learning flips each hill into a bowl. Together they show why `Φ` is a saddle.
- **`F_B` staircase over time:** the primary result plot (one downward step per transferred memory).
- **Replay raster / alignment heatmap:** `x_A`-vs-`m_p` cosine over time, showing one novel memory replayed at a time, in novelty order.
- **Optional minimax schematic:** A minimizes / B maximizes `Φ` — *with an explicit annotation* that the single-agent EFE descent is a separate object (per §9), so the figure does not imply the saddle and the EFE are the same thing.

---

## 12. Literature we build on

- **Predictive-coding associative memory (substrate).** Covariance-learning predictive-coding networks, *PLOS Computational Biology* 2023, DOI `10.1371/journal.pcbi.1010719` (Tang and colleagues). We use the **linear** covPCN variant: its stored patterns form a flat manifold (a hyperplane / line attractor) rather than deep point attractors, so A has **no strong intrinsic attractors** and its memories are surprise-gated. This is the property the whole model exploits.
- **Active inference / free-energy principle (Friston).** VFE for perception and learning; EFE and the epistemic-value (information-gain) decomposition for action (Friston et al. 2015, "Active inference and epistemic value"; Friston et al. 2017, "Active inference: a process theory"). Bogacz 2017 ("A tutorial on the free-energy framework…") for the predictive-coding math.
  - **EFE caveat (cite honestly):** Millidge, Tschantz & Buckley 2021, "Whence the Expected Free Energy?" — EFE is a constructed functional, not uniquely derived; the exploratory drive does not fall out of free-energy minimization by itself. This is why §8 marks the EFE mapping as structural rather than certain.
- **Prioritized replay.** Schaul et al. 2015, "Prioritized Experience Replay" — priority by TD error; here priority is the **student's own prediction error**.
- **Generative replay / continual learning.** Shin et al. 2017, "Continual Learning with Deep Generative Replay" — a teacher replays to a student; here the replay is *prioritized toward the gap*.
- **Complementary learning systems.** McClelland, McNaughton & O'Reilly 1995 — A ≈ fast teacher (hippocampus-like), B ≈ slow student (neocortex-like).

**One-line positioning:** this is *prioritized generative replay where the priority signal is the student's prediction error* — equivalently, an active-inference reading in which the student (B) forages for novelty in the teacher's (A's) memory space.
