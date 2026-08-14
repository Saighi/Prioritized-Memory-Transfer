# Paper outline — working document

**Working title:** PROVISIONAL, **Paul owns the wording**. Must contain "adversarially coupled"
(the reader should identify the system at a glance). Live version is in `latex/manuscript.tex`.

**Venue/format:** PLOS-style full paper (uses `Paper/example layout/manuscript.tex` template).

**Center of gravity:** **Memory transfer is the primitive.** Union and continual learning are built on it.

**THE SYSTEM IS AN "ADVERSARIALLY COUPLED PREDICTIVE-CODING ASSOCIATIVE MEMORY NETWORK" (2026-07-27).**
Framing decision, load-bearing everywhere:
- We do **NOT** present the full network as a predictive-coding network. It does not minimize a free
  energy, so calling it PC would be dangerous. The **components** are PC associative memories; the
  **coupling** is adversarial. (The title's "predictive-coding associative memories" therefore
  describes the parts and stays accurate.)
- The adversarial coupling coefficient is **κ (kappa)**, NOT `π_ST`, and it is **never called a
  precision**. Justification to state once in the paper: *a precision weights an error term in a free
  energy the agent MINIMIZES, and so reads as an inverse variance; κ weights a term the teacher
  MAXIMIZES. Same slot in the equation, same magnitude at the saddle, but not an inverse variance and
  no probabilistic reading.* Then one side line: *equivalently, κ = −π_TS corresponds to reversing the
  sign of the interface precision.*
- **κ is signed** (κ<0 adversarial/sleep, κ>0 ordinary PC coupling/wake; sign(κ) is still the
  wake/sleep switch), **plain, no subscript** (index κ_1, κ_2 only when interleaving needs several
  interfaces). Derivations use **|κ|**. Renaming is NOT the same as splitting the sign into a separate
  ±1 gate, which was rejected earlier and stays rejected.
- **The min-max saddle is the CENTRAL object**, the first thing introduced, and the whole dynamics is
  derived from it. See "Math placement" below for the reordered hierarchy.
- Potential stays **Φ** (not Θ).

**Union strategy:** interleaved only. The online additive sum is NOT presented as a method — its
rank-collapse (`α₁α₂E[x₁x₂ᵀ]` cross-term) appears once, as the justification for interleaving,
and (optionally) as a supplementary ablation.

---

## The ladder (one mechanism, three rungs)

1. **Transfer** — one teacher T → one student S. Novelty-gated, prioritized to what S lacks.
   Adversarial coupling `κ < 0` = sleep/replay; `κ > 0` = wake/recall.
2. **Union** — two teachers → one student, by **interleaved** rehearsal (rehearse one teacher per bout,
   never sum). Merges single-memory AND correlated teachers; back-and-forth cancels crosstalk.
3. **Continual learning** — a stream, via buffer → consolidate (interleaved) → download.
   Retains the whole stream where a buffer-only control catastrophically forgets.

---

## Section structure

### Abstract / Author summary
Hook = catastrophic forgetting + biological sleep replay (same audience as the AR paper).
Twist = priority is the **student's own prediction error**, derived from predictive coding,
not an external scheduler. Transfer is the primitive; continual learning is the payoff.

### Introduction
- Continual learning & catastrophic forgetting (reuse AR-paper scaffolding).
- Memory rehearsal / replay during sleep → the open question: how is replay *prioritized* and *autonomous*?
- Our answer: novelty-gated transfer in a **linear covPCN**, where memories are surprise-gated,
  not deep attractors — so a memory becomes a transient attractor for T only when S is surprised by it.
- Positioning: Tang et al. 2023 (substrate), Friston / active inference (VFE/EFE reading),
  Schaul (prioritized replay — here priority = student error), McClelland CLS (T≈hippocampus, S≈neocortex).

### Methods / Model
**ORDER CORRECTED 2026-07-27: circuit FIRST, Φ immediately after as a CHARACTERIZATION, never as the
opening definition.** Rationale (Paul's call after weighing it): Φ constrains exactly one thing, the
relation between the two cross-terms (`κ = −π_TS`); the rest of Φ is just "each network descends its
own energy", true with or without it. Opening with Φ would (i) promise a derivation but deliver one
sign condition, inviting the GAN comparison where the objective is a genuine design choice rather
than reverse-engineered, and (ii) **fight the abstract's own hook** ("standard PC architecture plus a
minimal modification"), which is architectural and only lands if the reader meets the circuit first.
NOTE: this does NOT undo saddle-first (D8) — that governs the *theorem* order, and Theorem 1 is still
the min-max, still first among formal results. Only the Models *prose* order changes. Tang does the
same: architecture in Models, theorems in Results.

1. Architecture: two covPCN associative memories, teacher below, student above, the interface.
2. The two free energies (native to the PC audience; no friction).
3. The coupling, noting **its sign is free**, and that reversing it turns error-suppression into
   error-seeking. **This is the mechanism, and it is far more vivid than Φ. It carries the intuition.**
4. *Then* the characterization: with the adversarial sign, and only at `κ = −π_TS`, the coupled
   system is exactly a zero-sum min-max on `Φ = F_T − F_S`.

- **Φ, stated at step 4 (not step 1):** `Φ(x_T, x_S, W_S) = F_T(x_T) − F_S(x_S, x_T; W_S)`, with
  `F_T = (π_T/2)‖M_T x_T‖²` and `F_S = (π_TS/2)‖x_S − x_T‖² + (π_S/2)‖M_S x_S‖²` (`M = I − W`).
  Flow: `τ_T ẋ_T = −∇_{x_T}Φ`, `τ_S ẋ_S = +∇_{x_S}Φ`, `Ẇ_S = +η P_0(∇_{W_S}Φ)`.
  **KEY GAIN: `κ = −π_TS` becomes a CONSEQUENCE, not a postulate.** By §8's converse (mixed partials:
  the cross-blocks `−κI` and `π_TS I` must be transposes), a single potential generates this
  descent–ascent flow *iff* `κ = −π_TS`. The adversarial coupling is therefore the unique one making
  the network a min-max on one function; it is not tuned.
  **Say WHY a single potential is worth wanting, or the uniqueness argument looks merely aesthetic:**
  `κ = −π_TS` is the point where both populations weight the shared error *identically but with
  opposite sign*, which is why the circulation vanishes there. That is a balance condition with
  physical content, not a preference for tidy formalism.
  **Three caveats to state (accepted by Paul 2026-07-27):** (i) this is the SLEEP dynamics; the wake
  counterpart should be both populations descending `F_T + F_S`, forcing `κ = +π_TS` by the same
  argument, but it is NOT written down in final_proofs.md, so **verify before asserting it**;
  (ii) the noise `ξ` is NOT generated by Φ (it is added, and it is what seeds empty directions);
  (iii) the constraints are not generated by Φ either (the norm leash makes the teacher block a
  projected flow `−P⊥∇_{x_T}Φ`, still monotone in Φ, hence an exact saddle on the sphere; the
  zero-diagonal `W_S` constraint is handled the same way by `P_0`).
- covPCN substrate recap (flat memory manifold); each population is a covPCN associative memory
  (cite Tang heavily), the coupling is a top-down PC interface term with the adversarial sign.
- **Present a GENERAL activation `f` from the start** (memory condition `f(m_p) = m_p`; identity and
  rectification-with-non-negative-patterns both satisfy it), then state the analysis is carried out
  for the linear instance. This makes linear *the analyzable instance of a general model*, so the
  rectified results are not bolted on at the end.
- Two-population wiring: **teacher BELOW, student ABOVE** (see hierarchy convention in the tracker),
  three error populations, interface error `ε_TS = x_S − x_T`.
- The three dynamics equations; **adversarial coupling `κ`** as the signed sleep/wake knob.
- **Operating point (changed 2026-07-27): `κ = −π_TS` for ALL simulations** (the exact zero-sum
  point), as the *default* value, still changeable if needed. Containment is then obtained by raising
  the teacher's self-precision `π_T` above `|κ|` by a hand-chosen factor, verified numerically
  (`manifold_leakage`, `terminal_occupancy`). Precision regime `π_TS > π_S` unchanged.
  *Rationale:* `π_T` acts essentially only normal to the manifold (on-manifold the teacher's
  self-error vanishes), so it should tighten containment without slowing transfer. **Paul does the
  code/numerical work; the factor must be measured, not assumed.**
- The interleaved union interface (rehearse one teacher per bout).
- The continual loop (buffer → consolidate → download).

### Results (the ladder + the theory rung + the nonlinear extension)
1. **Prioritized transfer works** → novelty-spectrum staircase + `x_T`-alignment heatmap.
   1b. **Targeted vs randomly-evoked replay** → prioritized replay makes the transfer *efficient*;
   randomly evoking the teacher gives slow, undirected transfer. (Committed to in the abstract.
   NEEDS new code + figure.)
2. **It is a saddle** → bowl-vs-hill cross-sections at the zero-sum point `κ = −π_TS`.
3. **Composition = memory union** → interleaved crosstalk cancellation (sawtooth to zero).
   One-line note on why summing fails (rank collapse) → motivates interleaving.
4. **Continual learning** → retention matrix vs. buffer-only control (catastrophic forgetting).
5. **Rectified networks replay individual memories** (NEW) → ReLU + non-negative patterns. Dedicated
   section, LAST, mirroring Tang's own final subsection ("Nonlinear covPCNs learn individual
   attractors"), where the nonlinearity likewise buys individual attractors over subspace behaviour.
   Content: transfer preserved; **cone confinement** as the seed-stable signature; MNIST "dreams";
   honest cost (~2x looser hold); the within-cone-linearization tie-back to the analysis.
   **Claim must be TIME-RESOLVED:** early replay lands near individual memories, drifting to mixtures
   as the student learns and cancels them (self-termination visible in trajectory geometry). The
   run-averaged participation ratio is not discriminative and must not carry the claim.

### Discussion
- Free-energy interpretation: the **components** descend free energies; the **coupled system does
  not** (it is adversarial). State honestly; no global VFE-minimization claim for the full network.
- Biological reading: hippocampus→neocortex, sleep replay, wake/sleep = sign of κ.
  **Teacher = bottom = hippocampus; student = top = neocortex** (see tracker's hierarchy convention).
  Precise Shin & Jadhav framing + the error-neuron-vs-representation-neuron distinction: see the
  tracker's Discussion entry, sourced from `neuroscience_interpretation.md`.
- Limitations: the **analysis** is linear ⇒ subspace/effective-rank statements (directions, not named
  patterns). The **demonstration** is not linear-only (see Results 5). Remaining open: saturating
  activations (untested; the tanh runs did not probe saturation), and episodic one-pattern-per-step
  replay.
- Relation to Tang and to the previous autonomous-retrieval paper.

### Conclusion
Short, like the AR paper.

---

## Figures

### Core (5)
| Fig | Content | Source |
|---|---|---|
| 1 | Architecture schematic — **student ABOVE, teacher BELOW**, three error populations, wiring, adversarial coupling κ, wake vs. sleep sign | new diagram |
| 2 | Saddle cross-sections — bowl (predictable dir) vs. hill (novel dir); learning flips hill→bowl | nb two_network/02, 06 |
| 3 | Novelty-spectrum staircase (primary transfer result) + `x_T`-alignment heatmap | nb two_network/01, 05 |
| 3b | **Targeted vs randomly-evoked replay** (efficiency comparison) — NEEDS NEW CODE | TBD |
| 4 | Memory union — interleaved crosstalk sawtooth to zero | nb interleaved/01, 02 |
| 5 | Continual learning — retention heatmap vs. buffer-only control collapsing to latest | nb continual/01 |
| 6 | **Rectified network** — cone confinement + MNIST dreams; time-resolved early(individual)→late(mixture) | nb memory_transfer_non_linear/01, 03 |

**NOTE (2026-07-27):** every existing figure/number was produced at the old `pi_ST="auto"` operating
point. Moving to `κ = −π_TS` invalidates the reported values, so the whole result set must be re-run
before figures go into the manuscript. (Paul does this.)

### Supplementary figures
- Additive source-rebalancing & self-termination (additive §19.3 / 19.8).
- Ablation grid A–I (additive §21).
- Stop-grad dendritic variant (`docs/stopgrad_dendritic.png`, nb two_network/07).
- Correlation / sparsity / timescale (τ_S/τ_T window) sweeps.
- **No-noise slow-mixing version** (annex — the transfer that relies on integrator imprecision
  rather than injected noise; tracked in TODO.md).
- (Optional) SMACOF/VFE landscape draping — intuition figure, talk/supp only.

---

## Math placement

**REORDERED 2026-07-27 — SADDLE FIRST.** The min-max is now the primary object and everything is
derived from it. Old chain was: descent → sign identity/ascent → landscape → prioritization → saddle.
**New chain: min-max saddle → ascent-on-surprise → landscape → prioritization.**
Derivations tracked in [maths/curriculum.md](maths/curriculum.md); the
**source of record** for statements + proofs is [maths/final_proofs.md](maths/final_proofs.md); the
fully-tutorialized versions live in [maths/detailed_proofs.md](maths/detailed_proofs.md).
Stated inline in the "Model and analysis" block (Tang-style), full proofs in Materials and methods.

### Core — the formal results (in body, in this order)
- **Model & definitions:** three state/error dynamics; **adversarial coupling `κ`** (signed
  sleep/wake), never called a precision; `F_S = (π_TS/2)‖ε_TS‖² + (π_S/2)‖ε_S‖²`, `F_T`;
  `π_TS > π_S` stated as an operating regime, NOT a theorem assumption.
  **Status note (must appear in the paper, REWRITTEN 2026-07-27):** the system is *adversarially
  coupled*, so **no global free-energy minimization is claimed for the full network**. The components
  are PC associative memories that descend their own free energies; the coupling is adversarial. At
  the operating point `κ = −π_TS` the whole system is an exact **zero-sum min-max on
  Φ = F_T − F_S`** (Theorem 1; note the FULL `F_S`, all three variables evolving, no adiabatic
  assumption). Active-inference link = resemblance only, discussion-only, never a
  free-energy-minimization claim.
- **Proposition 1 (setup, feeds Theorem 1)** — student inference & learning descend `F_S`
  (exact + projected gradient descent). Demoted from headline to setup by the reorder.
- **Theorem 1 — THE ADVERSARIAL MIN-MAX (was Proposition 2; now the first and central result).**
  **Statement (§8, verbatim level):** the deterministic sleep dynamics is the min-max flow of the
  single potential `Φ(x_T, x_S, W_S) = F_T(x_T) − F_S(x_S, x_T; W_S)` — teacher descends, student
  ascends in both `x_S` and `W_S`, each with its own mobility — **iff `κ = −π_TS`**. Uses the FULL
  `F_S`, with all three variables evolving (NOT the reduced `F_S^eq`, and no adiabatic assumption).
  The reduced form `Φ^eq = F_T − F_S^eq` is the adiabatic special case (§8 Remark 3).
  **VERIFIED 2026-07-27 against final_proofs.md §4 + §8.** Two corrections to earlier notes:
  1. **Theorem 1 needs LESS machinery than claimed, not more.** It does NOT absorb the
     envelope/strict-convexity step. §8's Proposition 2 holds with *all three variables evolving*
     ("neither A1a nor A1b is used"); its proof needs only `∇F_T` and the **sign identity**
     (§4 Step 1, valid at any `x_S`, settled or not). No adiabatic elimination, no uniqueness
     argument, no envelope. Those enter only at **Corollary 1**.
     *This is the real argument for saddle-first:* the min-max is the result requiring the least
     machinery, so the idealizations arrive later, attached to the corollary that actually needs
     them. Dependency chain: sign identity → **Theorem 1 (min-max)** → Corollary 1 (adiabatic +
     envelope) → Lemma 1 → Corollary 2 → remark.
  2. **`Φ_w` is NOT a "weighted minimax" and must never be written as one.** It is true only for the
     **teacher block alone, after adiabatic elimination** (where `x_S` is slaved to `x_T`, so being a
     gradient flow is nearly automatic); it is just §8 Remark 3 with the coefficient left general.
     For the **joint** `(x_T, x_S, W_S)` system, §8's (⇒) direction proves that when `κ ≠ −π_TS`
     **no C² potential** generates the descent–ascent flow with these fixed block mobilities
     (mixed-partials: cross-blocks `−κI` and `π_TS I` must be transposes ⇒ `κ = −π_TS`). The
     circulation `|κ+π_TS|√d` is exactly the measure of that joint non-integrability. Claiming a
     weighted minimax would contradict our own Proposition 2.
- **Corollary 1 — the adversarial push is gradient ascent on the student's surprise (was Theorem 1;
  now downstream).** Immediate from Theorem 1, since descending `−F_S^eq` is ascending `F_S^eq`:
  `u = κ ε_TS = (|κ|/π_TS)∇_{x_T}F_S^eq`. The *sign identity* stays the mechanical core (only the
  interface term of `F_S` contains `x_T`, so `∇_{x_T}F_S = −π_TS ε_TS`; teacher and student share one
  interface energy through opposite signs; `κ > 0` gives descent and no transfer). **Fast student
  state + frozen weights**, deterministic. Honest caveats: *local* ascent; the push rescales but does
  not *seed* an empty direction (noise seeds it).
- **Lemma 1** — the fast student, **mode by mode** (dynamics-first, the default proof): rotate into
  `S_S`'s eigenframe, settle one scalar tug-of-war per mode (`s_k* = (1−n_k)c_k`, error `−n_k c_k`,
  divide by a positive number — no inverses), reassemble — the novelty operator
  `N_S = U diag(n_k) Uᵀ` is *born* as the name of the reassembly (`ε_TS = −N_S x_T`,
  `x_S* = (I − N_S)x_T` — attenuated copy; equivalently the closed form
  `π_S S_S(π_TS I + π_S S_S)⁻¹`). The teacher's per-mode law `ċ_k = (|κ|/τ_T) n_k c_k` —
  **prioritization** — falls out inside the lemma, from the dynamics alone.
- **Corollary 2 (renumbered from Corollary 1) — the bridge: the landscape is the novelty-weighted quadratic.** The settled
  student's free energy is the novelty score: `F_S^eq(x_T) = (π_TS/2) x_Tᵀ N_S x_T`, so
  `∇F_S^eq = π_TS N_S x_T` (per direction interface `n²` + self `n(1−n)` = `n`; proof = a
  three-line dial-by-dial continuation of Lemma 1). This is where the two independent routes —
  the energy story (Thm 1 + Cor 1) and Lemma 1's dynamics story — provably meet (checked to 1.8e-15).
  Terminology fixed here: *novelty* = weight-level operator/spectrum (`N_S`, `n_k`), *surprise* =
  state-level scalar (`F_S`, `F_T`); no separate "teacher's surprise" is ever defined.
- **Remark — prioritization, the energetic reading (downstream of Thm 1 + Cor 1 + Cor 2).** The
  per-mode law derived dynamically in Lemma 1 is *also* gradient ascent on the quadratic landscape,
  decoupled along `N_S`'s eigenvectors — the two readings coincide (Cor 2's consistency). Learned
  (`n_k=0`) frozen, novel grown ∝ novelty, most-novel dominates (what the experiments measure);
  self-limiting `|κ|n_k ≤ (|κ|π_S/π_TS)‖M_S u_k‖²` — the push dies quadratically as the
  student learns; ratios survive the leash.
- **Manifold containment (empirical — no lemma, cut 2026-07-20; KNOB CHANGED 2026-07-27)** — since
  `κ = −π_TS` is now fixed by the zero-sum condition, containment is no longer obtained by shrinking
  `κ`. Instead **`π_T` is raised above `|κ|`** by a hand-chosen factor so the self-pull dominates off
  the manifold; the teacher's stay on its memory manifold is shown numerically (`manifold_leakage`,
  `terminal_occupancy`). The factor must be **measured, not assumed** (the old guard scaled like
  `1/σ²_min`, so the needed value depends on memory geometry). (Archived sufficient-condition lemma:
  maths/additional_proofs_not_in_paper.md — not for the paper.)
  *Bonus:* fixing `κ` definitionally removes the two `σ²_min`-derived failure modes we already hit
  (the continual-learning `spectral_gap` guard collapse, and the ReLU/tanh `pi_ST="auto"` → −8e-05
  confound that produced the wrong "saturating units break it" conclusion).
- **Proposition 2 — GONE as a separate result** (2026-07-27). Its content (the exact saddle) is now
  **Theorem 1**, promoted to first. Nothing else claims the number.
- **Scope remark** — the **analysis** is linear ⇒ subspace/effective rank, not named episodes. The
  **demonstration** covers linear and rectified (Results 5); for ReLU the linear analysis is the
  exact **within-cone linearization** (one copy per activation cone), not an approximation.

### Empirical (figures, NOT theorems)
- Completion figure (`U_Tᵀ N_S(t) U_T → 0`): "reliable completion across tested regimes", with the
  open point and the "steps are timescale/plotting artifacts" caveat stated.
- **Interleaving union + continual learning** — now entirely empirical (no theorem): the crosstalk
  sawtooth (merging teachers) and the retention-vs-buffer-only-forgetting heatmap. The primitive
  *composes*, demonstrated in simulation.
- Noise-driven symmetry breaking / cancellation escape: mechanism + figure, no proof.

### Supplementary (rigor + honesty)
- Full algebra for Prop 1, **Theorem 1 (min-max)**, Corollary 1 (ascent on surprise), Lemma 1,
  Corollary 2 (surprise identity), the prioritization remark; adiabatic elimination.
- **Off-zero-sum ablation** (`w = |κ|/π_TS ≠ 1`): the retired `"auto"` κ mode, `pi_ST_safety` and the
  `σ²_min`-derived guard live here now, not in the paper's operating description.
- Honest limitations note: completion of the coupled dynamics and finite-noise/step robustness are
  empirical; a full stochastic-approximation treatment is future work. No residual-bound theorems.

### Future work (flagged, not proven here)
- The **restricted-novelty / manifold analysis** (operator `A_S = U_Tᵀ N_S U_T`): once the teacher's own
  memory geometry is folded in, *which* surprising directions dominate and in what order. Theorem 1 only
  characterizes the instantaneous student push up the surprise; the manifold refinement is deferred.
- The **old self-extinction corollary** (`vᵀN_Sv ≤ (π_S/π_TS)‖M_Sv‖²`) is now a one-line bound inside
  the prioritization remark, not a standalone result. (The name **Corollary 1** now denotes the
  *surprise identity* `F_S^eq = (π_TS/2)x_TᵀN_Sx_T`, a core result in the body.)

### Dropped entirely (as *theorems*)
Reference-weight-distance spine; Barbalat/LaSalle idealized convergence; stochastic-approximation
robustness (imported "Theorem 2"); persistent-excitation / no-partial-trap; Bingham / mixing-time /
Freidlin–Wentzell. **Online additive network/method** — not even as a counterexample.
**The interleaving-union proposition** — cut; interleaving/union kept only as an *empirical*
figure (see above), no theorem. (Its number, **Proposition 2**, is now reused for the exact saddle.)

---

## Open items to settle before writing Results (from TODO.md)
- **Claim scope on VFE:** the *components* descend free energies; the *coupled system* does not. No
  integrated/dataset-level VFE-minimization claim anywhere (the adversarial framing makes this
  simpler than it used to be: we are no longer defending a PC reading of the whole network).
- ~~Verify the `Φ_w` generalization~~ **DONE 2026-07-27.** Result: valid for the reduced teacher block
  only, NOT a weighted minimax (§8 forbids it off the zero-sum point). Also corrected: Theorem 1 does
  not need the envelope step. See Theorem 1 above.
- Confirm the staircase step-count = effective rank story for the exact pattern sets used in Fig 3.
- **Renumber/rename pass on the maths documents** (`final_proofs.md`, `detailed_proofs.md`,
  `curriculum.md`, `cheatsheet.md`): `π_ST → κ` everywhere, and reorder to saddle-first
  (§8 Prop 2 → Theorem 1, old §4 Thm 1 → Corollary 1, §6 Cor 1 → Corollary 2). NOT yet done.
- Title: Paul owns it; must contain "adversarially coupled".
