# Paper outline — working document

**Working title:** *Memory transfer and continual learning in predictive-coding associative networks*
(alt, transfer-forward: *Prioritized memory transfer: a predictive-coding primitive for continual learning*)

**Venue/format:** PLOS-style full paper (uses `Paper/example layout/manuscript.tex` template).

**Center of gravity:** **Memory transfer is the primitive.** Union and continual learning are built on it.
The reversed-precision sleep/wake mechanism and the free-energy reading are the connective tissue.

**Union strategy:** interleaved only. The online additive sum is NOT presented as a method — its
rank-collapse (`α₁α₂E[x₁x₂ᵀ]` cross-term) appears once, as the justification for interleaving,
and (optionally) as a supplementary ablation.

---

## The ladder (one mechanism, three rungs)

1. **Transfer** — one teacher T → one student S. Novelty-gated, prioritized to what S lacks.
   Reversed precision `π_ST < 0` = sleep/replay; `π_ST > 0` = wake/recall.
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
- covPCN substrate recap (linear variant, flat memory manifold) — brief, cite Tang heavily.
- Two-population wiring: T above S, three error populations, interface error `ε_TS = x_S − x_T`.
- The three dynamics equations; **reversed precision** `π_ST` as the signed sleep/wake knob.
- The operating regime (precision `π_TS > π_S`; reversed precision kept well below `π_T` —
  code default: half the stability threshold) — stated here; containment shown numerically.
- The interleaved union interface (rehearse one teacher per bout).
- The continual loop (buffer → consolidate → download).

### Results (four blocks = the ladder + the theory rung)
1. **Prioritized transfer works** → novelty-spectrum staircase + `x_T`-alignment heatmap.
2. **It is a saddle / free-energy descent** → bowl-vs-hill cross-sections + exact-saddle condition `π_ST = −π_TS`.
3. **Composition = memory union** → interleaved crosstalk cancellation (sawtooth to zero).
   One-line note on why summing fails (rank collapse) → motivates interleaving.
4. **Continual learning** → retention matrix vs. buffer-only control (catastrophic forgetting).

### Discussion
- Free-energy interpretation: **VFE mapping exact and derived; EFE mapping structural** (state honestly, per spec §8).
- Biological reading: hippocampus→neocortex, sleep replay, wake/sleep = precision sign.
- Limitations: linear ⇒ **subspace** consolidation (directions, not named patterns);
  nonlinear / soft-WTA extension for episodic one-pattern-per-step replay.
- Relation to Tang and to the previous autonomous-retrieval paper.

### Conclusion
Short, like the AR paper.

---

## Figures

### Core (5)
| Fig | Content | Source |
|---|---|---|
| 1 | Architecture schematic — T-above-S, three error populations, wiring, wake vs. sleep sign flip | new diagram |
| 2 | Saddle cross-sections — bowl (predictable dir) vs. hill (novel dir); learning flips hill→bowl | nb two_network/02, 06 |
| 3 | Novelty-spectrum staircase (primary transfer result) + `x_T`-alignment heatmap | nb two_network/01, 05 |
| 4 | Memory union — interleaved crosstalk sawtooth to zero | nb interleaved/01, 02 |
| 5 | Continual learning — retention heatmap vs. buffer-only control collapsing to latest | nb continual/01 |

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

The ownable formal hierarchy (logical chain: descent → sign identity/ascent → landscape →
prioritization → saddle). Derivations tracked in [maths/curriculum.md](maths/curriculum.md); the
**source of record** for statements + proofs is [maths/final_proofs.md](maths/final_proofs.md); the
fully-tutorialized versions live in [maths/detailed_proofs.md](maths/detailed_proofs.md).
Stated inline in the "Model and analysis" block (Tang-style), full proofs in Materials and methods.

### Core — the formal results (in body, in this order)
- **Model & definitions:** three state/error dynamics; reversed precision `π_ST` (signed sleep/wake);
  `F_S = (π_TS/2)‖ε_TS‖² + (π_S/2)‖ε_S‖²`, `F_T`;
  `π_TS > π_S` stated as an operating regime, NOT a theorem assumption.
  **Status note (must appear in the paper):** the dynamics is the primitive, the energies are
  partial descriptors — wake is fully VFE (every force a gradient of a bounded free energy);
  the sleep flip is the one postulated, non-VFE element (no bounded energy owns the reversed
  interface term), characterized exactly by Thm 1 (ascent on the student's settled F_S =
  curiosity) and Prop 2 (zero-sum saddle at `π_ST = −π_TS`); active-inference link = resonance,
  speculative, discussion-only.
- **Proposition 1** — student inference & learning descend `F_S` (exact + projected gradient descent).
- **Theorem 1 — the push is gradient ascent on the student's surprise (sign identity + envelope; no
  `N_S`).** Step 1, *sign identity* (any `x_S`): only the interface term of `F_S` contains `x_T`, so
  `∇_{x_T}F_S = −π_TS ε_TS` and the push `u = π_ST ε_TS = (|π_ST|/π_TS)∇_{x_T}F_S|_{x_S}` — teacher
  and student share one interface energy through opposite-signed precisions; the sign of `π_ST` IS
  the wake/sleep switch (`π_ST > 0` gives descent, no transfer). Step 2, *envelope*: `F_S` strictly
  convex in `x_S` ⇒ unique settled state; differentiating through the minimizer is free, so
  `u = (|π_ST|/π_TS)∇F_S^eq` with `F_S^eq = min_{x_S}F_S` — **exact gradient ascent on the student's
  settled VFE**, every point. **Fast student state + frozen weights**, deterministic. Honest caveats:
  *local* ascent; the push rescales but does not *seed* an empty direction (noise seeds it).
- **Lemma 1** — the fast student, **mode by mode** (dynamics-first, the default proof): rotate into
  `S_S`'s eigenframe, settle one scalar tug-of-war per mode (`s_k* = (1−n_k)c_k`, error `−n_k c_k`,
  divide by a positive number — no inverses), reassemble — the novelty operator
  `N_S = U diag(n_k) Uᵀ` is *born* as the name of the reassembly (`ε_TS = −N_S x_T`,
  `x_S* = (I − N_S)x_T` — attenuated copy; equivalently the closed form
  `π_S S_S(π_TS I + π_S S_S)⁻¹`). The teacher's per-mode law `ċ_k = (|π_ST|/τ_T) n_k c_k` —
  **prioritization** — falls out inside the lemma, from the dynamics alone.
- **Corollary 1 — the bridge: the landscape is the novelty-weighted quadratic.** The settled
  student's free energy is the novelty score: `F_S^eq(x_T) = (π_TS/2) x_Tᵀ N_S x_T`, so
  `∇F_S^eq = π_TS N_S x_T` (per direction interface `n²` + self `n(1−n)` = `n`; proof = a
  three-line dial-by-dial continuation of Lemma 1). This is where the two independent routes —
  Thm 1's energy story and Lemma 1's dynamics story — provably meet (checked to 1.8e-15).
  Terminology fixed here: *novelty* = weight-level operator/spectrum (`N_S`, `n_k`), *surprise* =
  state-level scalar (`F_S`, `F_T`); no separate "teacher's surprise" is ever defined.
- **Remark — prioritization, the energetic reading (downstream of Thm 1 + Cor 1).** The per-mode
  law derived dynamically in Lemma 1 is *also* gradient ascent on the quadratic landscape,
  decoupled along `N_S`'s eigenvectors — the two readings coincide (Cor 1's consistency). Learned
  (`n_k=0`) frozen, novel grown ∝ novelty, most-novel dominates (what the experiments measure);
  self-limiting `|π_ST|n_k ≤ (|π_ST|π_S/π_TS)‖M_S u_k‖²` — the push dies quadratically as the
  student learns; ratios survive the leash.
- **Manifold containment (empirical — no lemma, cut 2026-07-20)** — the reversed precision is
  kept well below the teacher's self-precision (code default: half the stability threshold), so
  the self-pull dominates off the manifold and this **sets the precision-weighting regime**; the
  teacher's stay on its memory manifold is shown numerically (`manifold_leakage`,
  `terminal_occupancy`). (Archived sufficient-condition lemma:
  maths/additional_proofs_not_in_paper.md — not for the paper.)
- **Proposition 2** — exact `Φ = F_T − F_S^eq` saddle iff `π_ST = −π_TS`; active-inference-
  *reminiscent* (the shape of an agent–environment minimax, not a free-energy-minimization claim).
  Falls out of Theorem 1: the condition is mobility matching (`|π_ST| = π_TS`); teacher descends
  `F_T` while ascending `F_S`, student descends `F_S` (Prop 1) — a genuine minimax on one potential.
  (The number reuses the slot of the cut interleaving-union proposition.)
- **Scope remark** — linear ⇒ transfers a subspace/effective rank, not named episodes.

### Empirical (figures, NOT theorems)
- Completion figure (`U_Tᵀ N_S(t) U_T → 0`): "reliable completion across tested regimes", with the
  open point and the "steps are timescale/plotting artifacts" caveat stated.
- **Interleaving union + continual learning** — now entirely empirical (no theorem): the crosstalk
  sawtooth (merging teachers) and the retention-vs-buffer-only-forgetting heatmap. The primitive
  *composes*, demonstrated in simulation.
- Noise-driven symmetry breaking / cancellation escape: mechanism + figure, no proof.

### Supplementary (rigor + honesty)
- Full algebra for Prop 1, Theorem 1, Lemma 1, Corollary 1 (surprise identity), the prioritization
  remark, Prop 2; adiabatic elimination.
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
- **Claim scope on VFE:** paper should claim only what is proven — per-step exact VFE descent +
  idealized transfer theorem. Frame integrated/dataset-level VFE minimization as conjecture, not result.
  (TODO: "Does my model minimize VFE integrated across a long period?" / "prove we minimize VFE over the dataset".)
- Confirm the staircase step-count = effective rank story for the exact pattern sets used in Fig 3.
- Decide final title.
