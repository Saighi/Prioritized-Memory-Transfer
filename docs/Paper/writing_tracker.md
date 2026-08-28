# Paper writing tracker: prioritized reactivation and systems consolidation

**Live tracker started:** 2026-08-11

**Manuscript:** [latex/manuscript.tex](latex/manuscript.tex)

**Outline:** [outline.md](outline.md)

**Mathematical source of record:** [maths/minimax_cortical_memory_consolidation.md](maths/minimax_cortical_memory_consolidation.md)

**Tutorial proof:** [maths/minimax_cortical_memory_consolidation_tutorial.md](maths/minimax_cortical_memory_consolidation_tutorial.md)

**Archived preceding tracker:** [archive/pre_minimax_reframe_2026-08-11/writing_tracker.md](archive/pre_minimax_reframe_2026-08-11/writing_tracker.md)

## Current paper in one sentence

A worst-case consolidation objective makes the recipient memory system bias source reactivation toward its largest remaining deficit, fixes a positive teacher-side replay gain with an effective suppressive recipient contribution, and makes reactivation extinguish itself as the recipient learns.

## Structural template

Follow Tang et al. (2023): Abstract, Author summary, Introduction, Models, Results with declarative claim headings and compact mathematics, Discussion split into Summary, Relationship to other models, Relationship to experimental data, Limitations and future directions, Materials and methods, Supporting information. Full proofs belong in S1 Appendix.

The present paper may use slightly more mathematics than Tang because the normative derivation is the contribution. The core text must nevertheless contain only the mathematical objects needed to understand the mechanism and the two-line coupling-sign derivation.

## Locked decisions

- [x] **D1: Two-network scope only.** Three-system continual learning, interleaved union and the buffer-synthesis-storage architecture are moved to a separate project. Continual learning receives at most one future-application paragraph in Discussion.
- [x] **D2: Biology-first framing.** Abstract and Introduction begin from reactivation and systems consolidation, not catastrophic forgetting or a list of model results.
- [x] **D3: Worst-case objective is central.** The paper is organized around $\min_{W_S}\max_{x_T\in\mathbb S_T}\min_{x_S}F_S$ and the maximal settled recipient deficit $\mathcal D_{\max}$.
- [x] **D4: Remove the saddle construction.** No exact zero-sum proposition, no $\Phi=F_T-F_S$ narrative, no saddle figure, and no special emphasis on $\kappa=\pi_{TS}$.
- [x] **D5: Derive the replay gain.** With $\varepsilon_{TS}=x_T-x_S$ and $\nabla_{x_T}q=\pi_{TS}\varepsilon_{TS}^*$, ascent requires $\kappa=\mu_T\pi_{TS}>0$. $\kappa$ is a coupling gain, not a precision.
- [x] **D6: Short derivation in the main paper.** The envelope identity and sign comparison remain visible in the core text. Scalar-mode algebra, Rayleigh-Ritz, projected power iteration and weight-envelope details move to S1 Appendix.
- [x] **D7: Three named formal results at most.** Proposition 1 gives the settled deficit and spectral certificate; Theorem 1 gives the local implementation of worst-case consolidation; Corollary 1 gives completion and self-extinction.
- [x] **D8: Recipient-relative priority.** “Familiar content is replayed less” means content already represented by the recipient is deprioritized. It does not mean hippocampal source strength or reliability is irrelevant.
- [x] **D9: Awake-to-sleep circuit reuse.** Awake PFC-mediated suppression of unwanted or competing hippocampal retrieval is the circuit precedent. The hypothesis is that a related effective control motif is redeployed during sleep, with cortical competence replacing explicit behavioral goals.
- [x] **D10: Anatomical precision.** Describe an effective top-down suppressive influence. Do not claim a direct inhibitory PFC-to-hippocampus synapse or that PFC ripples have been shown to encode the model's deficit variable.
- [x] **D11: Primary efficiency comparison.** Compare autonomous prioritization, a greedy oracle and random teacher-memory selection under the same discrete consolidation protocol and identical recipient plasticity events.
- [x] **D12: Analysis and extension.** Formal analysis is linear. ReLU without bias and non-negative stored patterns is a piecewise-linear extension toward individual memories. It does not model ordered sequence replay.
- [x] **D13: Terminology.** *Replay* is a broad biological umbrella; *prioritized memory reactivation* describes model output; *reactivation of individual memories* describes the rectified extension; *sequential replay* is outside scope.
- [x] **D14: Hierarchy.** Teacher/source is the lower, hippocampal-like memory system. Student/recipient is the upper, neocortical-like memory system. PFC is interpreted as a possible control interface, not automatically as the complete storage substrate.
- [x] **D15: Preserve the previous paper.** Exact source snapshots are in [archive/pre_minimax_reframe_2026-08-11/](archive/pre_minimax_reframe_2026-08-11/).

## Style rules

- Sober, precise and biological. Distinguish observation, interpretation and model prediction.
- No em dash in `manuscript.tex`.
- No hard wrapping of prose paragraphs in `manuscript.tex`; use editor soft wrap.
- Preserve Paul's bracketed drafting notes as LaTeX comments until individually resolved.
- Preserve useful existing prose. Comment obsolete material rather than silently deleting it from the live drafting context; the archive remains the exact fallback.
- *Novelty* refers to the weight-level operator or spectrum ($N_S$, $A_S$ and their eigenvalues).
- *Deficit* refers to $q$ for one state or $\mathcal D_{\max}$ over the teacher manifold.
- Use *surprise* sparingly and only when explicitly identified with settled recipient energy.
- Never call $\kappa$ a precision.
- Do not claim global convergence, optimal total learning time, general nonlinear validity or explicit neural representation of $\mathcal D_{\max}$.

## Section tracker

Status: `[ ]` not started, `[-]` drafting, `[x]` structurally ready, `[L]` prose locked.

- [x] **Archive and scope reset.** Old outline, tracker and manuscript preserved in the dated archive.
- [-] **Title.** Provisional: *A minimax principle for prioritized memory reactivation during systems consolidation*. Revisit after figures and Discussion.
- [-] **Abstract.** Existing biological opening retained. Continual-learning sentence removed from the active scope but preserved as a comment. Remaining task: write one compact objective-to-biological-hypothesis paragraph without a results inventory.
- [ ] **Author summary.** Target 150 to 200 words, no equations.
- [-] **Introduction paragraph 1: intrinsic activity.** Core prose retained. Needs grammar, definition and citation audit.
- [-] **Introduction paragraph 2: replay/reactivation.** Core prose retained. Needs grammar, “hippocampo-cortical dialogue” citation and terminology audit.
- [-] **Introduction paragraph 3: systems consolidation and scope.** Core prose retained. Needs tightening and explicit exclusion of other sleep functions.
- [ ] **Introduction paragraph 4: selective reactivation evidence.** Build from primary studies on weak learning, novelty, recency, reward and links to new learning.
- [ ] **Introduction paragraph 5: model landscape and precise gap.** Compare utility, context, attractor, generative, recall-gated and generalization-based accounts without straw-manning them.
- [ ] **Introduction paragraph 6: awake PFC control and sleep reuse.** Lead from retrieval suppression and competitor suppression to Shin and Jadhav. Maintain anatomical and causal caveats.
- [ ] **Introduction paragraph 7: Tang substrate.** Two predictive-coding associative memories, local inference and local plasticity, teacher below and recipient above.
- [ ] **Introduction paragraph 8: objective and contribution.** Introduce the nested worst-case objective and preview recipient-driven selection and self-extinction.
- [x] **Models skeleton.** New subsection order and central equations are present in `manuscript.tex`.
- [ ] **Models prose.** Define the memory sphere, state-based comparison, recipient energy, constraints, timescales, rectified extension and discrete consolidation protocol.
- [x] **Analytical Results skeleton.** New proposition, theorem and corollary placeholders are present; old saddle material is commented out.
- [ ] **Proposition 1 final statement.** Verify notation and assumptions against the mathematical source of record.
- [ ] **Theorem 1 final statement.** Keep positive replay gain, projected source search and local worst-case descent distinct. State the simple-leading-eigenvalue and timescale assumptions.
- [ ] **Corollary 1 final statement.** Verify the zero-deficit equivalence and vanishing signals.
- [ ] **Results 3-memory demonstration.** Needs simulation outputs and caption-first design.
- [ ] **Results prioritization and self-extinction.** Needs direction-by-bout heat map, deficit spectrum and reactivation occupancy.
- [ ] **Results recipient familiarity.** Needs controlled naive, partial and complete recipient initialization.
- [ ] **Results weak-but-encoded memory reactivation.** Use fast hippocampal and slow cortical waking plasticity to create one well-consolidated memory and one source-supported but cortically weak memory; test preferential offline reactivation of the latter with presentation order counterbalanced.
- [ ] **Results autonomous versus oracle versus random.** Needs frozen protocol, implementation, figures and seed statistics.
- [ ] **Results robustness.** Core summary plus Supporting Information sweeps.
- [ ] **Results rectified extension.** Needs time-resolved individual-memory metric and honest final-transfer comparison.
- [ ] **Discussion summary.** One paragraph.
- [ ] **Discussion relationship to other models.** Comparison organized by priority variable and whether selection is autonomous.
- [ ] **Discussion relationship to experimental data.** Awake control, sleep PFC suppression and falsifiable recipient-competence predictions.
- [ ] **Discussion limitations and future directions.** Linear subspaces, piecewise-linear extension, no sequences, timescale separation, transformed codes, omitted salience and source reliability.
- [ ] **Conclusion.** Short biological and mechanistic close.
- [ ] **Materials and methods.** Reproducible dynamics, protocols, parameters, metrics and uncertainty.
- [ ] **S1 Appendix.** Complete proofs, assumptions, degeneracy handling and unmade claims.
- [ ] **References.** Audit every active `\cite{}` and add the comparison literature.

## Formal result ledger

### Proposition 1: settled recipient deficit

Must establish in the linear hard-constrained model:

- unique $x_S^*(x_T)$;
- $x_S^*=(I-N_S)x_T$;
- $q=(\pi_{TS}/2)x_T^\top N_Sx_T$;
- $\mathcal D_{\max}=(\pi_{TS}/2)\lambda_{\max}(U_T^\top N_SU_T)$;
- $q=0$ exactly for states stored by the recipient.

### Theorem 1: local implementation of worst-case consolidation

Must separate three claims and their assumptions:

1. recipient state descent implements $\min_{x_S}$ after settling;
2. projected source dynamics implement ascent on $q$ and require $\kappa=\mu_T\pi_{TS}>0$;
3. with the selected maximizing state, a simple leading discrepancy eigenvalue and separated timescales, recipient plasticity implements gradient descent on $\mathcal D_{\max}$ and is the locally steepest feasible first-order decrease for fixed small update norm.

### Corollary 1: completion and self-extinction

Must establish:

$$
\mathcal D_{\max}=0
\quad\Longleftrightarrow\quad
\mathcal U_T\subseteq\ker M_S,
$$

followed by vanishing interface error, recurrent recipient error, source search drive and plasticity signal.

## Figure tracker

| Figure | Design status | Data status | Caption status | Blocking issue |
|---|---|---|---|---|
| Fig 1: biological motif, circuit and objective | outlined | schematic partly exists | not started | decide exact use of existing schematic |
| Fig 2: three-memory transfer demonstration | outlined | likely reusable runs exist | not started | select one interpretable 3D condition |
| Fig 3: priority, self-extinction and familiarity | outlined | partial diagnostics exist | not started | add controlled recipient-familiarity experiment |
| Fig 4: waking strength and weak-memory reactivation | outlined | not run | not started | implement unequal waking plasticity with order counterbalancing |
| Fig 5: autonomous versus oracle versus random | protocol outlined | not run | not started | implement audited discrete protocol |
| Fig 6: rectified individual-memory extension | outlined | partial runs and animations exist | not started | add time-resolved nearest-memory metric |
| S1: coupling-sign ablation | outlined | possibly reusable | not started | align parameters with current objective |
| S2-S5: robustness and controls | outlined | partial | not started | choose minimum set after core figures |

## Literature comparison ledger

| Model or evidence class | What it already explains | Precise distinction or use here | Citation status |
|---|---|---|---|
| Mattar and Daw | normative reward-based gain and need | consolidation deficit rather than policy value | verify/add |
| Zhou, Kahana and Schapiro | autonomous context-driven replay | priority emerges from recipient competence | verify/add |
| Fiebig and Lansner | biologically detailed autonomous reinstatement | simpler recipient-deficit objective and analytical sign derivation | verify/add |
| Singh, Norman and Schapiro | autonomous sleep-stage interactions | priority not supplied by recipient worst-case reconstruction | verify/add |
| Spens and Burgess | replay trains generative cortical representations | adaptive recipient-driven selection of source states | verify/add |
| Lindsey and Litwin-Kumar | source reliability and recall-gated plasticity | distinguish source familiarity from recipient familiarity | verify/add |
| Go-CLS | regulates amount of consolidation for generalization | present mechanism selects which content comes next | verify/add |
| Awake retrieval suppression | PFC down-regulates hippocampal retrieval | circuit precedent for selective top-down control | verify/add |
| Retrieval-induced forgetting | PFC resolves mnemonic competition | awake content-selection precedent | verify/add |
| Shin and Jadhav | PFC ripples associated with suppression of CA1 activity and reactivation during NREM | sleep-state biological anchor, not proof of deficit coding | verify/add |

## Comparison protocol lock

Before running Fig 5, write a small protocol record in the experiment directory that fixes:

- identical teacher and recipient initializations across strategies;
- identical candidate memory set and activity normalization;
- identical recipient settling criterion;
- exactly one recipient update per bout;
- identical learning rate and no-autapse projection;
- frozen plasticity during selection and settling;
- random control clamps the teacher memory but never the recipient;
- oracle maximizes the same settled $q$ used to define $\mathcal D_{\max}$;
- predefined stopping thresholds and maximum bout count;
- predefined seed count and uncertainty summary;
- primary metric $\mathcal D_{\max}$ versus plasticity events;
- secondary metric $\mathcal D_{\max}$ versus cumulative weight-change norm.

## Citation and claim checks that must happen before prose lock

- [ ] Verify that “reactivation is considered to support systems consolidation” is cited with balanced reviews and primary evidence.
- [ ] Replace every `(REF)` in the manuscript.
- [ ] Add the five bibliography keys currently cited by active prose but absent from `references.bib`: `kitamura2017engrams`, `squire2015consolidation`, `tononi2014price`, `tononi2014sleep` and `xie2013sleep`.
- [ ] Verify the awake PFC claim at three levels: intentional retrieval stopping, suppression of competing memories, and plausible anatomical pathways.
- [ ] State that hippocampal GABA and local interneurons can implement the suppressive effect; do not imply inhibitory long-range PFC neurons by default.
- [ ] Describe Shin and Jadhav with “associated with” unless the specific experiment supports causal wording.
- [ ] Ensure the comparison section never says existing models “do not prioritize replay.”
- [ ] Distinguish source trace reliability from recipient consolidation.
- [ ] Check whether “memory transfer” is used only when recurrent recipient storage is actually measured.
- [ ] Check that all statements about individual memories are restricted to the rectified condition.

## Decisions still open

- [ ] **O1: final title.** Choose after Figs 2 to 5 are stable.
- [ ] **O2: exact cortical label.** Decide whether the main text says “neocortical recipient,” “cortical recipient,” or “PFC-like recipient” in each context. Default: neocortical recipient for storage, PFC control interface for biological feedback.
- [ ] **O3: Figure 1 composition.** Decide whether the three-memory demonstration begins in Fig 1 or Fig 2. Default: Fig 1 combines biological motif, circuit and objective; Fig 2 is the clean 3D demonstration.
- [ ] **O4: functional guarantee in the main text.** Decide whether the Lipschitz-readout consequence receives one sentence in Results or remains in S1 Appendix and Discussion.
- [ ] **O5: source-reliability extension.** Default: Discussion only, written as $P(m)\propto r_T(m)q_S(m)$ and explicitly not part of the tested model.

## Decision log

- **2026-08-11:** Archived the previous outline, tracker and manuscript before editing.
- **2026-08-11:** Reset scope to the two-network systems-consolidation mechanism. Continual-learning architecture and results moved out of the live paper.
- **2026-08-11:** Replaced the old saddle-first mathematical hierarchy with the worst-case recipient-deficit hierarchy.
- **2026-08-11:** Locked the main comparison as autonomous prioritized selection versus greedy oracle versus random teacher-memory selection.
- **2026-08-11:** Added awake PFC-mediated retrieval control as the circuit precedent and sleep PFC-associated CA1 suppression as the offline biological anchor.
- **2026-08-11:** Preserved existing manuscript prose and bracketed notes, commenting obsolete or unresolved material rather than silently discarding it.

## Next actions

1. Audit and write the references needed for the active Abstract and first three Introduction paragraphs.
2. Draft Introduction paragraphs 4 to 6: selective reactivation, comparison with existing models, and awake-to-sleep PFC control reuse.
3. Finalize the three concise formal statements against the mathematical source of record.
4. Freeze and implement the Fig 5 comparison protocol.
5. Build Fig 2 first, because the three-memory example should establish the behavior before the efficiency and robustness experiments.
