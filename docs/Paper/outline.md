# Paper outline: prioritized memory reactivation from a worst-case consolidation objective

**Status:** New live outline, 2026-08-11. The preceding saddle-first and continual-learning plan is preserved in [archive/pre_minimax_reframe_2026-08-11/outline.md](archive/pre_minimax_reframe_2026-08-11/outline.md).

**Structural model:** Tang et al. (2023), with a compact Models section, declarative Results subsections, mathematical statements next to the phenomena they explain, detailed proofs in S1 Appendix, and a Discussion organized as Summary, Relationship to other models, Relationship to experimental data, Limitations and future directions.

**Provisional title:** *A minimax principle for prioritized memory reactivation during systems consolidation*

Alternative titles to revisit after the Results are written:

- *Recipient-driven reactivation prioritizes memory transfer between associative networks*
- *Cortical memory deficits organize hippocampal reactivation during systems consolidation*
- *Adversarial coupling implements prioritized and self-terminating memory reactivation*

## 1. Paper identity

### Central biological question

During systems consolidation, what determines which hippocampal memory content is reactivated, and why should reactivation of that content decline once it becomes cortically supported?

### Central normative answer

Define the largest settled student free energy over the memory content supported by the teacher:

$$
\boxed{
\mathcal D_{\max}(W_S;W_T)
=
\max_{x_T\in\mathbb S_T}
\min_{x_S}
F_S(x_S,x_T;W_S)
}
$$

The formal claim concerns one plasticity event after the student and teacher states have settled:

$$
\Delta W_S
=
-\eta\nabla_{W_S}\mathcal D_{\max}.
$$

The nested operations have a direct mechanistic reading:

1. fast recipient inference evaluates how well a candidate source state can be reconstructed;
2. source-state dynamics search for the teacher-supported state with the largest settled recipient deficit;
3. one slow recipient-plasticity event takes a local gradient step that reduces the exposed worst-case value to first order.

### One-sentence contribution

The paper derives a two-network mechanism in which recipient deficits drive prioritized source reactivation, recipient learning progressively extinguishes the signal that selected each memory, and a positive teacher-side interface gain implements ascent while the recipient's negative contribution to the interface provides effective source suppression.

### Biological interpretation

- **Teacher/source:** a fast associative memory with a hippocampal interpretation.
- **Student/recipient:** a slower associative memory with a neocortical interpretation.
- **Top-down control channel:** an effective cortical or prefrontal influence on hippocampal activity. It is not claimed to be a direct monosynaptic inhibitory projection.
- **Awake precedent:** prefrontal control suppresses unwanted or competing hippocampal retrieval and helps determine which memory is recalled.
- **Sleep proposal:** related circuitry is reused offline, but the control variable is recipient competence rather than an explicit behavioral goal. Content already supported by cortex is suppressed; content exposing a cortical deficit remains eligible for reactivation.
- **Sleep evidence:** independent PFC ripples are associated with transient suppression of CA1 activity and assembly reactivation. This motivates the mapping but does not prove the model's specific deficit signal.

### Scope lock

The paper contains the two-network transfer primitive only:

- the worst-case consolidation objective;
- the derivation of the positive teacher-side replay gain;
- recipient-deficit-driven prioritized reactivation;
- progressive self-extinction as the recipient learns;
- reduced prioritization of content already familiar to the recipient;
- transfer dynamics and efficiency relative to random selection and a greedy oracle;
- linear analysis and a rectified-network extension toward individual memories;
- the awake and sleep prefrontal-hippocampal control literature.

The following material is reserved for a separate project:

- three-system continual learning;
- interleaved memory union;
- the buffer, synthesis and long-term-storage loop;
- continual-learning benchmarks and retention matrices.

Continual learning may appear in one short Discussion paragraph as a possible application of the transfer primitive. No continual-learning architecture, theorem, result or figure belongs in this paper.

### Explicit removals from the old paper

- Do not introduce a special zero-sum representation at $\kappa=\pi_{TS}$.
- Remove the saddle interpretation and the wake/sleep saddle figure.
- Do not derive the architecture from $\Phi=F_T-F_S$.
- Replace the old theorem and lemma sequence with the worst-case-deficit derivation below.
- Call $\kappa$ a signed effective coupling gain, not a precision.
- Do not claim that the complete coupled network minimizes a global variational free energy.

## 2. Narrative arc

The paper should move through one causal chain:

$$
\text{systems consolidation problem}
\longrightarrow
\text{worst recipient deficit}
\longrightarrow
\text{effective cortical suppression and replay ascent}
\longrightarrow
\text{prioritized reactivation}
\longrightarrow
\text{recipient plasticity}
\longrightarrow
\text{self-extinction}.
$$

The conceptual sentence to protect is:

> Reactivation is a closed-loop teaching process in which the recipient biases the source toward what the recipient still lacks.

Priority is therefore relational rather than intrinsic. The same hippocampal memory can be strongly prioritized for one recipient, weakly prioritized for another, and progressively deprioritized as it becomes consolidated.

## 3. Front matter

### Abstract

The abstract remains biology-led and should not become a list of every analytical and numerical result.

Five moves:

1. Reactivation during sleep and quiet wake is implicated in systems consolidation.
2. Reactivation is selective, but models commonly assign priority through recency, source strength, salience, context or reward, or leave content selection implicit.
3. Ask whether priority can instead emerge from the interaction between source and recipient memory systems.
4. Introduce the two-network worst-case consolidation objective and state, in one compact sentence, that its local implementation uses a positive teacher-side interface gain, while the recipient enters the interface negatively, to drive the source toward content the recipient represents poorly.
5. Close on the biological hypothesis: an inhibitory prefrontal-hippocampal control motif used for selective retrieval during wake may be redeployed during sleep to organize consolidation according to cortical competence.

Keep out of the abstract unless later required by the final data:

- the novelty spectrum;
- the random-versus-oracle comparison;
- detailed rectification results;
- continual learning;
- the removed saddle construction.

### Author summary

Explain the mechanism without formal notation:

- sleeping brains reactivate only part of their stored experience;
- the cortex need not passively receive a hippocampal replay schedule;
- its current knowledge can suppress redundant hippocampal content and leave poorly consolidated content active;
- the same transfer that removes the cortical deficit also removes the signal that caused reactivation;
- the model turns this idea into an analyzable two-network mechanism.

## 4. Introduction: paragraph plan

### Paragraph 1: intrinsic activity is biologically substantial

Retain the drafted metabolism, intrinsic-activity and phylogenetic-conservation opening. Tighten definitions and grammar later. Do not let this opening exceed one paragraph.

### Paragraph 2: replay and reactivation across brain systems

Retain the drafted progression from rodent hippocampal place-cell reactivation to human and extra-hippocampal observations, followed by the temporal coordination of hippocampal ripples with cortical slow oscillations and spindles. Define *memory replay* broadly at first use, while reserving *sequential replay* for ordered reinstatement that the model does not address.

### Paragraph 3: systems consolidation is the target phenomenon

Define systems consolidation as a progressive change in the neural support of memory retrieval. Frame the modeled problem as transfer of memory content from a temporary source to a recipient recurrent memory. Briefly acknowledge other proposed functions of sleep and replay, then state that they are outside the present scope.

### Paragraph 4: reactivation is selective

Establish the empirical motivation for prioritization. Relevant observations include effects of weak learning, novelty, recent experience, reward, future relevance and relationships to new learning. The point is not that one variable explains all replay, but that uniform random sampling is inadequate as a complete account.

### Paragraph 5: existing models solve complementary parts of the problem

Organize the comparison diplomatically:

- utility-based replay specifies which memories are behaviorally useful;
- context-driven and autonomous-attractor models explain how structured replay can emerge;
- generative and complementary-learning-systems models explain how replay trains cortical representations;
- recall-gated consolidation selects source traces according to source reliability or familiarity;
- generalization-based accounts regulate how much consolidation should occur.

The gap is narrower than “existing models do not prioritize replay.” The paper asks for one local closed-loop mechanism in which the recipient's own deficit selects source content and the same interaction drives recurrent storage transfer.

### Paragraph 6: awake prefrontal control supplies a circuit precedent

Introduce retrieval suppression and retrieval-induced forgetting. During awake, goal-directed retrieval, lateral and medial PFC regions exert an effective top-down suppressive influence on hippocampal retrieval, and prefrontal control helps suppress unwanted or competing memories. State the anatomical caveat: the effective influence can be mediated by cortical, entorhinal, thalamic and local hippocampal inhibitory circuitry.

Then make the paper's biological proposal explicit:

> During wake, current goals determine which hippocampal memories should be suppressed. During sleep, recipient competence may provide an endogenous control signal that suppresses already consolidated content.

Connect this proposal to sleep observations of PFC-associated suppression of CA1 activity and reactivation, with precise correlational wording.

### Paragraph 7: normative question and model substrate

Introduce two predictive-coding associative memories based on Tang et al. The source holds a memory manifold and remains fixed; the recipient settles and learns through local prediction-error dynamics. State that the mathematical analysis is linear and that the rectified extension is tested separately.

### Paragraph 8: contribution and roadmap

Introduce the worst-case consolidation objective. Explain that the sign of the teacher-side interface interaction is derived by differentiating the settled recipient deficit, rather than postulated from a predictive-coding hierarchy. Preview only the main biological consequences: recipient-driven prioritization, self-extinction and reduced reactivation of already consolidated content.

## 5. Models

Keep this section compact and equation-led, following Tang et al. Put proofs in S1 Appendix.

### 5.1 Two associative memory systems

- Define source state $x_T$, fixed recurrent weights $W_T$, recipient state $x_S$ and plastic recurrent weights $W_S$.
- Define $M_T=I-W_T$, $M_S=I-W_S$ in the linear analysis.
- Define the teacher memory subspace $\mathcal U_T=\ker M_T$ and the equal-activity teacher memory sphere

$$
\mathbb S_T=\{x_T\in\mathcal U_T:\|x_T\|=1\}.
$$

- Explain why the comparison is state-based, not weight-based: hippocampal and cortical networks need not share synaptic matrices, but they must be comparable through a target-state or functionally meaningful discrepancy.
- Generalization to transformed cortical codes belongs in Discussion.

### 5.2 Recipient energy, inference and plasticity

Define

$$
F_S(x_S,x_T;W_S)
=
\frac{\pi_{TS}}2\|x_T-x_S\|^2
+
\frac{\pi_S}{2}\|M_Sx_S\|^2.
$$

State the local recipient-state descent and recurrent plasticity rule. Avoid presenting basic gradient descent as a headline theorem.

### 5.3 Settled deficit for one source state

Define

$$
q(x_T;W_S)=\min_{x_S}F_S(x_S,x_T;W_S).
$$

Interpret $q$ as the recipient's best achievable reconstruction and recurrent-consistency deficit for that candidate source state.

### 5.4 Worst-case consolidation objective

Define

$$
\mathcal D_{\max}(W_S;W_T)
=
\max_{x_T\in\mathbb S_T}q(x_T;W_S),
$$

and state the local plasticity result

$$
\Delta W_S
=
-\eta\nabla_{W_S}\mathcal D_{\max}.
$$

Give the normative motivations briefly:

- future queries and the lifetime of the source trace are uncertain;
- the maximum provides a uniform guarantee over the teacher-supported memory manifold;
- reducing the maximum prevents the recipient from leaving one large blind spot hidden by good average performance.

### 5.5 Local implementation of selection and learning

State the three timescales

$$
\tau_S\ll\tau_T\ll1/\eta.
$$

The fast recipient evaluates, the intermediate source selects, and the slow weights learn. In the hard-constrained analysis source search is projected onto $\mathbb S_T$; in simulations recurrent source dynamics and normalization approximate this constraint and manifold leakage is measured.

### 5.6 Rectified extension

Introduce a general activation $f$ with the load-bearing condition $f(m_p)=m_p$ for stored patterns. Identity activation gives the analyzed linear model. ReLU without bias and non-negative patterns gives a piecewise-linear extension that can restrict activity to memory cones and support reactivation near individual memories. Do not claim general nonlinear validity.

### 5.7 Discrete consolidation protocol for efficiency comparisons

During each bout:

1. freeze recipient plasticity;
2. choose or generate a teacher state using one of the three selection strategies;
3. allow the recipient state to settle;
4. apply one identical small recipient-weight update;
5. repeat until a common transfer criterion is reached.

The recipient is never clamped. For the random control, a random teacher memory is clamped while the coupled recipient settles. The plasticity rule, step size, stopping condition and candidate memory set are identical across strategies.

## 6. Results

The Results should read as a sequence of biological and computational claims, not as a derivation dump. Short analytical statements appear beside the first figure they explain; complete proofs move to S1 Appendix.

### Part I: A worst-case objective derives recipient-driven reactivation

#### 6.1 The settled recipient deficit is a novelty-weighted quadratic

For the linear recipient, define $S_S=M_S^\top M_S$ and

$$
N_S=\pi_SS_S(\pi_{TS}I+\pi_SS_S)^{-1}.
$$

Then

$$
x_S^*=(I-N_S)x_T,
\qquad
q(x_T;W_S)=\frac{\pi_{TS}}2x_T^\top N_Sx_T.
$$

With $U_T$ spanning the teacher memory subspace and $A_S=U_T^\top N_SU_T$,

$$
\mathcal D_{\max}=\frac{\pi_{TS}}2\lambda_{\max}(A_S).
$$

Interpretation: the leading eigendirection of $A_S$ is the teacher-supported direction that the recipient represents least well.

#### 6.2 Maximizing settled deficit fixes the teacher-side interface drive

The core derivation must remain in the main paper:

$$
\nabla_{x_T}q(x_T;W_S)
=
+\pi_{TS}\varepsilon_{TS}^*,
\qquad
\varepsilon_{TS}=x_T-x_S.
$$

If the source interface contribution is

$$
\tau_T\dot x_T\big|_{\mathrm{interface}}=\kappa\varepsilon_{TS},
$$

then ascent on the recipient deficit requires

$$
\boxed{\kappa=\mu_T\pi_{TS}>0.}
$$

This is the elegant sign result. It requires no zero-sum potential. The magnitude includes the source search mobility $\mu_T$ and remains subject to the constraints of the implementation.

#### 6.3 Local plasticity reduces the selected worst-case deficit

At a simple leading discrepancy eigenvalue, the maximizing states are an antipodal pair that produce the same weight derivative. The local recipient rule implements gradient descent on $\mathcal D_{\max}$. Among sufficiently small feasible weight changes of the same norm, it gives the largest first-order decrease of the current worst-case bound.

#### 6.4 Completion makes selection and plasticity self-extinguishing

State

$$
\mathcal D_{\max}=0
\quad\Longleftrightarrow\quad
\mathcal U_T\subseteq\ker M_S.
$$

At completion, $x_S^*=x_T$, $\varepsilon_{TS}^*=0$ and the recurrent recipient error vanishes for all valid teacher states. Both source selection and recipient plasticity therefore terminate without an external stopping signal.

### Part II: The derived dynamics transfer and prioritize memory content

#### 6.5 Autonomous reactivation transfers a small memory manifold

Open the empirical Results with an elegant, readable demonstration using three memories in three dimensions.

- Show the three teacher memories and the source memory manifold.
- Overlay the teacher reactivation trajectory and the settled recipient trajectory.
- Show recipient weight or reconstruction evolution as a compact heat map.
- Show $\mathcal D_{\max}$ decreasing across bouts.
- Make the distinction between individual memories and linear combinations explicit.

This figure should let a reader understand the behavior before encountering a large parameter sweep.

#### 6.6 Reactivation follows recipient deficit and progressively changes target

Use a direction-by-bout heat map to show:

- the source initially occupies the direction with the largest recipient deficit;
- after the recipient learns it, its novelty eigenvalue and occupancy decline;
- another unresolved direction then dominates;
- the sequence ends when all supported deficits are small.

Do not call the staircase itself a theorem. The analytical claim concerns the instantaneous maximizing direction; finite-bout staircase dynamics are numerical.

#### 6.7 Familiarity is recipient-relative

Construct a controlled comparison in which the teacher memory is held fixed while recipient initialization changes.

- recipient naive to all teacher content;
- recipient already familiar with one direction;
- recipient already familiar with the complete teacher manifold.

Measure reactivation occupancy, selection time and plasticity magnitude. The prediction is not that source familiarity is irrelevant. It is that, conditional on content being supported by the source, priority falls with recipient competence.

#### 6.8 Differential waking learning explains preferential reactivation of successfully encoded weak memories

During waking, train one orthogonal memory until both the fast hippocampal source and slow cortical recipient represent it well, and train a second memory only until its hippocampal error is low while its cortical error remains high. During subsequent offline activity, the second memory should be preferentially reactivated because it is source-supported but still exposes a recipient deficit. Counterbalance presentation order so that the result is not reducible to recency.

#### 6.9 Autonomous prioritization approaches a greedy oracle and outperforms random reactivation

Compare:

1. **Autonomous prioritized network:** the teacher converges under discrepancy-driven dynamics.
2. **Greedy oracle:** directly select the teacher-manifold direction maximizing the current settled recipient deficit.
3. **Random unprioritized transfer:** select and clamp a random teacher memory while the connected recipient settles; never clamp the recipient.

Primary endpoint:

$$
\mathcal D_{\max}
\quad\text{versus number of identical plasticity events.}
$$

Secondary endpoints:

- $\mathcal D_{\max}$ versus cumulative Frobenius norm of weight change;
- novelty-spectrum evolution;
- events required to reach fixed deficit thresholds;
- gap between autonomous selection and the oracle's selected deficit;
- final transfer quality, to show that efficiency is not bought by a poorer endpoint.

Report uncertainty over seeds and use the same initial networks, candidate states, update magnitude and stopping rule across methods.

#### 6.10 Transfer remains reliable across memory geometries and dynamical regimes

Keep the core figure compact. Move broad sweeps to Supporting Information. Test at least:

- memory rank and ambient dimension;
- correlation between stored patterns;
- recipient initialization or partial familiarity;
- inference, selection and plasticity timescale separation;
- source-manifold leakage;
- noise or seed dependence.

The wording should be “reliable across the tested regimes,” not a global convergence claim.

#### 6.11 Rectification extends the mechanism toward individual memories

For ReLU units without bias and non-negative stored patterns:

- show transfer remains operational;
- visualize early reactivation near individual point memories;
- show how those targets lose attraction as they are learned;
- quantify distance to the nearest stored memory over time;
- show cone confinement and compare with the linear signed-mixture geometry;
- state the looser final reconstruction honestly.

Do not claim temporal or spatial sequence replay. Rectification addresses the geometry of which stored memories are expressed, not the ordered reproduction of experience.

## 7. Figure plan

| Figure | Main claim | Essential panels |
|---|---|---|
| **Fig 1. From biological control motif to the two-network mechanism** | Recipient cortex is an active controller of source reactivation | Awake PFC suppression of unwanted/competing hippocampal recall; proposed reuse during NREM; two-network circuit; three-timescale objective |
| **Fig 2. Autonomous transfer in an interpretable three-memory system** | The derived dynamics transfer a teacher memory manifold and then stop | 3D state trajectories; teacher and student memory geometry; reconstruction or weight heat map; $\mathcal D_{\max}$ and errors over bouts |
| **Fig 3. Recipient deficits organize priority and self-extinction** | Reactivation shifts away from learned content and is weaker for recipient-familiar content | deficit spectrum; reactivation occupancy heat map; naive versus partially familiar versus fully familiar recipient |
| **Fig 4. Waking learning strength predicts subsequent reactivation** | A successfully encoded but cortically weak memory is preferentially reactivated offline | waking exposure protocol; hippocampal and cortical reconstruction errors before sleep; subsequent memory-specific reactivation occupancy |
| **Fig 5. Prioritized reactivation is an efficient approximation to greedy transfer** | Autonomous selection approaches the oracle and tightens the worst-case guarantee faster than random selection | $\mathcal D_{\max}$ versus plasticity events; versus cumulative update; novelty spectra; time-to-threshold summary |
| **Fig 6. Rectified dynamics express individual memories** | The mechanism extends beyond linear subspaces toward point-like reactivation | early and late trajectories; nearest-memory distance; cone confinement; final transfer quality |

Suggested Supporting Information:

- **S1 Fig:** replay-ascent, zero-drive and replay-descent ablation.
- **S2 Fig:** rank, dimension, correlation and partial-familiarity sweeps.
- **S3 Fig:** timescale separation, noise and source-manifold leakage.
- **S4 Fig:** oracle implementation checks and random-baseline controls.
- **S5 Fig:** rectified-network seed robustness and time-resolved memory selectivity.
- **S1 Appendix:** complete analytical proofs and assumptions.

## 8. Mathematical hierarchy and placement

### Main paper

Use three named results at most.

1. **Proposition 1: settled deficit and spectral certificate.** State $x_S^*$, $N_S$, $q$, $A_S$ and $\mathcal D_{\max}$.
2. **Theorem 1: local implementation of worst-case consolidation.** Under hard teacher-manifold constraint and timescale separation, recipient inference implements the inner minimum, projected teacher dynamics implement ascent on the settled deficit, the positive teacher-side interface gain follows, and at a simple leading discrepancy eigenvalue recipient plasticity descends $\mathcal D_{\max}$.
3. **Corollary 1: completion and self-extinction.** State the zero-deficit equivalence and vanishing selection/plasticity signals.

In the main paper, show the two-line envelope derivative and coupling-sign derivation in full. This is the conceptual heart and is short enough to deserve the space.

### S1 Appendix

Move the following details out of the narrative flow:

- existence and uniqueness of the settled recipient response;
- scalar-mode derivation and reassembly of $N_S$;
- Rayleigh-Ritz derivation of $\mathcal D_{\max}$;
- reconstruction, recurrent-consistency and Lipschitz readout bounds;
- projected ascent and continuous-time power iteration;
- the two envelope steps for recipient weights;
- antipodal maximizers and the simple-leading-eigenvalue assumption;
- steepest feasible first-order descent with the no-autapse projection;
- exact assumptions, degeneracy caveats and deliberately unmade claims.

### Mathematical language to avoid

- no “saddle” terminology;
- no exact zero-sum proposition;
- no claim that $\kappa=\pi_{TS}$ is uniquely required;
- no global-optimal-time claim;
- no global convergence claim for arbitrary nonlinear networks;
- no claim that the brain explicitly computes the scalar $\mathcal D_{\max}$.

## 9. Discussion

Follow Tang et al.'s subsection structure.

### 9.1 Summary

One paragraph. Restate the closed loop: deficit selects content, selected content trains the recipient, learning removes the selecting signal.

### 9.2 Relationship to other models

Compare by the variable controlling reactivation and by whether selection is implemented autonomously:

- **Mattar and Daw:** gain and need for reward-based planning.
- **Zhou, Kahana and Schapiro:** autonomous context-driven reactivation.
- **Fiebig and Lansner; Singh, Norman and Schapiro:** autonomous hippocampal-cortical reinstatement and sleep-stage dynamics.
- **Spens and Burgess:** hippocampal replay training cortical generative models.
- **Lindsey and Litwin-Kumar:** source reliability and recall-gated long-term plasticity.
- **Go-CLS:** the amount of consolidation needed for generalization.

The defensible distinction is not that these models lack prioritization or autonomous replay. It is that the present model uses the recipient's own settled deficit to select source content and connects that selection to recurrent storage transfer through one nested objective.

Introduce the useful two-factor extension only as future work:

$$
P(m)\propto r_T(m)q_S(m),
$$

where $r_T$ is source reliability and $q_S$ is recipient deficit. This can reconcile greater replay of reliable source traces with declining replay after recipient consolidation. Do not add it to the core model yet.

### 9.3 Relationship to experimental data

Organize this subsection around predictions rather than analogy alone:

1. prefrontal or cortical similarity to a memory should suppress the corresponding hippocampal reactivation;
2. this relationship should persist after controlling for source strength, recency, salience and reward;
3. disrupting top-down suppression should increase redundant reactivation and reduce consolidation efficiency, rather than simply abolishing replay;
4. priority should return if a cortical representation is degraded or becomes outdated;
5. awake goal-driven retrieval control and sleep deficit-driven consolidation may reuse an effective suppressive circuit motif with different control variables.

State clearly that the model does not identify a direct inhibitory axon, does not establish that every PFC ripple carries a cortical deficit signal, and does not model the complete sleep oscillatory hierarchy.

### 9.4 Limitations and future directions

- linear memory manifolds represent subspaces rather than named episodes;
- the rectified extension is piecewise linear and restricted to non-negative patterns fixed by the activation;
- temporal sequence replay is not modeled;
- exact timescale separation and hard manifold projection idealize the simulated circuit;
- ties in the largest discrepancy eigenspace require a subgradient or set-valued treatment;
- transformed hippocampal and cortical codes require an explicit mapping or functionally relevant discrepancy;
- source reliability, emotion, reward, salience, context and neuromodulation are not modeled;
- continual learning is a future application and separate project.

## 10. Materials and methods

Include only information needed to reproduce the figures:

- memory construction and pattern distributions;
- state dynamics, integration scheme and stopping criteria;
- timescales, coupling gains and normalization;
- learning rule and no-autapse projection;
- definition and numerical computation of $N_S$, $A_S$ and $\mathcal D_{\max}$;
- prioritized, oracle and random protocols;
- familiarity manipulation;
- ReLU and non-negative-pattern protocol;
- seeds, uncertainty summaries and statistical comparisons;
- source-manifold leakage and convergence diagnostics.

## 11. Citation worklist

### Biological foundation

- Wilson and McNaughton on hippocampal reactivation.
- Human replay/reactivation studies during rest.
- Hippocampal ripple, cortical slow-oscillation and spindle coordination.
- Systems-consolidation and prefrontal-engram literature.

### Awake memory selection and inhibition

- Anderson and Green; Anderson, Bunce and Barbas.
- Benoit and Anderson.
- Schmitz et al. on hippocampal GABA.
- Wimber et al. on suppression of competing memories.
- Crespo-Garcia et al. on ACC detection and prefrontal control.

### Sleep top-down suppression

- Shin and Jadhav, with the exact distinction between association, suppression and causal selection.

### Reactivation models

- Mattar and Daw.
- Zhou, Kahana and Schapiro.
- Fiebig and Lansner.
- Singh, Norman and Schapiro.
- Spens and Burgess.
- Lindsey and Litwin-Kumar.
- Go-CLS.

### Model substrate

- Tang et al. on covariance-learning predictive-coding associative memories.

## 12. Completion criteria

The paper is structurally ready for submission drafting when:

- every abstract and introduction claim has a verified primary citation;
- the sign derivation and theorem assumptions match the source-of-record mathematics;
- the three selection strategies share a frozen, audited protocol;
- the main figures are complete and captions can be read independently;
- all continual-learning material is absent from the live manuscript;
- the text distinguishes source familiarity from recipient familiarity;
- every biological mapping is labeled as evidence, interpretation or prediction;
- the manuscript compiles without undefined references or stale section labels;
- S1 Appendix contains complete proofs while the main text retains the short sign derivation.
