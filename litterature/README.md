# Literature on memory reactivation and systems consolidation

This folder contains a deliberately focused corpus for positioning the present model. It emphasizes computational implementations of autonomous reactivation, replay selection, and hippocampal–neocortical transfer, together with a small set of empirical constraints on novelty, familiarity, and top-down control.

The three PDFs that were already at the project root have been renamed and moved here: Rao and Ballard (1999), Tang et al. (2023), and Shin and Jadhav (2024). The remaining papers were added because they are especially useful for establishing what current models do—and do not—derive from their internal dynamics.

## Reading key

- **Replay/reactivation only**: the paper explains how offline content is generated or selected, but does not implement learning in a distinct recipient memory system.
- **Systems consolidation**: reactivation in one system changes a second, slower-learning system.
- **Storage target** distinguishes learning in recurrent memory synapses from learning in feedforward/readout weights. “Not only a readout” means that distributed internal or generative connections change, even if the recipient is not a recurrent attractor network.
- **Recipient-relative priority** means that a memory is selected because of what the recipient system has or has not yet learned—not merely because the trace is recent, strong, rewarded, frequently experienced, or tagged at encoding.

## Main conclusion for the present paper

The literature supports a strong but carefully delimited novelty claim. Biologically explicit systems-consolidation models can generate reactivation autonomously. Fiebig and Lansner (2014) obtain spontaneous attractor reinstatement, and Singh, Norman, and Schapiro (2022) obtain autonomous transitions between hippocampal and cortical attractors. Their replay biases, however, arise mainly from source-trace strength, attractor competition, adaptation, or sleep-stage coupling. They do not compute a priority from the recipient network's current failure to represent a memory. Singh et al. are especially explicit: prioritizing weak or uncertain memories does not fall naturally out of their framework and would require an additional tagging mechanism.

There are important partial precedents, but their mechanisms are different. Barry and Love (2022) use a separate reinforcement-learning controller rewarded by improvements in a cortical model; Go-CLS supplies a normative rule for how much consolidation should occur but requires a supervisory estimate of generalization; Lindsey and Litwin-Kumar (2024) gate plasticity using an explicit recall/familiarity signal and generally favor repeated, already familiar traces; Spens and Burgess (2024) use recipient reconstruction error to determine what is encoded in detail, not which memory is selected during offline replay. CMR-replay (Zhou et al.) generates experience-dependent replay autonomously, but its biases are inherited from encoding strength, contextual associations, and an explicit activity-dependent suppression rule.

Across this core set, I did not find a prior biologically explicit systems-consolidation model in which **recipient-relative novelty is computed online by the local dynamics of the interacting memory systems and directly governs how long or how strongly a memory is reactivated**. This is the defensible point of novelty for the present architecture. It is narrower—and stronger—than claiming that earlier work never models prioritized replay or never uses network state.

## 01 — Foundations

### McClelland, McNaughton, and O’Reilly (1995)

**Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory.** *Psychological Review*, 102, 419–457. [DOI](https://doi.org/10.1037/0033-295X.102.3.419) · [PDF](01_foundations/McClelland_McNaughton_OReilly_1995_Complementary_learning_systems.pdf)

- **Scope:** Foundational theory and simulations of systems consolidation.
- **Reactivation:** Hippocampal traces are assumed to reinstate cortical patterns for slow, interleaved learning; replay order and initiation are not autonomously generated.
- **Priority:** No intrinsic recipient-relative selection mechanism.
- **Storage target:** Distributed neocortical connection weights—not merely a fixed-feature readout, but not a detailed recurrent cortical attractor implementation either.
- **Summary:** The paper establishes the complementary-learning-systems rationale: rapid, pattern-separated hippocampal learning protects new episodes, while gradual interleaved learning integrates their structure into overlapping neocortical representations without catastrophic interference. It is the conceptual baseline for transfer by reactivation, but not a mechanistic model of how replay events are initiated or prioritized.

### Rao and Ballard (1999)

**Predictive coding in the visual cortex: A functional interpretation of some extra-classical receptive-field effects.** *Nature Neuroscience*, 2, 79–87. [DOI](https://doi.org/10.1038/4580) · [PDF](01_foundations/Rao_Ballard_1999_Predictive_coding_visual_cortex.pdf)

- **Scope:** Predictive-coding foundation; neither replay nor systems consolidation.
- **Reactivation:** Not modeled.
- **Priority:** Prediction errors are computed locally within a hierarchical perception model, but they do not select memories for offline reactivation.
- **Storage target:** Learned generative/predictive connections in a recurrent hierarchical circuit; not a memory readout.
- **Summary:** Higher levels predict lower-level activity and residual errors drive inference and learning. The paper is relevant because it supplies the biologically grounded prediction/error architecture from which the present model derives its inter-system dynamics, not because it offers a replay mechanism.

### Tang et al. (2023)

**Recurrent predictive coding models for associative memory employing covariance learning.** *PLOS Computational Biology*, 19, e1010719. [DOI](https://doi.org/10.1371/journal.pcbi.1010719) · [PDF](01_foundations/Tang_et_al_2023_Recurrent_predictive_coding_associative_memory.pdf)

- **Scope:** Associative-memory implementation; not systems consolidation.
- **Reactivation:** Stored patterns can be retrieved as attractors, but the paper does not model offline replay scheduling between memory systems.
- **Priority:** None.
- **Storage target:** Recurrent synapses explicitly store covariance structure and associative memories.
- **Summary:** The paper reformulates covariance-learning predictive-coding networks so that covariance can be learned implicitly with local Hebbian plasticity and biologically plausible error representations. It is a direct architectural foundation for recurrent predictive-coding memory, but it leaves autonomous multi-memory reactivation and inter-system transfer open.

## 02 — Systems-consolidation models

### Káli and Dayan (2004)

**Off-line replay maintains declarative memories in a model of hippocampal-neocortical interactions.** *Nature Neuroscience*, 7, 286–294. [DOI](https://doi.org/10.1038/nn1202) · [PDF](02_systems_consolidation_models/Kali_Dayan_2004_Offline_replay_hippocampal_neocortical_interactions.pdf)

- **Scope:** Systems consolidation, with a strong emphasis on maintaining access to hippocampal episodic traces while cortical representations drift.
- **Reactivation:** Hippocampal patterns are assumed to become active intrinsically and are sampled randomly; the hippocampus itself is not explicitly implemented.
- **Priority:** No adaptive selection based on recipient state.
- **Storage target:** Plastic bidirectional neocortical connections and local cortical cleanup attractors; not only a readout. The model finds that episodic replay often preserves hippocampal–cortical registration rather than producing durable hippocampus-independent episodic storage.
- **Summary:** Replay keeps hippocampal indices aligned with changing cortical representational coordinates and contributes constructively to semantic recall. This is an important qualification of “transfer”: replay can maintain access and interpretation, not simply copy complete episodic traces into cortex.

### Fiebig and Lansner (2014)

**Memory consolidation from seconds to weeks: A three-stage neural network model with autonomous reinstatement dynamics.** *Frontiers in Computational Neuroscience*, 8, 64. [DOI](https://doi.org/10.3389/fncom.2014.00064) · [PDF](02_systems_consolidation_models/Fiebig_Lansner_2014_Autonomous_reinstatement_dynamics.pdf)

- **Scope:** Full systems consolidation across prefrontal, hippocampal, and neocortical stages.
- **Reactivation:** Autonomous attractor reinstatement arises from recurrent network dynamics, adaptation, and synaptic depression.
- **Priority:** Replay is biased by source-network properties. Stronger traces reactivate more often, weak traces may disappear, and less-correlated patterns can remain active longer. These effects are emergent, but they are not computed from the recipient cortex’s memory deficit.
- **Storage target:** Recurrent autoassociative synapses in all three memory modules, with plastic inter-module projections; decisively not a readout-only model.
- **Summary:** This is one of the most important mechanistic precedents because it turns replay into an intrinsic system property and uses it to drive consolidation over realistic timescale differences. Its content biases arise from attractor strength and competition in the reactivating network, rather than a mismatch evaluated through dialogue with the target network.

### Singh, Norman, and Schapiro (2022)

**A model of autonomous interactions between hippocampus and neocortex driving sleep-dependent memory consolidation.** *PNAS*, 119, e2123432119. [Article DOI](https://doi.org/10.1073/pnas.2123432119) · [stored bioRxiv preprint](02_systems_consolidation_models/Singh_Norman_Schapiro_2022_Autonomous_hippocampus_neocortex_interactions_preprint.pdf)

- **Scope:** Full hippocampal–neocortical systems consolidation.
- **Reactivation:** After one noise injection, recirculating activity, short-term synaptic depression, and oscillating inhibition produce autonomous transitions among attractors. NREM couples hippocampus and cortex and favors recent memories; REM allows cortex to revisit older attractors.
- **Priority:** Recency emerges largely from which system controls the dynamics and from relative attractor strength. The authors explicitly state that prioritization of weak or uncertain memories does not arise naturally and would need an additional tag.
- **Storage target:** Distributed cortical network connections shaped by local contrastive Hebbian learning; not merely a readout. The model also has input/output layers for its tasks, but consolidation restructures internal representations.
- **Summary:** This is the closest established model of fully autonomous inter-system replay. It explains event generation, switching, and useful offline learning, but not recipient-relative novelty selection. Its explicit limitation makes it particularly valuable for positioning the present model.

### Spens and Burgess (2024)

**A generative model of memory construction and consolidation.** *Nature Human Behaviour*, 8, 526–543. [DOI](https://doi.org/10.1038/s41562-023-01799-z) · [PDF](02_systems_consolidation_models/Spens_Burgess_2024_Generative_model_memory_construction_consolidation.pdf)

- **Scope:** Systems consolidation and reconstructive memory.
- **Reactivation:** Random inputs cause a modern Hopfield-network “teacher” to reactivate memories, which train neocortical variational-autoencoder “students.”
- **Priority:** Recipient reconstruction error determines which features require detailed hippocampal encoding and when a hippocampal trace becomes unnecessary. It does not prioritize which stored memory is replayed offline; replay sampling is random.
- **Storage target:** Recurrent/autoassociative storage in the hippocampal source, but feedforward generative-model parameters in the recipient; not a simple readout, yet not recurrent cortical memory synapses.
- **Summary:** The model tightly links consolidation to prediction and novelty, but the novelty calculation acts during encoding and memory allocation. It is therefore a strong conceptual neighbor, not a precedent for novelty-dependent replay selection emerging during offline inter-system interaction.

## 03 — Reactivation and prioritization models

### Mattar and Daw (2018)

**Prioritized memory access explains planning and hippocampal replay.** *Nature Neuroscience*, 21, 1609–1617. [DOI](https://doi.org/10.1038/s41593-018-0232-z) · [PDF](03_reactivation_prioritization_models/Mattar_Daw_2018_Prioritized_memory_access.pdf)

- **Scope:** Replay selection for reinforcement-learning planning; not systems consolidation.
- **Reactivation:** Candidate experiences are selected for Bellman backups.
- **Priority:** Explicit expected utility of an update, decomposed into gain and need. This is a powerful normative account, but the scheduler evaluates hypothetical future behavioral benefit rather than emerging from local memory-system dynamics.
- **Storage target:** Value or policy estimates/readouts, not recurrent synaptic memory transfer.
- **Summary:** The model explains many forward/reverse and reward-related replay phenomena by asking which backup would most improve future behavior. It is the canonical example of sophisticated but externally defined prioritization.

### Barry and Love (2022)

**A neural network account of memory replay and knowledge consolidation.** *Cerebral Cortex*, 33, 83–95. [DOI](https://doi.org/10.1093/cercor/bhac054) · [PDF](03_reactivation_prioritization_models/Barry_Love_2022_Neural_network_replay_knowledge_consolidation.pdf)

- **Scope:** Generative replay and cortical knowledge consolidation; a functional analogue of systems consolidation rather than a recurrent hippocampal–cortical circuit.
- **Reactivation:** Prototypes are sampled from class-specific activation distributions and injected at different depths of a VGG-like visual network.
- **Priority:** A separate reinforcement-learning controller selects a category and receives reward from its improvement of the visual model’s generalization performance. This does use recipient performance, but through an explicit global controller and evaluation loop.
- **Storage target:** Feedforward/convolutional synapses downstream of the replay-injection layer; no recurrent memory target. The controller itself learns a category-selection policy.
- **Summary:** This is the closest computational precedent for prioritizing weakly learned information according to benefit for a recipient network. Its distinction from the present model is mechanistic: the priority is computed by an auxiliary RL system, not by local dynamics intrinsic to the interacting memory architecture.

### Diekmann and Cheng (2023)

**A model of hippocampal replay driven by experience and environmental structure facilitates spatial learning.** *eLife*, 12, e82301. [DOI](https://doi.org/10.7554/eLife.82301) · [PDF](03_reactivation_prioritization_models/Diekmann_Cheng_2023_Experience_structure_replay.pdf)

- **Scope:** Hippocampal replay generation coupled to spatial reinforcement learning; not systems consolidation.
- **Reactivation:** The SFMA algorithm stochastically chains stored experience tuples.
- **Priority:** An explicit product of experience strength, structural similarity, and inhibition of return. Strength reflects experience statistics and reward; similarity comes from a learned/default representation.
- **Storage target:** Replay trains action values in an RL agent. The core implementation is not a recurrent synaptic transfer model, although the authors discuss possible attractor-network realizations.
- **Summary:** SFMA is more mechanistically economical than evaluating every possible update’s future utility, and it reproduces diverse replay sequences. Nonetheless, its priority equation and memory variables are explicitly specified rather than produced by a recipient network’s local mismatch dynamics.

### Sun et al. (2023): Go-CLS

**Organizing memories for generalization in complementary learning systems.** *Nature Neuroscience*, 26, 1438–1448. [DOI](https://doi.org/10.1038/s41593-023-01382-9) · [PDF](03_reactivation_prioritization_models/Sun_et_al_2023_GoCLS.pdf)

- **Scope:** Normative theory and neural-network model of systems consolidation.
- **Reactivation:** A recurrent Hopfield “notebook” can reactivate stored examples; simulations randomly sample memories for training.
- **Priority:** The amount of consolidation should depend on environmental predictability and stop when further transfer harms generalization. Fully implementing this requires a supervisory process, such as validation-error monitoring; the biological mechanism is left open.
- **Storage target:** Recurrent synapses in the hippocampal notebook, but predominantly linear feedforward/readout weights in the neocortical student.
- **Summary:** Go-CLS gives a principled answer to how much information should be transferred, showing that complete transfer can overfit noise. It does not supply a local autonomous mechanism that selects the currently most discrepant memory during inter-system replay.

### Lindsey and Litwin-Kumar (2024)

**Selective consolidation of learning and memory via recall-gated plasticity.** *eLife*, 13, e90793. [DOI](https://doi.org/10.7554/eLife.90793) · [PDF](03_reactivation_prioritization_models/Lindsey_LitwinKumar_2024_Recall_gated_consolidation.pdf)

- **Scope:** Selective systems consolidation, but not primarily a model of autonomous replay generation; the mechanism can operate online or be implemented through selective replay.
- **Reactivation:** Not generated by the model itself.
- **Priority:** Long-term plasticity is gated by overlap between the current candidate update and short-term-memory weights. This filters out unreliable one-off events and favors consistently repeated, strongly recalled/familiar patterns. The signal is recipient-network-related in a broad sense, but it has the opposite monotonicity from novelty prioritization.
- **Storage target:** The theory covers feedforward readouts, reinforcement-learning weights, and recurrent autoassociative synapses. In the recurrent simulation, long-term recurrent weights are updated, while familiarity is approximated by a separate linear readout.
- **Summary:** This paper is an important counterexample to any claim that network state never controls consolidation. The safer distinction is that it explicitly constructs a recall/familiarity gate and does not let recipient-relative novelty emerge from autonomous dialogue between reactivating and recipient networks.

### Zhou, Kahana, and Schapiro (2025/2026)

**A unifying account of replay as context-driven memory reactivation.** *eLife*, article 99931. [DOI](https://doi.org/10.7554/eLife.99931) · [PDF](03_reactivation_prioritization_models/Zhou_Kahana_Schapiro_2026_Context_driven_reactivation.pdf)

- **Scope:** Primarily an autonomous replay model, with local learning during replay and a proof-of-concept teacher-to-student transfer simulation.
- **Reactivation:** Bidirectional item–context associations and a drifting context state autonomously generate replay sequences from minimal initial activity.
- **Priority:** Recency, reward, and salience influence association strengths during encoding. An explicit activity-dependent suppression rule makes highly experienced items less likely to initiate replay, allowing weaker or older traces to reappear. Priority is therefore history/context based, not a live measure of the recipient system’s representational discrepancy.
- **Storage target:** Bidirectional item–context association matrices; the transfer demonstration trains a second model of the same kind. This is not a conventional recurrent cortical-attractor or readout-only implementation.
- **Summary:** CMR-replay is a strong recent account of autonomous content generation and explains apparently value-sensitive replay without computing future utility during replay. It is a key precedent for emergent-looking novelty/familiarity effects, but those effects depend on explicit encoding-rate and suppression assumptions rather than inhibitory predictive dialogue with a recipient network.

## 04 — Empirical constraints

### Rothschild, Eban, and Frank (2017)

**A cortical–hippocampal–cortical loop of information processing during memory consolidation.** *Nature Neuroscience*, 20, 251–259. [DOI](https://doi.org/10.1038/nn.4457) · [PDF](04_empirical_constraints/Rothschild_Eban_Frank_2017_Cortical_hippocampal_cortical_loop.pdf)

- **Scope:** Empirical; no computational replay or consolidation implementation.
- **Main result:** Auditory-cortical activity can precede and predict hippocampal sharp-wave-ripple content, while hippocampal activity predicts later cortical activity. This supports a cortical–hippocampal–cortical loop rather than exclusively hippocampus-to-cortex teaching.
- **Synapses/readout:** Not applicable.
- **Relevance:** Provides physiological support for bidirectional interaction and cortical influence over reactivation content, but does not establish that the cortex computes novelty or discrepancy in the specific way proposed here.

### Schapiro et al. (2018)

**Human hippocampal replay during rest prioritizes weakly learned information and predicts memory performance.** *Nature Communications*, 9, 3920. [DOI](https://doi.org/10.1038/s41467-018-06213-1) · [PDF](04_empirical_constraints/Schapiro_et_al_2018_Replay_prioritizes_weakly_learned_information.pdf)

- **Scope:** Empirical human fMRI study; no computational mechanism.
- **Main result:** Items remembered less well before rest showed more subsequent hippocampal replay, and greater replay predicted better later memory. The evidence therefore supports preferential processing of information that remains weak or vulnerable.
- **Synapses/readout:** Not applicable.
- **Relevance:** Strongly motivates a deficit-sensitive priority signal, but “weakly learned” is behavioral and does not identify whether the brain computes hippocampal trace weakness, cortical unfamiliarity, prediction error, or another latent variable.

### Huelin Gorriz, Takigawa, and Bendor (2023)

**The role of experience in prioritizing hippocampal replay.** *Nature Communications*, 14, 8157. [DOI](https://doi.org/10.1038/s41467-023-43939-z) · [PDF](04_empirical_constraints/HuelinGorriz_Takigawa_Bendor_2023_Experience_prioritizes_replay.pdf)

- **Scope:** Empirical rodent study; no computational implementation.
- **Main result:** Sleep replay increased with repeated experience but decreased as the environment became familiar; cumulative awake replay was the strongest predictor of later sleep replay. Experience therefore has competing effects rather than a single “more exposure means more replay” relationship.
- **Synapses/readout:** Not applicable.
- **Relevance:** This paper supports distinguishing raw recency or exposure from familiarity. It is compatible with the idea that a recent, poorly integrated memory receives more offline processing, while warning against treating recency and novelty as universally identical variables.

### Shin and Jadhav (2024)

**Prefrontal cortical ripples mediate top-down suppression of hippocampal reactivation during sleep memory consolidation.** *Current Biology*, 34. [DOI](https://doi.org/10.1016/j.cub.2024.05.018) · [PDF](04_empirical_constraints/Shin_Jadhav_2024_Prefrontal_ripples_top_down_suppression.pdf)

- **Scope:** Empirical rodent electrophysiology; no computational implementation.
- **Main result:** Independent prefrontal ripples suppress hippocampal activity and reactivation, whereas coordinated prefrontal–hippocampal ripples support reactivation. The temporal organization is consistent with top-down control of hippocampal replay.
- **Synapses/readout:** Not applicable.
- **Relevance:** This is direct biological support for the plausibility of a top-down inhibitory pathway. It does not by itself show that inhibition carries a predictive estimate or computes recipient-relative novelty, so the model remains a mechanistic hypothesis built on a supported circuit motif.

## Compact comparison

| Paper | Autonomous content generation? | Replay only or systems consolidation? | What controls selection/transfer? | Main learned substrate |
|---|---:|---|---|---|
| McClelland et al. 1995 | No | Systems consolidation | Imposed interleaving | Distributed cortical weights |
| Fiebig & Lansner 2014 | Yes | Systems consolidation | Source attractor strength, adaptation, competition | Recurrent synapses |
| Káli & Dayan 2004 | Assumed/random | Systems consolidation/access maintenance | Random hippocampal activation | Bidirectional cortical weights + cleanup attractors |
| Singh et al. 2022 | Yes | Systems consolidation | Attractor dynamics, depression, sleep stage | Distributed recurrent/interactive weights |
| Spens & Burgess 2024 | Randomly cued | Systems consolidation | Random replay; reconstruction error acts at encoding | Hopfield source + VAE target |
| Mattar & Daw 2018 | Algorithmic | Replay/planning only | Expected gain × need | Value/policy estimates |
| Barry & Love 2022 | Generated samples, selected by controller | Knowledge consolidation analogue | Auxiliary RL controller using recipient improvement | Feedforward/convolutional weights |
| Diekmann & Cheng 2023 | Algorithmic/stochastic | Replay + RL learning | Strength × similarity × inhibition | Experience/value representations |
| Go-CLS 2023 | Notebook can reactivate | Systems consolidation | Normative generalization criterion; supervisor required | Hopfield source + linear student weights |
| Lindsey & Litwin-Kumar 2024 | No | Systems consolidation/gated plasticity | Explicit STM recall/familiarity gate | Readout or recurrent synapses, task dependent |
| Zhou et al. 2025/2026 | Yes | Replay; proof-of-concept transfer | Context associations, encoding rates, suppression | Item–context association matrices |

## Direct implication for the abstract

A literature-aligned formulation should contrast the present mechanism with **recipient-relative prioritization**, rather than with prioritization in general. A concise defensible version is:

> Reactivation-based systems consolidation has been the subject of extensive modeling efforts, including biologically plausible models in which reactivation arises autonomously from intrinsic network dynamics. However, these models generally do not derive replay priorities from the recipient system’s current representation of a memory, leaving replay unprioritized or biasing it through factors such as trace strength, recency, context, or reward. Here, we show that a preference for recent, unfamiliar memories can instead emerge from the dialogue between the neural populations involved in consolidation: memories that remain poorly represented by the recipient network generate a stronger and more persistent reactivation drive.

This wording acknowledges genuine autonomous and state-dependent precedents while preserving the specific mechanistic novelty of the present model.
