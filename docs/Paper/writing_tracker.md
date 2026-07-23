# Paper writing tracker — LIVING document

Process/progress tracker for writing the paper. **Rich content plan lives in [outline.md](outline.md); full proofs in [maths/final_proofs.md](maths/final_proofs.md).** Keep this file lean — one line per part, checkboxes, decisions log, next action.

**Manuscript file:** [latex/manuscript.tex](latex/manuscript.tex) · **Bibliography:** [latex/references.bib](latex/references.bib)

---

## Structural template (Tang et al. 2023, PLOS Comput Biol)

Abstract → Author summary → Introduction → **Models** (equations only, one subsection per component) → **Results** (declarative-claim headings; theorems stated inline next to their figure) → **Discussion** (Summary → Relationship to other models → Relationship to experimental data → Future directions) → Materials & Methods / Supporting Information (proofs, sim details).

## OPEN DECISIONS (resolve before drafting the affected part)

- [ ] **D1 — Models/Results split.** Option A (Tang-faithful: `Models` equations-only, then theorems woven into `Results`) vs Option B (current skeleton: standalone "Model and analysis" theory block). *Claude recommends A.* → **awaiting Paul.**
- [ ] **D2 — Final title** (see candidates in outline.md).
- [ ] **D3 — Abstract now (provisional) or after body?** Claude suggests provisional now + revisit pass at end.

## Working method (per part)

1. Claude posts a **claims-ledger** (the exact factual claims the part must make) → Paul edits/approves.
2. Claude drafts **~3 options** differing only in phrasing/order, not claims.
3. Paul corrects; iterate to lock.
4. Recompile PDF; add any new references to `.bib` with a one-line "why cited"; tick the box.

Style card (enforce every draft): sober, informative, scientific; no overstatement / no creative leap. Fixed lexicon: *novelty* = weight-level operator (N_S, n_k); *surprise* = state-level scalar (F_S); *reversed precision* kept verbatim (not `s·π_ST`). Banned: dramatically / remarkably / breakthrough / novel-as-hype. Math = centered `$$` blocks, blank line before AND after.

---

## Parts checklist

Status: ☐ not started · ◐ drafting · ☑ locked

- [ ] **Title** — working: see outline.md; decide D2.
- [ ] **Abstract** — hook = catastrophic forgetting + sleep replay; twist = priority is the student's own prediction error; transfer = primitive, continual learning = payoff. (≤300 words)
- [ ] **Author summary** — lay framing of prioritized replay + continual learning. (150–200 words)
- [ ] **Introduction** — continual learning/forgetting; sleep replay + the open question (how is replay prioritized & autonomous?); our answer (novelty-gated transfer in linear covPCN); the ladder; positioning (Tang, Friston, Schaul, McClelland, prev paper).
- [ ] **Models** — two-population wiring; three dynamics eqns; reversed precision π_ST (signed sleep/wake); F_S, F_T; operating regime (π_TS>π_S, |π_ST| well below π_T); interleaved interface; continual loop.
- [ ] **Results 1 — Prioritized transfer** — Thm 1 + Lemma 1 (N_S) + Cor 1 (surprise identity) + prioritization remark inline; Fig: novelty staircase + x_T alignment.
- [ ] **Results 2 — Wake/sleep saddle** — Prop 2 (exact saddle iff π_ST=−π_TS) inline; Fig: bowl vs hill cross-sections.
- [ ] **Results 3 — Merging by interleaving** — EMPIRICAL (no theorem); Fig: crosstalk sawtooth → 0.
- [ ] **Results 4 — Continual learning** — EMPIRICAL; Fig: retention matrix vs buffer-only forgetting.
- [ ] **Discussion** — VFE exact/EFE structural (honest); saddle = active-inference-reminiscent; bio reading (hippocampus→neocortex, sleep replay, wake/sleep = precision sign); limitations (linear ⇒ subspace); relation to Tang + prev paper.
- [ ] **Conclusion** — short.
- [ ] **Materials & Methods** — full proofs (Prop 1, Thm 1, Lemma 1, Cor 1, Prop 2); adiabatic elimination; sim details/params/diagnostics.
- [ ] **Supporting Information** — no-noise slow-mixing annex; ablations; stop-grad/dendritic; sweeps.
- [ ] **References** — build `.bib` incrementally.
- [ ] **Figures** — 4 core (see outline.md figure table); caption-first.

## Decisions log (append as we go)

- 2026-07-23 — Confirmed Tang structure via PLOS article; created this tracker.

## Next action

Resolve **D1** (Models/Results split), then start the **Title** (D2).
