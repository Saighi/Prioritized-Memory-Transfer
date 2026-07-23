# Prioritized Memory Transfer — simulation

PyTorch implementation of the predictive-coding memory-transfer models. **One engine, three models**
— networks always communicate **two at a time**:

1. the **two-population** model (teacher T → student S) specified in
   [`two_population_memory_transfer_model.md`](two_population_memory_transfer_model.md) and analysed
   in [`analysis_stress_test.md`](analysis_stress_test.md);
2. **interleaved subspace addition** (two frozen teachers merged into one plastic network by
   rehearsing one teacher per replay bout);
3. a **continual-learning loop** (buffer → synthesis → storage) built on the interleaved merge.

All are instances of one composable **"network of networks" engine** (`prioritized_memory_transfer.macro`): predictive-coding
`Population` nodes wired by `CouplingInterface` edges into a `MacroNetwork` (edges can be switched
`active`/off, which is how the interleaving alternates teachers).

**Two-population.** A **teacher** T (a flat, linear covPCN associative memory, higher in the hierarchy)
and an empty **student** S below it are coupled. During **sleep / replay** — a **reversed (negative)
precision** `π_ST < 0` on the teacher's interface — noise stirs T; S's prediction error along memories
it lacks drives T into those memories; S learns them; and the replay/learning drive along each learned
direction extinguishes itself. The numerical runner still uses a fixed `n_steps` budget—there is no
automatic convergence detector. Because T is linear, what transfers is the memory **subspace** (the staircase has
~effective-rank steps, = `P` for orthonormal patterns) — see §9 of the analysis.

**Interleaved subspace addition.** Two frozen teachers, one plastic synthesis network; the synthesis
learns from **one teacher per replay bout**, alternating. The covariance it learns is
`Σ = p₁Σ₁ + p₂Σ₂` (no teacher cross-term), so the combined subspace `𝒰_Σ = 𝒰₁ + 𝒰₂` is built even
for single-memory, **correlated** teachers. The back-and-forth cancels crosstalk: rehearse T₁ →
nulls `m₁`, bumps `m₂`; rehearse T₂ → nulls `m₂`, bumps `m₁`; the bumps decay to zero (interleaved
rehearsal — the standard cure for catastrophic forgetting).

**Continual learning.** A stream of (correlated) memories is consolidated one at a time by three networks:
a fast **Buffer** (one-shot covPCN, overwritten each memory — the only network that ever learns from an
actual memory), a transient **Synthesis** workspace, and a slow **Storage**. Each cycle: write the new
memory into the Buffer, consolidate `Buffer + Storage → Synthesis` by **interleaved rehearsal** (alternate
rehearsing Buffer and Storage, so the single-memory Buffer merges cleanly and Storage's rehearsal cancels
crosstalk with correlated old memories), then download `Synthesis → Storage`. Storage accumulates the whole
stream (up to capacity `d−1`) while a buffer-only control catastrophically forgets everything but the latest.

## Setup

Uses the existing **`pytorch`** conda env (Python 3.10, torch 2.5, CUDA optional). One-time
editable install of the package (plus plotly/nbformat for the interactive figures):

```bash
conda run -n pytorch pip install -e . --no-deps
conda run -n pytorch pip install plotly nbformat
```

After this, `import prioritized_memory_transfer` works from anywhere in the environment.

## Run

Open [`notebooks/two_network/01_single_run.py`](notebooks/two_network/01_single_run.py) (or the
interleaved [`notebooks/subspace_addition/01_interleaved_single_run.py`](notebooks/subspace_addition/01_interleaved_single_run.py))
in VS Code, pick the **`pytorch`** interpreter as the kernel, and run the `# %%` cells top to bottom.
You get the spectral guards, the faithfulness self-checks, a full transfer run, a 6-panel static
dashboard, `W_S`-convergence snapshots, and interactive plotly figures.

Headless (no figures shown), e.g. to verify a notebook runs:

```bash
PMT_NO_SHOW=1 MPLBACKEND=Agg conda run -n pytorch --no-capture-output python notebooks/two_network/01_single_run.py
PMT_NO_SHOW=1 MPLBACKEND=Agg conda run -n pytorch --no-capture-output python notebooks/subspace_addition/01_interleaved_single_run.py
```

The checks are executable scripts with top-level assertions; pytest is not currently a declared
dependency. Run them directly:

```bash
conda run -n pytorch --no-capture-output python tests/smoke_test.py       # two-pop invariants, gradients (Prop 1), surprise identity (Cor 1), circulation (Prop 3), transfer
conda run -n pytorch --no-capture-output python tests/macro_test.py       # engine==2-pop equivalence
conda run -n pytorch --no-capture-output python tests/interleaved_test.py # interleaving builds the union of correlated single memories
conda run -n pytorch --no-capture-output python tests/continual_test.py   # interleaved continual retains a correlated stream
conda run -n pytorch --no-capture-output python tests/validation_test.py # invalid configs/graphs/patterns fail at their boundary
conda run -n pytorch --no-capture-output python tests/viz_test.py         # all single-run figures build
conda run -n pytorch --no-capture-output python scripts/run_findings.py   # run all 4 experiment notebooks headless (from the repo root)
```

Use `--no-capture-output`; plain `conda run` mangles tqdm progress bars. On Windows consoles, enable
UTF-8 before `scripts/run_findings.py` because its verdicts contain mathematical Unicode:

```powershell
$env:PYTHONUTF8 = "1"
conda run -n pytorch --no-capture-output python scripts/run_findings.py
```

## Layout

```
prioritized_memory_transfer/
  macro.py         the engine: Population + CouplingInterface + MacroNetwork (fwd/bwd/outer);
                   assembles rates, adiabatic solve, one unified step() — the LEGO layer
  config.py        ModelConfig / InterleavedConfig / ContinualConfig / SimConfig
  artifacts.py     typed memory/build diagnostics (replaces loosely-typed info dictionaries)
  memory.py        pattern generators, validated zero-diagonal covPCN fits, two-teacher geometries
  model.py         build_system (two-pop) -> MacroNetwork; TwoPopModel facade; simulate()
  interleaved.py   build_interleaved_synthesis + interleave_merge (one teacher per bout) -> MacroNetwork
  continual.py     ContinualLearner: write→interleaved-consolidate→download loop over a memory stream
  history.py       History + InterleavedHistory + ContinualHistory
  recall.py        AssociativeMemory: one network, clamped-query pattern completion (standalone)
  diagnostics.py   spectral gap, manifold/orthonormal bases, novelty operator & restricted spectrum,
                   transfer deficit, VFE/circulation checks
  viz_static.py / viz_interactive.py / viz_eigenspace.py   two-population figures
  viz_interleaved.py / viz_continual.py                    crosstalk-sawtooth / retention figures
  experiments.py   sweep helpers for the two-population experiment notebooks
notebooks/
  two_network/            the two-population model (01_single_run … 07_stopgrad_dendritic)
  subspace_addition/      the interleaved merge — 01_interleaved_single_run (crosstalk sawtooth),
                          02_interleaved_edge_cases (correlation sweep, multi-memory)
  continual_learning/     the buffer → synthesis → storage loop (interleaved consolidation)
    01_continual_single_stream.py   core demo (Storage keeps a correlated stream; buffer-only forgets)
    02_continual_edge_cases.py      storage-rehearsal ablation, capacity, correlated-vs-random
  associative_recall/     one-network clamped recall on MNIST (01_clamped_recall)
tests/
  smoke_test.py, macro_test.py, recall_test.py, interleaved_test.py, continual_test.py,
  validation_test.py, viz_test.py
scripts/
  probe_findings.py (parameter sweeps, prints verdicts), run_findings.py (runs the experiment
  notebooks headless) — exploration scripts, not tests
archive/
  retired material: the additive three-network (summed) model spec and notebooks, and the
  SMACOF embedding visualization (viz_embedding/) — kept for reference, not part of the paper
```

## Faithfulness (enforced by construction / asserted in checks)

- **Tied weights**: one matrix per population (`W_T`, `W_S`); top-down uses `M = I − W`,
  bottom-up uses `Mᵀ` (the same matrix transposed). The T↔S interface is identity both ways.
- **No autapses, ever**: `diag(W_T) = diag(W_S) = 0` at construction and after every update.
- **VFE**: the student's perception is exact descent on `F_S`; learning is the gradient projected
  onto the zero-diagonal weight subspace (autograd-verified before projection).
- **Saddle**: `circulation ≈ 0` only when `π_ST = −π_TS` (`ModelConfig(exact_saddle=True)`).
- **Memories in `ker M_T`**: construction measures `max_p ‖M_T m_p‖`, numerical rank, and
  conditioning, and rejects a pattern set that the no-autapse network cannot represent.
  `P ≤ d−1` is necessary but not sufficient for arbitrary sparse/adversarial patterns; it succeeds
  generically for the dense generated patterns used by the experiments.
- **Reversed precision (sleep/wake)**: the teacher's interface precision `π_ST` is *signed*.
  Transfer is simulated with a **reversed (negative) precision** `π_ST < 0` (the teacher
  maximizes the interface error — drive-to-disagree); `π_ST > 0` is ordinary inference/recall
  with no transfer drive. The sign of `π_ST` *is* the phase — there is no separate gate.

## Experiments (`notebooks/02..07`)

Each notebook turns an analytic claim from `analysis_stress_test.md` into a simulation-vs-theory
figure and prints a verdict. All are **confirmed numerically**:

- **Saddle** — circulation matches `|π_ST+π_TS|·√d` to machine precision; it vanishes only at
  `π_ST=−π_TS` (the teacher's reversed precision is the exact negative of the student's), so the
  single-potential saddle exists only there.
- **Stability** — both the off-manifold growth eigenvalue and the dynamical terminal
  manifold-occupancy turn over exactly at `|π_ST|* = π_T·σ²_min·(π_TS+π_S)/π_S`; below it T consolidates
  and stays quiescent on-manifold, above it T chases noise.
- **Timescale** — a known pattern is explained away (`‖ε_TS‖→0`) across the whole fast-S
  regime while a novel one stays at `n(1)`; the discrimination only collapses as `τ_S→τ_T`, so
  selection is spectral, not a speed race.
- **Subspace** — orthonormal `P=6` patterns give 6 staircase steps; correlated rank-3 patterns
  generated with `corr_rank=3, corr_noise=0`
  give 3 (= effective rank `< P`). The linear model transfers a *subspace*; discrete one-per-pattern
  (episodic) replay would need an added nonlinearity / soft-WTA.
- **Stop-gradient** — the Tang-style dendritic variant (drop the backward `Wᵀε` term) still
  transfers every direction; the price is a ~2× looser residual floor and a non-symmetric `N_S`.
