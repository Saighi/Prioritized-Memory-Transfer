"""Continual learning by consolidation (buffer -> synthesis -> storage), on the `src.macro` engine.

A complementary-learning-systems loop over THREE persistent networks (each a `Population` holding
an evolving zero-diagonal weight matrix):

  - Buffer  `B` : fast one-slot store. Each new memory is written by one-shot covPCN (`build_W_T`),
                  overwriting the previous one (interference erases it). This is the ONLY network
                  that ever learns from an actual memory; everything downstream learns by replay.
  - Storage `Z` : slow long-term store; starts empty (`W_Z = 0`).
  - Synthesis `S`: transient consolidation workspace.

Per new memory `m_k`:
  1. write `m_k` into `B` (one-shot covPCN);
  2. CONSOLIDATE  `B + Z -> S`  by **interleaved rehearsal**: `S` (warm-started from `Z`) is coupled
     to ONE network per replay bout, alternating Buffer and Storage. Never summing the two removes
     the cross-term that pins single-memory sources, and the back-and-forth is the co-excitation that
     cancels crosstalk between correlated memories (see `src.interleaved`);
  3. DOWNLOAD     `S -> Z`      (single-teacher transfer): Storage absorbs the union.

Over the stream Storage retains the whole subspace sum of all memories (up to capacity `d-1`), while
a buffer-only control catastrophically forgets everything but the latest.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from .config import ContinualConfig, SimConfig
from .diagnostics import orthonormal_basis, spectral_gap, transfer_deficit
from .history import ContinualHistory, InterleavedHistory
from .interleaved import interleave_merge
from .macro import CouplingInterface, MacroNetwork, Population, resolve_signed_precision
from .memory import build_W_T


def _make_memories(cfg: ContinualConfig) -> torch.Tensor:
    """A `(d, n_memories)` matrix of unit-norm memory columns for the stream.

    "correlated" (default): each memory shares a common component, so the memories are pairwise
    non-orthogonal (average cosine ~0.5) but still full rank `n` — the regime where interleaved
    rehearsal has to cancel crosstalk. "random": near-orthogonal unit vectors.
    """
    gen = torch.Generator().manual_seed(cfg.seed)
    d, n = cfg.d, cfg.n_memories
    if cfg.memory_kind == "random":
        M = torch.randn(d, n, generator=gen, dtype=cfg.dtype)
    elif cfg.memory_kind == "correlated":
        base = torch.randn(d, n, generator=gen, dtype=cfg.dtype)
        shared = torch.randn(d, 1, generator=gen, dtype=cfg.dtype)
        M = base + shared                       # common component -> correlated but full rank
    else:
        raise ValueError(f"unknown memory_kind={cfg.memory_kind!r} (pass memories= explicitly)")
    M = M / M.norm(dim=0, keepdim=True)
    return M.to(cfg.device)


class ContinualLearner:
    """Orchestrates the write -> consolidate -> download loop over a stream of memories.

    Pass an explicit `(d, n)` `memories` matrix to control the stream (e.g. revisits / correlated
    columns); otherwise it is generated from `cfg.memory_kind`.
    """

    def __init__(self, cfg: ContinualConfig, memories: Optional[torch.Tensor] = None) -> None:
        self.cfg = cfg
        self.d = cfg.d
        self.dtype = cfg.dtype
        self.device = cfg.device
        self.I = torch.eye(self.d, dtype=self.dtype, device=self.device)

        self.memories = memories if memories is not None else _make_memories(cfg)
        self.n_memories = self.memories.shape[1]

        z = torch.zeros(self.d, self.d, dtype=self.dtype, device=self.device)
        self.W_B = z.clone()      # buffer (overwritten each cycle)
        self.W_Z = z.clone()      # storage (accumulates)
        self.W_S = z.clone()      # synthesis (workspace)
        self.W_ctrl = z.clone()   # buffer-only control (catastrophic-forgetting baseline)

        self.hist = ContinualHistory(self.n_memories)
        self.last_consolidation_trace: Optional[InterleavedHistory] = None   # for the crosstalk viz
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed + 777)

    # ----- helpers -----
    def _M(self, W: torch.Tensor) -> torch.Tensor:
        return self.I - W

    def _resolve_rho(self, sources: List[Population], alphas: List[float]) -> float:
        """Reversed source-side precision inside the structure guard. The guard's sigma^2_min is the
        smallest *off-manifold* surprise eigenvalue, so we ignore the small-but-nonzero eigenvalues of
        the approximately-learned memory directions (tol=0.1) that would otherwise collapse it."""
        cfg = self.cfg
        Cnorm2 = sum(a * a for a in alphas)
        guard = min(cfg.pi_teacher * spectral_gap(p.S_op(), tol=0.1) for p in sources) / Cnorm2
        return resolve_signed_precision(cfg.rho, guard=guard, safety=cfg.rho_safety)

    def _source_pop(self, name: str, W: torch.Tensor) -> Population:
        cfg = self.cfg
        return Population(name, W.clone(), cfg.pi_teacher, cfg.tau_teacher,
                          plastic=False, sigma_xi=cfg.sigma_xi, r=cfg.r)

    def _plastic_pop(self, name: str, W_init: torch.Tensor) -> Population:
        cfg = self.cfg
        return Population(name, W_init.clone(), cfg.pi_S, cfg.tau_S, plastic=True, eta=cfg.eta)

    def _single_interface(self, target: str, source: Population) -> CouplingInterface:
        """An inactive single-source interface `target <- source` with rho resolved from the source's
        live weights (the interleaving driver switches it on for its bout)."""
        return CouplingInterface(target=target, sources=[source.name], alpha=[1.0],
                                 pi_I=self.cfg.pi_I, rho=self._resolve_rho([source], [1.0]),
                                 active=False)

    def _sim(self) -> SimConfig:
        return SimConfig(dt=self.cfg.dt, mode=self.cfg.mode, bout_steps=self.cfg.bout_steps,
                         s_substeps=1, progress=False)

    # ----- the three phase primitives -----
    def store_in_buffer(self, m: torch.Tensor) -> None:
        """One-shot covPCN of the single new memory; overwrites the buffer (and the control)."""
        col = (m / m.norm().clamp_min(1e-12)).reshape(self.d, 1)
        self.W_B = build_W_T(col, self.cfg)
        self.W_ctrl = self.W_B.clone()     # the baseline network only ever holds the latest memory

    def consolidate(self, k: int) -> None:
        """B (holds mₖ) + Z (holds m₀..m_{k-1}) -> S by INTERLEAVED rehearsal: S is coupled to one
        network per bout, alternating Buffer and Storage. S is warm-started from Z; Storage is
        rehearsed in as a co-teacher (unless `storage_support=False`). Records the per-bout crosstalk
        trace of this consolidation for visualization."""
        cfg = self.cfg
        storage_nonempty = k > 0
        B = self._source_pop("B", self.W_B)
        W_S_init = self.W_Z if (cfg.synthesis_warm_start and storage_nonempty) else \
            torch.zeros_like(self.W_Z)
        S = self._plastic_pop("S", W_S_init)

        pops: List[Population] = [B, S]
        interfaces = [self._single_interface("S", B)]
        order = ["B"]
        bases = {"B": orthonormal_basis(self.memories[:, k : k + 1])}
        if storage_nonempty and cfg.storage_support:
            Z = self._source_pop("Z", self.W_Z)
            pops = [B, Z, S]
            interfaces = [self._single_interface("S", B), self._single_interface("S", Z)]
            order = ["B", "Z"]
            bases["Z"] = orthonormal_basis(self.memories[:, :k])
        macro = MacroNetwork(pops, interfaces)

        # crosstalk trace: residual on the NEW (buffer) memory vs the OLD (storage) memories
        U_new = bases["B"]
        U_old = bases.get("Z", self.memories[:, :0])
        U_all = orthonormal_basis(self.memories[:, : k + 1])
        trace = InterleavedHistory(labels=("buffer", "storage"))

        def record(bout: int, active: int) -> None:
            trace.record(bout, active, S.M, U_new, U_old, U_all)

        self.W_S = interleave_merge(macro, order, bases, "S", self._sim(), self.gen,
                                    cfg.consolidate_bouts, record).clone()
        self.last_consolidation_trace = trace

    def download(self, k: int) -> None:
        """S (holds m₀..mₖ) -> Z, single-teacher transfer, Storage warm-started at its weights."""
        cfg = self.cfg
        teacher = self._source_pop("Sy", self.W_S)
        Z = self._plastic_pop("Z", self.W_Z)
        macro = MacroNetwork([teacher, Z], [self._single_interface("Z", teacher)])
        bases = {"Sy": orthonormal_basis(self.memories[:, : k + 1])}
        self.W_Z = interleave_merge(macro, ["Sy"], bases, "Z", self._sim(), self.gen,
                                    cfg.download_bouts).clone()

    # ----- the loop -----
    def add_memory(self, k: int) -> None:
        self.store_in_buffer(self.memories[:, k])
        self.consolidate(k)
        self.download(k)

        upto = self.memories[:, : k + 1]
        deficit = transfer_deficit(self._M(self.W_S), upto)
        self.hist.record(upto, self._M(self.W_Z), self._M(self.W_ctrl), deficit)

    def run(self, progress: bool = False) -> ContinualHistory:
        it = range(self.n_memories)
        if progress:
            try:
                from tqdm import tqdm
                it = tqdm(it, desc="continual[write/consolidate/download]")
            except Exception:
                pass
        for k in it:
            self.add_memory(k)
        return self.hist
