"""Recorded time series of a simulation. Kept light by recording every `record_every`
steps; weight matrices are stored only as a Frobenius distance trace plus a handful of
full snapshots for heatmaps.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from . import diagnostics as dg
from .model import TwoPopModel


class History:
    def __init__(self) -> None:
        self.t: List[float] = []
        self.F_T: List[float] = []
        self.F_S: List[float] = []
        self.Phi: List[float] = []
        self.eps_TS_norm: List[float] = []
        self.manifold_occ: List[float] = []
        self.align: List[torch.Tensor] = []          # (P,) per record
        self.novelty_spec: List[torch.Tensor] = []    # (k,) per record
        self.residual: List[torch.Tensor] = []        # (P,) per record
        self.WS_dist: List[float] = []                # ||W_S - W_T||_F
        self.x_T: List[torch.Tensor] = []             # (d,) per record
        self.weight_snaps: List[Dict] = []            # {'t', 'step', 'W_S'}

    def record(self, t: float, model: TwoPopModel, U_T: torch.Tensor, M: torch.Tensor) -> None:
        self.t.append(float(t))
        self.F_T.append(float(model.F_T()))
        self.F_S.append(float(model.F_S()))
        self.Phi.append(float(model.Phi()))
        self.eps_TS_norm.append(float(model.eps_TS().norm()))
        self.manifold_occ.append(dg.manifold_occupancy(model.x_T, U_T))
        self.align.append(dg.alignment(model.x_T, M).detach().cpu())
        self.novelty_spec.append(dg.restricted_novelty_spectrum(model, U_T).detach().cpu())
        self.residual.append(dg.per_direction_residual(model, M).detach().cpu())
        self.WS_dist.append(float((model.W_S - model.W_T).norm()))
        self.x_T.append(model.x_T.detach().cpu().clone())

    def snapshot_weights(self, t: float, step: int, model: TwoPopModel) -> None:
        self.weight_snaps.append(
            {"t": float(t), "step": int(step), "W_S": model.W_S.detach().cpu().clone()}
        )

    def to_numpy(self) -> Dict[str, np.ndarray]:
        return {
            "t": np.asarray(self.t),
            "F_T": np.asarray(self.F_T),
            "F_S": np.asarray(self.F_S),
            "Phi": np.asarray(self.Phi),
            "eps_TS_norm": np.asarray(self.eps_TS_norm),
            "manifold_occ": np.asarray(self.manifold_occ),
            "align": torch.stack(self.align).numpy(),
            "novelty_spec": torch.stack(self.novelty_spec).numpy(),
            "residual": torch.stack(self.residual).numpy(),
            "WS_dist": np.asarray(self.WS_dist),
            "x_T": torch.stack(self.x_T).numpy(),
        }


class ContinualHistory:
    """Per-cycle record of the continual-learning loop (`src.continual`).

    After adding memory `k`, records how well **Storage** still nulls every memory seen so far
    (`storage_residual[k][i] = ||M_Z m_i||`, low = retained), the Storage subspace dimension, the
    same residuals for the buffer-only **baseline** control (which forgets all but the latest), the
    consolidation deficit, and the fraction of past memories still retained.
    """

    def __init__(self, n_memories: int, tol: float = 0.3) -> None:
        self.n_memories = int(n_memories)
        self.tol = float(tol)
        self.storage_residual: List[torch.Tensor] = []    # per cycle: (k,) ||M_Z m_i||
        self.baseline_residual: List[torch.Tensor] = []    # per cycle: (k,) ||M_ctrl m_i||
        self.n_retained: List[int] = []                    # memories with residual < tol (accumulation)
        self.consolidation_deficit: List[float] = []
        self.retained_frac: List[float] = []
        self.baseline_retained_frac: List[float] = []

    def record(self, memories_upto: torch.Tensor, M_Z: torch.Tensor,
               M_ctrl: torch.Tensor, consolidation_deficit: float) -> None:
        res_Z = (M_Z @ memories_upto).norm(dim=0).detach().cpu()
        res_ctrl = (M_ctrl @ memories_upto).norm(dim=0).detach().cpu()
        self.storage_residual.append(res_Z)
        self.baseline_residual.append(res_ctrl)
        self.n_retained.append(int((res_Z < self.tol).sum()))
        self.consolidation_deficit.append(float(consolidation_deficit))
        self.retained_frac.append(float((res_Z < self.tol).double().mean()))
        self.baseline_retained_frac.append(float((res_ctrl < self.tol).double().mean()))

    def retention_matrix(self, which: str = "storage") -> np.ndarray:
        """(n_memories x n_cycles) matrix of residuals; entry [i,k] is memory i's residual at cycle
        k (NaN before it was added). `which` = "storage" or "baseline"."""
        rows = self.storage_residual if which == "storage" else self.baseline_residual
        n_cycles = len(rows)
        M = np.full((self.n_memories, n_cycles), np.nan)
        for k, res in enumerate(rows):
            M[: res.shape[0], k] = res.numpy()
        return M

    def to_numpy(self) -> Dict[str, np.ndarray]:
        return {
            "n_retained": np.asarray(self.n_retained),
            "consolidation_deficit": np.asarray(self.consolidation_deficit),
            "retained_frac": np.asarray(self.retained_frac),
            "baseline_retained_frac": np.asarray(self.baseline_retained_frac),
            "final_storage_residual": self.storage_residual[-1].numpy() if self.storage_residual else np.array([]),
            "final_baseline_residual": self.baseline_residual[-1].numpy() if self.baseline_residual else np.array([]),
            "retention_storage": self.retention_matrix("storage"),
            "retention_baseline": self.retention_matrix("baseline"),
        }


class InterleavedHistory:
    """Per-bout record of an interleaved merge (`src.interleaved`).

    Tracks the crosstalk-cancellation story: which teacher was rehearsed each bout, how well the
    plastic network nulls each teacher's memory (`resid1/resid2 = ||M_S U_k||_F`), and the deficit on
    the combined target subspace (`union_deficit`). For correlated memories `resid1/resid2` show a
    decaying sawtooth (each bout drops its own and bumps the other) converging to both nulled.
    """

    def __init__(self, labels: tuple = ("T1", "T2")) -> None:
        self.labels = labels
        self.bout: List[int] = []
        self.active: List[int] = []          # index of the teacher rehearsed this bout
        self.resid1: List[float] = []        # ||M_S U1||_F
        self.resid2: List[float] = []        # ||M_S U2||_F
        self.union_deficit: List[float] = []  # ||M_S U_Sigma||_F^2
        self.weight_snaps: List[Dict] = []

    def record(self, bout: int, active: int, M_S: torch.Tensor,
               U1: torch.Tensor, U2: torch.Tensor, U_Sigma: torch.Tensor) -> None:
        self.bout.append(int(bout))
        self.active.append(int(active))
        self.resid1.append(float((M_S @ U1).norm()))
        self.resid2.append(float((M_S @ U2).norm()))
        self.union_deficit.append(float((M_S @ U_Sigma).pow(2).sum()))

    def snapshot_weights(self, bout: int, W_S: torch.Tensor) -> None:
        self.weight_snaps.append({"bout": int(bout), "W_S": W_S.detach().cpu().clone()})

    def to_numpy(self) -> Dict[str, np.ndarray]:
        return {
            "bout": np.asarray(self.bout),
            "active": np.asarray(self.active),
            "resid1": np.asarray(self.resid1),
            "resid2": np.asarray(self.resid2),
            "union_deficit": np.asarray(self.union_deficit),
        }
