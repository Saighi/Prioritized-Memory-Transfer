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


class AdditiveHistory:
    """Recorded time series for the additive three-network model (`pmt.additive`).

    Tracks the spec's success observables (section 19): the transfer deficits `E_Sigma, E1, E2`,
    the restricted novelty spectrum on `U_Sigma`, the source-novelty contributions `J1, J2, J12`,
    the termination signals `||eps_Sigma||, ||eps_S||, ||dW_S||_F`, teacher manifold leakage
    `L1, L2` and occupancy, and the running mixture persistent-excitation margin `lambda_min`.
    """

    def __init__(self) -> None:
        self.t: List[float] = []
        self.E_Sigma: List[float] = []
        self.E1: List[float] = []
        self.E2: List[float] = []
        self.novelty_spec: List[torch.Tensor] = []   # (r_Sigma,) descending, on U_Sigma
        self.J1: List[float] = []
        self.J2: List[float] = []
        self.J12: List[float] = []
        self.eps_Sigma_norm: List[float] = []
        self.eps_S_norm: List[float] = []
        self.dWS_norm: List[float] = []
        self.L1: List[float] = []
        self.L2: List[float] = []
        self.occ1: List[float] = []
        self.occ2: List[float] = []
        self.mix_min_eig: List[float] = []
        self.WS_dist: List[float] = []                # ||W_S - (target)||_F, if provided
        self.weight_snaps: List[Dict] = []
        self._Sigma_y: object = None                  # running sum of y y^T
        self._n_y: int = 0

    def record(self, t: float, macro, info: Dict) -> None:
        from . import diagnostics as dg
        from .macro import fwd

        pops = macro.populations
        S = pops["S"]
        T1, T2 = pops["T1"], pops["T2"]
        U1, U2, U_Sigma = info["U1"], info["U2"], info["U_Sigma"]
        pi_I, pi_S = info["pi_I"], info["pi_S"]
        a1, a2 = info["alpha1"], info["alpha2"]

        M_S = S.M
        S_S = S.S_op()
        N_S = dg.additive_novelty_operator(S_S, pi_I, pi_S)

        # combined additive prediction y = a1 x1 + a2 x2 (identity coordinate maps)
        y = a1 * T1.x + a2 * T2.x

        self.t.append(float(t))
        self.E_Sigma.append(dg.transfer_deficit(M_S, U_Sigma))
        self.E1.append(dg.transfer_deficit(M_S, U1))
        self.E2.append(dg.transfer_deficit(M_S, U2))
        self.novelty_spec.append(dg.restricted_novelty_on(N_S, U_Sigma).detach().cpu())

        Nx1 = N_S @ T1.x
        Nx2 = N_S @ T2.x
        self.J1.append(float((a1 * Nx1).pow(2).sum()))
        self.J2.append(float((a2 * Nx2).pow(2).sum()))
        self.J12.append(float(2 * a1 * a2 * (Nx1 @ Nx2)))    # 2 a1 a2 x1^T N_S^2 x2

        eps_sigma_sq = 0.0
        for itf in macro.interfaces:
            eps_sigma_sq += float(itf.eps(pops).pow(2).sum())
        self.eps_Sigma_norm.append(float(eps_sigma_sq ** 0.5))
        self.eps_S_norm.append(float(fwd(M_S, S.x).norm()))
        self.dWS_norm.append(float(S.rate_W().norm()))

        self.L1.append(dg.manifold_leakage(T1.x, U1))
        self.L2.append(dg.manifold_leakage(T2.x, U2))
        self.occ1.append(dg.manifold_occupancy(T1.x, U1))
        self.occ2.append(dg.manifold_occupancy(T2.x, U2))

        yy = y.unsqueeze(-1) * y.unsqueeze(-2)
        self._Sigma_y = yy if self._Sigma_y is None else self._Sigma_y + yy
        self._n_y += 1
        self.mix_min_eig.append(dg.mixture_min_eig(self._Sigma_y / self._n_y, U_Sigma))

        if "W_target" in info:
            self.WS_dist.append(float((S.W - info["W_target"]).norm()))

    def snapshot_weights(self, t: float, step: int, macro) -> None:
        self.weight_snaps.append(
            {"t": float(t), "step": int(step), "W_S": macro.populations["S"].W.detach().cpu().clone()}
        )

    def to_numpy(self) -> Dict[str, np.ndarray]:
        out = {
            "t": np.asarray(self.t),
            "E_Sigma": np.asarray(self.E_Sigma),
            "E1": np.asarray(self.E1),
            "E2": np.asarray(self.E2),
            "novelty_spec": torch.stack(self.novelty_spec).numpy(),
            "J1": np.asarray(self.J1),
            "J2": np.asarray(self.J2),
            "J12": np.asarray(self.J12),
            "eps_Sigma_norm": np.asarray(self.eps_Sigma_norm),
            "eps_S_norm": np.asarray(self.eps_S_norm),
            "dWS_norm": np.asarray(self.dWS_norm),
            "L1": np.asarray(self.L1),
            "L2": np.asarray(self.L2),
            "occ1": np.asarray(self.occ1),
            "occ2": np.asarray(self.occ2),
            "mix_min_eig": np.asarray(self.mix_min_eig),
        }
        if self.WS_dist:
            out["WS_dist"] = np.asarray(self.WS_dist)
        return out


class ContinualHistory:
    """Per-cycle record of the continual-learning loop (`pmt.continual`).

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
    """Per-bout record of an interleaved merge (`pmt.interleaved`).

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
