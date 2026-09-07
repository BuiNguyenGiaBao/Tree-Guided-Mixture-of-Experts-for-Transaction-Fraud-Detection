from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ThresholdState:
    threshold: float
    objective: str
    min_precision: Optional[float]
    val_precision: float
    val_recall: float
    val_f1: float
    val_mcc: float
    tn: int
    fp: int
    fn: int
    tp: int


def _safe_arrays(y_true: Iterable[int], y_prob: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(list(y_true)).astype(int)
    y_prob = np.asarray(list(y_prob)).astype(float)
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")

    return y_true, y_prob


def _metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> ThresholdState:
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    return ThresholdState(
        threshold=float(threshold),
        objective="manual",
        min_precision=None,
        val_precision=float(precision),
        val_recall=float(recall),
        val_f1=float(f1),
        val_mcc=float(mcc),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def search_best_threshold(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    objective: str = "f1",
    min_precision: Optional[float] = None,
    n_grid: int = 500,
) -> ThresholdState:
    y_true, y_prob = _safe_arrays(y_true, y_prob)

    objective = objective.lower()
    if objective not in {"f1", "mcc", "recall", "precision_constrained_recall"}:
        raise ValueError("objective must be one of: f1, mcc, recall, precision_constrained_recall.")

    thresholds = np.unique(
        np.concatenate([
            np.linspace(0.001, 0.999, n_grid),
            np.quantile(y_prob, np.linspace(0.001, 0.999, 300)),
        ])
    )

    best = None
    best_score = -np.inf

    for threshold in thresholds:
        state = _metrics_at_threshold(y_true, y_prob, float(threshold))

        if min_precision is not None and state.val_precision < min_precision:
            continue

        if objective == "f1":
            score = state.val_f1
        elif objective == "mcc":
            score = state.val_mcc
        elif objective == "recall":
            score = state.val_recall
        else:
            score = state.val_recall

        if score > best_score:
            best_score = score
            best = state

    if best is None:
        best = ThresholdState(
            threshold=float("nan"),
            objective=objective,
            min_precision=min_precision,
            val_precision=0.0,
            val_recall=0.0,
            val_f1=0.0,
            val_mcc=0.0,
            tn=int((y_true == 0).sum()),
            fp=0,
            fn=int((y_true == 1).sum()),
            tp=0,
        )
    else:
        best.objective = objective
        best.min_precision = min_precision

    return best


def recall_at_precision(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    min_precision: float = 0.80,
) -> Dict[str, float]:
    y_true, y_prob = _safe_arrays(y_true, y_prob)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    valid = precision >= min_precision

    if not np.any(valid):
        return {"recall": 0.0, "precision": 0.0, "threshold": float("nan")}

    valid_indices = np.where(valid)[0]
    best_idx = valid_indices[np.argmax(recall[valid])]

    if best_idx >= len(thresholds):
        threshold = 1.0
    else:
        threshold = float(thresholds[best_idx])

    return {
        "recall": float(recall[best_idx]),
        "precision": float(precision[best_idx]),
        "threshold": threshold,
    }


def evaluate_fraud_metrics(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    threshold: float,
    min_precision: float = 0.80,
) -> Dict[str, float]:
    y_true, y_prob = _safe_arrays(y_true, y_prob)
    state = _metrics_at_threshold(y_true, y_prob, threshold)
    rap = recall_at_precision(y_true, y_prob, min_precision=min_precision)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = np.nan

    return {
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
        "ROC_AUC": float(roc_auc),
        "Fraud_Precision": state.val_precision,
        "Fraud_Recall": state.val_recall,
        "Fraud_F1": state.val_f1,
        "MCC": state.val_mcc,
        f"Recall@Precision>={min_precision:.2f}": float(rap["recall"]),
        f"Threshold@Precision>={min_precision:.2f}": float(rap["threshold"]),
        "Selected_Threshold": float(threshold),
        "TN": state.tn,
        "FP": state.fp,
        "FN": state.fn,
        "TP": state.tp,
    }


class TwoExpertSoftmaxGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class ThresholdedMoEGatedCNNDeepFM(nn.Module):
    def __init__(
        self,
        cnn_branch: nn.Module,
        deepfm_branch: nn.Module,
        cnn_embedding_dim: int,
        deepfm_embedding_dim: int,
        fusion_dim: int = 128,
        gate_hidden_dim: int = 64,
        dropout: float = 0.20,
        initial_threshold: float = 0.50,
    ):
        super().__init__()

        self.cnn_branch = cnn_branch
        self.deepfm_branch = deepfm_branch

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_embedding_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.deepfm_proj = nn.Sequential(
            nn.Linear(deepfm_embedding_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gate = TwoExpertSoftmaxGate(
            input_dim=fusion_dim * 2,
            hidden_dim=gate_hidden_dim,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

        self.register_buffer(
            "decision_threshold",
            torch.tensor(float(initial_threshold), dtype=torch.float32),
        )

        self.threshold_state: Optional[ThresholdState] = None

    def _cnn_embedding(self, x_cnn: torch.Tensor) -> torch.Tensor:
        if hasattr(self.cnn_branch, "encode"):
            return self.cnn_branch.encode(x_cnn)
        if hasattr(self.cnn_branch, "compute_embedding"):
            return self.cnn_branch.compute_embedding(x_cnn)
        if hasattr(self.cnn_branch, "get_embedding"):
            return self.cnn_branch.get_embedding(x_cnn)
        return self.cnn_branch(x_cnn)

    def _deepfm_embedding(
        self,
        x_cat: Optional[torch.Tensor] = None,
        x_num: Optional[torch.Tensor] = None,
        x_dense: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hasattr(self.deepfm_branch, "extract_embedding"):
            return self.deepfm_branch.extract_embedding(
                x_cat=x_cat,
                x_num=x_num,
                x_dense=x_dense,
            )
        if hasattr(self.deepfm_branch, "encode"):
            return self.deepfm_branch.encode(
                x_cat=x_cat,
                x_num=x_num,
                x_dense=x_dense,
            )
        return self.deepfm_branch(x_cat=x_cat, x_num=x_num, x_dense=x_dense)

    def forward(
        self,
        x_cnn: torch.Tensor,
        x_cat: Optional[torch.Tensor] = None,
        x_num: Optional[torch.Tensor] = None,
        x_dense: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        z_cnn_raw = self._cnn_embedding(x_cnn)
        z_deepfm_raw = self._deepfm_embedding(x_cat=x_cat, x_num=x_num, x_dense=x_dense)

        z_cnn = self.cnn_proj(z_cnn_raw)
        z_deepfm = self.deepfm_proj(z_deepfm_raw)

        gate_input = torch.cat([z_cnn, z_deepfm], dim=1)
        gate_weight = self.gate(gate_input)

        z_fused = gate_weight[:, 0:1] * z_cnn + gate_weight[:, 1:2] * z_deepfm
        logits = self.classifier(z_fused).view(-1)

        if not return_dict:
            return logits

        prob = torch.sigmoid(logits)
        pred = (prob >= self.decision_threshold.to(prob.device)).long()

        return {
            "logits": logits,
            "prob": prob,
            "pred": pred,
            "threshold": self.decision_threshold.detach().clone(),
            "embedding": z_fused,
            "z_cnn": z_cnn,
            "z_deepfm": z_deepfm,
            "gate_weight": gate_weight,
        }

    @torch.no_grad()
    def predict_proba(
        self,
        x_cnn: torch.Tensor,
        x_cat: Optional[torch.Tensor] = None,
        x_num: Optional[torch.Tensor] = None,
        x_dense: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        out = self.forward(
            x_cnn=x_cnn,
            x_cat=x_cat,
            x_num=x_num,
            x_dense=x_dense,
            return_dict=True,
        )
        return out["prob"]

    @torch.no_grad()
    def predict(
        self,
        x_cnn: torch.Tensor,
        x_cat: Optional[torch.Tensor] = None,
        x_num: Optional[torch.Tensor] = None,
        x_dense: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
    ) -> torch.Tensor:
        self.eval()
        prob = self.predict_proba(x_cnn=x_cnn, x_cat=x_cat, x_num=x_num, x_dense=x_dense)

        if threshold is None:
            threshold_tensor = self.decision_threshold.to(prob.device)
        else:
            threshold_tensor = torch.tensor(float(threshold), device=prob.device)

        return (prob >= threshold_tensor).long()

    def set_threshold(self, threshold: float) -> None:
        self.decision_threshold.fill_(float(threshold))

    def tune_threshold(
        self,
        y_val: Iterable[int],
        y_val_prob: Iterable[float],
        objective: str = "f1",
        min_precision: Optional[float] = None,
    ) -> ThresholdState:
        state = search_best_threshold(
            y_true=y_val,
            y_prob=y_val_prob,
            objective=objective,
            min_precision=min_precision,
        )

        if not np.isnan(state.threshold):
            self.set_threshold(state.threshold)

        self.threshold_state = state
        return state

    def threshold_summary(self) -> Dict[str, object]:
        if self.threshold_state is None:
            return {
                "threshold": float(self.decision_threshold.detach().cpu().item()),
                "is_tuned": False,
            }

        out = asdict(self.threshold_state)
        out["is_tuned"] = True
        return out


class ConcatCNNDeepFM(nn.Module):
    def __init__(
        self,
        cnn_branch: nn.Module,
        deepfm_branch: nn.Module,
        cnn_embedding_dim: int,
        deepfm_embedding_dim: int,
        fusion_dim: int = 128,
        dropout: float = 0.20,
        initial_threshold: float = 0.50,
    ):
        super().__init__()

        self.cnn_branch = cnn_branch
        self.deepfm_branch = deepfm_branch

        self.fusion = nn.Sequential(
            nn.Linear(cnn_embedding_dim + deepfm_embedding_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(fusion_dim, 1)

        self.register_buffer(
            "decision_threshold",
            torch.tensor(float(initial_threshold), dtype=torch.float32),
        )

        self.threshold_state: Optional[ThresholdState] = None

    def _cnn_embedding(self, x_cnn: torch.Tensor) -> torch.Tensor:
        if hasattr(self.cnn_branch, "encode"):
            return self.cnn_branch.encode(x_cnn)
        if hasattr(self.cnn_branch, "compute_embedding"):
            return self.cnn_branch.compute_embedding(x_cnn)
        if hasattr(self.cnn_branch, "get_embedding"):
            return self.cnn_branch.get_embedding(x_cnn)
        return self.cnn_branch(x_cnn)

    def _deepfm_embedding(
        self,
        x_cat: Optional[torch.Tensor] = None,
        x_num: Optional[torch.Tensor] = None,
        x_dense: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hasattr(self.deepfm_branch, "extract_embedding"):
            return self.deepfm_branch.extract_embedding(
                x_cat=x_cat,
                x_num=x_num,
                x_dense=x_dense,
            )
        if hasattr(self.deepfm_branch, "encode"):
            return self.deepfm_branch.encode(
                x_cat=x_cat,
                x_num=x_num,
                x_dense=x_dense,
            )
        return self.deepfm_branch(x_cat=x_cat, x_num=x_num, x_dense=x_dense)

    def forward(
        self,
        x_cnn: torch.Tensor,
        x_cat: Optional[torch.Tensor] = None,
        x_num: Optional[torch.Tensor] = None,
        x_dense: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        z_cnn = self._cnn_embedding(x_cnn)
        z_deepfm = self._deepfm_embedding(x_cat=x_cat, x_num=x_num, x_dense=x_dense)

        z_fused = self.fusion(torch.cat([z_cnn, z_deepfm], dim=1))
        logits = self.classifier(z_fused).view(-1)

        if not return_dict:
            return logits

        prob = torch.sigmoid(logits)
        pred = (prob >= self.decision_threshold.to(prob.device)).long()

        return {
            "logits": logits,
            "prob": prob,
            "pred": pred,
            "threshold": self.decision_threshold.detach().clone(),
            "embedding": z_fused,
        }

    def set_threshold(self, threshold: float) -> None:
        self.decision_threshold.fill_(float(threshold))

    def tune_threshold(
        self,
        y_val: Iterable[int],
        y_val_prob: Iterable[float],
        objective: str = "f1",
        min_precision: Optional[float] = None,
    ) -> ThresholdState:
        state = search_best_threshold(
            y_true=y_val,
            y_prob=y_val_prob,
            objective=objective,
            min_precision=min_precision,
        )

        if not np.isnan(state.threshold):
            self.set_threshold(state.threshold)

        self.threshold_state = state
        return state

    def threshold_summary(self) -> Dict[str, object]:
        if self.threshold_state is None:
            return {
                "threshold": float(self.decision_threshold.detach().cpu().item()),
                "is_tuned": False,
            }

        out = asdict(self.threshold_state)
        out["is_tuned"] = True
        return out
