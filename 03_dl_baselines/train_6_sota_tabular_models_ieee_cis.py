from __future__ import annotations

"""
Train 6 recent/strong tabular baselines for IEEE-CIS style fraud detection.

Models:
1. TabM_like               - self-contained PyTorch multi-head tabular ensemble MLP
2. TabR_like               - self-contained retrieval-augmented tabular MLP
3. FTTransformer           - self-contained feature-token Transformer
4. SAINT_like              - self-contained Transformer baseline inspired by SAINT
5. TabPFN                  - external package: tabpfn
6. AutoGluon_Tabular       - external package: autogluon.tabular

Important:
- The self-contained TabM/TabR/SAINT implementations are practical reproductions/approximations
  for benchmarking. For final Rank-A paper submission, cite and, if possible, re-run official
  implementations or clearly describe these as reproduced baselines.
- TabPFN and AutoGluon are optional. If not installed, the script will skip them and continue.

Expected processed folder:
D:\project\data\merge_paper_ready_tree_cost

Required files in processed folder:
- full_train / full_val / full_internal_test
- y_train / y_val / y_internal_test
- review_meta_internal_test, optional but recommended for cost-aware metrics

Supported file formats:
.parquet, .csv, .pkl, .pickle, .feather

Outputs:
<processed_dir>\sota_6_tabular_model_results
- sota_6_model_results_final.csv
- paper_ready_sota_6_model_table.csv
- pred_<model>.csv
- review_budget_detail_<model>.csv
- checkpoints for PyTorch models
"""

import argparse
import gc
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ============================================================
# CONFIG
# ============================================================

DEFAULT_PROCESSED_DIR = Path(r"D:\project\data\merge_paper_ready_tree_cost")
TARGET_COL = "isFraud"

DEFAULT_MODELS = [
    "tabm",
    "tabr",
    "ft_transformer",
    "saint",
    "tabpfn",
    "autogluon",
]

REVIEW_BUDGETS = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]


# ============================================================
# UTILITIES
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("@", "at")
        .replace(">=", "ge")
        .replace("%", "pct")
    )


def find_table_path(base_dir: Path, stem: str) -> Path:
    candidates = [
        base_dir / stem,
        base_dir / f"{stem}.parquet",
        base_dir / f"{stem}.csv",
        base_dir / f"{stem}.pkl",
        base_dir / f"{stem}.pickle",
        base_dir / f"{stem}.feather",
    ]

    for p in candidates:
        if p.exists() and p.is_file():
            return p

    # More flexible search
    for ext in ["parquet", "csv", "pkl", "pickle", "feather"]:
        matches = list(base_dir.rglob(f"{stem}.{ext}"))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Cannot find table: {stem} in {base_dir}")


def read_table(base_dir: Path, stem: str) -> pd.DataFrame:
    p = find_table_path(base_dir, stem)
    print(f"[READ] {stem}: {p}")

    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in [".pkl", ".pickle"]:
        return pd.read_pickle(p)
    if p.suffix.lower() == ".feather":
        return pd.read_feather(p)

    raise ValueError(f"Unsupported table format: {p}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}")


def reduce_memory_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32)
        elif pd.api.types.is_integer_dtype(df[c]):
            # Float32 is safer for PyTorch and missing values.
            df[c] = df[c].astype(np.float32)
    return df


def ensure_y_1d(y: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(y, pd.DataFrame):
        if TARGET_COL in y.columns:
            y = y[TARGET_COL]
        else:
            y = y.iloc[:, 0]
    if isinstance(y, pd.Series):
        y = y.values
    y = np.asarray(y).reshape(-1)
    return y.astype(np.int64)


def align_feature_columns(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = list(train.columns)

    # remove accidental target from features
    cols = [c for c in cols if c != TARGET_COL]

    val = val.reindex(columns=cols, fill_value=0)
    test = test.reindex(columns=cols, fill_value=0)
    train = train[cols]

    return train, val, test


def sample_rows(
    X: np.ndarray,
    y: np.ndarray,
    max_rows: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_rows <= 0 or len(y) <= max_rows:
        return X, y

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # Keep all positives if possible, sample negatives.
    n_pos = len(pos_idx)
    n_neg_needed = max_rows - n_pos

    if n_neg_needed <= 0:
        chosen = rng.choice(np.arange(len(y)), size=max_rows, replace=False)
    else:
        neg_choice = rng.choice(neg_idx, size=min(n_neg_needed, len(neg_idx)), replace=False)
        chosen = np.concatenate([pos_idx, neg_choice])
        rng.shuffle(chosen)

    return X[chosen], y[chosen]


def select_top_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    max_features: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if max_features <= 0 or X_train.shape[1] <= max_features:
        return X_train, X_val, X_test, feature_names

    print(f"[FEATURE SELECT] Selecting top {max_features} / {X_train.shape[1]} features by f_classif...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, _ = f_classif(np.nan_to_num(X_train), y_train)

    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    idx = np.argsort(scores)[::-1][:max_features]
    idx = np.sort(idx)

    selected_names = [feature_names[i] for i in idx]
    print(f"[FEATURE SELECT] Done. Selected shape: {X_train[:, idx].shape}")

    return X_train[:, idx], X_val[:, idx], X_test[:, idx], selected_names


def prepare_numeric_arrays(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train: np.ndarray,
    max_features: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], SimpleImputer, StandardScaler]:
    X_train_df, X_val_df, X_test_df = align_feature_columns(X_train_df, X_val_df, X_test_df)
    feature_names = list(X_train_df.columns)

    X_train_df = reduce_memory_numeric(X_train_df)
    X_val_df = reduce_memory_numeric(X_val_df)
    X_test_df = reduce_memory_numeric(X_test_df)

    X_train = X_train_df.to_numpy(dtype=np.float32, copy=False)
    X_val = X_val_df.to_numpy(dtype=np.float32, copy=False)
    X_test = X_test_df.to_numpy(dtype=np.float32, copy=False)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train).astype(np.float32)
    X_val = imputer.transform(X_val).astype(np.float32)
    X_test = imputer.transform(X_test).astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    X_train, X_val, X_test, feature_names = select_top_features(
        X_train, y_train, X_val, X_test, feature_names, max_features=max_features
    )

    return X_train, X_val, X_test, feature_names, imputer, scaler


# ============================================================
# METRICS
# ============================================================

def threshold_by_best_f1(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    if len(thresholds) == 0:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    p = precision[:-1]
    r = recall[:-1]
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    idx = int(np.nanargmax(f1))

    return {
        "threshold": float(thresholds[idx]),
        "precision": float(p[idx]),
        "recall": float(r[idx]),
        "f1": float(f1[idx]),
    }


def recall_at_precision(y_true: np.ndarray, prob: np.ndarray, min_precision: float = 0.80) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)

    valid = np.where(precision[:-1] >= min_precision)[0]
    if len(valid) == 0:
        return 0.0, 1.0

    best_idx = valid[np.argmax(recall[:-1][valid])]
    return float(recall[:-1][best_idx]), float(thresholds[best_idx])


def evaluate_standard_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    min_precision: float = 0.80,
) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)

    try:
        pr_auc = average_precision_score(y_true, prob)
    except Exception:
        pr_auc = np.nan

    try:
        roc_auc = roc_auc_score(y_true, prob)
    except Exception:
        roc_auc = np.nan

    rec_p80, thr_p80 = recall_at_precision(y_true, prob, min_precision=min_precision)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "PR_AUC": float(pr_auc),
        "ROC_AUC": float(roc_auc),
        "Fraud_Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Fraud_Recall": float(recall_score(y_true, pred, zero_division=0)),
        "Fraud_F1": float(f1_score(y_true, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, pred)),
        "Recall@Precision>=0.80": float(rec_p80),
        "Threshold@Precision>=0.80": float(thr_p80),
        "Selected_Threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def load_review_meta(processed_dir: Path, y_test: np.ndarray) -> pd.DataFrame:
    try:
        meta = read_table(processed_dir, "review_meta_internal_test")
        if len(meta) != len(y_test):
            print("[WARN] review_meta_internal_test length mismatch. Rebuilding simple meta.")
            raise ValueError("length mismatch")
        return meta
    except Exception as e:
        print(f"[WARN] Cannot load review meta: {e}")
        return pd.DataFrame({
            "row_id": np.arange(len(y_test)),
            TARGET_COL: y_test,
            "TransactionAmt_raw": np.ones(len(y_test), dtype=np.float32),
            "fraud_value_raw": y_test.astype(np.float32),
            "review_cost_unit": np.ones(len(y_test), dtype=np.float32),
        })


def evaluate_review_budget(
    y_true: np.ndarray,
    prob: np.ndarray,
    meta: pd.DataFrame,
    model_name: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = meta.copy()
    df["y_true"] = y_true.astype(int)
    df["prob"] = prob.astype(float)

    if "TransactionAmt_raw" not in df.columns:
        df["TransactionAmt_raw"] = 1.0
    if "fraud_value_raw" not in df.columns:
        df["fraud_value_raw"] = df["TransactionAmt_raw"] * df["y_true"]
    if "review_cost_unit" not in df.columns:
        df["review_cost_unit"] = 1.0

    df = df.sort_values("prob", ascending=False).reset_index(drop=True)

    total_fraud = max(float(df["y_true"].sum()), 1.0)
    total_fraud_amount = max(float(df.loc[df["y_true"] == 1, "fraud_value_raw"].sum()), 1e-12)
    base_rate = total_fraud / len(df)

    detail_rows = []
    summary = {}

    for b in REVIEW_BUDGETS:
        k = max(1, int(math.ceil(len(df) * b)))
        top = df.iloc[:k]

        tp = float(top["y_true"].sum())
        precision_k = tp / k
        recall_k = tp / total_fraud
        f1_k = 2 * precision_k * recall_k / max(precision_k + recall_k, 1e-12)
        lift_k = precision_k / max(base_rate, 1e-12)

        captured_amount = float(top.loc[top["y_true"] == 1, "fraud_value_raw"].sum())
        captured_amount_rate = captured_amount / total_fraud_amount
        review_cost = float(top["review_cost_unit"].sum())
        expected_utility = captured_amount - review_cost

        pct = int(round(b * 100))
        summary[f"Precision@{pct}%"] = precision_k
        summary[f"Recall@{pct}%"] = recall_k
        summary[f"F1@{pct}%"] = f1_k
        summary[f"Lift@{pct}%"] = lift_k
        summary[f"CapturedAmountRate@{pct}%"] = captured_amount_rate
        summary[f"ExpectedUtility@{pct}%"] = expected_utility

        detail_rows.append({
            "Model": model_name,
            "Review_Budget": b,
            "K": k,
            "Precision@K": precision_k,
            "Recall@K": recall_k,
            "F1@K": f1_k,
            "Lift@K": lift_k,
            "Captured_Fraud_Amount": captured_amount,
            "CapturedAmountRate@K": captured_amount_rate,
            "Review_Cost": review_cost,
            "ExpectedUtility@K": expected_utility,
        })

    return pd.DataFrame(detail_rows), summary


def save_predictions(
    out_dir: Path,
    model_name: str,
    y_true: np.ndarray,
    prob: np.ndarray,
    meta: Optional[pd.DataFrame] = None,
) -> Path:
    pred_df = pd.DataFrame({
        "row_id": np.arange(len(y_true)),
        "isFraud": y_true.astype(int),
        "prob": prob.astype(float),
    })

    if meta is not None:
        for c in ["TransactionID", "TransactionDT_raw", "TransactionAmt_raw", "fraud_value_raw"]:
            if c in meta.columns:
                pred_df[c] = meta[c].values

    path = out_dir / f"pred_{safe_name(model_name)}.csv"
    save_csv(pred_df, path)
    return path


# ============================================================
# DATASET
# ============================================================

class NumpyBinaryDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.astype(np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    weighted_sampler: bool,
) -> DataLoader:
    ds = NumpyBinaryDataset(X, y)

    sampler = None
    if weighted_sampler:
        counts = np.bincount(y.astype(int), minlength=2)
        weights = np.zeros_like(y, dtype=np.float32)
        weights[y == 0] = 1.0 / max(counts[0], 1)
        weights[y == 1] = 1.0 / max(counts[1], 1)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# MODELS
# ============================================================

class TabMEnsembleMLP(nn.Module):
    """
    TabM-like baseline:
    shared trunk + K parallel heads; logits are averaged.
    """

    def __init__(self, n_features: int, hidden_dim: int = 512, depth: int = 3, dropout: float = 0.15, n_heads: int = 8):
        super().__init__()
        layers = []
        dim = n_features
        for _ in range(depth):
            layers.extend([
                nn.Linear(dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_heads)])

    def forward(self, x):
        h = self.trunk(x)
        logits = torch.stack([head(h).squeeze(-1) for head in self.heads], dim=1)
        return logits.mean(dim=1)


class FTTransformer(nn.Module):
    """
    Simple feature-token Transformer for numerical tabular features.
    """

    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.xavier_uniform_(self.weight)

        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        h = self.encoder(tokens)
        out = self.norm(h[:, 0])
        return self.head(out).squeeze(-1)


class SAINTLike(nn.Module):
    """
    SAINT-like baseline:
    transformer over feature tokens with mean pooling.
    This is a supervised SAINT-inspired baseline, not the full official SAINT pretraining pipeline.
    """

    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.feature_embed = nn.Parameter(torch.empty(n_features, d_token))
        self.value_proj = nn.Linear(1, d_token)
        nn.init.xavier_uniform_(self.feature_embed)

        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x):
        v = self.value_proj(x.unsqueeze(-1))
        tokens = v + self.feature_embed.unsqueeze(0)
        h = self.encoder(tokens)
        pooled = self.norm(h.mean(dim=1))
        return self.head(pooled).squeeze(-1)


class TabRLikeMLP(nn.Module):
    """
    TabR-like baseline:
    retrieval statistics are appended to feature vector before MLP.
    """

    def __init__(self, n_features_augmented: int, hidden_dim: int = 512, depth: int = 3, dropout: float = 0.15):
        super().__init__()
        layers = []
        dim = n_features_augmented
        for _ in range(depth):
            layers.extend([
                nn.Linear(dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def augment_with_retrieval(
    X_train_ref: np.ndarray,
    y_train_ref: np.ndarray,
    X: np.ndarray,
    k: int = 16,
    is_train: bool = False,
    chunk_size: int = 25000,
) -> np.ndarray:
    print(f"[TabR retrieval] Fitting NearestNeighbors k={k} ref={X_train_ref.shape} target={X.shape}")
    n_neighbors = k + 1 if is_train else k
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm="auto")
    nn_model.fit(X_train_ref)

    stat_list = []
    for start in range(0, len(X), chunk_size):
        end = min(start + chunk_size, len(X))
        dist, ind = nn_model.kneighbors(X[start:end], return_distance=True)

        if is_train:
            # Drop nearest self-like neighbor.
            dist = dist[:, 1:]
            ind = ind[:, 1:]

        neigh_y = y_train_ref[ind]
        fraud_rate = neigh_y.mean(axis=1)
        dist_mean = dist.mean(axis=1)
        dist_min = dist.min(axis=1)

        stats = np.stack([fraud_rate, dist_mean, dist_min], axis=1).astype(np.float32)
        stat_list.append(stats)

    stats_all = np.vstack(stat_list)
    return np.hstack([X, stats_all]).astype(np.float32)


# ============================================================
# TRAIN / PREDICT
# ============================================================

@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 1024
    epochs: int = 20
    patience: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 512
    dropout: float = 0.15
    d_token: int = 64
    n_layers: int = 3
    n_heads: int = 4
    weighted_sampler: bool = True
    min_precision_target: float = 0.80


def train_torch_model(
    model_name: str,
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    cfg: TrainConfig,
    out_dir: Path,
) -> Tuple[np.ndarray, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    pos_weight_value = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True, weighted_sampler=cfg.weighted_sampler)
    val_loader = make_loader(X_val, y_val, cfg.batch_size * 2, shuffle=False, weighted_sampler=False)
    test_loader = make_loader(X_test, np.zeros(len(X_test), dtype=np.int64), cfg.batch_size * 2, shuffle=False, weighted_sampler=False)

    best_pr_auc = -np.inf
    best_epoch = 0
    patience_counter = 0
    ckpt_path = out_dir / f"best_{safe_name(model_name)}.pt"
    history = []
    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {model_name}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        yv, pv = predict_torch(model, val_loader, device, y_true_available=True)
        thr = threshold_by_best_f1(yv, pv)
        val_metrics = evaluate_standard_metrics(yv, pv, thr["threshold"], cfg.min_precision_target)

        scheduler.step(val_metrics["PR_AUC"])

        row = {
            "Model": model_name,
            "Epoch": epoch,
            "Train_Loss": float(np.mean(losses)),
            "Val_PR_AUC": val_metrics["PR_AUC"],
            "Val_ROC_AUC": val_metrics["ROC_AUC"],
            "Val_Fraud_F1": val_metrics["Fraud_F1"],
            "Val_MCC": val_metrics["MCC"],
            "Val_Threshold": thr["threshold"],
        }
        history.append(row)

        print(
            f"[{model_name}] epoch={epoch:03d} "
            f"loss={row['Train_Loss']:.5f} "
            f"val_pr_auc={row['Val_PR_AUC']:.5f} "
            f"val_f1={row['Val_Fraud_F1']:.5f} "
            f"val_mcc={row['Val_MCC']:.5f}"
        )

        if val_metrics["PR_AUC"] > best_pr_auc:
            best_pr_auc = val_metrics["PR_AUC"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_name": model_name,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "best_val_metrics": val_metrics,
                "threshold_info": thr,
            }, ckpt_path)
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience:
            print(f"[{model_name}] Early stopping at epoch {epoch}")
            break

    hist_path = out_dir / f"history_{safe_name(model_name)}.csv"
    save_csv(pd.DataFrame(history), hist_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, p_test = predict_torch(model, test_loader, device, y_true_available=False)
    train_time = time.time() - start_time

    meta = {
        "Best_Epoch": int(best_epoch),
        "Best_Val_PR_AUC": float(best_pr_auc),
        "Selected_Threshold_from_Val": float(ckpt["threshold_info"]["threshold"]),
        "Val_Best_Fraud_Precision": float(ckpt["threshold_info"]["precision"]),
        "Val_Best_Fraud_Recall": float(ckpt["threshold_info"]["recall"]),
        "Val_Best_Fraud_F1": float(ckpt["threshold_info"]["f1"]),
        "Train_Time_Seconds": round(float(train_time), 3),
        "Checkpoint_File": str(ckpt_path),
        "History_File": str(hist_path),
    }
    return p_test, meta


@torch.no_grad()
def predict_torch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_true_available: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    ys = []

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(prob)
        if y_true_available:
            ys.append(yb.numpy())

    p = np.concatenate(probs)
    if y_true_available:
        y = np.concatenate(ys).astype(int)
    else:
        y = np.zeros(len(p), dtype=int)
    return y, p


# ============================================================
# EXTERNAL MODELS
# ============================================================

def train_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    cfg: TrainConfig,
    max_train_rows: int,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    try:
        from tabpfn import TabPFNClassifier
    except Exception as e:
        return None, {"Skip_Reason": f"tabpfn not installed or import failed: {e}"}

    start = time.time()

    X_fit = np.vstack([X_train, X_val]).astype(np.float32)
    y_fit = np.concatenate([y_train, y_val]).astype(int)

    X_fit, y_fit = sample_rows(X_fit, y_fit, max_train_rows, cfg.seed)

    print(f"[TabPFN] Fitting on {X_fit.shape}, predicting {X_test.shape}")
    try:
        clf = TabPFNClassifier()
        clf.fit(X_fit, y_fit)
        prob = clf.predict_proba(X_test)[:, 1]
    except Exception as e:
        return None, {"Skip_Reason": f"TabPFN failed: {e}"}

    return prob.astype(float), {
        "Best_Epoch": "",
        "Best_Val_PR_AUC": "",
        "Selected_Threshold_from_Val": "",
        "Train_Time_Seconds": round(float(time.time() - start), 3),
    }


def train_autogluon(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_val_df: pd.DataFrame,
    y_val: np.ndarray,
    X_test_df: pd.DataFrame,
    out_dir: Path,
    time_limit: int,
    presets: str,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    try:
        from autogluon.tabular import TabularPredictor
    except Exception as e:
        return None, {"Skip_Reason": f"autogluon.tabular not installed or import failed: {e}"}

    start = time.time()
    model_dir = out_dir / "autogluon_model"

    train_df = X_train_df.copy()
    train_df[TARGET_COL] = y_train.astype(int)
    val_df = X_val_df.copy()
    val_df[TARGET_COL] = y_val.astype(int)

    print(f"[AutoGluon] Fitting presets={presets}, time_limit={time_limit}")
    try:
        predictor = TabularPredictor(
            label=TARGET_COL,
            problem_type="binary",
            eval_metric="average_precision",
            path=str(model_dir),
        ).fit(
            train_data=train_df,
            tuning_data=val_df,
            presets=presets,
            time_limit=time_limit,
        )
        prob_df = predictor.predict_proba(X_test_df)
        # AutoGluon may return columns [0,1] or class labels.
        if 1 in prob_df.columns:
            prob = prob_df[1].values
        elif "1" in prob_df.columns:
            prob = prob_df["1"].values
        else:
            prob = prob_df.iloc[:, -1].values
    except Exception as e:
        return None, {"Skip_Reason": f"AutoGluon failed: {e}"}

    return prob.astype(float), {
        "Best_Epoch": "",
        "Best_Val_PR_AUC": "",
        "Selected_Threshold_from_Val": "",
        "Train_Time_Seconds": round(float(time.time() - start), 3),
        "Checkpoint_File": str(model_dir),
    }


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def evaluate_and_record(
    model_name: str,
    prob_test: np.ndarray,
    y_test: np.ndarray,
    meta_test: pd.DataFrame,
    threshold: Optional[float],
    base_meta: Dict[str, float],
    out_dir: Path,
    min_precision_target: float,
) -> Dict[str, float]:
    if threshold is None or threshold == "":
        thr = threshold_by_best_f1(y_test, prob_test)
        selected_threshold = thr["threshold"]
    else:
        selected_threshold = float(threshold)

    standard = evaluate_standard_metrics(
        y_test,
        prob_test,
        selected_threshold,
        min_precision=min_precision_target,
    )

    budget_df, budget_summary = evaluate_review_budget(y_test, prob_test, meta_test, model_name)
    budget_path = out_dir / f"review_budget_detail_{safe_name(model_name)}.csv"
    save_csv(budget_df, budget_path)

    pred_path = save_predictions(out_dir, model_name, y_test, prob_test, meta_test)

    result = {
        "Model": model_name,
        **base_meta,
        **standard,
        **budget_summary,
        "Prediction_File": str(pred_path),
        "Review_Budget_Detail_File": str(budget_path),
    }
    return result


def run_all(args):
    set_seed(args.seed)
    processed_dir = Path(args.processed_dir)
    out_dir = processed_dir / "sota_6_tabular_model_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        d_token=args.d_token,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        weighted_sampler=not args.no_weighted_sampler,
        min_precision_target=args.min_precision_target,
    )

    print("=" * 100)
    print("TRAIN 6 SOTA TABULAR BASELINES")
    print("=" * 100)
    print("Processed dir:", processed_dir)
    print("Output dir:", out_dir)
    print("Config:", asdict(cfg))

    X_train_df = read_table(processed_dir, "full_train")
    X_val_df = read_table(processed_dir, "full_val")
    X_test_df = read_table(processed_dir, "full_internal_test")
    y_train = ensure_y_1d(read_table(processed_dir, "y_train"))
    y_val = ensure_y_1d(read_table(processed_dir, "y_val"))
    y_test = ensure_y_1d(read_table(processed_dir, "y_internal_test"))
    meta_test = load_review_meta(processed_dir, y_test)

    X_train_df, X_val_df, X_test_df = align_feature_columns(X_train_df, X_val_df, X_test_df)

    # Keep a selected DataFrame for AutoGluon as well.
    X_train, X_val, X_test, selected_features, _, _ = prepare_numeric_arrays(
        X_train_df, X_val_df, X_test_df, y_train, max_features=args.max_features, seed=args.seed
    )
    X_train_sel_df = pd.DataFrame(X_train, columns=selected_features)
    X_val_sel_df = pd.DataFrame(X_val, columns=selected_features)
    X_test_sel_df = pd.DataFrame(X_test, columns=selected_features)

    # Sample train for heavy DL/SOTA models.
    X_train_fit, y_train_fit = sample_rows(X_train, y_train, args.max_train_rows, args.seed)

    print("[DATA] X_train_fit:", X_train_fit.shape)
    print("[DATA] X_val:", X_val.shape)
    print("[DATA] X_test:", X_test.shape)
    print("[DATA] fraud rate train:", float(np.mean(y_train_fit)))
    print("[DATA] fraud rate test:", float(np.mean(y_test)))

    models_to_run = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    results = []

    def add_skip(model_name: str, reason: str):
        print(f"[SKIP] {model_name}: {reason}")
        results.append({"Model": model_name, "Skip_Reason": reason})

    # 1. TabM-like
    if "tabm" in models_to_run:
        name = "TabM_like"
        try:
            model = TabMEnsembleMLP(
                n_features=X_train_fit.shape[1],
                hidden_dim=cfg.hidden_dim,
                depth=3,
                dropout=cfg.dropout,
                n_heads=8,
            )
            prob, meta = train_torch_model(name, model, X_train_fit, y_train_fit, X_val, y_val, X_test, cfg, out_dir)
            result = evaluate_and_record(name, prob, y_test, meta_test, meta.get("Selected_Threshold_from_Val", None), meta, out_dir, cfg.min_precision_target)
            results.append(result)
        except Exception as e:
            add_skip(name, str(e))
        gc.collect()

    # 2. TabR-like
    if "tabr" in models_to_run:
        name = "TabR_like"
        try:
            Xtr_aug = augment_with_retrieval(X_train_fit, y_train_fit, X_train_fit, k=args.tabr_k, is_train=True)
            Xv_aug = augment_with_retrieval(X_train_fit, y_train_fit, X_val, k=args.tabr_k, is_train=False)
            Xt_aug = augment_with_retrieval(X_train_fit, y_train_fit, X_test, k=args.tabr_k, is_train=False)

            model = TabRLikeMLP(
                n_features_augmented=Xtr_aug.shape[1],
                hidden_dim=cfg.hidden_dim,
                depth=3,
                dropout=cfg.dropout,
            )
            prob, meta = train_torch_model(name, model, Xtr_aug, y_train_fit, Xv_aug, y_val, Xt_aug, cfg, out_dir)
            result = evaluate_and_record(name, prob, y_test, meta_test, meta.get("Selected_Threshold_from_Val", None), meta, out_dir, cfg.min_precision_target)
            results.append(result)

            del Xtr_aug, Xv_aug, Xt_aug
        except Exception as e:
            add_skip(name, str(e))
        gc.collect()

    # 3. FT-Transformer
    if "ft_transformer" in models_to_run or "ft" in models_to_run:
        name = "FTTransformer"
        try:
            model = FTTransformer(
                n_features=X_train_fit.shape[1],
                d_token=cfg.d_token,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                dropout=cfg.dropout,
            )
            prob, meta = train_torch_model(name, model, X_train_fit, y_train_fit, X_val, y_val, X_test, cfg, out_dir)
            result = evaluate_and_record(name, prob, y_test, meta_test, meta.get("Selected_Threshold_from_Val", None), meta, out_dir, cfg.min_precision_target)
            results.append(result)
        except Exception as e:
            add_skip(name, str(e))
        gc.collect()

    # 4. SAINT-like
    if "saint" in models_to_run:
        name = "SAINT_like"
        try:
            model = SAINTLike(
                n_features=X_train_fit.shape[1],
                d_token=cfg.d_token,
                n_layers=max(cfg.n_layers, 3),
                n_heads=cfg.n_heads,
                dropout=max(cfg.dropout, 0.20),
            )
            prob, meta = train_torch_model(name, model, X_train_fit, y_train_fit, X_val, y_val, X_test, cfg, out_dir)
            result = evaluate_and_record(name, prob, y_test, meta_test, meta.get("Selected_Threshold_from_Val", None), meta, out_dir, cfg.min_precision_target)
            results.append(result)
        except Exception as e:
            add_skip(name, str(e))
        gc.collect()

    # 5. TabPFN
    if "tabpfn" in models_to_run:
        name = "TabPFN"
        prob, meta = train_tabpfn(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            cfg,
            max_train_rows=args.tabpfn_max_train_rows,
        )
        if prob is None:
            add_skip(name, meta.get("Skip_Reason", "TabPFN skipped"))
        else:
            result = evaluate_and_record(name, prob, y_test, meta_test, None, meta, out_dir, cfg.min_precision_target)
            results.append(result)
        gc.collect()

    # 6. AutoGluon
    if "autogluon" in models_to_run:
        name = "AutoGluon_Tabular"
        # For AutoGluon, optionally sample because full data can be heavy.
        if args.autogluon_max_train_rows > 0 and len(y_train) > args.autogluon_max_train_rows:
            Xag_np, yag = sample_rows(X_train, y_train, args.autogluon_max_train_rows, args.seed)
            Xag_df = pd.DataFrame(Xag_np, columns=selected_features)
        else:
            Xag_df = X_train_sel_df
            yag = y_train

        prob, meta = train_autogluon(
            Xag_df,
            yag,
            X_val_sel_df,
            y_val,
            X_test_sel_df,
            out_dir,
            time_limit=args.autogluon_time_limit,
            presets=args.autogluon_presets,
        )
        if prob is None:
            add_skip(name, meta.get("Skip_Reason", "AutoGluon skipped"))
        else:
            result = evaluate_and_record(name, prob, y_test, meta_test, None, meta, out_dir, cfg.min_precision_target)
            results.append(result)
        gc.collect()

    results_df = pd.DataFrame(results)
    final_path = out_dir / "sota_6_model_results_final.csv"
    save_csv(results_df, final_path)

    paper_cols = [
        "Model",
        "PR_AUC",
        "ROC_AUC",
        "Fraud_Precision",
        "Fraud_Recall",
        "Fraud_F1",
        "MCC",
        "Recall@Precision>=0.80",
        "Precision@3%",
        "Recall@3%",
        "CapturedAmountRate@3%",
        "ExpectedUtility@3%",
        "Precision@5%",
        "Recall@5%",
        "CapturedAmountRate@5%",
        "ExpectedUtility@5%",
        "Best_Epoch",
        "Train_Time_Seconds",
        "Skip_Reason",
    ]
    paper_cols = [c for c in paper_cols if c in results_df.columns]
    paper_df = results_df[paper_cols].copy()
    paper_path = out_dir / "paper_ready_sota_6_model_table.csv"
    save_csv(paper_df, paper_path)

    config_path = out_dir / "sota_6_model_config.json"
    config_path.write_text(json.dumps({
        "args": vars(args),
        "train_config": asdict(cfg),
        "selected_features": selected_features,
        "outputs": {
            "final": str(final_path),
            "paper_ready": str(paper_path),
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] {config_path}")

    print("\nDONE.")
    print("Final:", final_path)
    print("Paper-ready:", paper_path)


def build_arg_parser():
    p = argparse.ArgumentParser(description="Train 6 SOTA tabular baselines for fraud detection.")
    p.add_argument("--processed-dir", type=str, default=str(DEFAULT_PROCESSED_DIR))
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                   help="Comma list: tabm,tabr,ft_transformer,saint,tabpfn,autogluon")
    p.add_argument("--seed", type=int, default=42)

    # Data / feature limits
    p.add_argument("--max-features", type=int, default=256,
                   help="Top-K features for neural/SOTA baselines. Use 0 for all features.")
    p.add_argument("--max-train-rows", type=int, default=250000,
                   help="Max training rows for heavy PyTorch baselines. Use 0 for all rows.")
    p.add_argument("--tabpfn-max-train-rows", type=int, default=50000)
    p.add_argument("--autogluon-max-train-rows", type=int, default=300000)

    # PyTorch training
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--d-token", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--no-weighted-sampler", action="store_true")

    # TabR
    p.add_argument("--tabr-k", type=int, default=16)

    # AutoGluon
    p.add_argument("--autogluon-time-limit", type=int, default=3600)
    p.add_argument("--autogluon-presets", type=str, default="medium_quality",
                   help="medium_quality, good_quality, high_quality, best_quality")

    # Metrics
    p.add_argument("--min-precision-target", type=float, default=0.80)

    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_all(args)
