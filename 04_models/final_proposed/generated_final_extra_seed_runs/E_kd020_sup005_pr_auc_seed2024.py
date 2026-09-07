from __future__ import annotations

import sys
from pathlib import Path as _PathForImport
_THIS_DIR = _PathForImport(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
for _p in [str(_THIS_DIR), str(_PARENT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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

from cnn_branch_updated import TabularCNNBranch
from deepfm_branch_updated import DeepFMBranch
from fraud_losses import BinaryFocalLoss, CombinedFocalSupConLoss


# ============================================================
# CONFIG – SINGLE PROPOSED MODEL ONLY
# ============================================================

# Folder output của:
# data_cleaning_ieee_cis_tree_guided_cost_ready_memory_fixed_v2.py
PROCESSED_DIR = Path(r"D:\project\data\merge_paper_ready_tree_cost")

OUTPUT_DIR = PROCESSED_DIR / "proposed_tree_guided_moe_final_3seeds" / "E_kd020_sup005_pr_auc_seed2024"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEACHER_DIR = PROCESSED_DIR / "teacher_signals"

TARGET_COL = "isFraud"
RANDOM_STATE = 2024

# DL experiments.
# Có thể chạy ít trước bằng cách comment bớt.
RUN_EXPERIMENTS = [
    "tree_guided_moe_kd_focal_supcon",
]

# Training
BATCH_SIZE = 1024
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0

# Nếu máy yếu hoặc hay NaN, giữ USE_AMP=False.
USE_AMP = False
USE_WEIGHTED_SAMPLER = True
NUM_WORKERS = 0

# Nếu muốn chạy thử nhanh:
SUBSET_TRAIN_N = None
SUBSET_VAL_N = None
SUBSET_TEST_N = None

# Threshold and metrics
THRESHOLD_OBJECTIVE = "f1"
MIN_PRECISION_TARGET = 0.80
THRESHOLD_MIN_PRECISION = None

# Review-budget metrics
REVIEW_BUDGETS = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
REVIEW_COST_PER_CASE = 1.0
FALSE_POSITIVE_COST = 0.0

# Model sizes
CNN_EMBED_DIM = 128
CNN_CONV_CHANNELS = 128
CNN_KERNEL_SIZE = 3
CNN_BILINEAR_RANK = 32
CNN_OUT_DIM = 128
CNN_SEQ_LENGTH = 10

DEEPFM_EMBED_DIM = 16
DEEPFM_DENSE_NUM_FIELDS = 8
DEEPFM_BRANCH_OUT_DIM = 128
DEEPFM_HIDDEN = [256, 128]

FUSION_DIM = 128
GATE_HIDDEN_DIM = 64
DROPOUT = 0.30

# Loss weights
FOCAL_ALPHA = 0.85
FOCAL_GAMMA = 2.0
LAMBDA_SUPCON = 0.005
SUPCON_TEMPERATURE = 0.10

# Distillation
KD_WEIGHT = 0.2
KD_TEMPERATURE = 2.0


# ============================================================
# REPRODUCIBILITY AND IO
# ============================================================

def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_file(base_name: str, folder: Path = PROCESSED_DIR) -> Path:
    for ext in [".parquet", ".csv.gz", ".csv"]:
        p = folder / f"{base_name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {base_name} in {folder}")


def read_table(base_name: str, folder: Path = PROCESSED_DIR) -> pd.DataFrame:
    path = resolve_file(base_name, folder)
    print(f"[LOAD] {base_name}: {path}")
    if path.name.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col == TARGET_COL:
            df[col] = df[col].astype("int8")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def maybe_subset_by_y(df: pd.DataFrame, n: Optional[int], seed: int = RANDOM_STATE) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df
    if TARGET_COL not in df.columns:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    fraud = df[df[TARGET_COL] == 1]
    normal = df[df[TARGET_COL] == 0]
    n_normal = max(n - len(fraud), 0)
    normal_sample = normal.sample(n=min(n_normal, len(normal)), random_state=seed)
    out = pd.concat([fraud, normal_sample], axis=0).sample(frac=1.0, random_state=seed)
    return out.reset_index(drop=True)


def safe_prob(y_prob):
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(y_prob, 0.0, 1.0)


# ============================================================
# METRICS
# ============================================================

def threshold_by_best_f1(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = safe_prob(y_prob)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    p = precision[:-1]
    r = recall[:-1]
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)

    best_idx = int(np.nanargmax(f1))
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(p[best_idx]),
        "recall": float(r[best_idx]),
        "f1": float(f1[best_idx]),
    }


def recall_at_precision(y_true, y_prob, min_precision=0.80):
    y_true = np.asarray(y_true).astype(int)
    y_prob = safe_prob(y_prob)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    valid = precision >= min_precision

    if not np.any(valid):
        return {
            "recall": 0.0,
            "precision": np.nan,
            "threshold": np.nan,
        }

    valid_idx = np.where(valid)[0]
    best_i = valid_idx[np.argmax(recall[valid])]
    threshold = 1.0 if best_i >= len(thresholds) else thresholds[best_i]

    return {
        "recall": float(recall[best_i]),
        "precision": float(precision[best_i]),
        "threshold": float(threshold),
    }


def evaluate_fraud_metrics(y_true, y_prob, threshold, min_precision=0.80):
    y_true = np.asarray(y_true).astype(int)
    y_prob = safe_prob(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = np.nan

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = np.nan

    rap = recall_at_precision(y_true, y_prob, min_precision=min_precision)

    return {
        "PR_AUC": float(pr_auc),
        "ROC_AUC": float(roc_auc),
        "Fraud_Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Fraud_Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Fraud_F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        f"Recall@Precision>={min_precision:.2f}": float(rap["recall"]),
        f"Threshold@Precision>={min_precision:.2f}": float(rap["threshold"]) if np.isfinite(rap["threshold"]) else np.nan,
        "Selected_Threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def review_budget_metrics(
    y_true,
    y_prob,
    amount,
    budgets=REVIEW_BUDGETS,
    review_cost_per_case=REVIEW_COST_PER_CASE,
    false_positive_cost=FALSE_POSITIVE_COST,
):
    y_true = np.asarray(y_true).astype(int)
    y_prob = safe_prob(y_prob)

    amount = np.asarray(amount, dtype=float)
    amount = np.nan_to_num(amount, nan=0.0, posinf=0.0, neginf=0.0)
    amount = np.clip(amount, 0.0, None)

    n = len(y_true)
    total_fraud = int(y_true.sum())
    total_fraud_amount = float((amount * y_true).sum())
    base_fraud_rate = total_fraud / max(n, 1)
    order = np.argsort(-y_prob)

    rows = []

    for budget in budgets:
        k = max(1, int(np.ceil(n * budget)))
        selected = order[:k]

        selected_flag = np.zeros(n, dtype=np.int8)
        selected_flag[selected] = 1

        tp = int(((selected_flag == 1) & (y_true == 1)).sum())
        fp = int(((selected_flag == 1) & (y_true == 0)).sum())
        fn = int(((selected_flag == 0) & (y_true == 1)).sum())
        tn = int(((selected_flag == 0) & (y_true == 0)).sum())

        precision_k = tp / max(k, 1)
        recall_k = tp / max(total_fraud, 1)
        f1_k = 2 * precision_k * recall_k / max(precision_k + recall_k, 1e-12)
        lift_k = precision_k / max(base_fraud_rate, 1e-12)

        captured_amount = float((amount[selected] * y_true[selected]).sum())
        captured_amount_rate = captured_amount / max(total_fraud_amount, 1e-12)

        review_cost = review_cost_per_case * k
        fp_cost = false_positive_cost * fp
        expected_utility = captured_amount - review_cost - fp_cost

        rows.append({
            "Review_Budget": float(budget),
            "Review_Budget_Percent": float(budget * 100),
            "Review_Count": int(k),
            "TP_at_K": tp,
            "FP_at_K": fp,
            "FN_at_K": fn,
            "TN_at_K": tn,
            "Precision@K": float(precision_k),
            "Recall@K": float(recall_k),
            "F1@K": float(f1_k),
            "Lift@K": float(lift_k),
            "Captured_Fraud_Amount@K": float(captured_amount),
            "Captured_Fraud_Amount_Rate@K": float(captured_amount_rate),
            "Review_Cost@K": float(review_cost),
            "False_Positive_Cost@K": float(fp_cost),
            "Expected_Utility@K": float(expected_utility),
        })

    return pd.DataFrame(rows)


def flatten_budget_metrics(budget_df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for _, row in budget_df.iterrows():
        pct = int(round(row["Review_Budget_Percent"]))
        out[f"Precision@{pct}%"] = float(row["Precision@K"])
        out[f"Recall@{pct}%"] = float(row["Recall@K"])
        out[f"F1@{pct}%"] = float(row["F1@K"])
        out[f"Lift@{pct}%"] = float(row["Lift@K"])
        out[f"CapturedAmountRate@{pct}%"] = float(row["Captured_Fraud_Amount_Rate@K"])
        out[f"ExpectedUtility@{pct}%"] = float(row["Expected_Utility@K"])
    return out


# ============================================================
# DATASET
# ============================================================

class FraudMultiInputDataset(Dataset):
    def __init__(
        self,
        cnn_df: pd.DataFrame,
        cat_df: pd.DataFrame,
        num_df: pd.DataFrame,
        teacher_df: Optional[pd.DataFrame] = None,
    ):
        for name, df in [("cnn_df", cnn_df), ("cat_df", cat_df), ("num_df", num_df)]:
            if TARGET_COL not in df.columns:
                raise ValueError(f"{TARGET_COL} not found in {name}")

        y_cnn = cnn_df[TARGET_COL].astype(int).to_numpy()
        y_cat = cat_df[TARGET_COL].astype(int).to_numpy()
        y_num = num_df[TARGET_COL].astype(int).to_numpy()

        if not (np.array_equal(y_cnn, y_cat) and np.array_equal(y_cnn, y_num)):
            raise ValueError("Labels in cnn/cat/num files are not aligned.")

        self.y = y_cnn.astype("float32")

        self.x_cnn = cnn_df.drop(columns=[TARGET_COL]).to_numpy(dtype="float32", copy=True)
        self.x_num = num_df.drop(columns=[TARGET_COL]).to_numpy(dtype="float32", copy=True)

        x_cat = cat_df.drop(columns=[TARGET_COL]).to_numpy(dtype="int64", copy=True)
        x_cat = x_cat + 1
        x_cat[x_cat < 0] = 0
        self.x_cat = x_cat.astype("int64")

        if teacher_df is not None and "teacher_prob" in teacher_df.columns:
            teacher_prob = pd.to_numeric(teacher_df["teacher_prob"], errors="coerce").fillna(0.0).to_numpy(dtype="float32")
            teacher_logit = pd.to_numeric(teacher_df.get("teacher_logit", pd.Series(np.zeros(len(teacher_df)))), errors="coerce").fillna(0.0).to_numpy(dtype="float32")
            if len(teacher_prob) != len(self.y):
                raise ValueError("teacher_signal length does not match dataset length.")
            self.teacher_prob = np.clip(teacher_prob, 0.0, 1.0).astype("float32")
            self.teacher_logit = teacher_logit.astype("float32")
        else:
            self.teacher_prob = np.zeros_like(self.y, dtype="float32")
            self.teacher_logit = np.zeros_like(self.y, dtype="float32")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x_cnn": torch.from_numpy(self.x_cnn[idx]),
            "x_cat": torch.from_numpy(self.x_cat[idx]),
            "x_num": torch.from_numpy(self.x_num[idx]),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "teacher_prob": torch.tensor(self.teacher_prob[idx], dtype=torch.float32),
            "teacher_logit": torch.tensor(self.teacher_logit[idx], dtype=torch.float32),
        }


def load_split(split: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cnn = downcast_df(read_table(f"cnn_{split}"))
    cat = downcast_df(read_table(f"deepfm_cat_{split}"))
    num = downcast_df(read_table(f"deepfm_num_{split}"))
    return cnn, cat, num


def load_teacher_split(split: str) -> Optional[pd.DataFrame]:
    # split: train, val, internal_test
    base = f"teacher_signal_{split}"
    try:
        return downcast_df(read_table(base, TEACHER_DIR))
    except Exception as e:
        print(f"[WARN] Teacher signal not found for {split}: {e}")
        return None


def build_loaders():
    train_cnn, train_cat, train_num = load_split("train")
    val_cnn, val_cat, val_num = load_split("val")
    test_cnn, test_cat, test_num = load_split("internal_test")

    train_teacher = load_teacher_split("train")
    val_teacher = load_teacher_split("val")
    test_teacher = load_teacher_split("internal_test")

    if SUBSET_TRAIN_N is not None:
        idx = maybe_subset_by_y(train_cnn[[TARGET_COL]].reset_index(), SUBSET_TRAIN_N)["index"].to_numpy()
        train_cnn = train_cnn.iloc[idx].reset_index(drop=True)
        train_cat = train_cat.iloc[idx].reset_index(drop=True)
        train_num = train_num.iloc[idx].reset_index(drop=True)
        if train_teacher is not None:
            train_teacher = train_teacher.iloc[idx].reset_index(drop=True)

    if SUBSET_VAL_N is not None:
        idx = maybe_subset_by_y(val_cnn[[TARGET_COL]].reset_index(), SUBSET_VAL_N)["index"].to_numpy()
        val_cnn = val_cnn.iloc[idx].reset_index(drop=True)
        val_cat = val_cat.iloc[idx].reset_index(drop=True)
        val_num = val_num.iloc[idx].reset_index(drop=True)
        if val_teacher is not None:
            val_teacher = val_teacher.iloc[idx].reset_index(drop=True)

    if SUBSET_TEST_N is not None:
        idx = maybe_subset_by_y(test_cnn[[TARGET_COL]].reset_index(), SUBSET_TEST_N)["index"].to_numpy()
        test_cnn = test_cnn.iloc[idx].reset_index(drop=True)
        test_cat = test_cat.iloc[idx].reset_index(drop=True)
        test_num = test_num.iloc[idx].reset_index(drop=True)
        if test_teacher is not None:
            test_teacher = test_teacher.iloc[idx].reset_index(drop=True)

    cat_feature_cols = [c for c in train_cat.columns if c != TARGET_COL]
    categorical_cardinalities = []
    for col in cat_feature_cols:
        max_code = int(train_cat[col].max())
        categorical_cardinalities.append(max_code + 2)

    dims = {
        "cnn_dim": train_cnn.shape[1] - 1,
        "deepfm_num_dim": train_num.shape[1] - 1,
        "num_cat": len(cat_feature_cols),
        "train_size": len(train_cnn),
        "val_size": len(val_cnn),
        "test_size": len(test_cnn),
    }

    train_ds = FraudMultiInputDataset(train_cnn, train_cat, train_num, train_teacher)
    val_ds = FraudMultiInputDataset(val_cnn, val_cat, val_num, val_teacher)
    test_ds = FraudMultiInputDataset(test_cnn, test_cat, test_num, test_teacher)

    if USE_WEIGHTED_SAMPLER:
        y = train_ds.y.astype(int)
        class_count = np.bincount(y, minlength=2)
        class_weight = 1.0 / np.maximum(class_count, 1)
        sample_weight = class_weight[y]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weight, dtype=torch.double),
            num_samples=len(sample_weight),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print("DIMS:", dims)
    print("Train fraud ratio:", float(train_ds.y.mean()))
    print("Val fraud ratio:", float(val_ds.y.mean()))
    print("Test fraud ratio:", float(test_ds.y.mean()))
    print("Teacher signal available train:", bool(np.any(train_ds.teacher_prob > 0)))

    return train_loader, val_loader, test_loader, dims, categorical_cardinalities, test_ds


# ============================================================
# MODEL WRAPPERS
# ============================================================

class DeepFMCompatWrapper(nn.Module):
    def __init__(self, deepfm: nn.Module):
        super().__init__()
        self.deepfm = deepfm
        self.output_dim = getattr(deepfm, "output_dim", None)

    def extract_embedding(
        self,
        x_cat: Optional[torch.Tensor] = None,
        x_num: Optional[torch.Tensor] = None,
        x_dense: Optional[torch.Tensor] = None,
        cat_x: Optional[torch.Tensor] = None,
        num_x: Optional[torch.Tensor] = None,
        dense_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cat_x is None:
            cat_x = x_cat
        if num_x is None:
            num_x = x_num
        if dense_x is None:
            dense_x = x_dense
        return self.deepfm.extract_embedding(cat_x=cat_x, num_x=num_x, dense_x=dense_x)

    def forward(self, *args, **kwargs):
        return self.extract_embedding(*args, **kwargs)


class CNNOnlyClassifier(nn.Module):
    def __init__(self, cnn_branch: nn.Module):
        super().__init__()
        self.cnn_branch = cnn_branch

    def forward(self, x_cnn, x_cat=None, x_num=None, x_dense=None, teacher_logit=None, return_dict=True):
        logits, z_cnn, _ = self.cnn_branch(x_cnn, return_embedding=True)
        logits = logits.view(-1)
        prob = torch.sigmoid(logits)
        if not return_dict:
            return logits
        return {"logits": logits, "prob": prob, "embedding": z_cnn}


class DeepFMOnlyClassifier(nn.Module):
    def __init__(self, deepfm: nn.Module):
        super().__init__()
        self.deepfm = deepfm

    def forward(self, x_cnn=None, x_cat=None, x_num=None, x_dense=None, teacher_logit=None, return_dict=True):
        logits, z_fm = self.deepfm(cat_x=x_cat, num_x=None, dense_x=x_dense, return_embedding=True)
        logits = logits.view(-1)
        prob = torch.sigmoid(logits)
        if not return_dict:
            return logits
        return {"logits": logits, "prob": prob, "embedding": z_fm}


class ConcatCNNDeepFM(nn.Module):
    def __init__(self, cnn_branch, deepfm_branch, cnn_dim, deepfm_dim, fusion_dim=128, dropout=0.30):
        super().__init__()
        self.cnn_branch = cnn_branch
        self.deepfm_branch = deepfm_branch
        self.fusion = nn.Sequential(
            nn.Linear(cnn_dim + deepfm_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, x_cnn, x_cat, x_num=None, x_dense=None, teacher_logit=None, return_dict=True):
        _, z_cnn, _ = self.cnn_branch(x_cnn, return_embedding=True)
        z_fm = self.deepfm_branch.extract_embedding(x_cat=x_cat, x_dense=x_dense)
        z = torch.cat([z_cnn, z_fm], dim=1)
        logits = self.fusion(z).view(-1)
        prob = torch.sigmoid(logits)
        if not return_dict:
            return logits
        return {"logits": logits, "prob": prob, "embedding": z}


class MoEGatedCNNDeepFM(nn.Module):
    def __init__(
        self,
        cnn_branch,
        deepfm_branch,
        cnn_dim=128,
        deepfm_dim=128,
        fusion_dim=128,
        gate_hidden_dim=64,
        dropout=0.30,
        tree_guided=False,
    ):
        super().__init__()
        self.cnn_branch = cnn_branch
        self.deepfm_branch = deepfm_branch
        self.tree_guided = tree_guided

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fm_proj = nn.Sequential(
            nn.Linear(deepfm_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        gate_in = fusion_dim * 2 + (1 if tree_guided else 0)
        self.gate = nn.Sequential(
            nn.Linear(gate_in, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 2),
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, x_cnn, x_cat, x_num=None, x_dense=None, teacher_logit=None, return_dict=True):
        _, z_cnn_raw, _ = self.cnn_branch(x_cnn, return_embedding=True)
        z_fm_raw = self.deepfm_branch.extract_embedding(x_cat=x_cat, x_dense=x_dense)

        z_cnn = self.cnn_proj(z_cnn_raw)
        z_fm = self.fm_proj(z_fm_raw)

        if self.tree_guided:
            if teacher_logit is None:
                teacher_logit = torch.zeros(z_cnn.size(0), device=z_cnn.device)
            guide = teacher_logit.view(-1, 1)
            gate_input = torch.cat([z_cnn, z_fm, guide], dim=1)
        else:
            gate_input = torch.cat([z_cnn, z_fm], dim=1)

        gate_weight = F.softmax(self.gate(gate_input), dim=1)
        z = gate_weight[:, 0:1] * z_cnn + gate_weight[:, 1:2] * z_fm

        logits = self.classifier(z).view(-1)
        prob = torch.sigmoid(logits)

        if not return_dict:
            return logits

        return {
            "logits": logits,
            "prob": prob,
            "embedding": z,
            "gate_weight": gate_weight,
        }


def make_cnn_branch(cnn_dim: int) -> nn.Module:
    return TabularCNNBranch(
        tabular_dim=cnn_dim,
        embed_dim=CNN_EMBED_DIM,
        conv_channels=CNN_CONV_CHANNELS,
        kernel_size=CNN_KERNEL_SIZE,
        bilinear_rank=CNN_BILINEAR_RANK,
        bilinear_out_dim=CNN_OUT_DIM,
        num_classes=1,
        seq_length=CNN_SEQ_LENGTH,
        dropout=DROPOUT,
    )


def make_deepfm_branch(cat_cardinalities: List[int], deepfm_num_dim: int) -> nn.Module:
    return DeepFMBranch(
        num_classes=2,
        categorical_cardinalities=cat_cardinalities,
        num_numerical=0,
        embed_dim=DEEPFM_EMBED_DIM,
        deep_hidden=DEEPFM_HIDDEN,
        dropout=DROPOUT,
        dense_in_dim=deepfm_num_dim,
        dense_num_fields=DEEPFM_DENSE_NUM_FIELDS,
        branch_out_dim=DEEPFM_BRANCH_OUT_DIM,
    )


def build_model(experiment: str, dims: Dict[str, int], cat_cardinalities: List[int]) -> nn.Module:
    cnn = make_cnn_branch(dims["cnn_dim"])
    deepfm = make_deepfm_branch(cat_cardinalities, dims["deepfm_num_dim"])
    deepfm_compat = DeepFMCompatWrapper(deepfm)

    if experiment.startswith("cnn_only"):
        return CNNOnlyClassifier(cnn)

    if experiment.startswith("deepfm_only"):
        return DeepFMOnlyClassifier(deepfm)

    if experiment.startswith("concat"):
        return ConcatCNNDeepFM(
            cnn_branch=cnn,
            deepfm_branch=deepfm_compat,
            cnn_dim=CNN_OUT_DIM,
            deepfm_dim=DEEPFM_BRANCH_OUT_DIM,
            fusion_dim=FUSION_DIM,
            dropout=DROPOUT,
        )

    if experiment.startswith("tree_guided_moe"):
        return MoEGatedCNNDeepFM(
            cnn_branch=cnn,
            deepfm_branch=deepfm_compat,
            cnn_dim=CNN_OUT_DIM,
            deepfm_dim=DEEPFM_BRANCH_OUT_DIM,
            fusion_dim=FUSION_DIM,
            gate_hidden_dim=GATE_HIDDEN_DIM,
            dropout=DROPOUT,
            tree_guided=True,
        )

    if experiment.startswith("moe"):
        return MoEGatedCNNDeepFM(
            cnn_branch=cnn,
            deepfm_branch=deepfm_compat,
            cnn_dim=CNN_OUT_DIM,
            deepfm_dim=DEEPFM_BRANCH_OUT_DIM,
            fusion_dim=FUSION_DIM,
            gate_hidden_dim=GATE_HIDDEN_DIM,
            dropout=DROPOUT,
            tree_guided=False,
        )

    raise ValueError(f"Unknown experiment: {experiment}")


# ============================================================
# LOSSES
# ============================================================

def kd_loss_with_logits(student_logits, teacher_prob, temperature=2.0):
    teacher_prob = torch.clamp(teacher_prob.float(), 1e-6, 1.0 - 1e-6)
    teacher_logit = torch.log(teacher_prob / (1.0 - teacher_prob))
    s = student_logits.float() / temperature
    t = torch.sigmoid(teacher_logit / temperature)
    return F.binary_cross_entropy_with_logits(s, t) * (temperature ** 2)


def compute_loss(experiment: str, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = out["logits"].view(-1)
    y = batch["y"].float()
    teacher_prob = batch["teacher_prob"].float()

    if "focal" in experiment:
        cls_loss = BinaryFocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)(logits, y)
    else:
        cls_loss = F.binary_cross_entropy_with_logits(logits, y)

    kd = torch.zeros((), device=logits.device)
    if "kd" in experiment:
        kd = kd_loss_with_logits(logits, teacher_prob, temperature=KD_TEMPERATURE)

    supcon = torch.zeros((), device=logits.device)
    if "supcon" in experiment:
        # Reuse existing supervised contrastive implementation through combined class.
        # Classification part from this combined loss is ignored here to avoid double counting.
        comb = CombinedFocalSupConLoss(
            focal_alpha=FOCAL_ALPHA,
            focal_gamma=FOCAL_GAMMA,
            lambda_supcon=1.0,
            temperature=SUPCON_TEMPERATURE,
            fraud_anchor_weight=2.0,
        ).to(logits.device)
        loss_dict = comb(logits, y, features=out.get("embedding"))
        supcon = loss_dict["supcon_loss"]

    total = cls_loss + KD_WEIGHT * kd + LAMBDA_SUPCON * supcon

    if not torch.isfinite(total):
        raise FloatingPointError("NaN/Inf loss detected.")

    return total, {
        "classification_loss": float(cls_loss.detach().cpu()),
        "kd_loss": float(kd.detach().cpu()),
        "supcon_loss": float(supcon.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }


# ============================================================
# TRAINING
# ============================================================

def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def forward_model(model, batch):
    return model(
        x_cnn=batch["x_cnn"],
        x_cat=batch["x_cat"].long(),
        x_dense=batch["x_num"],
        teacher_logit=batch["teacher_logit"],
        return_dict=True,
    )


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total = {"loss": 0.0, "classification_loss": 0.0, "kd_loss": 0.0, "supcon_loss": 0.0}
    n = 0

    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda", enabled=True):
                out = forward_model(model, batch)
                loss, loss_log = compute_loss(model.experiment_name, out, batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = forward_model(model, batch)
            loss, loss_log = compute_loss(model.experiment_name, out, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        bs = int(batch["y"].size(0))
        total["loss"] += loss_log["total_loss"] * bs
        total["classification_loss"] += loss_log["classification_loss"] * bs
        total["kd_loss"] += loss_log["kd_loss"] * bs
        total["supcon_loss"] += loss_log["supcon_loss"] * bs
        n += bs

    return {k: v / max(n, 1) for k, v in total.items()}


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    ys, probs = [], []
    gate_weights = []

    for batch in loader:
        batch = move_batch(batch, device)
        out = forward_model(model, batch)
        ys.append(batch["y"].detach().cpu().numpy())
        probs.append(out["prob"].view(-1).detach().cpu().numpy())
        if "gate_weight" in out:
            gate_weights.append(out["gate_weight"].detach().cpu().numpy())

    y = np.concatenate(ys)
    p = np.concatenate(probs)
    gw = np.concatenate(gate_weights) if gate_weights else None
    return y, p, gw


def attach_meta(pred_df: pd.DataFrame, meta_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if meta_df is None:
        return pred_df

    out = pred_df.copy()
    for col in ["row_id", "TransactionID", "TransactionDT_raw", "TransactionAmt_raw"]:
        if col in meta_df.columns and len(meta_df) == len(out):
            out[col] = meta_df[col].values

    front = [c for c in ["row_id", "TransactionID", "TransactionDT_raw", "TransactionAmt_raw"] if c in out.columns]
    return out[front + [c for c in out.columns if c not in front]]


def run_experiment(experiment, train_loader, val_loader, test_loader, dims, cat_cardinalities, test_ds, review_meta_test, device):
    print("\n" + "=" * 80)
    print("EXPERIMENT:", experiment)
    print("=" * 80)

    model = build_model(experiment, dims, cat_cardinalities).to(device)
    model.experiment_name = experiment

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    scaler = torch.amp.GradScaler("cuda") if (USE_AMP and device.type == "cuda") else None

    ckpt_path = OUTPUT_DIR / f"best_{experiment}.pt"
    history_path = OUTPUT_DIR / f"history_{experiment}.csv"

    best_val_pr_auc = -np.inf
    best_epoch = 0
    patience_counter = 0
    history = []
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        try:
            train_log = train_one_epoch(model, train_loader, optimizer, device, scaler)
        except FloatingPointError as e:
            print("[STOP]", e)
            break

        y_val, val_prob, _ = collect_predictions(model, val_loader, device)
        thr_info = threshold_by_best_f1(y_val, val_prob)
        val_metrics = evaluate_fraud_metrics(y_val, val_prob, thr_info["threshold"], min_precision=MIN_PRECISION_TARGET)

        scheduler.step(val_metrics["PR_AUC"])

        row = {
            "experiment": experiment,
            "epoch": epoch,
            "train_loss": train_log["loss"],
            "train_classification_loss": train_log["classification_loss"],
            "train_kd_loss": train_log["kd_loss"],
            "train_supcon_loss": train_log["supcon_loss"],
            "val_PR_AUC": val_metrics["PR_AUC"],
            "val_ROC_AUC": val_metrics["ROC_AUC"],
            "val_Fraud_Precision": val_metrics["Fraud_Precision"],
            "val_Fraud_Recall": val_metrics["Fraud_Recall"],
            "val_Fraud_F1": val_metrics["Fraud_F1"],
            "val_MCC": val_metrics["MCC"],
            f"val_Recall@Precision>={MIN_PRECISION_TARGET:.2f}": val_metrics[f"Recall@Precision>={MIN_PRECISION_TARGET:.2f}"],
            "val_threshold": thr_info["threshold"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(time.time() - epoch_start, 2),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={row['train_loss']:.5f} | "
            f"kd={row['train_kd_loss']:.5f} | "
            f"supcon={row['train_supcon_loss']:.5f} | "
            f"val_pr_auc={row['val_PR_AUC']:.5f} | "
            f"val_f1={row['val_Fraud_F1']:.5f} | "
            f"val_rec={row['val_Fraud_Recall']:.5f} | "
            f"val_thr={row['val_threshold']:.5f} | "
            f"time={row['epoch_seconds']}s"
        )

        if val_metrics["PR_AUC"] > best_val_pr_auc:
            best_val_pr_auc = val_metrics["PR_AUC"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "experiment": experiment,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "dims": dims,
                "categorical_cardinalities": cat_cardinalities,
                "config": {
                    "LEARNING_RATE": LEARNING_RATE,
                    "WEIGHT_DECAY": WEIGHT_DECAY,
                    "KD_WEIGHT": KD_WEIGHT,
                    "LAMBDA_SUPCON": LAMBDA_SUPCON,
                    "FOCAL_ALPHA": FOCAL_ALPHA,
                    "FOCAL_GAMMA": FOCAL_GAMMA,
                },
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    pd.DataFrame(history).to_csv(history_path, index=False)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_val, val_prob, _ = collect_predictions(model, val_loader, device)
    thr_info = threshold_by_best_f1(y_val, val_prob)
    selected_threshold = thr_info["threshold"]

    y_test, test_prob, gate_weights = collect_predictions(model, test_loader, device)
    test_metrics = evaluate_fraud_metrics(
        y_test,
        test_prob,
        selected_threshold,
        min_precision=MIN_PRECISION_TARGET,
    )

    amount = None
    if review_meta_test is not None and "TransactionAmt_raw" in review_meta_test.columns and len(review_meta_test) == len(y_test):
        amount = pd.to_numeric(review_meta_test["TransactionAmt_raw"], errors="coerce").fillna(0).to_numpy(dtype=float)
    else:
        amount = np.zeros(len(y_test), dtype=float)

    budget_df = review_budget_metrics(y_test, test_prob, amount)
    budget_flat = flatten_budget_metrics(budget_df)

    pred_df = pd.DataFrame({
        "y_true": y_test.astype(int),
        "y_prob": test_prob.astype(float),
        "y_pred": (test_prob >= selected_threshold).astype(int),
    })

    if gate_weights is not None:
        pred_df["gate_cnn_weight"] = gate_weights[:, 0]
        pred_df["gate_deepfm_weight"] = gate_weights[:, 1]

    pred_df = attach_meta(pred_df, review_meta_test)

    pred_path = OUTPUT_DIR / f"pred_{experiment}.csv"
    budget_path = OUTPUT_DIR / f"review_budget_detail_{experiment}.csv"

    pred_df.to_csv(pred_path, index=False)
    budget_df.insert(0, "Model", experiment)
    budget_df.to_csv(budget_path, index=False)

    result = {
        "Model": experiment,
        "Best_Epoch": int(best_epoch),
        "Config_Name": "E_kd020_sup005_pr_auc_seed2024",
        "Config_Seed": 2024,
        "Config_KD_Weight": 0.2,
        "Config_Lambda_SupCon": 0.005,
        "Best_Val_PR_AUC": float(best_val_pr_auc),
        "Selected_Threshold_from_Val": float(selected_threshold),
        "Val_Best_Fraud_Precision": float(thr_info["precision"]),
        "Val_Best_Fraud_Recall": float(thr_info["recall"]),
        "Val_Best_Fraud_F1": float(thr_info["f1"]),
        "Train_Time_Seconds": round(time.time() - start_time, 3),
        "Checkpoint_File": str(ckpt_path),
        "History_File": str(history_path),
        "Prediction_File": str(pred_path),
        "Review_Budget_Detail_File": str(budget_path),
        **test_metrics,
        **budget_flat,
    }

    print("\nFINAL TEST RESULT")
    for k in [
        "PR_AUC", "ROC_AUC", "Fraud_Precision", "Fraud_Recall", "Fraud_F1",
        "MCC", f"Recall@Precision>={MIN_PRECISION_TARGET:.2f}",
        "Precision@5%", "Recall@5%", "CapturedAmountRate@5%",
        "Selected_Threshold", "TN", "FP", "FN", "TP",
    ]:
        if k in result:
            print(k, ":", result[k])

    return result


# ============================================================
# MAIN
# ============================================================

def save_paper_table(results_df: pd.DataFrame):
    paper_cols = [
        "Model",
        "PR_AUC",
        "ROC_AUC",
        "Fraud_Precision",
        "Fraud_Recall",
        "Fraud_F1",
        "MCC",
        f"Recall@Precision>={MIN_PRECISION_TARGET:.2f}",
        "Precision@1%",
        "Recall@1%",
        "CapturedAmountRate@1%",
        "ExpectedUtility@1%",
        "Precision@3%",
        "Recall@3%",
        "CapturedAmountRate@3%",
        "ExpectedUtility@3%",
        "Precision@5%",
        "Recall@5%",
        "CapturedAmountRate@5%",
        "ExpectedUtility@5%",
        "Precision@10%",
        "Recall@10%",
        "CapturedAmountRate@10%",
        "ExpectedUtility@10%",
        "Selected_Threshold_from_Val",
        "TN", "FP", "FN", "TP",
        "Best_Epoch",
        "Best_Val_PR_AUC",
        "Train_Time_Seconds",
    ]
    paper_cols = [c for c in paper_cols if c in results_df.columns]
    results_df[paper_cols].to_csv(OUTPUT_DIR / "paper_ready_proposed_only_cost_aware_table.csv", index=False)


def main():
    set_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Running ONLY proposed model: tree_guided_moe_kd_focal_supcon")
    print("Output dir:", OUTPUT_DIR)

    train_loader, val_loader, test_loader, dims, cat_cardinalities, test_ds = build_loaders()

    try:
        review_meta_test = read_table("review_meta_internal_test")
        if SUBSET_TEST_N is not None:
            idx = maybe_subset_by_y(review_meta_test[[TARGET_COL]].reset_index(), SUBSET_TEST_N)["index"].to_numpy()
            review_meta_test = review_meta_test.iloc[idx].reset_index(drop=True)
    except Exception as e:
        print("[WARN] Could not load review_meta_internal_test:", e)
        review_meta_test = None

    all_results = []

    for experiment in RUN_EXPERIMENTS:
        if "kd" in experiment and not TEACHER_DIR.exists():
            print("[SKIP] KD experiment needs teacher_signals folder:", experiment)
            continue

        try:
            result = run_experiment(
                experiment=experiment,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                dims=dims,
                cat_cardinalities=cat_cardinalities,
                test_ds=test_ds,
                review_meta_test=review_meta_test,
                device=device,
            )
            all_results.append(result)

            running_df = pd.DataFrame(all_results).sort_values("PR_AUC", ascending=False)
            running_df.to_csv(OUTPUT_DIR / "proposed_only_results_running.csv", index=False)
            with open(OUTPUT_DIR / "proposed_only_results_running.json", "w", encoding="utf-8") as f:
                json.dump(running_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
            save_paper_table(running_df)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[ERROR] Experiment failed: {experiment}")
            print(type(e).__name__, str(e))

    if not all_results:
        raise RuntimeError("No DL experiment completed successfully.")

    results_df = pd.DataFrame(all_results).sort_values("PR_AUC", ascending=False)

    final_csv = OUTPUT_DIR / "proposed_only_results_final.csv"
    final_json = OUTPUT_DIR / "proposed_only_results_final.json"

    results_df.to_csv(final_csv, index=False)
    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    save_paper_table(results_df)

    print("\nSaved final results:")
    print(final_csv)
    print(final_json)
    print(OUTPUT_DIR / "paper_ready_proposed_only_cost_aware_table.csv")
    print(results_df)


if __name__ == "__main__":
    main()
