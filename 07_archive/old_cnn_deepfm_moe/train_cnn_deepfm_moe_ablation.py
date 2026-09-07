from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sklearn.metrics import average_precision_score, roc_auc_score

from cnn_branch_updated import TabularCNNBranch
from deepfm_branch_updated import DeepFMBranch
from fraud_losses import BinaryFocalLoss, CombinedFocalSupConLoss
from moe_gated_fusion import (
    ConcatCNNDeepFM,
    ThresholdedMoEGatedCNNDeepFM,
    evaluate_fraud_metrics,
    search_best_threshold,
)


# ============================================================
# CONFIG
# ============================================================

PROCESSED_DIR = Path(r"D:\project\data\merge_paper_ready")
OUTPUT_DIR = PROCESSED_DIR / "dl_moe_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "isFraud"
RANDOM_STATE = 42

RUN_EXPERIMENTS = [
    "cnn_only_bce",
    "deepfm_only_bce",
    "concat_bce",
    "moe_bce",
    "moe_focal",
    "moe_focal_supcon",
]

BATCH_SIZE = 1024
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0

USE_WEIGHTED_SAMPLER = True
USE_AMP = True
NUM_WORKERS = 0

MIN_PRECISION_TARGET = 0.80
THRESHOLD_OBJECTIVE = "f1"
THRESHOLD_MIN_PRECISION = None

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
DROPOUT = 0.25

SUBSET_TRAIN_N = None
SUBSET_VAL_N = None
SUBSET_TEST_N = None


# ============================================================
# UTILITIES
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
    df = df.copy()
    for col in df.columns:
        if col == TARGET_COL:
            df[col] = df[col].astype("int8")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def maybe_subset(df: pd.DataFrame, n: Optional[int], seed: int = RANDOM_STATE) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df
    if TARGET_COL not in df.columns:
        return df.sample(n=n, random_state=seed)
    fraud = df[df[TARGET_COL] == 1]
    normal = df[df[TARGET_COL] == 0]
    n_normal = max(n - len(fraud), 0)
    normal_sample = normal.sample(n=min(n_normal, len(normal)), random_state=seed)
    out = pd.concat([fraud, normal_sample], axis=0).sample(frac=1.0, random_state=seed)
    return out.reset_index(drop=True)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================
# DATASET
# ============================================================

class FraudMultiInputDataset(Dataset):
    def __init__(self, cnn_df: pd.DataFrame, cat_df: pd.DataFrame, num_df: pd.DataFrame):
        if TARGET_COL not in cnn_df.columns:
            raise ValueError(f"{TARGET_COL} not found in cnn_df")
        if TARGET_COL not in cat_df.columns:
            raise ValueError(f"{TARGET_COL} not found in cat_df")
        if TARGET_COL not in num_df.columns:
            raise ValueError(f"{TARGET_COL} not found in num_df")

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

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x_cnn": torch.from_numpy(self.x_cnn[idx]),
            "x_cat": torch.from_numpy(self.x_cat[idx]),
            "x_num": torch.from_numpy(self.x_num[idx]),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


def load_split(split: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cnn = downcast_df(read_table(f"cnn_{split}"))
    cat = downcast_df(read_table(f"deepfm_cat_{split}"))
    num = downcast_df(read_table(f"deepfm_num_{split}"))
    return cnn, cat, num


def build_loaders() -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], List[int]]:
    train_cnn, train_cat, train_num = load_split("train")
    val_cnn, val_cat, val_num = load_split("val")
    test_cnn, test_cat, test_num = load_split("internal_test")

    if SUBSET_TRAIN_N is not None:
        idx = maybe_subset(train_cnn[[TARGET_COL]].reset_index(), SUBSET_TRAIN_N)["index"].to_numpy()
        train_cnn = train_cnn.iloc[idx].reset_index(drop=True)
        train_cat = train_cat.iloc[idx].reset_index(drop=True)
        train_num = train_num.iloc[idx].reset_index(drop=True)

    if SUBSET_VAL_N is not None:
        idx = maybe_subset(val_cnn[[TARGET_COL]].reset_index(), SUBSET_VAL_N)["index"].to_numpy()
        val_cnn = val_cnn.iloc[idx].reset_index(drop=True)
        val_cat = val_cat.iloc[idx].reset_index(drop=True)
        val_num = val_num.iloc[idx].reset_index(drop=True)

    if SUBSET_TEST_N is not None:
        idx = maybe_subset(test_cnn[[TARGET_COL]].reset_index(), SUBSET_TEST_N)["index"].to_numpy()
        test_cnn = test_cnn.iloc[idx].reset_index(drop=True)
        test_cat = test_cat.iloc[idx].reset_index(drop=True)
        test_num = test_num.iloc[idx].reset_index(drop=True)

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

    train_ds = FraudMultiInputDataset(train_cnn, train_cat, train_num)
    val_ds = FraudMultiInputDataset(val_cnn, val_cat, val_num)
    test_ds = FraudMultiInputDataset(test_cnn, test_cat, test_num)

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
    print("Categorical cardinalities sample:", categorical_cardinalities[:10])
    print("Train fraud ratio:", float(train_ds.y.mean()))
    print("Val fraud ratio:", float(val_ds.y.mean()))
    print("Test fraud ratio:", float(test_ds.y.mean()))

    return train_loader, val_loader, test_loader, dims, categorical_cardinalities


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
        self.decision_threshold = torch.tensor(0.5, dtype=torch.float32)

    def forward(
        self,
        x_cnn: torch.Tensor,
        x_cat: Optional[torch.Tensor] = None,
        x_num: Optional[torch.Tensor] = None,
        x_dense: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        logits, z_cnn, _ = self.cnn_branch(x_cnn, return_embedding=True)
        logits = logits.view(-1)
        if not return_dict:
            return logits
        prob = torch.sigmoid(logits)
        pred = (prob >= self.decision_threshold.to(prob.device)).long()
        return {"logits": logits, "prob": prob, "pred": pred, "embedding": z_cnn}

    def set_threshold(self, threshold: float):
        self.decision_threshold = torch.tensor(float(threshold), dtype=torch.float32)


class DeepFMOnlyClassifier(nn.Module):
    def __init__(self, deepfm: nn.Module):
        super().__init__()
        self.deepfm = deepfm
        self.decision_threshold = torch.tensor(0.5, dtype=torch.float32)

    def forward(
        self,
        x_cnn: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
        x_num: Optional[torch.Tensor] = None,
        x_dense: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        logits, z_fm = self.deepfm(cat_x=x_cat, num_x=None, dense_x=x_dense, return_embedding=True)
        logits = logits.view(-1)
        if not return_dict:
            return logits
        prob = torch.sigmoid(logits)
        pred = (prob >= self.decision_threshold.to(prob.device)).long()
        return {"logits": logits, "prob": prob, "pred": pred, "embedding": z_fm}

    def set_threshold(self, threshold: float):
        self.decision_threshold = torch.tensor(float(threshold), dtype=torch.float32)


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
            cnn_embedding_dim=CNN_OUT_DIM,
            deepfm_embedding_dim=DEEPFM_BRANCH_OUT_DIM,
            fusion_dim=FUSION_DIM,
            dropout=DROPOUT,
        )

    if experiment.startswith("moe"):
        return ThresholdedMoEGatedCNNDeepFM(
            cnn_branch=cnn,
            deepfm_branch=deepfm_compat,
            cnn_embedding_dim=CNN_OUT_DIM,
            deepfm_embedding_dim=DEEPFM_BRANCH_OUT_DIM,
            fusion_dim=FUSION_DIM,
            gate_hidden_dim=GATE_HIDDEN_DIM,
            dropout=DROPOUT,
            initial_threshold=0.5,
        )

    raise ValueError(f"Unknown experiment: {experiment}")


def build_criterion(experiment: str, train_loader: DataLoader, device: torch.device):
    if experiment.endswith("focal_supcon"):
        return CombinedFocalSupConLoss(
            focal_alpha=0.85,
            focal_gamma=2.0,
            lambda_supcon=0.05,
            temperature=0.10,
            fraud_anchor_weight=2.0,
        )

    if experiment.endswith("focal"):
        return BinaryFocalLoss(alpha=0.85, gamma=2.0)

    return nn.BCEWithLogitsLoss()


# ============================================================
# TRAIN / EVAL
# ============================================================

def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def forward_model(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return model(
        x_cnn=batch["x_cnn"],
        x_cat=batch["x_cat"].long(),
        x_dense=batch["x_num"],
        return_dict=True,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_supcon = 0.0
    n = 0

    for batch in loader:
        batch = move_batch(batch, device)
        y = batch["y"].float()

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = forward_model(model, batch)
                logits = out["logits"].view(-1)
                if isinstance(criterion, CombinedFocalSupConLoss):
                    loss_dict = criterion(logits, y, features=out.get("embedding"))
                    loss = loss_dict["loss"]
                    cls_loss = loss_dict["classification_loss"]
                    supcon_loss = loss_dict["supcon_loss"]
                else:
                    loss = criterion(logits, y)
                    cls_loss = loss.detach()
                    supcon_loss = torch.zeros((), device=device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = forward_model(model, batch)
            logits = out["logits"].view(-1)
            if isinstance(criterion, CombinedFocalSupConLoss):
                loss_dict = criterion(logits, y, features=out.get("embedding"))
                loss = loss_dict["loss"]
                cls_loss = loss_dict["classification_loss"]
                supcon_loss = loss_dict["supcon_loss"]
            else:
                loss = criterion(logits, y)
                cls_loss = loss.detach()
                supcon_loss = torch.zeros((), device=device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        bs = y.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        total_cls += float(cls_loss.detach().cpu()) * bs
        total_supcon += float(supcon_loss.detach().cpu()) * bs
        n += bs

    return {
        "loss": total_loss / max(n, 1),
        "classification_loss": total_cls / max(n, 1),
        "supcon_loss": total_supcon / max(n, 1),
    }


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    probs = []

    for batch in loader:
        batch = move_batch(batch, device)
        out = forward_model(model, batch)
        prob = out["prob"].view(-1)
        ys.append(batch["y"].detach().cpu().numpy())
        probs.append(prob.detach().cpu().numpy())

    return np.concatenate(ys), np.concatenate(probs)


def evaluate_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    y_true, y_prob = collect_predictions(model, loader, device)

    if threshold is None:
        threshold_state = search_best_threshold(
            y_true=y_true,
            y_prob=y_prob,
            objective=THRESHOLD_OBJECTIVE,
            min_precision=THRESHOLD_MIN_PRECISION,
        )
        threshold = threshold_state.threshold

    metrics = evaluate_fraud_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        min_precision=MIN_PRECISION_TARGET,
    )

    try:
        metrics["PR_AUC"] = float(average_precision_score(y_true, y_prob))
        metrics["ROC_AUC"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass

    return metrics


def run_experiment(
    experiment: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    dims: Dict[str, int],
    cat_cardinalities: List[int],
    device: torch.device,
) -> Dict[str, object]:
    print("\n" + "=" * 80)
    print("EXPERIMENT:", experiment)
    print("=" * 80)

    model = build_model(experiment, dims, cat_cardinalities).to(device)
    criterion = build_criterion(experiment, train_loader, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        verbose=True,
    )

    scaler = torch.cuda.amp.GradScaler() if (USE_AMP and device.type == "cuda") else None

    best_val_pr_auc = -np.inf
    best_epoch = 0
    patience_counter = 0
    ckpt_path = OUTPUT_DIR / f"best_{experiment}.pt"

    history = []
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        train_log = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)

        y_val, val_prob = collect_predictions(model, val_loader, device)
        val_threshold_state = search_best_threshold(
            y_true=y_val,
            y_prob=val_prob,
            objective=THRESHOLD_OBJECTIVE,
            min_precision=THRESHOLD_MIN_PRECISION,
        )
        val_metrics = evaluate_fraud_metrics(
            y_true=y_val,
            y_prob=val_prob,
            threshold=val_threshold_state.threshold,
            min_precision=MIN_PRECISION_TARGET,
        )

        scheduler.step(val_metrics["PR_AUC"])

        row = {
            "experiment": experiment,
            "epoch": epoch,
            "train_loss": train_log["loss"],
            "train_classification_loss": train_log["classification_loss"],
            "train_supcon_loss": train_log["supcon_loss"],
            "val_PR_AUC": val_metrics["PR_AUC"],
            "val_ROC_AUC": val_metrics["ROC_AUC"],
            "val_Fraud_Precision": val_metrics["Fraud_Precision"],
            "val_Fraud_Recall": val_metrics["Fraud_Recall"],
            "val_Fraud_F1": val_metrics["Fraud_F1"],
            "val_MCC": val_metrics["MCC"],
            f"val_Recall@Precision>={MIN_PRECISION_TARGET:.2f}": val_metrics[f"Recall@Precision>={MIN_PRECISION_TARGET:.2f}"],
            "val_threshold": val_threshold_state.threshold,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(time.time() - epoch_start, 2),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={row['train_loss']:.5f} | "
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
            torch.save(
                {
                    "experiment": experiment,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dims": dims,
                    "categorical_cardinalities": cat_cardinalities,
                    "config": {
                        "BATCH_SIZE": BATCH_SIZE,
                        "EPOCHS": EPOCHS,
                        "LEARNING_RATE": LEARNING_RATE,
                        "WEIGHT_DECAY": WEIGHT_DECAY,
                        "MIN_PRECISION_TARGET": MIN_PRECISION_TARGET,
                        "THRESHOLD_OBJECTIVE": THRESHOLD_OBJECTIVE,
                        "THRESHOLD_MIN_PRECISION": THRESHOLD_MIN_PRECISION,
                    },
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    history_df = pd.DataFrame(history)
    history_path = OUTPUT_DIR / f"history_{experiment}.csv"
    history_df.to_csv(history_path, index=False)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_val, val_prob = collect_predictions(model, val_loader, device)
    best_threshold_state = search_best_threshold(
        y_true=y_val,
        y_prob=val_prob,
        objective=THRESHOLD_OBJECTIVE,
        min_precision=THRESHOLD_MIN_PRECISION,
    )
    selected_threshold = best_threshold_state.threshold

    if hasattr(model, "set_threshold") and not math.isnan(selected_threshold):
        model.set_threshold(selected_threshold)
    elif hasattr(model, "tune_threshold"):
        model.tune_threshold(y_val, val_prob, objective=THRESHOLD_OBJECTIVE, min_precision=THRESHOLD_MIN_PRECISION)

    y_test, test_prob = collect_predictions(model, test_loader, device)
    test_metrics = evaluate_fraud_metrics(
        y_true=y_test,
        y_prob=test_prob,
        threshold=selected_threshold,
        min_precision=MIN_PRECISION_TARGET,
    )

    pred_path = OUTPUT_DIR / f"pred_{experiment}.csv"
    pd.DataFrame({
        "y_true": y_test.astype(int),
        "y_prob": test_prob,
        "y_pred": (test_prob >= selected_threshold).astype(int),
    }).to_csv(pred_path, index=False)

    result = {
        "Model": experiment,
        "Best_Epoch": best_epoch,
        "Best_Val_PR_AUC": float(best_val_pr_auc),
        "Selected_Threshold_from_Val": float(selected_threshold),
        "Val_Best_Fraud_Precision": best_threshold_state.val_precision,
        "Val_Best_Fraud_Recall": best_threshold_state.val_recall,
        "Val_Best_Fraud_F1": best_threshold_state.val_f1,
        "Val_Best_MCC": best_threshold_state.val_mcc,
        "Train_Time_Seconds": round(time.time() - start_time, 3),
        "Checkpoint_File": str(ckpt_path),
        "History_File": str(history_path),
        "Prediction_File": str(pred_path),
        **test_metrics,
    }

    print("\nFINAL TEST RESULT:")
    for k in [
        "PR_AUC", "ROC_AUC", "Fraud_Precision", "Fraud_Recall",
        "Fraud_F1", "MCC", f"Recall@Precision>={MIN_PRECISION_TARGET:.2f}",
        "Selected_Threshold", "TN", "FP", "FN", "TP"
    ]:
        print(k, ":", result[k])

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, test_loader, dims, cat_cardinalities = build_loaders()

    all_results = []
    for experiment in RUN_EXPERIMENTS:
        try:
            result = run_experiment(
                experiment=experiment,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                dims=dims,
                cat_cardinalities=cat_cardinalities,
                device=device,
            )
            all_results.append(result)

            results_df = pd.DataFrame(all_results).sort_values("PR_AUC", ascending=False)
            results_df.to_csv(OUTPUT_DIR / "dl_moe_results_running.csv", index=False)
            with open(OUTPUT_DIR / "dl_moe_results_running.json", "w", encoding="utf-8") as f:
                json.dump(results_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[ERROR] Experiment failed: {experiment}")
            print(type(e).__name__, str(e))

    results_df = pd.DataFrame(all_results).sort_values("PR_AUC", ascending=False)
    final_csv = OUTPUT_DIR / "dl_moe_results_final.csv"
    final_json = OUTPUT_DIR / "dl_moe_results_final.json"
    results_df.to_csv(final_csv, index=False)
    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print("\nSaved final results:")
    print(final_csv)
    print(final_json)
    print(results_df)


if __name__ == "__main__":
    main()
