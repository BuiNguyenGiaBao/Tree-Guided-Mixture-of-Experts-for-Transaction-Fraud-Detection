
from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

@dataclass
class TrainConfig:
    processed_dir: str = r"D:\project\data\merge_paper_ready"
    output_dir_name: str = "unified_moe_results"

    target_col: str = "isFraud"
    random_state: int = 42

    batch_size: int = 1024
    epochs: int = 20
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 0

    use_balanced_sampler: bool = True

    cnn_embed_dim: int = 128
    cnn_conv_channels: int = 128
    cnn_kernel_size: int = 3
    cnn_bilinear_rank: int = 32
    cnn_out_dim: int = 128
    cnn_seq_length: int = 10

    deepfm_embed_dim: int = 16
    deepfm_hidden_1: int = 256
    deepfm_hidden_2: int = 128
    deepfm_branch_out_dim: int = 128
    deepfm_dense_num_fields: int = 4

    fusion_dim: int = 128
    gate_hidden_dim: int = 64
    dropout: float = 0.25

    focal_alpha: float = 0.85
    focal_gamma: float = 2.0
    lambda_supcon: float = 0.05
    supcon_temperature: float = 0.10
    fraud_anchor_weight: float = 2.0

    threshold_objective: str = "f1"
    threshold_min_precision: Optional[float] = None
    recall_at_precision_target: float = 0.80

    # For first debugging run, set this to e.g. 100000. For final paper run, keep None.
    sample_train_n: Optional[int] = None

    save_best_model: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# UTILITIES
# ============================================================

def seed_everything(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_file(base_name: str, folder: Path) -> Path:
    for ext in [".parquet", ".csv.gz", ".csv"]:
        p = folder / f"{base_name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {base_name}.parquet/.csv.gz/.csv in {folder}")


def read_table(base_name: str, folder: Path) -> pd.DataFrame:
    path = resolve_file(base_name, folder)
    print(f"[LOAD] {base_name}: {path}")
    if path.name.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def downcast_df(df: pd.DataFrame, target_col: str = "isFraud") -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c == target_col:
            df[c] = df[c].astype("int8")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def maybe_sample_train(
    cnn_df: pd.DataFrame,
    cat_df: pd.DataFrame,
    num_df: pd.DataFrame,
    target_col: str,
    sample_train_n: Optional[int],
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if sample_train_n is None or sample_train_n >= len(cnn_df):
        return cnn_df, cat_df, num_df

    y = cnn_df[target_col].values.astype(int)
    rng = np.random.default_rng(random_state)

    fraud_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]

    n_fraud = len(fraud_idx)
    n_normal_needed = max(sample_train_n - n_fraud, 0)
    normal_sample = rng.choice(normal_idx, size=min(n_normal_needed, len(normal_idx)), replace=False)

    selected = np.concatenate([fraud_idx, normal_sample])
    rng.shuffle(selected)

    return (
        cnn_df.iloc[selected].reset_index(drop=True),
        cat_df.iloc[selected].reset_index(drop=True),
        num_df.iloc[selected].reset_index(drop=True),
    )


# ============================================================
# DATASET
# ============================================================

class FraudFusionDataset(Dataset):
    def __init__(
        self,
        cnn_df: pd.DataFrame,
        cat_df: pd.DataFrame,
        num_df: pd.DataFrame,
        target_col: str = "isFraud",
    ):
        if len(cnn_df) != len(cat_df) or len(cnn_df) != len(num_df):
            raise ValueError("cnn_df, cat_df, num_df must have the same number of rows.")

        self.target_col = target_col

        self.y = cnn_df[target_col].values.astype(np.float32)

        self.x_cnn = cnn_df.drop(columns=[target_col], errors="ignore").values.astype(np.float32)
        self.x_cat = cat_df.drop(columns=[target_col], errors="ignore").values.astype(np.int64)
        self.x_num = num_df.drop(columns=[target_col], errors="ignore").values.astype(np.float32)

        # OrdinalEncoder unknown category is -1, shift to 0 for embeddings.
        self.x_cat = np.maximum(self.x_cat + 1, 0).astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x_cnn": torch.tensor(self.x_cnn[idx], dtype=torch.float32),
            "x_cat": torch.tensor(self.x_cat[idx], dtype=torch.long),
            "x_num": torch.tensor(self.x_num[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


def categorical_cardinalities_from_train(cat_df: pd.DataFrame, target_col: str = "isFraud") -> List[int]:
    X = cat_df.drop(columns=[target_col], errors="ignore").copy()
    cards = []
    for c in X.columns:
        vals = pd.to_numeric(X[c], errors="coerce").fillna(-1).astype(int).values
        vals = np.maximum(vals + 1, 0)
        cards.append(int(vals.max()) + 1)
    return cards


def build_loaders(cfg: TrainConfig):
    processed_dir = Path(cfg.processed_dir)

    cnn_train = downcast_df(read_table("cnn_train", processed_dir), cfg.target_col)
    cnn_val = downcast_df(read_table("cnn_val", processed_dir), cfg.target_col)
    cnn_test = downcast_df(read_table("cnn_internal_test", processed_dir), cfg.target_col)

    cat_train = downcast_df(read_table("deepfm_cat_train", processed_dir), cfg.target_col)
    cat_val = downcast_df(read_table("deepfm_cat_val", processed_dir), cfg.target_col)
    cat_test = downcast_df(read_table("deepfm_cat_internal_test", processed_dir), cfg.target_col)

    num_train = downcast_df(read_table("deepfm_num_train", processed_dir), cfg.target_col)
    num_val = downcast_df(read_table("deepfm_num_val", processed_dir), cfg.target_col)
    num_test = downcast_df(read_table("deepfm_num_internal_test", processed_dir), cfg.target_col)

    cnn_train, cat_train, num_train = maybe_sample_train(
        cnn_train, cat_train, num_train,
        target_col=cfg.target_col,
        sample_train_n=cfg.sample_train_n,
        random_state=cfg.random_state,
    )

    train_ds = FraudFusionDataset(cnn_train, cat_train, num_train, cfg.target_col)
    val_ds = FraudFusionDataset(cnn_val, cat_val, num_val, cfg.target_col)
    test_ds = FraudFusionDataset(cnn_test, cat_test, num_test, cfg.target_col)

    if cfg.use_balanced_sampler:
        y = train_ds.y.astype(int)
        class_count = np.bincount(y, minlength=2)
        class_weight = 1.0 / np.maximum(class_count, 1)
        sample_weight = class_weight[y]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weight, dtype=torch.double),
            num_samples=len(sample_weight),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    dims = {
        "cnn_dim": train_ds.x_cnn.shape[1],
        "num_dim": train_ds.x_num.shape[1],
        "cat_dim": train_ds.x_cat.shape[1],
        "categorical_cardinalities": categorical_cardinalities_from_train(cat_train, cfg.target_col),
        "train_fraud_ratio": float(train_ds.y.mean()),
        "val_fraud_ratio": float(val_ds.y.mean()),
        "test_fraud_ratio": float(test_ds.y.mean()),
    }

    return train_loader, val_loader, test_loader, dims


# ============================================================
# CNN MIX BRANCH
# ============================================================

class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_t = x.transpose(1, 2)
        scores = self.attention(x_t)
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.transpose(1, 2), dim=2)
        return pooled, weights


class LowRankBilinear(nn.Module):
    def __init__(self, in1_features: int, in2_features: int, out_features: int, rank: int):
        super().__init__()
        self.U1 = nn.Linear(in1_features, rank, bias=False)
        self.U2 = nn.Linear(in2_features, rank, bias=False)
        self.V = nn.Linear(rank, out_features)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.V(self.U1(x1) * self.U2(x2))


class CNNMixBranch(nn.Module):
    def __init__(
        self,
        tabular_dim: int,
        embed_dim: int = 128,
        conv_channels: int = 128,
        kernel_size: int = 3,
        bilinear_rank: int = 32,
        out_dim: int = 128,
        seq_length: int = 10,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.tabular_dim = int(tabular_dim)
        self.out_dim = int(out_dim)
        self.seq_length = int(seq_length)
        self.embed_dim = int(embed_dim)

        self.tabular_embed = nn.Sequential(
            nn.Linear(tabular_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.feature_projection = nn.Linear(embed_dim, embed_dim * seq_length)

        self.conv1 = nn.Conv1d(embed_dim, conv_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(conv_channels)

        self.attention_pool = AttentionPooling(conv_channels)
        self.low_rank_bilinear = LowRankBilinear(conv_channels, conv_channels, out_dim, bilinear_rank)

    def encode(self, x: torch.Tensor, return_attention: bool = False):
        if x.dim() != 2 or x.size(1) != self.tabular_dim:
            raise ValueError(f"x must have shape (B, {self.tabular_dim}).")

        b = x.size(0)
        h = self.tabular_embed(x)
        h = self.feature_projection(h).view(b, self.seq_length, self.embed_dim).transpose(1, 2)

        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.conv2(h)))

        pooled, attn = self.attention_pool(h)
        z = F.relu(self.low_rank_bilinear(pooled, pooled))

        if return_attention:
            return z, attn
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


# ============================================================
# DEEPFM MIX BRANCH
# ============================================================

class FactorizationMachine(nn.Module):
    def forward(self, emb: torch.Tensor, values: Optional[torch.Tensor] = None) -> torch.Tensor:
        if values is not None:
            emb = emb * values
        sum_emb = emb.sum(dim=1)
        sum_square = sum_emb * sum_emb
        square_sum = (emb * emb).sum(dim=1)
        return 0.5 * (sum_square - square_sum).sum(dim=1, keepdim=True)


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float = 0.25):
        super().__init__()
        layers = []
        d = int(in_dim)
        for h in hidden_dims:
            layers += [
                nn.Linear(d, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            d = int(h)
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepFMMixBranch(nn.Module):
    def __init__(
        self,
        categorical_cardinalities: List[int],
        num_numerical: int,
        embed_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        dense_num_fields: int = 4,
        branch_out_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()

        hidden_dims = hidden_dims or [256, 128]

        self.categorical_cardinalities = [int(max(c, 1)) for c in categorical_cardinalities]
        self.num_cat = len(self.categorical_cardinalities)
        self.num_num = int(num_numerical)
        self.embed_dim = int(embed_dim)
        self.dense_num_fields = int(dense_num_fields)

        self.linear_cat = nn.ModuleList([nn.Embedding(card, 1) for card in self.categorical_cardinalities])
        self.linear_num = nn.Linear(self.num_num, 1, bias=False)

        self.fm = FactorizationMachine()
        self.fm_cat_emb = nn.ModuleList([nn.Embedding(card, embed_dim) for card in self.categorical_cardinalities])
        self.fm_num_emb = nn.Parameter(torch.randn(self.num_num, embed_dim) * 0.01)

        self.dense_proj = nn.Linear(self.num_num, embed_dim * dense_num_fields, bias=False)

        total_fields = self.num_cat + self.num_num + self.dense_num_fields
        self.mlp = MLPBlock(embed_dim * total_fields, hidden_dims, dropout=dropout)

        raw_dim = 2 + self.mlp.out_dim
        self.output_dim = int(branch_out_dim)
        self.branch_projection = nn.Sequential(
            nn.Linear(raw_dim, branch_out_dim),
            nn.BatchNorm1d(branch_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _build_field_embeddings(self, x_cat: torch.Tensor, x_num: torch.Tensor):
        embs = []
        vals = []

        if self.num_cat > 0:
            if x_cat is None or x_cat.size(1) != self.num_cat:
                raise ValueError(f"x_cat must have shape (B, {self.num_cat}).")
            for j, emb_layer in enumerate(self.fm_cat_emb):
                xj = x_cat[:, j].clamp(min=0, max=emb_layer.num_embeddings - 1)
                e = emb_layer(xj)
                embs.append(e.unsqueeze(1))
                vals.append(torch.ones(e.size(0), 1, 1, device=e.device, dtype=e.dtype))

        if x_num is None or x_num.size(1) != self.num_num:
            raise ValueError(f"x_num must have shape (B, {self.num_num}).")

        e_num = self.fm_num_emb.unsqueeze(0).expand(x_num.size(0), -1, -1)
        embs.append(e_num)
        vals.append(x_num.unsqueeze(-1))

        e_dense = self.dense_proj(x_num).view(x_num.size(0), self.dense_num_fields, self.embed_dim)
        embs.append(e_dense)
        vals.append(torch.ones(x_num.size(0), self.dense_num_fields, 1, device=x_num.device, dtype=x_num.dtype))

        field_emb = torch.cat(embs, dim=1)
        values = torch.cat(vals, dim=1)
        return field_emb, values

    def _linear_part(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        out = self.linear_num(x_num)

        if self.num_cat > 0:
            for j, emb in enumerate(self.linear_cat):
                xj = x_cat[:, j].clamp(min=0, max=emb.num_embeddings - 1)
                out = out + emb(xj)

        return out

    def extract_embedding(
        self,
        x_cat: Optional[torch.Tensor] = None,
        x_num: Optional[torch.Tensor] = None,
        x_dense: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_dense is not None and x_num is None:
            x_num = x_dense
        if x_num is None:
            raise ValueError("x_num is required.")

        linear_out = self._linear_part(x_cat, x_num)
        field_emb, values = self._build_field_embeddings(x_cat, x_num)
        fm_out = self.fm(field_emb, values)
        deep_h = self.mlp(field_emb.reshape(field_emb.size(0), -1))

        z = torch.cat([linear_out, fm_out, deep_h], dim=1)
        return self.branch_projection(z)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        return self.extract_embedding(x_cat=x_cat, x_num=x_num)


# ============================================================
# MOE FUSION
# ============================================================

class TwoExpertSoftmaxGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class CNNDeepFMMoE(nn.Module):
    def __init__(
        self,
        cnn_branch: CNNMixBranch,
        deepfm_branch: DeepFMMixBranch,
        cnn_dim: int = 128,
        deepfm_dim: int = 128,
        fusion_dim: int = 128,
        gate_hidden_dim: int = 64,
        dropout: float = 0.25,
        initial_threshold: float = 0.50,
    ):
        super().__init__()

        self.cnn_branch = cnn_branch
        self.deepfm_branch = deepfm_branch

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.deepfm_proj = nn.Sequential(
            nn.Linear(deepfm_dim, fusion_dim),
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

        self.projection_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
        )

        self.register_buffer("decision_threshold", torch.tensor(float(initial_threshold), dtype=torch.float32))
        self.threshold_state = None

    def forward(
        self,
        x_cnn: torch.Tensor,
        x_cat: torch.Tensor,
        x_num: torch.Tensor,
        return_dict: bool = True,
    ):
        z_cnn_raw = self.cnn_branch.encode(x_cnn)
        z_fm_raw = self.deepfm_branch.extract_embedding(x_cat=x_cat, x_num=x_num)

        z_cnn = self.cnn_proj(z_cnn_raw)
        z_fm = self.deepfm_proj(z_fm_raw)

        gate_input = torch.cat([z_cnn, z_fm], dim=1)
        gate_weight = self.gate(gate_input)

        z_fused = gate_weight[:, 0:1] * z_cnn + gate_weight[:, 1:2] * z_fm
        logits = self.classifier(z_fused).view(-1)
        proj = F.normalize(self.projection_head(z_fused), dim=1)

        if not return_dict:
            return logits

        prob = torch.sigmoid(logits)
        pred = (prob >= self.decision_threshold.to(prob.device)).long()

        return {
            "logits": logits,
            "prob": prob,
            "pred": pred,
            "embedding": z_fused,
            "projection": proj,
            "gate_weight": gate_weight,
            "z_cnn": z_cnn,
            "z_fm": z_fm,
            "threshold": self.decision_threshold.detach().clone(),
        }

    def set_threshold(self, threshold: float):
        self.decision_threshold.fill_(float(threshold))

    def tune_threshold(self, y_val: Iterable[int], y_val_prob: Iterable[float], objective: str = "f1", min_precision: Optional[float] = None):
        state = search_best_threshold(y_val, y_val_prob, objective=objective, min_precision=min_precision)
        if not np.isnan(state.threshold):
            self.set_threshold(state.threshold)
        self.threshold_state = state
        return state


# ============================================================
# LOSSES
# ============================================================

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.85, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.float().view(-1)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)

        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.10, fraud_anchor_weight: float = 2.0, eps: float = 1e-8):
        super().__init__()
        self.temperature = float(temperature)
        self.fraud_anchor_weight = float(fraud_anchor_weight)
        self.eps = float(eps)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError("features must have shape (B, D).")

        device = features.device
        labels = labels.view(-1).long()
        features = F.normalize(features, dim=1)

        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        b = features.size(0)
        self_mask = torch.eye(b, dtype=torch.bool, device=device)

        positive_mask = labels.view(-1, 1).eq(labels.view(1, -1))
        positive_mask = positive_mask.masked_fill(self_mask, False).float()

        contrast_mask = (~self_mask).float()
        exp_logits = torch.exp(logits) * contrast_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)

        positive_count = positive_mask.sum(dim=1)
        valid_anchor = positive_count > 0

        if valid_anchor.sum() == 0:
            return torch.zeros((), device=device, dtype=features.dtype)

        loss_per_anchor = -(positive_mask * log_prob).sum(dim=1) / (positive_count + self.eps)
        loss_per_anchor = loss_per_anchor[valid_anchor]

        valid_labels = labels[valid_anchor]
        weights = torch.ones_like(loss_per_anchor)
        weights = torch.where(valid_labels == 1, torch.full_like(weights, self.fraud_anchor_weight), weights)

        return (loss_per_anchor * weights).sum() / weights.sum().clamp_min(self.eps)


class FocalSupConLoss(nn.Module):
    def __init__(
        self,
        focal_alpha: float = 0.85,
        focal_gamma: float = 2.0,
        lambda_supcon: float = 0.05,
        temperature: float = 0.10,
        fraud_anchor_weight: float = 2.0,
    ):
        super().__init__()
        self.lambda_supcon = float(lambda_supcon)
        self.focal = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.supcon = SupervisedContrastiveLoss(temperature=temperature, fraud_anchor_weight=fraud_anchor_weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        cls_loss = self.focal(logits, labels)

        if features is None or self.lambda_supcon <= 0:
            supcon_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        else:
            supcon_loss = self.supcon(features, labels)

        total = cls_loss + self.lambda_supcon * supcon_loss

        return {
            "loss": total,
            "classification_loss": cls_loss.detach(),
            "supcon_loss": supcon_loss.detach(),
        }


# ============================================================
# METRICS AND THRESHOLD
# ============================================================

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


def safe_arrays(y_true: Iterable[int], y_prob: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(list(y_true)).astype(int)
    y_prob = np.asarray(list(y_prob)).astype(float)
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    return y_true, y_prob


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> ThresholdState:
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
    y_true, y_prob = safe_arrays(y_true, y_prob)

    thresholds = np.unique(
        np.concatenate([
            np.linspace(0.001, 0.999, n_grid),
            np.quantile(y_prob, np.linspace(0.001, 0.999, 300)),
        ])
    )

    best = None
    best_score = -np.inf

    for th in thresholds:
        st = metrics_at_threshold(y_true, y_prob, float(th))

        if min_precision is not None and st.val_precision < min_precision:
            continue

        if objective == "mcc":
            score = st.val_mcc
        elif objective == "recall":
            score = st.val_recall
        elif objective == "precision_constrained_recall":
            score = st.val_recall
        else:
            score = st.val_f1

        if score > best_score:
            best_score = score
            best = st

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


def recall_at_precision(y_true: Iterable[int], y_prob: Iterable[float], min_precision: float = 0.80) -> Dict[str, float]:
    y_true, y_prob = safe_arrays(y_true, y_prob)
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


def evaluate_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float, min_precision: float = 0.80) -> Dict[str, float]:
    y_true, y_prob = safe_arrays(y_true, y_prob)
    st = metrics_at_threshold(y_true, y_prob, threshold)
    rap = recall_at_precision(y_true, y_prob, min_precision=min_precision)

    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = np.nan

    return {
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
        "ROC_AUC": float(roc),
        "Fraud_Precision": st.val_precision,
        "Fraud_Recall": st.val_recall,
        "Fraud_F1": st.val_f1,
        "MCC": st.val_mcc,
        f"Recall@Precision>={min_precision:.2f}": float(rap["recall"]),
        f"Threshold@Precision>={min_precision:.2f}": float(rap["threshold"]),
        "Selected_Threshold": float(threshold),
        "TN": st.tn,
        "FP": st.fp,
        "FN": st.fn,
        "TP": st.tp,
    }


# ============================================================
# TRAINING
# ============================================================

def build_model(cfg: TrainConfig, dims: Dict) -> CNNDeepFMMoE:
    cnn = CNNMixBranch(
        tabular_dim=dims["cnn_dim"],
        embed_dim=cfg.cnn_embed_dim,
        conv_channels=cfg.cnn_conv_channels,
        kernel_size=cfg.cnn_kernel_size,
        bilinear_rank=cfg.cnn_bilinear_rank,
        out_dim=cfg.cnn_out_dim,
        seq_length=cfg.cnn_seq_length,
        dropout=cfg.dropout,
    )

    deepfm = DeepFMMixBranch(
        categorical_cardinalities=dims["categorical_cardinalities"],
        num_numerical=dims["num_dim"],
        embed_dim=cfg.deepfm_embed_dim,
        hidden_dims=[cfg.deepfm_hidden_1, cfg.deepfm_hidden_2],
        dense_num_fields=cfg.deepfm_dense_num_fields,
        branch_out_dim=cfg.deepfm_branch_out_dim,
        dropout=cfg.dropout,
    )

    model = CNNDeepFMMoE(
        cnn_branch=cnn,
        deepfm_branch=deepfm,
        cnn_dim=cfg.cnn_out_dim,
        deepfm_dim=cfg.deepfm_branch_out_dim,
        fusion_dim=cfg.fusion_dim,
        gate_hidden_dim=cfg.gate_hidden_dim,
        dropout=cfg.dropout,
    )

    return model


def run_epoch(model, loader, criterion, optimizer, device: str) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_cls = 0.0
    total_sup = 0.0
    n = 0

    for batch in loader:
        x_cnn = batch["x_cnn"].to(device)
        x_cat = batch["x_cat"].to(device)
        x_num = batch["x_num"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)

        out = model(x_cnn=x_cnn, x_cat=x_cat, x_num=x_num, return_dict=True)
        losses = criterion(out["logits"], y, out["projection"])

        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = y.size(0)
        total_loss += float(losses["loss"].detach().cpu()) * bs
        total_cls += float(losses["classification_loss"].detach().cpu()) * bs
        total_sup += float(losses["supcon_loss"].detach().cpu()) * bs
        n += bs

    return {
        "loss": total_loss / max(n, 1),
        "classification_loss": total_cls / max(n, 1),
        "supcon_loss": total_sup / max(n, 1),
    }


@torch.no_grad()
def predict_loader(model, loader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    labels = []

    for batch in loader:
        x_cnn = batch["x_cnn"].to(device)
        x_cat = batch["x_cat"].to(device)
        x_num = batch["x_num"].to(device)
        y = batch["y"].cpu().numpy().astype(int)

        out = model(x_cnn=x_cnn, x_cat=x_cat, x_num=x_num, return_dict=True)
        p = out["prob"].detach().cpu().numpy()

        probs.append(p)
        labels.append(y)

    return np.concatenate(labels), np.concatenate(probs)


def train_unified_model(cfg: TrainConfig) -> Dict[str, object]:
    seed_everything(cfg.random_state)

    processed_dir = Path(cfg.processed_dir)
    output_dir = processed_dir / cfg.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, dims = build_loaders(cfg)

    print("DIMS:", json.dumps({k: v for k, v in dims.items() if k != "categorical_cardinalities"}, indent=2))
    print("Num categorical fields:", len(dims["categorical_cardinalities"]))
    print("Device:", cfg.device)

    model = build_model(cfg, dims).to(cfg.device)

    criterion = FocalSupConLoss(
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
        lambda_supcon=cfg.lambda_supcon,
        temperature=cfg.supcon_temperature,
        fraud_anchor_weight=cfg.fraud_anchor_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_pr_auc = -np.inf
    best_epoch = -1
    bad_epochs = 0
    history = []

    best_path = output_dir / "best_unified_cnn_deepfm_moe.pt"

    for epoch in range(1, cfg.epochs + 1):
        start = time.time()

        train_stats = run_epoch(model, train_loader, criterion, optimizer, cfg.device)
        y_val, p_val = predict_loader(model, val_loader, cfg.device)

        val_pr_auc = average_precision_score(y_val, p_val)
        val_roc_auc = roc_auc_score(y_val, p_val)

        scheduler.step(val_pr_auc)

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_cls_loss": train_stats["classification_loss"],
            "train_supcon_loss": train_stats["supcon_loss"],
            "val_PR_AUC": float(val_pr_auc),
            "val_ROC_AUC": float(val_roc_auc),
            "time_seconds": round(time.time() - start, 2),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={row['train_loss']:.5f} | "
            f"cls={row['train_cls_loss']:.5f} | "
            f"supcon={row['train_supcon_loss']:.5f} | "
            f"val_PR_AUC={row['val_PR_AUC']:.5f} | "
            f"val_ROC_AUC={row['val_ROC_AUC']:.5f} | "
            f"time={row['time_seconds']}s"
        )

        if val_pr_auc > best_pr_auc:
            best_pr_auc = float(val_pr_auc)
            best_epoch = epoch
            bad_epochs = 0
            if cfg.save_best_model:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": asdict(cfg),
                        "dims": dims,
                        "best_epoch": best_epoch,
                        "best_val_pr_auc": best_pr_auc,
                    },
                    best_path,
                )
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
            break

    if cfg.save_best_model and best_path.exists():
        ckpt = torch.load(best_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model_state_dict"])

    y_val, p_val = predict_loader(model, val_loader, cfg.device)
    threshold_state = model.tune_threshold(
        y_val=y_val,
        y_val_prob=p_val,
        objective=cfg.threshold_objective,
        min_precision=cfg.threshold_min_precision,
    )

    y_test, p_test = predict_loader(model, test_loader, cfg.device)
    test_metrics = evaluate_metrics(
        y_true=y_test,
        y_prob=p_test,
        threshold=float(model.decision_threshold.detach().cpu()),
        min_precision=cfg.recall_at_precision_target,
    )

    result = {
        "Model": "Unified CNNMix + DeepFMMix + MoE + Focal + SupCon",
        "Best_Epoch": best_epoch,
        "Best_Val_PR_AUC": best_pr_auc,
        "Threshold_Objective": cfg.threshold_objective,
        "Threshold_Min_Precision": cfg.threshold_min_precision,
        **test_metrics,
        "Train_Fraud_Ratio": dims["train_fraud_ratio"],
        "Val_Fraud_Ratio": dims["val_fraud_ratio"],
        "Test_Fraud_Ratio": dims["test_fraud_ratio"],
    }

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    result_df = pd.DataFrame([result])
    result_df.to_csv(output_dir / "unified_moe_test_metrics.csv", index=False)

    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_prob": p_test,
        "y_pred": (p_test >= result["Selected_Threshold"]).astype(int),
    })
    pred_df.to_csv(output_dir / "unified_moe_test_predictions.csv", index=False)

    with open(output_dir / "unified_moe_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    print("\nFINAL TEST METRICS")
    print(result_df.T)

    return {
        "model": model,
        "result": result,
        "history": history,
        "output_dir": str(output_dir),
        "best_model_path": str(best_path),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    cfg = TrainConfig(
        processed_dir=r"D:\project\data\merge_paper_ready",
        epochs=20,
        patience=5,
        batch_size=1024,
        lambda_supcon=0.05,
        threshold_objective="f1",
        threshold_min_precision=None,
        sample_train_n=None,
    )

    train_unified_model(cfg)
