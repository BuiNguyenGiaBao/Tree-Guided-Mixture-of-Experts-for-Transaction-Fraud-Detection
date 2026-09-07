from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_branch_updated import TabularCNNBranch
from deepfm_branch_updated import DeepFMBranch


# ============================================================
# Proposed model:
# Tree-guided CNNMix + DeepFMMix + MoE + KD + Focal + SupCon
# ============================================================


@dataclass
class ProposedModelConfig:
    # CNNMix branch
    cnn_embed_dim: int = 128
    cnn_conv_channels: int = 128
    cnn_kernel_size: int = 3
    cnn_bilinear_rank: int = 32
    cnn_out_dim: int = 128
    cnn_seq_length: int = 10

    # DeepFMMix branch
    deepfm_embed_dim: int = 16
    deepfm_dense_num_fields: int = 8
    deepfm_branch_out_dim: int = 128
    deepfm_hidden: Tuple[int, int] = (256, 128)

    # MoE fusion
    fusion_dim: int = 128
    gate_hidden_dim: int = 64
    dropout: float = 0.30

    # Loss
    focal_alpha: float = 0.85
    focal_gamma: float = 2.0
    kd_weight: float = 0.30
    kd_temperature: float = 2.0
    lambda_supcon: float = 0.01
    supcon_temperature: float = 0.10


class DeepFMCompatWrapper(nn.Module):
    """
    Adapter để gọi DeepFMBranch bằng cùng interface với MoE.
    """

    def __init__(self, deepfm: nn.Module):
        super().__init__()
        self.deepfm = deepfm

    def extract_embedding(
        self,
        x_cat: Optional[torch.Tensor] = None,
        x_dense: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.deepfm.extract_embedding(
            cat_x=x_cat,
            num_x=None,
            dense_x=x_dense,
        )

    def forward(
        self,
        x_cat: Optional[torch.Tensor] = None,
        x_dense: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.extract_embedding(x_cat=x_cat, x_dense=x_dense)


class TreeGuidedCNNDeepFMMoE(nn.Module):
    """
    Proposed architecture:

    CNNMix branch
    + DeepFMMix branch
    + Tree-guided MoE gate using teacher_logit
    + Binary classifier

    Inputs:
        x_cnn: dense numeric tensor for CNNMix branch, shape [B, cnn_dim]
        x_cat: categorical code tensor for DeepFM, shape [B, n_cat]
        x_dense: dense numeric tensor for DeepFM dense side, shape [B, deepfm_num_dim]
        teacher_logit: teacher tree logit, shape [B]

    Output dict:
        logits
        prob
        embedding
        gate_weight
        z_cnn
        z_deepfm
    """

    def __init__(
        self,
        cnn_branch: nn.Module,
        deepfm_branch: nn.Module,
        cnn_dim: int = 128,
        deepfm_dim: int = 128,
        fusion_dim: int = 128,
        gate_hidden_dim: int = 64,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.cnn_branch = cnn_branch
        self.deepfm_branch = deepfm_branch

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.deepfm_proj = nn.Sequential(
            nn.Linear(deepfm_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # +1 is the teacher logit guidance.
        gate_input_dim = fusion_dim * 2 + 1

        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
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

    def forward(
        self,
        x_cnn: torch.Tensor,
        x_cat: torch.Tensor,
        x_dense: torch.Tensor,
        teacher_logit: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        _, z_cnn_raw, _ = self.cnn_branch(x_cnn, return_embedding=True)
        z_deepfm_raw = self.deepfm_branch.extract_embedding(
            x_cat=x_cat,
            x_dense=x_dense,
        )

        z_cnn = self.cnn_proj(z_cnn_raw)
        z_deepfm = self.deepfm_proj(z_deepfm_raw)

        if teacher_logit is None:
            teacher_logit = torch.zeros(
                z_cnn.size(0),
                dtype=z_cnn.dtype,
                device=z_cnn.device,
            )

        teacher_guide = teacher_logit.view(-1, 1).to(dtype=z_cnn.dtype)

        gate_input = torch.cat(
            [z_cnn, z_deepfm, teacher_guide],
            dim=1,
        )

        gate_weight = F.softmax(self.gate(gate_input), dim=1)

        # Adaptive expert fusion.
        z_fused = gate_weight[:, 0:1] * z_cnn + gate_weight[:, 1:2] * z_deepfm

        logits = self.classifier(z_fused).view(-1)
        prob = torch.sigmoid(logits)

        if not return_dict:
            return logits

        return {
            "logits": logits,
            "prob": prob,
            "embedding": z_fused,
            "gate_weight": gate_weight,
            "z_cnn": z_cnn,
            "z_deepfm": z_deepfm,
        }


# ============================================================
# Loss components
# ============================================================


class BinaryFocalLoss(nn.Module):
    """
    Binary focal loss for imbalanced fraud detection.
    """

    def __init__(
        self,
        alpha: float = 0.85,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.float().view(-1)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        prob = torch.sigmoid(logits)
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)

        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow(1.0 - pt, self.gamma)

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss for binary labels.
    """

    def __init__(
        self,
        temperature: float = 0.10,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if features is None:
            return torch.zeros((), device=labels.device)

        features = F.normalize(features, p=2, dim=1)
        labels = labels.view(-1, 1).long()

        batch_size = features.size(0)

        if batch_size <= 1:
            return torch.zeros((), device=features.device)

        similarity = torch.matmul(features, features.T) / self.temperature

        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)

        label_mask = torch.eq(labels, labels.T).float().to(features.device)
        positive_mask = label_mask * logits_mask

        # Stabilize softmax.
        similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()

        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)

        positives_per_row = positive_mask.sum(dim=1)
        valid = positives_per_row > 0

        if valid.sum() == 0:
            return torch.zeros((), device=features.device)

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positives_per_row + self.eps)
        loss = -mean_log_prob_pos[valid].mean()
        return loss


def kd_loss_with_logits(
    student_logits: torch.Tensor,
    teacher_prob: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Binary knowledge-distillation loss.

    teacher_prob is the soft probability produced by the tree teacher,
    usually LightGBM / CatBoost / XGBoost.
    """

    student_logits = student_logits.view(-1).float()
    teacher_prob = torch.clamp(teacher_prob.view(-1).float(), 1e-6, 1.0 - 1e-6)

    teacher_logit = torch.log(teacher_prob / (1.0 - teacher_prob))

    student_scaled = student_logits / temperature
    teacher_scaled_prob = torch.sigmoid(teacher_logit / temperature)

    return F.binary_cross_entropy_with_logits(
        student_scaled,
        teacher_scaled_prob,
    ) * (temperature ** 2)


def proposed_total_loss(
    outputs: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    teacher_prob: torch.Tensor,
    config: ProposedModelConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Full proposed loss:

    L = Focal(y, student)
        + kd_weight * KD(student, teacher)
        + lambda_supcon * SupCon(embedding, y)
    """

    logits = outputs["logits"].view(-1)
    y_true = y_true.float().view(-1)
    teacher_prob = teacher_prob.float().view(-1)

    focal_loss = BinaryFocalLoss(
        alpha=config.focal_alpha,
        gamma=config.focal_gamma,
    )(logits, y_true)

    kd = kd_loss_with_logits(
        student_logits=logits,
        teacher_prob=teacher_prob,
        temperature=config.kd_temperature,
    )

    supcon = SupervisedContrastiveLoss(
        temperature=config.supcon_temperature,
    )(outputs.get("embedding"), y_true)

    total = focal_loss + config.kd_weight * kd + config.lambda_supcon * supcon

    return total, {
        "total_loss": total.detach(),
        "focal_loss": focal_loss.detach(),
        "kd_loss": kd.detach(),
        "supcon_loss": supcon.detach(),
    }


# ============================================================
# Builder
# ============================================================


def build_proposed_tree_guided_model(
    cnn_input_dim: int,
    deepfm_dense_input_dim: int,
    categorical_cardinalities: List[int],
    config: Optional[ProposedModelConfig] = None,
) -> TreeGuidedCNNDeepFMMoE:
    """
    Build full proposed model.

    Args:
        cnn_input_dim:
            Number of numeric features used by the CNNMix branch.

        deepfm_dense_input_dim:
            Number of dense numeric features used by DeepFMMix dense side.

        categorical_cardinalities:
            List of cardinalities for categorical features.
            Values should include +1 or +2 if categorical codes reserve 0 for unknown.

        config:
            ProposedModelConfig.

    Returns:
        TreeGuidedCNNDeepFMMoE model.
    """

    if config is None:
        config = ProposedModelConfig()

    cnn_branch = TabularCNNBranch(
        tabular_dim=cnn_input_dim,
        embed_dim=config.cnn_embed_dim,
        conv_channels=config.cnn_conv_channels,
        kernel_size=config.cnn_kernel_size,
        bilinear_rank=config.cnn_bilinear_rank,
        bilinear_out_dim=config.cnn_out_dim,
        num_classes=1,
        seq_length=config.cnn_seq_length,
        dropout=config.dropout,
    )

    deepfm = DeepFMBranch(
        num_classes=2,
        categorical_cardinalities=categorical_cardinalities,
        num_numerical=0,
        embed_dim=config.deepfm_embed_dim,
        deep_hidden=list(config.deepfm_hidden),
        dropout=config.dropout,
        dense_in_dim=deepfm_dense_input_dim,
        dense_num_fields=config.deepfm_dense_num_fields,
        branch_out_dim=config.deepfm_branch_out_dim,
    )

    deepfm_branch = DeepFMCompatWrapper(deepfm)

    model = TreeGuidedCNNDeepFMMoE(
        cnn_branch=cnn_branch,
        deepfm_branch=deepfm_branch,
        cnn_dim=config.cnn_out_dim,
        deepfm_dim=config.deepfm_branch_out_dim,
        fusion_dim=config.fusion_dim,
        gate_hidden_dim=config.gate_hidden_dim,
        dropout=config.dropout,
    )

    return model


# ============================================================
# Minimal example
# ============================================================


if __name__ == "__main__":
    # Example only. Replace these dimensions with your processed data dimensions.
    batch_size = 8
    cnn_dim = 797
    dense_dim = 797
    categorical_cardinalities = [100, 50, 20]

    config = ProposedModelConfig()
    model = build_proposed_tree_guided_model(
        cnn_input_dim=cnn_dim,
        deepfm_dense_input_dim=dense_dim,
        categorical_cardinalities=categorical_cardinalities,
        config=config,
    )

    x_cnn = torch.randn(batch_size, cnn_dim)
    x_dense = torch.randn(batch_size, dense_dim)
    x_cat = torch.randint(0, 10, (batch_size, len(categorical_cardinalities)))
    teacher_logit = torch.randn(batch_size)
    teacher_prob = torch.sigmoid(teacher_logit)
    y = torch.randint(0, 2, (batch_size,)).float()

    outputs = model(
        x_cnn=x_cnn,
        x_cat=x_cat,
        x_dense=x_dense,
        teacher_logit=teacher_logit,
    )

    loss, logs = proposed_total_loss(
        outputs=outputs,
        y_true=y,
        teacher_prob=teacher_prob,
        config=config,
    )

    print("logits:", outputs["logits"].shape)
    print("prob:", outputs["prob"].shape)
    print("gate_weight:", outputs["gate_weight"].shape)
    print("loss:", float(loss))
    print({k: float(v) for k, v in logs.items()})
