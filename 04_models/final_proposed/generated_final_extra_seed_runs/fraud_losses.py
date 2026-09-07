from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.85,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.float().view(-1)

        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

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
    def __init__(
        self,
        temperature: float = 0.10,
        fraud_anchor_weight: float = 2.0,
        eps: float = 1e-8,
    ):
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

        batch_size = features.size(0)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

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
        weights = torch.where(
            valid_labels == 1,
            torch.full_like(weights, self.fraud_anchor_weight),
            weights,
        )

        return (loss_per_anchor * weights).sum() / weights.sum().clamp_min(self.eps)


class CombinedFocalSupConLoss(nn.Module):
    def __init__(
        self,
        focal_alpha: float = 0.85,
        focal_gamma: float = 2.0,
        lambda_supcon: float = 0.05,
        temperature: float = 0.10,
        fraud_anchor_weight: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lambda_supcon = float(lambda_supcon)
        self.focal = BinaryFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
        self.supcon = SupervisedContrastiveLoss(
            temperature=temperature,
            fraud_anchor_weight=fraud_anchor_weight,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cls_loss = self.focal(logits, labels)

        if features is None or self.lambda_supcon <= 0:
            supcon_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        else:
            supcon_loss = self.supcon(features, labels)

        total_loss = cls_loss + self.lambda_supcon * supcon_loss

        return {
            "loss": total_loss,
            "classification_loss": cls_loss.detach(),
            "supcon_loss": supcon_loss.detach(),
        }


def get_loss_preset(name: str = "focal_supcon") -> CombinedFocalSupConLoss:
    name = name.lower()

    if name == "focal":
        return CombinedFocalSupConLoss(
            focal_alpha=0.85,
            focal_gamma=2.0,
            lambda_supcon=0.0,
        )

    if name == "focal_supcon":
        return CombinedFocalSupConLoss(
            focal_alpha=0.85,
            focal_gamma=2.0,
            lambda_supcon=0.05,
            temperature=0.10,
            fraud_anchor_weight=2.0,
        )

    if name == "high_recall":
        return CombinedFocalSupConLoss(
            focal_alpha=0.90,
            focal_gamma=2.5,
            lambda_supcon=0.05,
            temperature=0.10,
            fraud_anchor_weight=2.5,
        )

    raise ValueError("name must be one of: focal, focal_supcon, high_recall.")
