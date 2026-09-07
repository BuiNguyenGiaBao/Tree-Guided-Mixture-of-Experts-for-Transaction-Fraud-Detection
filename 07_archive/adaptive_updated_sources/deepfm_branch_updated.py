"""
deepfm_branch_updated.py

Updated DeepFM module for two purposes:
1) standalone DeepFM classification, and
2) the second branch in an adaptive CNN-DeepFM fusion model.

Main DeepFM branch flow:
Categorical / numerical / dense input -> field embeddings ->
Linear + FM second-order interaction + Deep MLP -> branch representation.

The file also includes:
- AdaptiveGatedFusion
- AdaptiveCNNDeepFM, which combines a CNN branch and a DeepFM branch
- FocalLoss for imbalanced fraud detection
- SupervisedContrastiveLoss for representation separation
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizationMachine(nn.Module):
    """Second-order factorization machine interaction."""

    def __init__(self):
        super().__init__()

    def forward(self, emb: torch.Tensor, values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            emb: Field embeddings with shape (B, F, K).
            values: Optional field values with shape (B, F, 1).
        """
        if values is not None:
            emb = emb * values

        sum_emb = emb.sum(dim=1)                 # (B, K)
        sum_square = sum_emb * sum_emb           # (B, K)
        square_sum = (emb * emb).sum(dim=1)      # (B, K)
        fm = 0.5 * (sum_square - square_sum).sum(dim=1, keepdim=True)  # (B, 1)
        return fm


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        d = int(in_dim)
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = int(h)
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepFM(nn.Module):
    """
    DeepFM with an explicit branch representation.

    Standalone behavior:
        logits = deepfm(cat_x, num_x, dense_x)

    Branch behavior:
        z_fm = deepfm.extract_embedding(cat_x, num_x, dense_x)

    The branch embedding is:
        concat(linear_out, fm_out, deep_hidden)
    optionally projected to branch_out_dim.
    """

    def __init__(
        self,
        num_classes: int,
        categorical_cardinalities: Optional[List[int]] = None,
        num_numerical: int = 0,
        embed_dim: int = 16,
        deep_hidden: Optional[List[int]] = None,
        dropout: float = 0.2,
        dense_in_dim: Optional[int] = None,
        dense_num_fields: int = 4,
        use_bias: bool = True,
        branch_out_dim: Optional[int] = None,
    ):
        super().__init__()

        deep_hidden = deep_hidden or [128, 64]

        self.num_classes = int(num_classes)
        self.categorical_cardinalities = categorical_cardinalities or []
        self.num_cat = len(self.categorical_cardinalities)
        self.num_num = int(num_numerical)
        self.embed_dim = int(embed_dim)
        self.dense_in_dim = dense_in_dim
        self.dense_num_fields = int(dense_num_fields) if dense_in_dim is not None else 0

        # Linear part
        self.linear_cat = nn.ModuleList([nn.Embedding(card, 1) for card in self.categorical_cardinalities])
        self.linear_num = nn.Linear(self.num_num, 1, bias=False) if self.num_num > 0 else None
        self.linear_dense = nn.Linear(self.dense_in_dim, 1, bias=False) if self.dense_in_dim is not None else None
        self.linear_bias = nn.Parameter(torch.zeros(1)) if use_bias else None

        # FM part
        self.fm = FactorizationMachine()
        self.fm_cat_emb = nn.ModuleList([nn.Embedding(card, self.embed_dim) for card in self.categorical_cardinalities])
        self.fm_num_emb = nn.Parameter(torch.randn(self.num_num, self.embed_dim) * 0.01) if self.num_num > 0 else None
        self.fm_dense_proj = (
            nn.Linear(self.dense_in_dim, self.embed_dim * self.dense_num_fields, bias=False)
            if self.dense_in_dim is not None else None
        )

        # Deep part
        total_fields = self.num_cat + self.num_num + self.dense_num_fields
        if total_fields <= 0:
            raise ValueError("DeepFM requires at least one field.")
        deep_in = self.embed_dim * total_fields
        self.mlp = MLP(deep_in, deep_hidden, dropout=dropout)

        raw_representation_dim = 2 + self.mlp.out_dim  # linear_out + fm_out + deep hidden
        self.raw_representation_dim = raw_representation_dim
        self.output_dim = int(branch_out_dim) if branch_out_dim is not None else raw_representation_dim

        self.branch_projection = None
        if branch_out_dim is not None:
            self.branch_projection = nn.Sequential(
                nn.Linear(raw_representation_dim, self.output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # Standalone classifier head based on branch representation
        standalone_out_dim = 1 if self.num_classes == 2 else self.num_classes
        self.classifier = nn.Linear(self.output_dim, standalone_out_dim)

    def _build_field_embeddings(
        self,
        cat_x: Optional[torch.Tensor],
        num_x: Optional[torch.Tensor],
        dense_x: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embs: List[torch.Tensor] = []
        vals: List[torch.Tensor] = []

        # Categorical fields
        if self.num_cat > 0:
            if cat_x is None:
                raise ValueError("cat_x is required because categorical_cardinalities is not empty.")
            if cat_x.dim() != 2 or cat_x.size(1) != self.num_cat:
                raise ValueError(f"cat_x must be (B, {self.num_cat}).")

            for j, emb_layer in enumerate(self.fm_cat_emb):
                e = emb_layer(cat_x[:, j])       # (B, K)
                embs.append(e.unsqueeze(1))      # (B, 1, K)
                vals.append(torch.ones(e.size(0), 1, 1, device=e.device, dtype=e.dtype))

        # Numerical fields
        if self.num_num > 0:
            if num_x is None:
                raise ValueError("num_x is required because num_numerical > 0.")
            if num_x.dim() != 2 or num_x.size(1) != self.num_num:
                raise ValueError(f"num_x must be (B, {self.num_num}).")

            e_num = self.fm_num_emb.unsqueeze(0).expand(num_x.size(0), -1, -1)  # (B, N, K)
            embs.append(e_num)
            vals.append(num_x.unsqueeze(-1))                                    # (B, N, 1)

        # Dense input is split into multiple FM fields so FM can learn interactions.
        if self.dense_in_dim is not None:
            if dense_x is None:
                raise ValueError("dense_x is required because dense_in_dim is set.")
            if dense_x.dim() != 2 or dense_x.size(1) != self.dense_in_dim:
                raise ValueError(f"dense_x must be (B, {self.dense_in_dim}).")

            e_dense = self.fm_dense_proj(dense_x)  # (B, dense_num_fields * K)
            e_dense = e_dense.view(dense_x.size(0), self.dense_num_fields, self.embed_dim)
            embs.append(e_dense)
            vals.append(torch.ones(dense_x.size(0), self.dense_num_fields, 1, device=dense_x.device, dtype=dense_x.dtype))

        if len(embs) == 0:
            raise ValueError("At least one of (categorical, numerical, dense) must be provided.")

        field_emb = torch.cat(embs, dim=1)  # (B, F_total, K)
        values = torch.cat(vals, dim=1)     # (B, F_total, 1)
        return field_emb, values

    def _linear_part(
        self,
        cat_x: Optional[torch.Tensor],
        num_x: Optional[torch.Tensor],
        dense_x: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if cat_x is not None:
            batch_size, device = cat_x.size(0), cat_x.device
        elif num_x is not None:
            batch_size, device = num_x.size(0), num_x.device
        elif dense_x is not None:
            batch_size, device = dense_x.size(0), dense_x.device
        else:
            raise ValueError("At least one of cat_x, num_x, dense_x must be provided.")

        out = torch.zeros(batch_size, 1, device=device)

        if self.num_cat > 0 and cat_x is not None:
            for j, emb1 in enumerate(self.linear_cat):
                out = out + emb1(cat_x[:, j])

        if self.linear_num is not None and num_x is not None:
            out = out + self.linear_num(num_x)

        if self.linear_dense is not None and dense_x is not None:
            out = out + self.linear_dense(dense_x)

        if self.linear_bias is not None:
            out = out + self.linear_bias

        return out

    def extract_embedding(
        self,
        cat_x: Optional[torch.Tensor] = None,
        num_x: Optional[torch.Tensor] = None,
        dense_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the DeepFM branch representation z_fm."""
        linear_out = self._linear_part(cat_x, num_x, dense_x)              # (B, 1)
        field_emb, values = self._build_field_embeddings(cat_x, num_x, dense_x)
        fm_out = self.fm(field_emb, values)                                # (B, 1)

        deep_in = field_emb.reshape(field_emb.size(0), -1)                 # (B, F*K)
        deep_h = self.mlp(deep_in)                                         # (B, H)

        z_fm = torch.cat([linear_out, fm_out, deep_h], dim=1)              # (B, 2 + H)
        if self.branch_projection is not None:
            z_fm = self.branch_projection(z_fm)                            # (B, branch_out_dim)
        return z_fm

    def forward(
        self,
        cat_x: Optional[torch.Tensor] = None,
        num_x: Optional[torch.Tensor] = None,
        dense_x: Optional[torch.Tensor] = None,
        return_embedding: bool = False,
    ):
        z_fm = self.extract_embedding(cat_x=cat_x, num_x=num_x, dense_x=dense_x)
        logits = self.classifier(z_fm)
        if return_embedding:
            return logits, z_fm
        return logits


class AdaptiveGatedFusion(nn.Module):
    """Adaptive fusion: z = g * z_cnn + (1 - g) * z_fm."""

    def __init__(self, fusion_dim: int, dropout: float = 0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )

    def forward(self, z_cnn: torch.Tensor, z_fm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if z_cnn.shape != z_fm.shape:
            raise ValueError(f"z_cnn and z_fm must have the same shape, got {z_cnn.shape} and {z_fm.shape}.")
        g = self.gate(torch.cat([z_cnn, z_fm], dim=1))
        z = g * z_cnn + (1.0 - g) * z_fm
        return z, g


class AdaptiveCNNDeepFM(nn.Module):
    """
    Paper-level two-branch model:
    CNN branch + DeepFM branch + adaptive gated fusion + classifier.

    This class accepts a CNN branch object and a DeepFM branch object, so it does
    not depend on a specific file name for the CNN module.
    """

    def __init__(
        self,
        cnn_branch: nn.Module,
        deepfm_branch: DeepFM,
        cnn_dim: int,
        fm_dim: int,
        fusion_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        projection_dim: Optional[int] = 64,
    ):
        super().__init__()
        self.cnn_branch = cnn_branch
        self.deepfm_branch = deepfm_branch
        self.num_classes = int(num_classes)

        self.proj_cnn = nn.Sequential(
            nn.Linear(cnn_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proj_fm = nn.Sequential(
            nn.Linear(fm_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fusion = AdaptiveGatedFusion(fusion_dim=fusion_dim, dropout=dropout)
        self.classifier = nn.Linear(fusion_dim, 1 if self.num_classes == 2 else self.num_classes)

        self.projection_head = None
        if projection_dim is not None:
            self.projection_head = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, projection_dim),
            )

    def forward(
        self,
        dense_cnn_x: torch.Tensor,
        cat_x: Optional[torch.Tensor] = None,
        num_x: Optional[torch.Tensor] = None,
        dense_fm_x: Optional[torch.Tensor] = None,
        return_embedding: bool = False,
        return_gate: bool = False,
        return_projection: bool = False,
    ):
        # CNN branch representation
        if hasattr(self.cnn_branch, "encode"):
            z_cnn = self.cnn_branch.encode(dense_cnn_x, return_attention=False)
        elif hasattr(self.cnn_branch, "compute_embedding"):
            z_cnn = self.cnn_branch.compute_embedding(dense_cnn_x)
        else:
            raise AttributeError("cnn_branch must provide encode() or compute_embedding().")

        # DeepFM branch representation
        z_fm = self.deepfm_branch.extract_embedding(cat_x=cat_x, num_x=num_x, dense_x=dense_fm_x)

        z_cnn = self.proj_cnn(z_cnn)
        z_fm = self.proj_fm(z_fm)
        z_fused, gate = self.fusion(z_cnn, z_fm)

        logits = self.classifier(z_fused)

        outputs = [logits]
        if return_embedding:
            outputs.append(z_fused)
        if return_gate:
            outputs.append(gate)
        if return_projection:
            if self.projection_head is None:
                raise ValueError("projection_head is disabled because projection_dim=None.")
            outputs.append(F.normalize(self.projection_head(z_fused), dim=1))

        if len(outputs) == 1:
            return logits
        return tuple(outputs)


class FocalLoss(nn.Module):
    """Binary/multiclass focal loss for imbalanced classification."""

    def __init__(self, alpha: Optional[float] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.size(-1) == 1:
            targets_float = targets.float().view(-1, 1)
            bce = F.binary_cross_entropy_with_logits(logits, targets_float, reduction="none")
            probs = torch.sigmoid(logits)
            pt = probs * targets_float + (1.0 - probs) * (1.0 - targets_float)
            loss = (1.0 - pt).pow(self.gamma) * bce
            if self.alpha is not None:
                alpha_t = self.alpha * targets_float + (1.0 - self.alpha) * (1.0 - targets_float)
                loss = alpha_t * loss
        else:
            ce = F.cross_entropy(logits, targets.long(), reduction="none")
            pt = torch.exp(-ce)
            loss = (1.0 - pt).pow(self.gamma) * ce
            if self.alpha is not None:
                loss = self.alpha * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss using labels.

    Use with the normalized projection returned by:
        logits, z, projection = model(..., return_embedding=True, return_projection=True)
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError("features must have shape (B, D).")

        device = features.device
        labels = labels.view(-1, 1)
        features = F.normalize(features, dim=1)

        batch_size = features.size(0)
        similarity = torch.matmul(features, features.T) / self.temperature

        # Remove self-comparison for numerical stability.
        logits_mask = torch.ones_like(similarity, device=device) - torch.eye(batch_size, device=device)
        positive_mask = (labels == labels.T).float().to(device) * logits_mask

        similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)

        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positive_count + self.eps)
        loss = -mean_log_prob_pos[valid].mean()
        return loss


# Alias for clearer naming when used only as the second branch.
DeepFMBranch = DeepFM
