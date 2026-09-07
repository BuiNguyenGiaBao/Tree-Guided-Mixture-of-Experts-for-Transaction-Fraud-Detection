"""
cnn_branch_updated.py

CNN branch for tabular data.
Updated from the original TabularCNNNetwork so it can be used as:
1) a standalone classifier, or
2) the first branch in an adaptive CNN-DeepFM fusion model.

Main flow:
Tabular input -> embedding -> pseudo-sequence projection -> Conv1D -> Attention pooling
-> Low-rank bilinear self-interaction -> branch embedding -> classifier.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Attention pooling over the pseudo-sequence dimension."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor with shape (batch, channels, length).

        Returns:
            pooled: Tensor with shape (batch, channels).
            attn_weights: Tensor with shape (batch, length, 1).
        """
        x_t = x.transpose(1, 2)                     # (B, L, C)
        attn_scores = self.attention(x_t)           # (B, L, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.sum(x * attn_weights.transpose(1, 2), dim=2)  # (B, C)
        return pooled, attn_weights


class LowRankBilinear(nn.Module):
    """Low-rank bilinear interaction module."""

    def __init__(self, in1_features: int, in2_features: int, out_features: int, rank: int):
        super().__init__()
        self.rank = int(rank)
        self.U1 = nn.Linear(in1_features, rank, bias=False)
        self.U2 = nn.Linear(in2_features, rank, bias=False)
        self.V = nn.Linear(rank, out_features)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        u1 = self.U1(x1)                # (B, rank)
        u2 = self.U2(x2)                # (B, rank)
        interaction = u1 * u2           # Hadamard product, (B, rank)
        output = self.V(interaction)    # (B, out_features)
        return output


class TabularCNNNetwork(nn.Module):
    """
    CNN branch for tabular classification and feature extraction.

    This class keeps the original standalone behavior:
        logits, attn_weights = model(x)

    New branch behavior:
        z_cnn, attn_weights = model.encode(x, return_attention=True)
        z_cnn = model.get_embedding(x)
    """

    def __init__(
        self,
        tabular_dim: int,
        embed_dim: int,
        conv_channels: int,
        kernel_size: int,
        bilinear_rank: int,
        bilinear_out_dim: int,
        num_classes: int,
        seq_length: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.tabular_dim = int(tabular_dim)
        self.embed_dim = int(embed_dim)
        self.seq_length = int(seq_length)
        self.conv_channels = int(conv_channels)
        self.bilinear_rank = int(bilinear_rank)
        self.output_dim = int(bilinear_out_dim)
        self.num_classes = int(num_classes)

        # 1. Tabular embedding layer
        self.tabular_embed = nn.Sequential(
            nn.Linear(self.tabular_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 2. Project a tabular vector into a pseudo-sequence
        self.feature_projection = nn.Linear(self.embed_dim, self.embed_dim * self.seq_length)

        # 3. CNN layers over the pseudo-sequence
        self.conv1 = nn.Conv1d(self.embed_dim, self.conv_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(self.conv_channels)
        self.conv2 = nn.Conv1d(self.conv_channels, self.conv_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(self.conv_channels)

        # 4. Attention pooling
        self.attention_pool = AttentionPooling(self.conv_channels)

        # 5. Low-rank bilinear self-interaction
        self.low_rank_bilinear = LowRankBilinear(
            in1_features=self.conv_channels,
            in2_features=self.conv_channels,
            out_features=self.output_dim,
            rank=self.bilinear_rank,
        )

        # 6. Standalone classification head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.output_dim, self.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor, return_attention: bool = False):
        """
        Extract the CNN branch representation.

        Args:
            x: Dense tabular tensor with shape (batch, tabular_dim).
            return_attention: If True, also return attention weights.

        Returns:
            z_cnn: Tensor with shape (batch, bilinear_out_dim).
            attn_weights: Optional tensor with shape (batch, seq_length, 1).
        """
        if x.dim() != 2 or x.size(1) != self.tabular_dim:
            raise ValueError(f"x must have shape (B, {self.tabular_dim}).")

        batch_size = x.size(0)

        embedded = self.tabular_embed(x)                             # (B, embed_dim)
        projected = self.feature_projection(embedded)                # (B, embed_dim * seq_length)
        seq = projected.view(batch_size, self.seq_length, self.embed_dim)
        seq = seq.transpose(1, 2)                                    # (B, embed_dim, seq_length)

        conv1_out = F.relu(self.bn1(self.conv1(seq)))                # (B, conv_channels, seq_length)
        conv2_out = F.relu(self.bn2(self.conv2(conv1_out)))          # (B, conv_channels, seq_length)

        pooled, attn_weights = self.attention_pool(conv2_out)        # (B, conv_channels)
        z_cnn = self.low_rank_bilinear(pooled, pooled)               # (B, bilinear_out_dim)
        z_cnn = F.relu(z_cnn)

        if return_attention:
            return z_cnn, attn_weights
        return z_cnn

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """
        Standalone classifier forward pass.

        If return_embedding=True, returns (logits, z_cnn, attn_weights).
        Otherwise, returns (logits, attn_weights) for backward compatibility.
        """
        z_cnn, attn_weights = self.encode(x, return_attention=True)
        logits = self.output(self.dropout(z_cnn))

        if return_embedding:
            return logits, z_cnn, attn_weights
        return logits, attn_weights

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            _, attn_weights = self.forward(x)
        return attn_weights.squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x, return_attention=False)

    def get_embedding(self, x: torch.Tensor, detach: bool = True) -> torch.Tensor:
        if detach:
            self.eval()
            with torch.no_grad():
                return self.compute_embedding(x)
        return self.compute_embedding(x)


# Alias for clearer paper-level naming.
TabularCNNBranch = TabularCNNNetwork
