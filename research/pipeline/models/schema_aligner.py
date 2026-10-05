"""
SchemaAligner — maps heterogeneous metadata into a fixed-size patient
embedding regardless of which fields the source dataset records.

Input (from stage 4 dataloader):
    meta      : (B, 19)  float, canonical layout
    meta_mask : (B, 19)  float, 1.0 = present, 0.0 = missing

Output:
    patient_emb : (B, ALIGNED_DIM)  float

Mechanism:
    For each feature index i:
        x_i = meta[:, i] if mask[:, i]==1 else absent_embed[i]
    Then MLP -> Linear -> patient_emb

The "absent_embed" is a learnable scalar per feature: distinct from
zero, so the downstream model can tell "feature missing" apart from
"feature present and equal to zero" (which the dataloader documents
as a real distinction).

Per-feature dropout (used by V11) randomly zeroes the mask of some
features during training, forcing the model to be robust to deployment
configurations that omit certain fields.
"""
from __future__ import annotations

import torch
import torch.nn as nn


META_DIM: int = 19          # MUST match dataloader META_DIM
ALIGNED_DIM: int = 64       # canonical patient-embedding width


class SchemaAligner(nn.Module):
    def __init__(self, in_dim: int = META_DIM, out_dim: int = ALIGNED_DIM,
                 hidden_dim: int = 128, dropout: float = 0.1,
                 feature_dropout_p: float = 0.0) -> None:
        super().__init__()
        if in_dim != META_DIM:
            raise ValueError(f"in_dim {in_dim} != META_DIM {META_DIM}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.feature_dropout_p = float(feature_dropout_p)

        # one scalar per feature: the "absent" value
        self.absent_embed = nn.Parameter(torch.zeros(in_dim))

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, meta: torch.Tensor, meta_mask: torch.Tensor
                 ) -> torch.Tensor:
        if meta.shape[-1] != self.in_dim or meta_mask.shape[-1] != self.in_dim:
            raise ValueError(
                f"SchemaAligner expected last dim {self.in_dim}; "
                f"got meta={tuple(meta.shape)} mask={tuple(meta_mask.shape)}"
            )

        # Per-feature dropout (only during training, only if p > 0)
        if self.training and self.feature_dropout_p > 0.0:
            keep = (torch.rand_like(meta_mask) > self.feature_dropout_p).float()
            meta_mask = meta_mask * keep

        # blend: where present use meta; where absent use the learned scalar
        absent = self.absent_embed.view(1, -1).expand_as(meta)
        x = meta * meta_mask + absent * (1.0 - meta_mask)
        return self.proj(x)
