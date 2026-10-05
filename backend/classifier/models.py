"""DermaLens Hybrid Classifier Architecture
=========================================
Dual-backbone deep learning (EfficientNet-B4 + Swin Transformer V2)
with clinical patient metadata MLP fusion and optional auxiliary supervision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm
from backend.classifier.config import NUM_CLASSES


class DermaLensHybrid(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        meta_dim: int = 13,
        mode: str = "full",
        pretrained: bool = True,
    ):
        """
        Mode options:
          'effnet_only' — Single-branch EfficientNet-B4 baseline
          'swin_only'   — Single-branch Swin Transformer V2 baseline
          'image_only'  — Dual-backbone image fusion (no metadata)
          'full'        — Hybrid image + 13-d clinical patient metadata fusion
        """
        super().__init__()
        self.mode = mode

        if mode in ("effnet_only", "image_only", "full"):
            self.effnet = timm.create_model(
                "efficientnet_b4", pretrained=pretrained, num_classes=0
            )
            eff_dim = self.effnet.num_features  # 1792

        if mode in ("swin_only", "image_only", "full"):
            self.swin = timm.create_model(
                "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
                pretrained=pretrained,
                num_classes=0,
            )
            swin_dim = self.swin.num_features  # 1024

        if mode == "full":
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            self.meta_aux_head = nn.Linear(32, num_classes)

        # Compute fusion dim based on mode
        fusion_dim = {
            "effnet_only": 1792,
            "swin_only": 1024,
            "image_only": 1792 + 1024,
            "full": 1792 + 1024 + 32,
        }[mode]

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        # Auto-freeze backbones for image_only and full modes
        if mode in ("image_only", "full"):
            self.freeze_backbones()

    def freeze_backbones(self):
        """Freeze earlier backbone stages while keeping upper feature extraction layers trainable."""
        frozen_count = 0

        if hasattr(self, "effnet"):
            for name, param in self.effnet.named_parameters():
                if not name.startswith("blocks.6") and not name.startswith("conv_head") and not name.startswith("bn2"):
                    param.requires_grad = False
                    frozen_count += 1

        if hasattr(self, "swin"):
            for name, param in self.swin.named_parameters():
                if not name.startswith("layers.3") and not name.startswith("norm"):
                    param.requires_grad = False
                    frozen_count += 1

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"[FREEZE] Frozen {frozen_count} param groups | "
            f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M total"
        )

    def forward(
        self,
        img: torch.Tensor,
        meta: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        features = []
        aux_logits = None

        if self.mode in ("effnet_only", "image_only", "full"):
            features.append(self.effnet(img))

        if self.mode in ("swin_only", "image_only", "full"):
            features.append(self.swin(img))

        if self.mode == "full":
            if meta is None:
                raise ValueError(
                    "full mode requires metadata tensor (13-d). "
                    "Got meta=None. Use mode='image_only' if metadata "
                    "is unavailable, or pass a zero tensor as fallback."
                )
            meta_feat = self.meta_mlp(meta)
            features.append(meta_feat)
            if hasattr(self, "meta_aux_head"):
                aux_logits = self.meta_aux_head(meta_feat)
        elif meta is not None:
            import warnings
            warnings.warn(
                f"Metadata tensor passed to '{self.mode}' mode — "
                "metadata is only used in 'full' mode. Ignoring.",
                stacklevel=2,
            )

        logits = self.classifier(torch.cat(features, dim=1))
        if return_aux:
            return logits, aux_logits
        return logits


# Backward-compatible alias for existing checkpoints and scripts
LesionIQHybrid = DermaLensHybrid
