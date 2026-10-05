"""
Dual-backbone wrapper around timm models.

We use `timm` for both EfficientNet-B4 and SwinV2-Base. To keep this
file useful without network access at code-review time, we lazy-import
timm and provide a thin fallback (tiny CNN+ViT) that lets the model
classes be imported and unit-tested on CPU with no weights download.

The fallback is ONLY for smoke tests. Real training imports timm.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def _try_import_timm():
    try:
        import timm
        return timm
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────────
# Fallback tiny backbones (used only when timm is unavailable)
# ─────────────────────────────────────────────────────────────────────

class _TinyCNN(nn.Module):
    """Mini stand-in for EfficientNet during smoke tests."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.feature_dim = 128

    def forward_stem(self, x): return self.stem(x)
    def forward_block1(self, x): return self.block1(x)
    def forward_block2(self, x): return self.block2(x)

    def forward_features(self, x):
        """Return (B, C, H, W) feature map."""
        return self.block2(self.block1(self.stem(x)))

    def forward(self, x):
        h = self.forward_features(x)
        return F.adaptive_avg_pool2d(h, 1).flatten(1)


class _TinyViT(nn.Module):
    """Mini stand-in for SwinV2 during smoke tests."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 96, 8, stride=8)  # downsample 8x
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, 96))  # capped at 16x16=256
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=96, nhead=4, dim_feedforward=192,
                batch_first=True, norm_first=True),
            num_layers=2,
        )
        self.norm = nn.LayerNorm(96)
        self.feature_dim = 96

    def token_dim(self): return 96

    def tokens(self, x):
        """Return (B, T, C) patch tokens BEFORE the encoder."""
        h = self.patch_embed(x)  # (B, 96, h, w)
        B, C, H, W = h.shape
        t = h.flatten(2).transpose(1, 2)  # (B, T, C)
        n = min(t.shape[1], self.pos_embed.shape[1])
        t[:, :n] = t[:, :n] + self.pos_embed[:, :n]
        return t

    def encode(self, t):
        return self.norm(self.encoder(t))

    def forward_features(self, x):
        return self.encode(self.tokens(x))

    def forward(self, x):
        t = self.forward_features(x)
        return t.mean(dim=1)


# ─────────────────────────────────────────────────────────────────────
# DualBackbone
# ─────────────────────────────────────────────────────────────────────

class DualBackbone(nn.Module):
    """Wraps EfficientNet-B4 (CNN branch) + SwinV2-Base (transformer branch).

    Exposes the points needed by all 12 injection variants:

        cnn_features(B,3,H,W) ->  cnn_feat       (B, C_b, h, w)
        cnn_pool              ->  (B, eff_dim)

        swin_feature_tokens(...) -> patch_tokens (B, L, C)  (post-encoder)
        swin_features(...)       -> patch_tokens (B, L, C)  (alias)
        swin_pool                -> (B, swin_dim)

    Token-fusion variants (V4/V8/V11) consume the post-encoder token
    sequence; injecting before the encoder is unsafe on timm SwinV2
    (windowed attention + patch merging assume a square grid).
    """

    def __init__(self, use_timm: bool = True,
                 pretrained: bool = False) -> None:
        super().__init__()
        timm = _try_import_timm() if use_timm else None
        self._using_timm = timm is not None and use_timm

        if self._using_timm:
            try:
                self.effnet = timm.create_model(
                    "efficientnet_b4",
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=(2, 3, 4),
                )
                eff_info = self.effnet.feature_info[-1]
                self.eff_dim = eff_info["num_chs"]
            except Exception as e:
                warnings.warn(f"timm EfficientNet load failed: {e}; using fallback")
                self._using_timm = False

        if not self._using_timm:
            self.effnet = _TinyCNN()
            self.eff_dim = self.effnet.feature_dim

        if self._using_timm:
            try:
                self.swin = timm.create_model(
                    "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
                    pretrained=pretrained, features_only=False,
                )
                # timm's swin returns features through forward_features
                self.swin_dim = getattr(self.swin, "num_features", 1024)
                self.token_dim = self.swin_dim
            except Exception as e:
                warnings.warn(f"timm Swin load failed: {e}; using fallback")
                self.swin = _TinyViT()
                self.swin_dim = self.swin.feature_dim
                self.token_dim = self.swin_dim
        else:
            self.swin = _TinyViT()
            self.swin_dim = self.swin.feature_dim
            self.token_dim = self.swin_dim

    # ───── EffNet path ─────

    def cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return final-stage CNN feature map (B, C, h, w)."""
        if hasattr(self.effnet, "forward_features"):
            return self.effnet.forward_features(x)
        # timm features_only mode returns a list of feature maps
        feats = self.effnet(x)
        return feats[-1]

    def cnn_pool(self, feat: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(feat, 1).flatten(1)

    # ───── Swin path ─────

    def swin_feature_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Post-encoder flat token sequence (B, L, C). Always 3-D.

        Robust to the three shapes timm / the fallback can return:
          - (B, H, W, C)  channels-last  (timm SwinV2)        -> (B, H*W, C)
          - (B, C, H, W)  channels-first (some timm builds)   -> (B, H*W, C)
          - (B, L, C)     already flat   (the tiny fallback)  -> unchanged

        The previous implementation assumed channels-first and would
        mis-reshape the channels-last SwinV2 output, corrupting every
        variant under real timm. This single robust path fixes that and
        is what the token-fusion variants (V4/V8/V11) consume.
        """
        if hasattr(self.swin, "forward_features"):
            feat = self.swin.forward_features(x)
        else:
            feat = self.swin(x)

        if feat.dim() == 4:
            B, d1, d2, d3 = feat.shape
            if d3 == self.swin_dim:        # (B, H, W, C) channels-last
                feat = feat.reshape(B, d1 * d2, d3)
            elif d1 == self.swin_dim:      # (B, C, H, W) channels-first
                feat = feat.flatten(2).transpose(1, 2)
            else:                          # unknown ordering: assume last is C
                feat = feat.reshape(B, d1 * d2, d3)
        elif feat.dim() == 3:
            pass                            # already (B, L, C)
        elif feat.dim() == 2:
            feat = feat.unsqueeze(1)        # (B, 1, C)
        else:
            raise ValueError(f"Unexpected Swin feature rank {feat.dim()}")
        return feat

    def swin_features(self, x: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias: returns (B, L, C) tokens."""
        return self.swin_feature_tokens(x)

    def swin_pool(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 3:
            return feat.mean(dim=1)
        return feat
