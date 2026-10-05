"""
Twelve metadata-fusion variants (V0–V11) sharing one DualBackbone.

Each variant is a standalone nn.Module with the call signature:

    forward(image, meta, meta_mask) -> (B, K) logits

The variants are registered in VARIANT_REGISTRY for the orchestrator
to discover by string id.

Variants:
    V0  M0 concat       D0 (classifier)        — hackathon baseline
    V1  M1 FiLM         D1 (feature)
    V2  M1 FiLM         D2 (block, cnn last stage)
    V3  M2 cross-attn   D1 (feature)
    V4  M3 token        D2 (transformer)
    V5  M4 hypernet     D0
    V6  M5 CBN          D2 (cnn)
    V7  M6 gated        D1
    V8  M7 hybrid       D2 (FiLM CNN + tokens transformer)
    V9  M0 concat       D3 (stem, control)
    V10 M1 FiLM         D3 (stem)
    V11 M7 hybrid + per-feature dropout p=0.3
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema_aligner import SchemaAligner, ALIGNED_DIM, META_DIM
from .injectors import (
    LateConcat, FiLM, CrossAttention, TokenFusion,
    Hypernetwork, ConditionalBN, GatedFusion,
)
from .backbones import DualBackbone


N_CLASSES_DEFAULT = 8


# ─────────────────────────────────────────────────────────────────────
# Shared classifier head
# ─────────────────────────────────────────────────────────────────────

def _classifier_head(in_dim: int, n_classes: int = N_CLASSES_DEFAULT,
                     dropout: float = 0.4) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(512, n_classes),
    )


# ─────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────

class _Base(nn.Module):
    """Shared structure: backbone + aligner + (variant-specific) injection."""

    variant_id: str = ""

    def __init__(self, *, n_classes: int = N_CLASSES_DEFAULT,
                 use_timm: bool = True, pretrained: bool = False,
                 feature_dropout_p: float = 0.0) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.backbone = DualBackbone(use_timm=use_timm,
                                     pretrained=pretrained)
        self.aligner = SchemaAligner(
            in_dim=META_DIM, out_dim=ALIGNED_DIM,
            feature_dropout_p=feature_dropout_p,
        )


# ─────────────────────────────────────────────────────────────────────
# V0 — late concat (baseline)
# ─────────────────────────────────────────────────────────────────────

class V0_LateConcat(_Base):
    variant_id = "V0"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.injector = LateConcat()
        self.head = _classifier_head(
            self.backbone.eff_dim + self.backbone.swin_dim + ALIGNED_DIM,
            self.n_classes,
        )

    def forward(self, image, meta, meta_mask):
        emb = self.aligner(meta, meta_mask)
        cnn_feat = self.backbone.cnn_pool(self.backbone.cnn_features(image))
        swin_feat = self.backbone.swin_pool(self.backbone.swin_features(image))
        img_feat = torch.cat([cnn_feat, swin_feat], dim=1)
        return self.head(self.injector(img_feat, emb))


# ─────────────────────────────────────────────────────────────────────
# V1 — FiLM at feature level (post-pool, pre-classifier)
# ─────────────────────────────────────────────────────────────────────

class V1_FiLM_Feature(_Base):
    variant_id = "V1"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        feat_dim = self.backbone.eff_dim + self.backbone.swin_dim
        self.film = FiLM(ALIGNED_DIM, feat_dim)
        self.head = _classifier_head(feat_dim, self.n_classes)

    def forward(self, image, meta, meta_mask):
        emb = self.aligner(meta, meta_mask)
        cnn_feat = self.backbone.cnn_pool(self.backbone.cnn_features(image))
        swin_feat = self.backbone.swin_pool(self.backbone.swin_features(image))
        img_feat = torch.cat([cnn_feat, swin_feat], dim=1)
        return self.head(self.film(img_feat, emb))


# ─────────────────────────────────────────────────────────────────────
# V2 — FiLM at block level (CNN last stage, pre-pool)
# ─────────────────────────────────────────────────────────────────────

class V2_FiLM_Block(_Base):
    variant_id = "V2"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.cnn_film = FiLM(ALIGNED_DIM, self.backbone.eff_dim)
        self.head = _classifier_head(
            self.backbone.eff_dim + self.backbone.swin_dim, self.n_classes)

    def forward(self, image, meta, meta_mask):
        emb = self.aligner(meta, meta_mask)
        cnn_map = self.backbone.cnn_features(image)
        cnn_map = self.cnn_film(cnn_map, emb)  # modulate channels of (B,C,H,W)
        cnn_feat = self.backbone.cnn_pool(cnn_map)
        swin_feat = self.backbone.swin_pool(self.backbone.swin_features(image))
        return self.head(torch.cat([cnn_feat, swin_feat], dim=1))


# ─────────────────────────────────────────────────────────────────────
# V3 — Cross-attention (meta as Q, CNN feature map tokens as KV)
# ─────────────────────────────────────────────────────────────────────

class V3_CrossAttention(_Base):
    variant_id = "V3"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.cross = CrossAttention(
            emb_dim=ALIGNED_DIM, kv_dim=self.backbone.eff_dim, num_heads=4)
        self.head = _classifier_head(
            self.backbone.eff_dim + self.backbone.swin_dim, self.n_classes)

    def forward(self, image, meta, meta_mask):
        emb = self.aligner(meta, meta_mask)
        cnn_map = self.backbone.cnn_features(image)
        attn_feat = self.cross(cnn_map, emb)  # (B, eff_dim)
        cnn_feat = self.backbone.cnn_pool(cnn_map) + attn_feat
        swin_feat = self.backbone.swin_pool(self.backbone.swin_features(image))
        return self.head(torch.cat([cnn_feat, swin_feat], dim=1))


# ─────────────────────────────────────────────────────────────────────
# V4 — Token-level fusion on the post-encoder Swin token sequence
# ─────────────────────────────────────────────────────────────────────

class V4_TokenInject(_Base):
    variant_id = "V4"

    def __init__(self, n_tokens: int = 4, **kw) -> None:
        super().__init__(**kw)
        # TokenFusion appends metadata tokens to the post-encoder patch
        # tokens and runs one self-attention block so the patches absorb
        # patient context — gradient-carrying on real timm Swin.
        self.token_fusion = TokenFusion(
            ALIGNED_DIM, self.backbone.swin_dim, n_tokens=n_tokens)
        self.head = _classifier_head(
            self.backbone.eff_dim + self.backbone.swin_dim, self.n_classes)

    def forward(self, image, meta, meta_mask):
        emb = self.aligner(meta, meta_mask)
        patch_tokens = self.backbone.swin_feature_tokens(image)  # (B, L, C)
        swin_feat = self.token_fusion(patch_tokens, emb)         # (B, C)
        cnn_feat = self.backbone.cnn_pool(self.backbone.cnn_features(image))
        return self.head(torch.cat([cnn_feat, swin_feat], dim=1))


# ─────────────────────────────────────────────────────────────────────
# V5 — Hypernetwork classifier
# ─────────────────────────────────────────────────────────────────────

class V5_Hypernetwork(_Base):
    variant_id = "V5"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        img_dim = self.backbone.eff_dim + self.backbone.swin_dim
        self.hypernet = Hypernetwork(ALIGNED_DIM, img_dim, self.n_classes)

    def forward(self, image, meta, meta_mask):
        emb = self.aligner(meta, meta_mask)
        cnn_feat = self.backbone.cnn_pool(self.backbone.cnn_features(image))
        swin_feat = self.backbone.swin_pool(self.backbone.swin_features(image))
        img_feat = torch.cat([cnn_feat, swin_feat], dim=1)
        return self.hypernet(img_feat, emb)


# ─────────────────────────────────────────────────────────────────────
# V6 — Conditional BN at CNN block level
# ─────────────────────────────────────────────────────────────────────

class V6_CBN(_Base):
    variant_id = "V6"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.cbn = ConditionalBN(ALIGNED_DIM, self.backbone.eff_dim)
        self.head = _classifier_head(
            self.backbone.eff_dim + self.backbone.swin_dim, self.n_classes)

    def forward(self, image, meta, meta_mask):
        emb = self.aligner(meta, meta_mask)
        cnn_map = self.backbone.cnn_features(image)
        cnn_map = self.cbn(cnn_map, emb)
        cnn_feat = self.backbone.cnn_pool(cnn_map)
        swin_feat = self.backbone.swin_pool(self.backbone.swin_features(image))
        return self.head(torch.cat([cnn_feat, swin_feat], dim=1))


# ─────────────────────────────────────────────────────────────────────
# V7 — Gated fusion at feature level
# ─────────────────────────────────────────────────────────────────────

class V7_Gated(_Base):
    variant_id = "V7"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        feat_dim = self.backbone.eff_dim + self.backbone.swin_dim
        self.gated = GatedFusion(ALIGNED_DIM, feat_dim)
        self.head = _classifier_head(feat_dim, self.n_classes)

    def forward(self, image, meta, meta_mask):
        emb = self.aligner(meta, meta_mask)
        cnn_feat = self.backbone.cnn_pool(self.backbone.cnn_features(image))
        swin_feat = self.backbone.swin_pool(self.backbone.swin_features(image))
        img_feat = torch.cat([cnn_feat, swin_feat], dim=1)
        return self.head(self.gated(img_feat, emb))


# ─────────────────────────────────────────────────────────────────────
# V8 — Hybrid: FiLM on CNN block + token injection on transformer
# ─────────────────────────────────────────────────────────────────────

class V8_Hybrid(_Base):
    variant_id = "V8"

    def __init__(self, n_tokens: int = 4, **kw) -> None:
        super().__init__(**kw)
        self.cnn_film = FiLM(ALIGNED_DIM, self.backbone.eff_dim)
        self.token_fusion = TokenFusion(
            ALIGNED_DIM, self.backbone.swin_dim, n_tokens=n_tokens)
        self.head = _classifier_head(
            self.backbone.eff_dim + self.backbone.swin_dim, self.n_classes)

    def forward(self, image, meta, meta_mask):
        emb = self.aligner(meta, meta_mask)
        # CNN branch with FiLM modulation
        cnn_map = self.backbone.cnn_features(image)
        cnn_map = self.cnn_film(cnn_map, emb)
        cnn_feat = self.backbone.cnn_pool(cnn_map)
        # Swin branch with post-encoder token fusion
        patch_tokens = self.backbone.swin_feature_tokens(image)  # (B, L, C)
        swin_feat = self.token_fusion(patch_tokens, emb)         # (B, C)
        return self.head(torch.cat([cnn_feat, swin_feat], dim=1))


# ─────────────────────────────────────────────────────────────────────
# V9 — Stem-level concat (USELESS control variant)
# ─────────────────────────────────────────────────────────────────────

class V9_StemConcat(_Base):
    variant_id = "V9"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        # Reduce embedding to 3 extra channels per pixel
        self.proj = nn.Linear(ALIGNED_DIM, 3)
        self.head = _classifier_head(
            self.backbone.eff_dim + self.backbone.swin_dim, self.n_classes)

    def forward(self, image, meta, meta_mask):
        emb = self.aligner(meta, meta_mask)
        # broadcast 3 meta channels across full HxW
        extra = self.proj(emb).unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, image.shape[2], image.shape[3])
        # mix into the input (additive small perturbation)
        x = image + 0.05 * extra
        cnn_feat = self.backbone.cnn_pool(self.backbone.cnn_features(x))
        swin_feat = self.backbone.swin_pool(self.backbone.swin_features(x))
        return self.head(torch.cat([cnn_feat, swin_feat], dim=1))


# ─────────────────────────────────────────────────────────────────────
# V10 — Stem-level FiLM
# ─────────────────────────────────────────────────────────────────────

class V10_StemFiLM(_Base):
    variant_id = "V10"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.input_film = FiLM(ALIGNED_DIM, 3)
        self.head = _classifier_head(
            self.backbone.eff_dim + self.backbone.swin_dim, self.n_classes)

    def forward(self, image, meta, meta_mask):
        emb = self.aligner(meta, meta_mask)
        x = self.input_film(image, emb)  # FiLM on input channels
        cnn_feat = self.backbone.cnn_pool(self.backbone.cnn_features(x))
        swin_feat = self.backbone.swin_pool(self.backbone.swin_features(x))
        return self.head(torch.cat([cnn_feat, swin_feat], dim=1))


# ─────────────────────────────────────────────────────────────────────
# V11 — V8 + per-feature dropout p=0.3 (missing-metadata robustness)
# ─────────────────────────────────────────────────────────────────────

class V11_HybridDropout(V8_Hybrid):
    variant_id = "V11"

    def __init__(self, **kw) -> None:
        kw["feature_dropout_p"] = 0.3
        super().__init__(**kw)


# ─────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────

VARIANT_REGISTRY: dict[str, type] = {
    "V0": V0_LateConcat,
    "V1": V1_FiLM_Feature,
    "V2": V2_FiLM_Block,
    "V3": V3_CrossAttention,
    "V4": V4_TokenInject,
    "V5": V5_Hypernetwork,
    "V6": V6_CBN,
    "V7": V7_Gated,
    "V8": V8_Hybrid,
    "V9": V9_StemConcat,
    "V10": V10_StemFiLM,
    "V11": V11_HybridDropout,
}

ALL_VARIANT_IDS = tuple(VARIANT_REGISTRY.keys())


def build_variant(variant_id: str, *, n_classes: int = N_CLASSES_DEFAULT,
                  use_timm: bool = True, pretrained: bool = False
                  ) -> nn.Module:
    if variant_id not in VARIANT_REGISTRY:
        raise ValueError(
            f"Unknown variant '{variant_id}'. Available: {ALL_VARIANT_IDS}")
    cls = VARIANT_REGISTRY[variant_id]
    return cls(n_classes=n_classes, use_timm=use_timm, pretrained=pretrained)
