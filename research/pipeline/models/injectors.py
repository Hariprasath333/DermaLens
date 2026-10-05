"""
Metadata-injection mechanisms.

Each injector takes:
    img_feat     : image feature tensor at some injection point
    patient_emb  : (B, ALIGNED_DIM) from SchemaAligner

and returns the same-shaped img_feat with metadata fused in.

All injectors are stateless w.r.t. the SchemaAligner (which lives one
level up in the model). They expect a pre-projected patient_emb of
ALIGNED_DIM dimensions.

Shape conventions:
    - "feature-level" tensors are (B, C, H, W) for CNN or (B, T, C)
      for transformer patch tokens
    - "classifier-level" tensors are (B, C)

The injectors document which shape they expect; using the wrong one
raises a clear ValueError.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
# M0 — Late concat (the hackathon baseline)
# ─────────────────────────────────────────────────────────────────────

class LateConcat(nn.Module):
    """Concat at classifier level: (B, C) + (B, P) -> (B, C+P)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, img_feat: torch.Tensor,
                patient_emb: torch.Tensor) -> torch.Tensor:
        if img_feat.dim() != 2 or patient_emb.dim() != 2:
            raise ValueError(
                f"LateConcat expects 2D tensors; got img={img_feat.shape} "
                f"emb={patient_emb.shape}")
        return torch.cat([img_feat, patient_emb], dim=1)


# ─────────────────────────────────────────────────────────────────────
# M1 — FiLM (Feature-wise Linear Modulation)
# ─────────────────────────────────────────────────────────────────────

class FiLM(nn.Module):
    """y = gamma(meta) * x + beta(meta), per-channel.

    Supports (B, C, H, W) CNN feature maps and (B, T, C) transformer
    tokens. The channel dim is always at position 1 for CNN, last for
    transformer. We detect by tensor rank.
    """

    def __init__(self, emb_dim: int, channels: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(emb_dim, channels)
        self.beta_proj = nn.Linear(emb_dim, channels)
        # init: gamma ≈ 1, beta ≈ 0 so initial behaviour is identity
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, patient_emb: torch.Tensor
                 ) -> torch.Tensor:
        gamma = self.gamma_proj(patient_emb)
        beta = self.beta_proj(patient_emb)
        if x.dim() == 4:   # (B, C, H, W)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            return x * gamma + beta
        elif x.dim() == 3:  # (B, T, C)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            return x * gamma + beta
        elif x.dim() == 2:  # (B, C)
            return x * gamma + beta
        else:
            raise ValueError(f"FiLM: unsupported rank {x.dim()}")


# ─────────────────────────────────────────────────────────────────────
# M2 — Cross-attention (metadata as Q, image tokens as K/V)
# ─────────────────────────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """Metadata attends over image patches; returns a single token
    (B, embed_dim) that's added to the pooled image features.
    """

    def __init__(self, emb_dim: int, kv_dim: int, num_heads: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads")
        # Project patient_emb to a sequence of 1 query token at dim kv_dim
        self.q_proj = nn.Linear(emb_dim, kv_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=kv_dim, num_heads=num_heads, dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(kv_dim)

    def forward(self, img_feat: torch.Tensor, patient_emb: torch.Tensor,
                 ) -> torch.Tensor:
        """img_feat is either (B, C, H, W) or (B, T, C); returns (B, C)."""
        if img_feat.dim() == 4:
            B, C, H, W = img_feat.shape
            tokens = img_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        elif img_feat.dim() == 3:
            tokens = img_feat
            C = tokens.shape[-1]
        else:
            raise ValueError(f"CrossAttention bad rank: {img_feat.dim()}")

        q = self.q_proj(patient_emb).unsqueeze(1)  # (B, 1, C)
        out, _ = self.attn(q, tokens, tokens)
        return self.norm(out.squeeze(1))


# ─────────────────────────────────────────────────────────────────────
# M3 — Token-level fusion (transformer-native)
# ─────────────────────────────────────────────────────────────────────

class TokenFusion(nn.Module):
    """Fuse metadata tokens into a POST-encoder patch-token sequence.

    Why post-encoder (not pre-encoder): timm SwinV2 uses windowed
    attention + patch merging, which assume a square H×W grid. Appending
    tokens before the encoder breaks those reshapes. Operating on the
    flat post-encoder token sequence (B, L, C) sidesteps that entirely
    and works on any backbone.

    Mechanism:
        meta_tokens = Linear(emb) -> (B, n_meta, C)
        seq = [patch_tokens ; meta_tokens]            # (B, L+n, C)
        seq = LayerNorm(seq + SelfAttention(seq))     # patches absorb meta
        out = mean(seq[:, :L])                          # pool patches only

    Gradient flows emb -> meta_tokens -> attention -> patch reps -> out,
    so the metadata genuinely conditions the representation (verified by
    the gradient-flow guard in run.py selftest).
    """

    def __init__(self, emb_dim: int, token_dim: int, n_tokens: int = 4,
                 num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        # token_dim must divide evenly for MHA; fall back to 1 head if not
        if token_dim % num_heads != 0:
            num_heads = 1
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.meta_proj = nn.Linear(emb_dim, n_tokens * token_dim)
        self.attn = nn.MultiheadAttention(
            token_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, patch_tokens: torch.Tensor,
                patient_emb: torch.Tensor) -> torch.Tensor:
        """patch_tokens: (B, L, C) post-encoder tokens. Returns (B, C)."""
        if patch_tokens.dim() != 3:
            raise ValueError(
                f"TokenFusion expects (B, L, C); got {patch_tokens.shape}")
        B, L, C = patch_tokens.shape
        if C != self.token_dim:
            raise ValueError(
                f"TokenFusion token_dim {self.token_dim} != patch dim {C}")
        meta_tokens = self.meta_proj(patient_emb).view(B, self.n_tokens, C)
        seq = torch.cat([patch_tokens, meta_tokens], dim=1)   # (B, L+n, C)
        attended, _ = self.attn(seq, seq, seq)
        seq = self.norm(seq + attended)
        return seq[:, :L].mean(dim=1)                          # (B, C)


# ─────────────────────────────────────────────────────────────────────
# M4 — Hypernetwork classifier (meta generates classifier weights)
# ─────────────────────────────────────────────────────────────────────

class Hypernetwork(nn.Module):
    """Patient-conditioned classifier head: W(meta) @ img_feat + b(meta).

    Interpretable: weights vary per patient.
    """

    def __init__(self, emb_dim: int, img_feat_dim: int, n_classes: int
                 ) -> None:
        super().__init__()
        self.img_feat_dim = img_feat_dim
        self.n_classes = n_classes
        # Low-rank decomposition to keep param count tractable:
        # W = U @ diag(d(meta)) @ V where U,V are static, d is meta-dep
        rank = min(64, img_feat_dim, n_classes * 4)
        self.U = nn.Parameter(torch.randn(img_feat_dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, n_classes) * 0.01)
        self.d_proj = nn.Linear(emb_dim, rank)
        self.b_proj = nn.Linear(emb_dim, n_classes)

    def forward(self, img_feat: torch.Tensor, patient_emb: torch.Tensor
                 ) -> torch.Tensor:
        if img_feat.dim() != 2:
            raise ValueError(f"Hypernetwork expects (B,C); got {img_feat.shape}")
        d = self.d_proj(patient_emb)  # (B, rank)
        # logits_b = (img_feat @ U) * d @ V  + b
        proj = img_feat @ self.U          # (B, rank)
        modulated = proj * d              # (B, rank)
        logits = modulated @ self.V       # (B, n_classes)
        return logits + self.b_proj(patient_emb)


# ─────────────────────────────────────────────────────────────────────
# M5 — Conditional BatchNorm
# ─────────────────────────────────────────────────────────────────────

class ConditionalBN(nn.Module):
    """BN whose affine gamma,beta are produced by patient_emb.

    Wraps a vanilla BN by stripping its affine and feeding meta-derived
    affine parameters per-channel.
    """

    def __init__(self, emb_dim: int, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, affine=False, eps=eps)
        self.gamma_proj = nn.Linear(emb_dim, channels)
        self.beta_proj = nn.Linear(emb_dim, channels)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, patient_emb: torch.Tensor
                 ) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"ConditionalBN expects (B,C,H,W); got {x.shape}")
        x = self.bn(x)
        gamma = self.gamma_proj(patient_emb).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_proj(patient_emb).unsqueeze(-1).unsqueeze(-1)
        return x * gamma + beta


# ─────────────────────────────────────────────────────────────────────
# M6 — Gated fusion
# ─────────────────────────────────────────────────────────────────────

class GatedFusion(nn.Module):
    """y = g(meta) * img + (1-g(meta)) * proj(meta), channel-wise.

    The gate is sigmoid-bounded; init biased to mostly use image
    features (so untrained behaviour ~ image-only).
    """

    def __init__(self, emb_dim: int, channels: int) -> None:
        super().__init__()
        self.gate = nn.Linear(emb_dim, channels)
        self.meta_proj = nn.Linear(emb_dim, channels)
        nn.init.constant_(self.gate.bias, 4.0)   # sigmoid(4) ≈ 0.98

    def forward(self, x: torch.Tensor, patient_emb: torch.Tensor
                 ) -> torch.Tensor:
        g = torch.sigmoid(self.gate(patient_emb))
        m_proj = self.meta_proj(patient_emb)
        if x.dim() == 4:
            g = g.unsqueeze(-1).unsqueeze(-1)
            m_proj = m_proj.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 3:
            g = g.unsqueeze(1)
            m_proj = m_proj.unsqueeze(1)
        elif x.dim() == 2:
            pass
        else:
            raise ValueError(f"GatedFusion bad rank: {x.dim()}")
        return g * x + (1 - g) * m_proj


# ─────────────────────────────────────────────────────────────────────
# M7 — Hybrid (FiLM on CNN branch + TokenFusion on transformer branch)
# ─────────────────────────────────────────────────────────────────────
# The hybrid is composed directly inside V8/V11 (variants.py) from a
# FiLM + TokenFusion pair — no separate wrapper module is needed.
# ─────────────────────────────────────────────────────────────────────
