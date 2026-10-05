"""Metadata-aware dermoscopy model variants."""
from .schema_aligner import SchemaAligner, ALIGNED_DIM
from .injectors import (
    LateConcat, FiLM, CrossAttention, TokenFusion,
    Hypernetwork, ConditionalBN, GatedFusion,
)
from .variants import (
    VARIANT_REGISTRY, build_variant, ALL_VARIANT_IDS,
)
__all__ = [
    "SchemaAligner", "ALIGNED_DIM",
    "LateConcat", "FiLM", "CrossAttention", "TokenFusion",
    "Hypernetwork", "ConditionalBN", "GatedFusion",
    "VARIANT_REGISTRY", "build_variant", "ALL_VARIANT_IDS",
]
