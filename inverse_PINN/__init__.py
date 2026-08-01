"""Modular inverse PINN experiments for the two-dimensional waveguide."""

from .config import Case, InverseConfig
from .variants import ALL_VARIANTS, VariantSpec, parse_variant

__all__ = [
    "ALL_VARIANTS",
    "Case",
    "InverseConfig",
    "VariantSpec",
    "parse_variant",
]
