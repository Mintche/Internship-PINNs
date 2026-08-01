"""Canonical inverse-PINN variant names and feature flags."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


ARCHITECTURES = ("fourier", "fourier_modified")
FORMULATIONS = ("total", "scattered")
MODIFIERS = ("field_adweights", "material_adweights", "tv")


@dataclass(frozen=True)
class VariantSpec:
    name: str
    architecture: str
    formulation: str
    field_adweights: bool
    material_adweights: bool
    total_variation: bool

    @property
    def modified(self) -> bool:
        return self.architecture == "fourier_modified"

    @property
    def scattered(self) -> bool:
        return self.formulation == "scattered"


def _variant_name(
    architecture: str,
    formulation: str,
    field_adweights: bool,
    material_adweights: bool,
    total_variation: bool,
) -> str:
    pieces = [architecture, formulation]
    if field_adweights:
        pieces.append("field_adweights")
    if material_adweights:
        pieces.append("material_adweights")
    if total_variation:
        pieces.append("tv")
    return "_".join(pieces)


ALL_VARIANTS = tuple(
    _variant_name(architecture, formulation, field, material, tv)
    for architecture, formulation in product(ARCHITECTURES, FORMULATIONS)
    for field, material, tv in product((False, True), repeat=3)
)
BASE_VARIANTS = tuple(
    _variant_name(architecture, formulation, False, False, False)
    for architecture, formulation in product(ARCHITECTURES, FORMULATIONS)
)


def parse_variant(name: str) -> VariantSpec:
    """Parse a canonical name, rejecting reordered or unknown modifiers."""
    name = str(name)
    if name not in ALL_VARIANTS:
        raise ValueError(
            f"Unknown inverse PINN variant {name!r}; expected one of {ALL_VARIANTS}"
        )
    architecture = "fourier_modified" if name.startswith("fourier_modified_") else "fourier"
    rest = name.removeprefix(architecture + "_")
    formulation = "scattered" if rest.startswith("scattered") else "total"
    return VariantSpec(
        name=name,
        architecture=architecture,
        formulation=formulation,
        field_adweights="_field_adweights" in name,
        material_adweights="_material_adweights" in name,
        total_variation=name.endswith("_tv"),
    )


def variant_label(name: str) -> str:
    spec = parse_variant(name)
    pieces = ["Fourier modified" if spec.modified else "Fourier", spec.formulation]
    if spec.field_adweights:
        pieces.append("field adweights")
    if spec.material_adweights:
        pieces.append("material adweights")
    if spec.total_variation:
        pieces.append("TV")
    return " · ".join(pieces)
