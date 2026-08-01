from __future__ import annotations

import json

import pytest

from inverse_PINN.config import InverseConfig
from inverse_PINN.variants import ALL_VARIANTS, parse_variant
from tests.inverse_pinn_test_utils import make_inverse_config


def test_canonical_variants_are_strict_and_combinable():
    assert len(ALL_VARIANTS) == 32
    combined = parse_variant(
        "fourier_modified_scattered_field_adweights_material_adweights_tv"
    )
    assert combined.modified and combined.scattered
    assert combined.field_adweights and combined.material_adweights
    assert combined.total_variation
    with pytest.raises(ValueError):
        parse_variant("fourier_total_tv_field_adweights")
    with pytest.raises(ValueError):
        parse_variant("classical_total")


def test_strict_json_contrast_and_free_packages(tmp_path):
    path = make_inverse_config(
        tmp_path,
        frequencies=(1000.0, 1200.0),
        package_frequencies=((1000.0,), (1000.0, 1200.0), (1200.0,)),
    )
    config = InverseConfig.from_json(path)
    assert config.geometry.c_min == pytest.approx(238.0)
    assert config.geometry.c_max == pytest.approx(408.0)
    assert [len(package.cases) for package in config.training_packages] == [1, 2, 1]
    assert len(config.all_cases) == 2

    raw = json.loads(path.read_text())
    raw["unknown"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unknown"):
        InverseConfig.from_json(path)


def test_evanescent_incident_mode_is_rejected(tmp_path):
    path = make_inverse_config(tmp_path)
    raw = json.loads(path.read_text())
    raw["training_packages"][0]["cases"]["-1"]["1200"] = [20]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="evanescent"):
        InverseConfig.from_json(path)

