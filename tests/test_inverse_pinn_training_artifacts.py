from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from inverse_PINN.config import InverseConfig
from inverse_PINN.checkpoints import load_pressure_checkpoint
from inverse_PINN.plot_results import plot_campaign
from inverse_PINN.training import run_training
from tests.inverse_pinn_test_utils import make_inverse_config


REQUIRED = {
    "pressure_loss_history.csv",
    "material_loss_history.csv",
    "pressure_gradient_history.csv",
    "material_snapshot_diagnostics.csv",
    "material_gradient_cosines.csv",
    "adweights_history.csv",
    "sigma_history.csv",
    "timing.csv",
    "summary.json",
    "celerity_snapshots.npz",
    "pressure_weights_best.npz",
    "slowness_weights_best.npz",
    "pressure_metrics.csv",
    "celerity_metrics.json",
}


@pytest.mark.parametrize(
    "variant",
    [
        "fourier_total_field_adweights_material_adweights_tv",
        "fourier_scattered",
    ],
)
def test_total_and_scattered_smoke_artifacts_and_snapshot_atomicity(tmp_path, variant):
    config = InverseConfig.from_json(
        make_inverse_config(tmp_path / variant, warmup_steps=1, inverse_steps=1)
    )
    run = run_training(config, variant, 0)
    package = next((run / "packages").iterdir())
    assert REQUIRED <= {path.name for path in package.iterdir()}
    summary = json.loads((package / "summary.json").read_text())
    assert summary["snapshot_steps"] == [1]
    with np.load(package / "celerity_snapshots.npz", allow_pickle=False) as archive:
        assert archive["celerity"].shape == (1, 3, 4)
        assert archive["steps"].tolist() == [1]
    with (package / "material_snapshot_diagnostics.csv").open() as stream:
        diagnostic_steps = {int(row["step"]) for row in csv.DictReader(stream)}
    with (package / "material_gradient_cosines.csv").open() as stream:
        cosine_steps = {int(row["step"]) for row in csv.DictReader(stream)}
    assert diagnostic_steps == cosine_steps == {1}
    material_loss_rows = list(
        csv.DictReader((package / "material_loss_history.csv").open())
    )
    aggregate_rows = [
        row for row in csv.DictReader(
            (package / "material_snapshot_diagnostics.csv").open()
        )
        if row["kind"] == "aggregate"
    ]
    if variant.endswith("_tv"):
        assert material_loss_rows[0]["tv"] != ""
        assert aggregate_rows[0]["tv_gradient_l2_norm"] != ""
    else:
        assert material_loss_rows[0]["tv"] == ""
        assert material_loss_rows[0]["weighted_tv"] == ""
        assert aggregate_rows[0]["tv"] == ""
        assert aggregate_rows[0]["tv_gradient_l2_norm"] == ""
    with pytest.raises(FileExistsError):
        run_training(config, variant, 0)

    if "total" in variant:
        outputs = plot_campaign(run, cosines=True)
        pdfs = list(outputs[0].glob("*.pdf"))
        assert len(pdfs) == 8
        assert all(path.stat().st_size > 100 for path in pdfs)


def test_alternating_lbfgs_miniature_and_best_checkpoint(tmp_path):
    config = InverseConfig.from_json(
        make_inverse_config(
            tmp_path / "lbfgs", inverse_steps=0, snapshot_fractions=(),
            lbfgs_cycles=1, lbfgs_field_steps=1, lbfgs_material_steps=1,
        )
    )
    run = run_training(config, "fourier_total", 4)
    package = next((run / "packages").iterdir())
    phases = {
        row["phase"]
        for row in csv.DictReader((package / "pressure_loss_history.csv").open())
    }
    assert "inverse_lbfgs_field_cycle1" in phases
    assert "inverse_lbfgs_material_cycle1" in phases
    assert (package / "pressure_weights_best.npz").is_file()
    assert (package / "slowness_weights_best.npz").is_file()


@pytest.mark.parametrize(
    ("variant", "expected_warmups"),
    [
        ("fourier_total", [(0, "im1_f1000_m0"), (1, "im1_f1200_m0")]),
        ("fourier_scattered", [(1, "im1_f1200_m0")]),
    ],
)
def test_curriculum_pretrains_only_new_cases_and_keeps_seen_frozen(
    tmp_path, monkeypatch, variant, expected_warmups
):
    config = InverseConfig.from_json(
        make_inverse_config(
            tmp_path / variant,
            frequencies=(1000.0, 1200.0),
            package_frequencies=((1000.0,), (1000.0, 1200.0), (1200.0,)),
            warmup_steps=1, inverse_steps=0, snapshot_fractions=(),
        )
    )
    calls = []

    def fake_warmup(**kwargs):
        calls.append((kwargs["package_index"], kwargs["case"].id))
        return kwargs["field_model"], 0.0, 0.0

    monkeypatch.setattr("inverse_PINN.training._run_warmup", fake_warmup)
    run = run_training(config, variant, 2)
    assert calls == expected_warmups
    last_package = sorted((run / "packages").iterdir())[-1]
    models, _, _ = load_pressure_checkpoint(last_package / "pressure_weights_best.npz")
    assert len(models) == 2
