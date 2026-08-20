from __future__ import annotations

from pathlib import Path

import pytest

from inverse_PINN import cli
from tests.inverse_pinn_test_utils import make_inverse_config


def test_campaign_cartesian_product_and_refuses_overwrite(tmp_path, monkeypatch):
    config_path = make_inverse_config(tmp_path / "case", inverse_steps=0, snapshot_fractions=())
    calls = []

    def fake_run(config, variant, seed, *, output_parent=None):
        calls.append((variant, seed, Path(output_parent)))
        directory = Path(output_parent) / f"{variant}_seed{seed}"
        directory.mkdir()
        return directory

    monkeypatch.setattr(cli, "run_training", fake_run)
    arguments = [
        "campaign", "--config", str(config_path),
        "--variants", "fourier_total,fourier_scattered", "--seeds", "0,1",
    ]
    assert cli.main(arguments) == 0
    assert [(variant, seed) for variant, seed, _ in calls] == [
        ("fourier_total", 0), ("fourier_total", 1),
        ("fourier_scattered", 0), ("fourier_scattered", 1),
    ]
    with pytest.raises(FileExistsError):
        cli.main(arguments)
