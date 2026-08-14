from __future__ import annotations

import json

import pytest

from inverse_PINN.config import InverseConfig
from inverse_PINN.optimizers import learning_rate_schedule
from tests.inverse_pinn_test_utils import make_inverse_config


def test_field_and_material_learning_rate_schedule(tmp_path):
    config = InverseConfig.from_json(make_inverse_config(tmp_path))
    optimization = config.optimization

    for initial_rate in (
        optimization.field_learning_rate,
        optimization.material_learning_rate,
    ):
        schedule = learning_rate_schedule(initial_rate, optimization)
        assert float(schedule(9_999)) == pytest.approx(initial_rate)
        assert float(schedule(10_000)) == pytest.approx(initial_rate)
        assert float(schedule(15_000)) == pytest.approx(
            initial_rate * (1.0 + optimization.cosine_decay_alpha) / 2.0
        )
        assert float(schedule(20_000)) == pytest.approx(
            initial_rate * optimization.cosine_decay_alpha
        )
        assert float(schedule(30_000)) == pytest.approx(
            initial_rate * optimization.cosine_decay_alpha
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"cosine_decay_start": 20_000, "consine_decay_stop": 20_000},
            "consine_decay_stop",
        ),
        ({"cosine_decay_alpha": 1.1}, "cosine_decay_alpha"),
    ],
)
def test_cosine_decay_configuration_is_validated(tmp_path, updates, message):
    path = make_inverse_config(tmp_path)
    raw = json.loads(path.read_text())
    raw["optimization"].update(updates)
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match=message):
        InverseConfig.from_json(path)
