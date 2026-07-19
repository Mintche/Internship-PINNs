from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from tools.resolve_checkpoint_with_fem import (
    _case_strings,
    _checkpoint_kind,
    _select_cases,
    _validate_boundary_sampling,
)


class ResolveCheckpointWithFEMTest(unittest.TestCase):
    def test_select_cases_filters_available_checkpoint_cases(self) -> None:
        available = [(600.0, 0), (600.0, 1), (1200.0, 0)]
        self.assertEqual(
            _select_cases(available, frequencies=[600.0], modes=[1]),
            [(600.0, 1)],
        )
        with self.assertRaisesRegex(ValueError, "No checkpoint case"):
            _select_cases(available, frequencies=[900.0], modes=None)

    def test_case_strings_requires_a_cartesian_product(self) -> None:
        self.assertEqual(
            _case_strings([(600.0, 0), (600.0, 1)]),
            ("600", "0,1"),
        )
        with self.assertRaisesRegex(ValueError, "Cartesian"):
            _case_strings([(600.0, 0), (1200.0, 1)])

    def test_checkpoint_kind_can_be_explicit_or_inferred(self) -> None:
        from pathlib import Path

        self.assertEqual(_checkpoint_kind(Path("scattered_case.npz"), "auto"), "scattered")
        self.assertEqual(_checkpoint_kind(Path("uv_case.npz"), "auto"), "total")
        self.assertEqual(_checkpoint_kind(Path("scattered_case.npz"), "total"), "total")

    def test_boundary_sampling_must_match(self) -> None:
        observed = SimpleNamespace(
            x_left=-2.0,
            x_right=2.0,
            y_left=np.array([0.0, 0.3, 0.6]),
            y_right=np.array([0.0, 0.3, 0.6]),
        )
        resolved = SimpleNamespace(
            x_left=-2.0,
            x_right=2.0,
            y_left=np.array([0.0, 0.3, 0.6]),
            y_right=np.array([0.0, 0.3, 0.6]),
        )
        _validate_boundary_sampling(observed, resolved)

        resolved.y_right = np.array([0.0, 0.2, 0.6])
        with self.assertRaisesRegex(ValueError, "sampling mismatch"):
            _validate_boundary_sampling(observed, resolved)


if __name__ == "__main__":
    unittest.main()
