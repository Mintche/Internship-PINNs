import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tools.data_loader import FEMFieldCase, SymmetricCOOMatrix
from tools.plot_fem_convergence import (
    compute_convergence,
    create_convergence_figure,
    interpolate_on_reference,
)


class _FEMData:
    def __init__(self, x, y, values):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.size = self.x.size
        self._case = FEMFieldCase(
            frequency=1200.0,
            mode=0,
            incidence=-1,
            k0=1.0,
            node_ids=np.arange(self.size),
            x=self.x,
            y=self.y,
            values=np.asarray(values, dtype=np.complex128),
        )

    def case(self, frequency, mode, incidence=-1):
        self.test_case = (frequency, mode, incidence)
        return self._case


def _diagonal_matrix(size, diagonal):
    indices = np.arange(size, dtype=np.int64)
    return SymmetricCOOMatrix(
        size=size,
        rows=indices,
        columns=indices,
        values=np.full(size, diagonal, dtype=np.float64),
    )


class FEMConvergenceTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_complex_affine_field_is_interpolated_exactly(self):
        coarse_x = np.asarray([0.0, 1.0, 1.0, 0.0])
        coarse_y = np.asarray([0.0, 0.0, 1.0, 1.0])
        fine_x, fine_y = np.meshgrid(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
        fine_x = fine_x.ravel()
        fine_y = fine_y.ravel()

        field = lambda x, y: 1.0 + 2.0 * x - y + 1j * (x + 3.0 * y)
        source = _FEMData(coarse_x, coarse_y, field(coarse_x, coarse_y))._case
        reference = _FEMData(fine_x, fine_y, field(fine_x, fine_y))._case

        np.testing.assert_allclose(
            interpolate_on_reference(source, reference), reference.values
        )

    def test_errors_use_reference_matrices_and_native_dof_counts(self):
        coarse_x = np.asarray([0.0, 1.0, 1.0, 0.0])
        coarse_y = np.asarray([0.0, 0.0, 1.0, 1.0])
        fine_x, fine_y = np.meshgrid(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
        fine_x = fine_x.ravel()
        fine_y = fine_y.ravel()
        exact = lambda x, y: 2.0 + x + 1j * y

        coarse = _FEMData(
            coarse_x, coarse_y, exact(coarse_x, coarse_y) + 0.25
        )
        middle_x = np.asarray([0.0, 1.0, 1.0, 0.0, 0.5, 0.5])
        middle_y = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        middle = _FEMData(
            middle_x, middle_y, exact(middle_x, middle_y) + 0.10
        )
        reference = _FEMData(fine_x, fine_y, exact(fine_x, fine_y))
        mass = _diagonal_matrix(reference.size, 1.0)
        stiffness = _diagonal_matrix(reference.size, 2.0)

        result = compute_convergence([coarse, middle, reference], mass, stiffness)

        np.testing.assert_array_equal(result.degrees_of_freedom, [4, 6])
        self.assertEqual(result.reference_degrees_of_freedom, 9)
        expected_l2 = np.asarray([0.25, 0.10]) * np.sqrt(9.0) / np.linalg.norm(
            reference._case.values
        )
        np.testing.assert_allclose(result.relative_l2_errors, expected_l2)
        np.testing.assert_allclose(result.relative_h1_errors, expected_l2)

    def test_figure_uses_logarithmic_axes(self):
        result = type(
            "Result",
            (),
            {
                "degrees_of_freedom": np.asarray([100, 400]),
                "relative_l2_errors": np.asarray([0.1, 0.02]),
                "relative_h1_errors": np.asarray([0.2, 0.05]),
                "reference_degrees_of_freedom": 1600,
            },
        )()

        figure = create_convergence_figure(result)

        self.assertEqual(figure.axes[0].get_xscale(), "log")
        self.assertEqual(figure.axes[0].get_yscale(), "log")


if __name__ == "__main__":
    unittest.main()
