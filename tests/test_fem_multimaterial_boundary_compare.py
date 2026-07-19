import csv
import subprocess
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from tools.compare_boundary_fields import (
    compare_cases,
    create_comparison_figure,
    incident_wave,
)
from tools.compare_pinn_fem import compute_misfit_metrics
from tools.data_loader import (
    FEMFieldData,
    WaveguideBoundaryData,
    load_symmetric_coo_matrix,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FEM_DIR = REPOSITORY_ROOT / "FEM"
GENERATOR = FEM_DIR / "generate_pinn_data.x"


def _write_boundary_csv(path, *, frequency, k0, mode, x, y, values):
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["f", "k0", "mode", "x", "y", "Re_U", "Im_U"])
        for yi, value in zip(y, values):
            writer.writerow([frequency, k0, mode, x, yi, value.real, value.imag])


def _write_boundary_pair(
    directory,
    prefix,
    *,
    frequency=600.0,
    k0=None,
    mode=1,
    y=None,
    scattered_left=None,
    scattered_right=None,
):
    if k0 is None:
        k0 = 2.0 * np.pi * frequency / 340.0
    if y is None:
        y = np.asarray([0.0, 0.3, 0.6], dtype=np.float64)
    if scattered_left is None:
        scattered_left = np.asarray([1.0 + 0.5j, 1.5 - 0.25j, 0.8 + 0.1j])
    if scattered_right is None:
        scattered_right = np.asarray([0.7 - 0.2j, 1.1 + 0.4j, 1.4 - 0.3j])

    x_left = -1.0
    x_right = 1.0
    incident_left = incident_wave(
        x_left,
        y,
        k0=k0,
        mode=mode,
        y_min=float(y.min()),
        height=float(y.max() - y.min()),
    )
    incident_right = incident_wave(
        x_right,
        y,
        k0=k0,
        mode=mode,
        y_min=float(y.min()),
        height=float(y.max() - y.min()),
    )

    left_path = Path(directory) / f"{prefix}_left.csv"
    right_path = Path(directory) / f"{prefix}_right.csv"
    _write_boundary_csv(
        left_path,
        frequency=frequency,
        k0=k0,
        mode=mode,
        x=x_left,
        y=y,
        values=incident_left + scattered_left,
    )
    _write_boundary_csv(
        right_path,
        frequency=frequency,
        k0=k0,
        mode=mode,
        x=x_right,
        y=y,
        values=incident_right + scattered_right,
    )
    return left_path, right_path


def _read_numeric_csv(path):
    return pd.read_csv(path).to_numpy(dtype=np.float64)


class FEMMultiMaterialGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(
            ["make", "-C", str(FEM_DIR)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _run_generator(
        self,
        output_dir,
        *material_args,
        defect_name="equiv",
        mesh=None,
        frequencies="400",
        modes="0",
        number_of_data_points="5",
    ):
        if mesh is None:
            mesh = FEM_DIR / "data" / "test_us_barhalf_centree.msh"
        command = [
            str(GENERATOR),
            "--mesh",
            str(mesh),
            "--defectname",
            defect_name,
            "--freqs",
            frequencies,
            "--modes",
            modes,
            "--outputdir",
            str(output_dir),
            "--c0",
            "340",
            "--numberofdatapoints",
            number_of_data_points,
            *material_args,
        ]
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_legacy_contrast_matches_single_tag_contrast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy"
            tagged = root / "tagged"
            legacy.mkdir()
            tagged.mkdir()

            self._run_generator(legacy, "--contrast", "0.8")
            self._run_generator(tagged, "--tag-contrasts", "2:0.8")

            pairs = (
                (
                    legacy / "pinn_boundary_left_equiv_ratio0p8.csv",
                    tagged / "pinn_boundary_left_equiv.csv",
                ),
                (
                    legacy / "pinn_boundary_right_equiv_ratio0p8.csv",
                    tagged / "pinn_boundary_right_equiv.csv",
                ),
                (
                    legacy / "fem_field_equiv_ratio0p8.csv",
                    tagged / "fem_field_equiv.csv",
                ),
                (legacy / "Mass_matrix_equiv.csv", tagged / "Mass_matrix_equiv.csv"),
                (legacy / "Stiff_matrix_equiv.csv", tagged / "Stiff_matrix_equiv.csv"),
            )
            for legacy_path, tagged_path in pairs:
                self.assertTrue(legacy_path.is_file(), legacy_path)
                self.assertTrue(tagged_path.is_file(), tagged_path)
                np.testing.assert_allclose(
                    _read_numeric_csv(legacy_path),
                    _read_numeric_csv(tagged_path),
                    rtol=1e-11,
                    atol=1e-11,
                )

    def test_multi_tag_contrast_smoke_writes_defect_name_only_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "output"
            output_dir.mkdir()
            mesh_path = root / "tag3.msh"
            source = (FEM_DIR / "data" / "test_us_barhalf_centree.msh").read_text()
            lines = source.splitlines()
            replaced = False
            for index, line in enumerate(lines):
                tokens = line.split()
                if len(tokens) > 4 and tokens[1] == "2" and tokens[3] == "2":
                    tokens[3] = "3"
                    lines[index] = " ".join(tokens)
                    replaced = True
                    break
            self.assertTrue(replaced, "test mesh should contain at least one defect triangle")
            mesh_path.write_text("\n".join(lines) + "\n")

            self._run_generator(
                output_dir,
                "--tag-contrasts",
                "2:0.8,3:0.9",
                defect_name="barhalf_sym",
                mesh=mesh_path,
            )

            self.assertTrue((output_dir / "pinn_boundary_left_barhalf_sym.csv").is_file())
            self.assertTrue((output_dir / "pinn_boundary_right_barhalf_sym.csv").is_file())
            self.assertTrue((output_dir / "fem_field_barhalf_sym.csv").is_file())

    def test_rejects_mixed_legacy_and_tag_contrasts(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    str(GENERATOR),
                    "--mesh",
                    str(FEM_DIR / "data" / "test_us_barhalf_centree.msh"),
                    "--defectname",
                    "bad",
                    "--freqs",
                    "400",
                    "--modes",
                    "0",
                    "--outputdir",
                    directory,
                    "--c0",
                    "340",
                    "--contrast",
                    "0.8",
                    "--tag-contrasts",
                    "2:0.8",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactement un de --contrast ou --tag-contrasts", result.stderr)

    def test_homogeneous_contrast_reproduces_incident_modes(self):
        """The complete FEM/DtN export must reduce to the analytic incident field."""
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            self._run_generator(
                output_dir,
                "--contrast",
                "1",
                defect_name="homogeneous",
                frequencies="600",
                modes="0,1,2",
                number_of_data_points="31",
            )

            suffix = "homogeneous_ratio1"
            fem_data = FEMFieldData(output_dir / f"fem_field_{suffix}.csv")
            mass_matrix = load_symmetric_coo_matrix(
                output_dir / "Mass_matrix_homogeneous.csv",
                expected_size=fem_data.size,
            )
            stiffness_matrix = load_symmetric_coo_matrix(
                output_dir / "Stiff_matrix_homogeneous.csv",
                expected_size=fem_data.size,
            )
            boundary_data = WaveguideBoundaryData(
                output_dir / f"pinn_boundary_left_{suffix}.csv",
                output_dir / f"pinn_boundary_right_{suffix}.csv",
            )

            nodal_l2_relative = []
            nodal_h1_relative = []
            boundary_l2_relative = []
            y_min = float(fem_data.y.min())
            height = float(fem_data.y.max() - y_min)

            for frequency, mode in fem_data.available_cases:
                fem_case = fem_data.case(frequency, mode)
                amplitude = np.sqrt((1.0 if mode == 0 else 2.0) / height)
                transverse_wavenumber = mode * np.pi / height
                beta = np.sqrt(fem_case.k0**2 - transverse_wavenumber**2 + 0j)
                exact_nodes = (
                    amplitude
                    * np.cos(transverse_wavenumber * (fem_data.y - y_min))
                    * np.exp(1j * beta * fem_data.x)
                )
                metrics = compute_misfit_metrics(
                    exact_nodes,
                    fem_case.values,
                    mass_matrix,
                    stiffness_matrix,
                )
                nodal_l2_relative.append(metrics.l2_relative)
                nodal_h1_relative.append(metrics.h1_relative)

                pair = boundary_data.get_pair(mode, frequency)
                exact_left = incident_wave(
                    pair.x_left,
                    pair.y_left,
                    k0=pair.k0,
                    mode=mode,
                    y_min=y_min,
                    height=height,
                )
                exact_right = incident_wave(
                    pair.x_right,
                    pair.y_right,
                    k0=pair.k0,
                    mode=mode,
                    y_min=y_min,
                    height=height,
                )
                fem_boundary = np.concatenate(
                    (
                        pair.u_re_left + 1j * pair.u_im_left,
                        pair.u_re_right + 1j * pair.u_im_right,
                    )
                )
                exact_boundary = np.concatenate((exact_left, exact_right))
                boundary_l2_relative.append(
                    np.linalg.norm(fem_boundary - exact_boundary)
                    / np.linalg.norm(exact_boundary)
                )

            self.assertLess(max(nodal_l2_relative), 5e-4)
            self.assertLess(max(nodal_h1_relative), 1e-3)
            self.assertLess(max(boundary_l2_relative), 1e-3)


class BoundaryComparatorTests(unittest.TestCase):
    def test_identical_scattered_boundaries_have_zero_misfit_and_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            left_a, right_a = _write_boundary_pair(directory, "a")
            left_b, right_b = _write_boundary_pair(directory, "b")
            data_a = WaveguideBoundaryData(left_a, right_a)
            data_b = WaveguideBoundaryData(left_b, right_b)

            results = compare_cases(data_a, data_b, [600.0], [1])

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].metrics.relative_l2, 0.0)
            self.assertEqual(results[0].metrics.symmetric_relative_l2, 0.0)
            self.assertEqual(results[0].metrics.rms_absolute, 0.0)
            self.assertEqual(results[0].metrics.max_absolute, 0.0)
            figure = create_comparison_figure(results[0])
            figure.canvas.draw()

    def test_known_scattered_boundary_perturbation_metrics(self):
        scattered_left = np.asarray([1.0 + 0.5j, 1.5 - 0.25j, 0.8 + 0.1j])
        scattered_right = np.asarray([0.7 - 0.2j, 1.1 + 0.4j, 1.4 - 0.3j])
        delta_left = np.asarray([0.1 - 0.05j, -0.2 + 0.03j, 0.04 + 0.2j])
        delta_right = np.asarray([-0.03 + 0.12j, 0.08 - 0.05j, -0.1 + 0.01j])

        with tempfile.TemporaryDirectory() as directory:
            left_a, right_a = _write_boundary_pair(
                directory,
                "a",
                scattered_left=scattered_left,
                scattered_right=scattered_right,
            )
            left_b, right_b = _write_boundary_pair(
                directory,
                "b",
                scattered_left=scattered_left + delta_left,
                scattered_right=scattered_right + delta_right,
            )
            data_a = WaveguideBoundaryData(left_a, right_a)
            data_b = WaveguideBoundaryData(left_b, right_b)

            result = compare_cases(data_a, data_b, [600.0], [1])[0]

        combined_ref = np.concatenate((scattered_left, scattered_right))
        combined_comparison = np.concatenate(
            (scattered_left + delta_left, scattered_right + delta_right)
        )
        combined_delta = np.concatenate((delta_left, delta_right))
        self.assertAlmostEqual(
            result.metrics.relative_l2,
            np.linalg.norm(combined_delta) / np.linalg.norm(combined_ref),
        )
        self.assertAlmostEqual(
            result.metrics.symmetric_relative_l2,
            2.0
            * np.linalg.norm(combined_delta)
            / (np.linalg.norm(combined_ref) + np.linalg.norm(combined_comparison)),
        )
        self.assertAlmostEqual(
            result.metrics.rms_absolute,
            np.sqrt(np.mean(np.abs(combined_delta) ** 2)),
        )
        self.assertAlmostEqual(result.metrics.max_absolute, np.max(np.abs(combined_delta)))
        self.assertAlmostEqual(
            result.metrics.left_relative_l2,
            np.linalg.norm(delta_left) / np.linalg.norm(scattered_left),
        )
        self.assertAlmostEqual(
            result.metrics.right_relative_l2,
            np.linalg.norm(delta_right) / np.linalg.norm(scattered_right),
        )

    def test_missing_cases_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            left_a, right_a = _write_boundary_pair(directory, "a")
            left_b, right_b = _write_boundary_pair(directory, "b")
            data_a = WaveguideBoundaryData(left_a, right_a)
            data_b = WaveguideBoundaryData(left_b, right_b)

            with self.assertRaisesRegex(ValueError, "unavailable"):
                compare_cases(data_a, data_b, [700.0], [1])

    def test_mismatched_k0_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            left_a, right_a = _write_boundary_pair(directory, "a")
            left_b, right_b = _write_boundary_pair(directory, "b", k0=12.0)
            data_a = WaveguideBoundaryData(left_a, right_a)
            data_b = WaveguideBoundaryData(left_b, right_b)

            with self.assertRaisesRegex(ValueError, "k0 mismatch"):
                compare_cases(data_a, data_b, [600.0], [1])

    def test_mismatched_boundary_grid_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            left_a, right_a = _write_boundary_pair(directory, "a")
            left_b, right_b = _write_boundary_pair(
                directory,
                "b",
                y=np.asarray([0.0, 0.25, 0.6], dtype=np.float64),
            )
            data_a = WaveguideBoundaryData(left_a, right_a)
            data_b = WaveguideBoundaryData(left_b, right_b)

            with self.assertRaisesRegex(ValueError, "y-grid mismatch"):
                compare_cases(data_a, data_b, [600.0], [1])


if __name__ == "__main__":
    unittest.main()
