from __future__ import annotations

import csv
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.data_loader import FEMFieldData, WaveguideBoundaryData


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FEM_ROOT = REPOSITORY_ROOT / "FEM"
GENERATOR = FEM_ROOT / "generate_pinn_data.x"


class FEMNodalMaterialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(["make", "-C", str(FEM_ROOT)], check=True, capture_output=True)

    def test_constant_nodal_speed_matches_homogeneous_tag_material(self) -> None:
        reference = FEMFieldData(
            REPOSITORY_ROOT / "FEM/pinn_data/fem_field_barhalf_ratio0p8.csv"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nodal_csv = root / "sound_speed.csv"
            with nodal_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("node_id", "x", "y", "c"))
                writer.writerows(
                    (index, x, y, 340.0)
                    for index, (x, y) in enumerate(zip(reference.x, reference.y))
                )

            common = (
                "--mesh",
                str(FEM_ROOT / "data/test_us_barhalf_centree.msh"),
                "--freqs",
                "600",
                "--modes",
                "0",
                "--c0",
                "340",
                "--numberofdatapoints",
                "31",
            )
            legacy_dir = root / "legacy"
            nodal_dir = root / "nodal"
            subprocess.run(
                (
                    str(GENERATOR),
                    *common,
                    "--defectname",
                    "legacy",
                    "--outputdir",
                    str(legacy_dir),
                    "--contrast",
                    "1.0",
                ),
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                (
                    str(GENERATOR),
                    *common,
                    "--defectname",
                    "nodal",
                    "--outputdir",
                    str(nodal_dir),
                    "--nodal-sound-speed",
                    str(nodal_csv),
                ),
                check=True,
                capture_output=True,
                text=True,
            )

            legacy_field = FEMFieldData(legacy_dir / "fem_field_legacy_ratio1.csv")
            nodal_field = FEMFieldData(nodal_dir / "fem_field_nodal.csv")
            legacy_values = legacy_field.case(600.0, 0).values
            nodal_values = nodal_field.case(600.0, 0).values
            field_error = np.linalg.norm(nodal_values - legacy_values) / np.linalg.norm(
                legacy_values
            )

            legacy_boundary = WaveguideBoundaryData(
                legacy_dir / "pinn_boundary_left_legacy_ratio1.csv",
                legacy_dir / "pinn_boundary_right_legacy_ratio1.csv",
            ).get_pair(0, 600.0)
            nodal_boundary = WaveguideBoundaryData(
                nodal_dir / "pinn_boundary_left_nodal.csv",
                nodal_dir / "pinn_boundary_right_nodal.csv",
            ).get_pair(0, 600.0)
            legacy_trace = np.concatenate(
                (
                    legacy_boundary.u_re_left + 1j * legacy_boundary.u_im_left,
                    legacy_boundary.u_re_right + 1j * legacy_boundary.u_im_right,
                )
            )
            nodal_trace = np.concatenate(
                (
                    nodal_boundary.u_re_left + 1j * nodal_boundary.u_im_left,
                    nodal_boundary.u_re_right + 1j * nodal_boundary.u_im_right,
                )
            )
            trace_error = np.linalg.norm(nodal_trace - legacy_trace) / np.linalg.norm(
                legacy_trace
            )
            self.assertLess(field_error, 1e-11)
            self.assertLess(trace_error, 1e-11)


if __name__ == "__main__":
    unittest.main()
