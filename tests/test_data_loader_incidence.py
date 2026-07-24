import tempfile
import unittest
from pathlib import Path

from tools.data_loader import FEMFieldData, WaveguideBoundaryData


def write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class IncidenceStrictDataLoaderTests(unittest.TestCase):
    def test_fem_field_requires_incidence_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_text(
                Path(directory) / "field.csv",
                "f,k0,mode,node_id,x,y,Re_U,Im_U\n"
                "600,11.1,0,0,-1,0,1,0\n",
            )
            with self.assertRaisesRegex(ValueError, "missing columns"):
                FEMFieldData(path)

    def test_boundary_requires_incidence_column(self):
        with tempfile.TemporaryDirectory() as directory:
            left = write_text(
                Path(directory) / "left.csv",
                "f,k0,mode,x,y,Re_U,Im_U\n"
                "600,11.1,0,-1,0,1,0\n",
            )
            right = write_text(
                Path(directory) / "right.csv",
                "f,k0,mode,x,y,Re_U,Im_U\n"
                "600,11.1,0,1,0,0,1\n",
            )
            with self.assertRaisesRegex(ValueError, "missing columns"):
                WaveguideBoundaryData(left, right)

    def test_invalid_incidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_text(
                Path(directory) / "field.csv",
                "incidence,f,k0,mode,node_id,x,y,Re_U,Im_U\n"
                "0,600,11.1,0,0,-1,0,1,0\n",
            )
            with self.assertRaisesRegex(ValueError, "invalid incidence"):
                FEMFieldData(path)

    def test_double_incidence_field_cases_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_text(
                Path(directory) / "field.csv",
                "incidence,f,k0,mode,node_id,x,y,Re_U,Im_U\n"
                "-1,600,11.1,0,0,-1,0,1,0\n"
                "-1,600,11.1,0,1,1,0,0,1\n"
                "1,600,11.1,0,0,-1,0,2,0\n"
                "1,600,11.1,0,1,1,0,0,2\n",
            )
            data = FEMFieldData(path)
            self.assertEqual(data.available_triplets, ((600.0, 0, -1), (600.0, 0, 1)))
            self.assertEqual(data.case(600.0, 0, -1).incidence, -1)
            self.assertEqual(data.case(600.0, 0, 1).incidence, 1)
            self.assertEqual(data.case(600.0, 0, 1).values[0], 2.0 + 0.0j)

    def test_double_incidence_boundary_pairs_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            left = write_text(
                Path(directory) / "left.csv",
                "incidence,f,k0,mode,x,y,Re_U,Im_U\n"
                "-1,600,11.1,0,-1,0,1,0\n"
                "-1,600,11.1,0,-1,1,0,1\n"
                "1,600,11.1,0,-1,0,2,0\n"
                "1,600,11.1,0,-1,1,0,2\n",
            )
            right = write_text(
                Path(directory) / "right.csv",
                "incidence,f,k0,mode,x,y,Re_U,Im_U\n"
                "-1,600,11.1,0,1,0,3,0\n"
                "-1,600,11.1,0,1,1,0,3\n"
                "1,600,11.1,0,1,0,4,0\n"
                "1,600,11.1,0,1,1,0,4\n",
            )
            data = WaveguideBoundaryData(left, right)
            self.assertEqual(data.available_triplets, ((0, 600.0, -1), (0, 600.0, 1)))
            self.assertEqual(data.get_pair(0, 600.0, -1).incidence, -1)
            self.assertEqual(data.get_pair(0, 600.0, 1).incidence, 1)
            self.assertEqual(data.get_pair(0, 600.0, 1).u_re_left[0], 2.0)


if __name__ == "__main__":
    unittest.main()
