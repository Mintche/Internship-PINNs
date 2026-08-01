"""Strict FEM, boundary, matrix, and P2 mesh loading for inverse runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Case, GeometryConfig, TruthRegion


BOUNDARY_COLUMNS = ("incidence", "f", "k0", "mode", "x", "y", "Re_U", "Im_U")
FIELD_COLUMNS = (
    "incidence", "f", "k0", "mode", "node_id", "x", "y", "Re_U", "Im_U",
)


@dataclass(frozen=True)
class BoundaryTrace:
    case: Case
    k0: float
    x_left: float
    y_left: np.ndarray
    values_left: np.ndarray
    x_right: float
    y_right: np.ndarray
    values_right: np.ndarray


@dataclass(frozen=True)
class FEMCase:
    case: Case
    k0: float
    node_ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class MeshP2:
    node_ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    triangles: np.ndarray
    references: np.ndarray

    def subtriangles(self) -> np.ndarray:
        """Split every P2 element into four P1 triangles."""
        nodes = self.triangles
        return np.concatenate(
            (
                nodes[:, (0, 3, 5)],
                nodes[:, (3, 1, 4)],
                nodes[:, (5, 4, 2)],
                nodes[:, (3, 4, 5)],
            ),
            axis=0,
        )


@dataclass(frozen=True)
class SymmetricCOOMatrix:
    size: int
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector)
        if vector.shape != (self.size,):
            raise ValueError(f"Expected vector shape ({self.size},), got {vector.shape}")
        result = np.zeros(self.size, dtype=np.result_type(vector.dtype, self.values.dtype))
        np.add.at(result, self.rows, self.values * vector[self.columns])
        off = self.rows != self.columns
        np.add.at(result, self.columns[off], self.values[off] * vector[self.rows[off]])
        return result

    def quadratic_form(self, vector: np.ndarray) -> float:
        value = np.vdot(vector, self.matvec(vector))
        tolerance = 1e-9 * max(1.0, abs(float(value.real)))
        if abs(float(value.imag)) > tolerance or float(value.real) < -tolerance:
            raise ValueError(f"Invalid quadratic form {value}")
        return max(float(value.real), 0.0)


@dataclass(frozen=True)
class InverseDataset:
    boundaries: dict[Case, BoundaryTrace]
    fem_cases: dict[Case, FEMCase]
    mass: SymmetricCOOMatrix
    stiffness: SymmetricCOOMatrix
    mesh: MeshP2


def _numeric_frame(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(set(required) - set(frame))
    if missing or frame.empty:
        raise ValueError(f"Invalid data file {path}; missing={missing}, empty={frame.empty}")
    frame = frame.loc[:, list(required)].copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path} contains non-finite values")
    return frame


def _integer_column(frame: pd.DataFrame, column: str, path: Path) -> None:
    values = frame[column].to_numpy(dtype=np.float64)
    if not np.equal(values, np.rint(values)).all():
        raise ValueError(f"Column {column} in {path} must contain integers")
    frame[column] = np.rint(values).astype(np.int64)


def _load_boundary_side(path: Path) -> dict[Case, tuple[float, float, np.ndarray, np.ndarray]]:
    frame = _numeric_frame(path, BOUNDARY_COLUMNS)
    _integer_column(frame, "incidence", path)
    _integer_column(frame, "mode", path)
    if not set(frame["incidence"]).issubset({-1, 1}) or (frame["mode"] < 0).any():
        raise ValueError(f"Invalid incidence or mode in {path}")
    duplicate = frame.duplicated(["incidence", "f", "mode", "y"], keep=False)
    if duplicate.any():
        raise ValueError(f"Duplicate boundary ordinate in {path}")
    result = {}
    for (incidence, frequency, mode), subset in frame.groupby(
        ["incidence", "f", "mode"], sort=True
    ):
        subset = subset.sort_values("y", kind="stable")
        x = subset["x"].to_numpy(dtype=np.float64)
        k0 = subset["k0"].to_numpy(dtype=np.float64)
        if not np.allclose(x, x[0], rtol=0.0, atol=1e-8):
            raise ValueError(f"Boundary x is not constant in {path}")
        if not np.allclose(k0, k0[0], rtol=1e-8, atol=1e-9):
            raise ValueError(f"Boundary k0 is not constant in {path}")
        case = Case(float(frequency), int(mode), int(incidence))
        values = (
            subset["Re_U"].to_numpy(dtype=np.float64)
            + 1j * subset["Im_U"].to_numpy(dtype=np.float64)
        )
        result[case] = (
            float(k0[0]),
            float(x[0]),
            subset["y"].to_numpy(dtype=np.float64),
            values,
        )
    return result


def load_boundaries(left: Path, right: Path, cases: Iterable[Case]) -> dict[Case, BoundaryTrace]:
    left_data = _load_boundary_side(left)
    right_data = _load_boundary_side(right)
    if set(left_data) != set(right_data):
        raise ValueError("Boundary files do not contain the same cases")
    result = {}
    for requested in cases:
        matches = [candidate for candidate in left_data if candidate.mode == requested.mode and candidate.incidence == requested.incidence and np.isclose(candidate.frequency, requested.frequency)]
        if len(matches) != 1:
            raise ValueError(f"Boundary case {requested.id} is unavailable")
        case = matches[0]
        lk0, lx, ly, lv = left_data[case]
        rk0, rx, ry, rv = right_data[case]
        if not np.isclose(lk0, rk0, rtol=1e-8, atol=1e-9):
            raise ValueError(f"Inconsistent boundary k0 for {case.id}")
        result[requested] = BoundaryTrace(requested, lk0, lx, ly, lv, rx, ry, rv)
    return result


def load_fem_fields(path: Path, cases: Iterable[Case]) -> dict[Case, FEMCase]:
    frame = _numeric_frame(path, FIELD_COLUMNS)
    for name in ("incidence", "mode", "node_id"):
        _integer_column(frame, name, path)
    available = {}
    reference_coordinates = None
    for (incidence, frequency, mode), subset in frame.groupby(
        ["incidence", "f", "mode"], sort=True
    ):
        subset = subset.sort_values("node_id", kind="stable")
        ids = subset["node_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(ids, np.arange(ids.size)):
            raise ValueError(f"Non-contiguous FEM node IDs for f={frequency}, mode={mode}")
        x = subset["x"].to_numpy(dtype=np.float64)
        y = subset["y"].to_numpy(dtype=np.float64)
        coordinates = np.column_stack((x, y))
        if reference_coordinates is None:
            reference_coordinates = coordinates
        elif not np.allclose(coordinates, reference_coordinates, rtol=0.0, atol=1e-12):
            raise ValueError("All FEM cases must share the same P2 node grid")
        k0_values = subset["k0"].to_numpy(dtype=np.float64)
        if not np.allclose(k0_values, k0_values[0], rtol=1e-8, atol=1e-9):
            raise ValueError("FEM k0 is not constant within a case")
        case = Case(float(frequency), int(mode), int(incidence))
        available[case] = FEMCase(
            case,
            float(k0_values[0]),
            ids,
            x,
            y,
            subset["Re_U"].to_numpy(dtype=np.float64)
            + 1j * subset["Im_U"].to_numpy(dtype=np.float64),
        )
    result = {}
    for requested in cases:
        matches = [candidate for candidate in available if candidate.mode == requested.mode and candidate.incidence == requested.incidence and np.isclose(candidate.frequency, requested.frequency)]
        if len(matches) != 1:
            raise ValueError(f"FEM case {requested.id} is unavailable")
        result[requested] = available[matches[0]]
    return result


def load_symmetric_matrix(path: Path, expected_size: int) -> SymmetricCOOMatrix:
    frame = _numeric_frame(path, ("row", "col", "value"))
    for name in ("row", "col"):
        _integer_column(frame, name, path)
    rows = frame["row"].to_numpy(dtype=np.int64)
    columns = frame["col"].to_numpy(dtype=np.int64)
    values = frame["value"].to_numpy(dtype=np.float64)
    if (rows < columns).any() or (rows < 0).any() or (columns < 0).any():
        raise ValueError(f"{path} must store a non-negative lower triangle")
    if int(rows.max()) + 1 != expected_size or int(columns.max()) >= expected_size:
        raise ValueError(f"Matrix {path} does not have expected size {expected_size}")
    if frame.duplicated(["row", "col"]).any():
        raise ValueError(f"Duplicate COO coefficient in {path}")
    order = np.lexsort((columns, rows))
    return SymmetricCOOMatrix(expected_size, rows[order], columns[order], values[order])


def load_mesh(nodes_path: Path, triangles_path: Path) -> MeshP2:
    nodes = _numeric_frame(nodes_path, ("node_id", "x", "y", "ref"))
    _integer_column(nodes, "node_id", nodes_path)
    nodes = nodes.sort_values("node_id", kind="stable")
    ids = nodes["node_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(ids, np.arange(ids.size)):
        raise ValueError("Mesh node IDs must be contiguous")
    triangle_columns = ("triangle_id", "n0", "n1", "n2", "n3", "n4", "n5", "ref")
    triangles = _numeric_frame(triangles_path, triangle_columns)
    for name in triangle_columns:
        _integer_column(triangles, name, triangles_path)
    triangles = triangles.sort_values("triangle_id", kind="stable")
    triangle_ids = triangles["triangle_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(triangle_ids, np.arange(triangle_ids.size)):
        raise ValueError("Mesh triangle IDs must be contiguous")
    connectivity = triangles[[f"n{index}" for index in range(6)]].to_numpy(dtype=np.int64)
    if connectivity.min() < 0 or connectivity.max() >= ids.size:
        raise ValueError("Mesh connectivity contains invalid node IDs")
    return MeshP2(
        ids,
        nodes["x"].to_numpy(dtype=np.float64),
        nodes["y"].to_numpy(dtype=np.float64),
        connectivity,
        triangles["ref"].to_numpy(dtype=np.int64),
    )


def load_inverse_dataset(config: object) -> InverseDataset:
    cases = config.all_cases
    boundaries = load_boundaries(config.data.boundary_left, config.data.boundary_right, cases)
    fem_cases = load_fem_fields(config.data.fem_field, cases)
    first = fem_cases[cases[0]]
    mass = load_symmetric_matrix(config.data.mass_matrix, first.node_ids.size)
    stiffness = load_symmetric_matrix(config.data.stiffness_matrix, first.node_ids.size)
    mesh = load_mesh(config.data.mesh_nodes, config.data.mesh_triangles)
    if mesh.node_ids.size != first.node_ids.size or not np.allclose(mesh.x, first.x, atol=1e-12, rtol=0.0) or not np.allclose(mesh.y, first.y, atol=1e-12, rtol=0.0):
        raise ValueError("Configured mesh and FEM field use different node grids")
    geometry = config.geometry
    for case, boundary in boundaries.items():
        if not np.isclose(boundary.x_left, -geometry.half_length, atol=1e-7, rtol=0.0) or not np.isclose(boundary.x_right, geometry.half_length, atol=1e-7, rtol=0.0):
            raise ValueError(f"Boundary coordinates for {case.id} do not match the guide")
    return InverseDataset(boundaries, fem_cases, mass, stiffness, mesh)


def truth_sound_speed(
    geometry: GeometryConfig, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Evaluate ordered synthetic regions; later regions overwrite earlier ones."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    result = np.full(x.shape, geometry.c0, dtype=np.float64)
    for region in geometry.truth_regions:
        if region.shape == "circle":
            assert region.center is not None and region.radius is not None
            mask = (x - region.center[0]) ** 2 + (y - region.center[1]) ** 2 <= region.radius**2
        else:
            assert region.bounds is not None
            xmin, xmax, ymin, ymax = region.bounds
            mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        result[mask] = geometry.c0 * region.speed_ratio
    return result
