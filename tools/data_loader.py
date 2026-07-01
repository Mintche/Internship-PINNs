from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("f", "k0", "mode", "x", "y", "Re_U", "Im_U")
FEM_FIELD_COLUMNS = ("f", "k0", "mode", "node_id", "x", "y", "Re_U", "Im_U")
MATRIX_COLUMNS = ("row", "col", "value")


def format_ratio_label(ratio: float) -> str:
    """Return the filename label shared with the C++ generator."""
    ratio = float(ratio)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("ratio must be a finite, strictly positive number")
    token = f"{ratio:.15f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"ratio{token}"


@dataclass(frozen=True)
class BoundaryPair:
    mode: int
    frequency: float
    k0: float
    x_left: float
    y_left: np.ndarray
    u_re_left: np.ndarray
    u_im_left: np.ndarray
    x_right: float
    y_right: np.ndarray
    u_re_right: np.ndarray
    u_im_right: np.ndarray


@dataclass(frozen=True)
class _BoundaryTrace:
    k0: float
    x: float
    y: np.ndarray
    u_re: np.ndarray
    u_im: np.ndarray


@dataclass(frozen=True)
class FEMFieldCase:
    frequency: float
    mode: int
    k0: float
    node_ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class SymmetricCOOMatrix:
    """Real symmetric matrix stored once on its lower triangle."""

    size: int
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector)
        if vector.ndim != 1 or vector.shape[0] != self.size:
            raise ValueError(
                f"Expected a vector of shape ({self.size},), got {vector.shape}"
            )

        result = np.zeros(self.size, dtype=np.result_type(vector.dtype, self.values.dtype))
        np.add.at(result, self.rows, self.values * vector[self.columns])
        off_diagonal = self.rows != self.columns
        np.add.at(
            result,
            self.columns[off_diagonal],
            self.values[off_diagonal] * vector[self.rows[off_diagonal]],
        )
        return result

    def quadratic_form(self, vector: np.ndarray) -> float:
        vector = np.asarray(vector)
        value = np.vdot(vector, self.matvec(vector))
        tolerance = 1e-10 * max(1.0, abs(float(value.real)))
        if abs(float(value.imag)) > tolerance:
            raise ValueError(f"Quadratic form has a non-negligible imaginary part: {value}")
        if float(value.real) < -tolerance:
            raise ValueError(f"Quadratic form is negative: {value.real}")
        return max(float(value.real), 0.0)


class WaveguideBoundaryData:
    """Load and validate a combined left/right FEM boundary dataset."""

    def __init__(self, left_filepath: str, right_filepath: str):
        self.left_filepath = Path(left_filepath)
        self.right_filepath = Path(right_filepath)

        left = self._load_side(self.left_filepath, "left")
        right = self._load_side(self.right_filepath, "right")

        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            missing_left = sorted(right_keys - left_keys)
            missing_right = sorted(left_keys - right_keys)
            raise ValueError(
                "Left/right boundary files do not contain the same (mode, frequency) pairs; "
                f"missing on left: {missing_left}, missing on right: {missing_right}"
            )

        self._pairs = {}
        for mode, frequency in sorted(left_keys):
            left_trace = left[(mode, frequency)]
            right_trace = right[(mode, frequency)]
            if not np.isclose(left_trace.k0, right_trace.k0, rtol=1e-6, atol=1e-7):
                raise ValueError(
                    f"Inconsistent k0 for mode {mode}, frequency {frequency}: "
                    f"{left_trace.k0} versus {right_trace.k0}"
                )
            self._pairs[(mode, frequency)] = BoundaryPair(
                mode=mode,
                frequency=frequency,
                k0=left_trace.k0,
                x_left=left_trace.x,
                y_left=left_trace.y,
                u_re_left=left_trace.u_re,
                u_im_left=left_trace.u_im,
                x_right=right_trace.x,
                y_right=right_trace.y,
                u_re_right=right_trace.u_re,
                u_im_right=right_trace.u_im,
            )

        self.available_pairs = tuple(sorted(self._pairs))
        self.modes = np.asarray(sorted({mode for mode, _ in self.available_pairs}), dtype=np.int32)

    @staticmethod
    def _load_side(filepath: Path, side: str):
        if not filepath.is_file():
            raise FileNotFoundError(f"Missing {side} boundary file: {filepath}")

        dataframe = pd.read_csv(filepath)
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"{filepath} is missing columns: {missing_columns}")
        if dataframe.empty:
            raise ValueError(f"{filepath} contains no boundary data")

        dataframe = dataframe.loc[:, REQUIRED_COLUMNS].copy()
        for column in REQUIRED_COLUMNS:
            try:
                dataframe[column] = pd.to_numeric(dataframe[column], errors="raise")
            except (TypeError, ValueError) as error:
                raise ValueError(f"Column {column!r} in {filepath} is not numeric") from error

        numeric_values = dataframe.to_numpy(dtype=np.float64)
        if not np.isfinite(numeric_values).all():
            raise ValueError(f"{filepath} contains NaN or infinite values")
        if (dataframe["f"] <= 0.0).any():
            raise ValueError(f"{filepath} contains a non-positive frequency")

        modes = dataframe["mode"].to_numpy(dtype=np.float64)
        if not np.equal(modes, np.rint(modes)).all() or (modes < 0.0).any():
            raise ValueError(f"{filepath} contains an invalid mode index")
        dataframe["mode"] = np.rint(modes).astype(np.int64)

        duplicate_mask = dataframe.duplicated(subset=["mode", "f", "y"], keep=False)
        if duplicate_mask.any():
            duplicate = dataframe.loc[duplicate_mask, ["mode", "f", "y"]].iloc[0].tolist()
            raise ValueError(f"Duplicate boundary sample {duplicate} in {filepath}")

        traces = {}
        for (mode, frequency), subset in dataframe.groupby(["mode", "f"], sort=True):
            subset = subset.sort_values("y", kind="stable")
            x_values = subset["x"].to_numpy(dtype=np.float64)
            k0_values = subset["k0"].to_numpy(dtype=np.float64)
            if not np.allclose(x_values, x_values[0], rtol=0.0, atol=1e-7):
                raise ValueError(
                    f"Boundary x is not constant for mode {mode}, frequency {frequency} in {filepath}"
                )
            if not np.allclose(k0_values, k0_values[0], rtol=1e-7, atol=1e-7):
                raise ValueError(
                    f"k0 is not constant for mode {mode}, frequency {frequency} in {filepath}"
                )

            key = (int(mode), float(frequency))
            traces[key] = _BoundaryTrace(
                k0=float(k0_values[0]),
                x=float(x_values[0]),
                y=subset["y"].to_numpy(dtype=np.float32, copy=True),
                u_re=subset["Re_U"].to_numpy(dtype=np.float32, copy=True),
                u_im=subset["Im_U"].to_numpy(dtype=np.float32, copy=True),
            )
        return traces

    def _resolve_key(self, mode: int, frequency: float):
        exact_key = (int(mode), float(frequency))
        if exact_key in self._pairs:
            return exact_key

        tolerance = max(1e-5, abs(float(frequency)) * 1e-7)
        candidates = [
            key for key in self.available_pairs
            if key[0] == int(mode) and np.isclose(key[1], frequency, rtol=0.0, atol=tolerance)
        ]
        if len(candidates) == 1:
            return candidates[0]
        raise KeyError(f"No boundary data for mode {mode}, frequency {frequency}")

    def has_pair(self, mode: int, frequency: float) -> bool:
        try:
            self._resolve_key(mode, frequency)
        except KeyError:
            return False
        return True

    def get_pair(self, mode: int, frequency: float) -> BoundaryPair:
        return self._pairs[self._resolve_key(mode, frequency)]

    def frequencies_for_mode(self, mode: int) -> np.ndarray:
        frequencies = [frequency for candidate_mode, frequency in self.available_pairs
                       if candidate_mode == int(mode)]
        return np.asarray(frequencies, dtype=np.float32)


class FEMFieldData:
    """Load nodal complex FEM fields sharing one physical P2 node ordering."""

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        if not self.filepath.is_file():
            raise FileNotFoundError(f"Missing FEM field file: {self.filepath}")

        dataframe = pd.read_csv(self.filepath)
        missing_columns = [
            column for column in FEM_FIELD_COLUMNS if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(f"{self.filepath} is missing columns: {missing_columns}")
        if dataframe.empty:
            raise ValueError(f"{self.filepath} contains no FEM field data")

        dataframe = dataframe.loc[:, list(FEM_FIELD_COLUMNS)].copy()
        for column in FEM_FIELD_COLUMNS:
            try:
                dataframe[column] = pd.to_numeric(dataframe[column], errors="raise")
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Column {column!r} in {self.filepath} is not numeric"
                ) from error
        if not np.isfinite(dataframe.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{self.filepath} contains NaN or infinite values")
        if (dataframe["f"] <= 0.0).any() or (dataframe["k0"] <= 0.0).any():
            raise ValueError(f"{self.filepath} contains a non-positive frequency or k0")

        modes = dataframe["mode"].to_numpy(dtype=np.float64)
        node_ids = dataframe["node_id"].to_numpy(dtype=np.float64)
        if not np.equal(modes, np.rint(modes)).all() or (modes < 0.0).any():
            raise ValueError(f"{self.filepath} contains an invalid mode index")
        if not np.equal(node_ids, np.rint(node_ids)).all() or (node_ids < 0.0).any():
            raise ValueError(f"{self.filepath} contains an invalid node_id")
        dataframe["mode"] = np.rint(modes).astype(np.int64)
        dataframe["node_id"] = np.rint(node_ids).astype(np.int64)

        duplicate_mask = dataframe.duplicated(
            subset=["f", "mode", "node_id"], keep=False
        )
        if duplicate_mask.any():
            duplicate = dataframe.loc[
                duplicate_mask, ["f", "mode", "node_id"]
            ].iloc[0].tolist()
            raise ValueError(f"Duplicate FEM node sample {duplicate} in {self.filepath}")

        self._cases: dict[tuple[float, int], FEMFieldCase] = {}
        reference_x = None
        reference_y = None
        reference_node_ids = None

        for (frequency, mode), subset in dataframe.groupby(["f", "mode"], sort=True):
            subset = subset.sort_values("node_id", kind="stable")
            case_node_ids = subset["node_id"].to_numpy(dtype=np.int64, copy=True)
            expected_ids = np.arange(case_node_ids.size, dtype=np.int64)
            if not np.array_equal(case_node_ids, expected_ids):
                raise ValueError(
                    f"node_id must be contiguous from 0 for f={frequency}, mode={mode}"
                )

            x = subset["x"].to_numpy(dtype=np.float64, copy=True)
            y = subset["y"].to_numpy(dtype=np.float64, copy=True)
            if np.unique(np.column_stack((x, y)), axis=0).shape[0] != x.size:
                raise ValueError(
                    f"Duplicate physical node coordinates for f={frequency}, mode={mode}"
                )

            k0_values = subset["k0"].to_numpy(dtype=np.float64)
            if not np.allclose(k0_values, k0_values[0], rtol=1e-7, atol=1e-7):
                raise ValueError(
                    f"k0 is not constant for f={frequency}, mode={mode}"
                )

            if reference_node_ids is None:
                reference_node_ids = case_node_ids
                reference_x = x
                reference_y = y
            elif (
                not np.array_equal(case_node_ids, reference_node_ids)
                or not np.allclose(x, reference_x, rtol=0.0, atol=1e-12)
                or not np.allclose(y, reference_y, rtol=0.0, atol=1e-12)
            ):
                raise ValueError(
                    "All FEM cases must share the same node ordering and coordinates"
                )

            key = (float(frequency), int(mode))
            self._cases[key] = FEMFieldCase(
                frequency=key[0],
                mode=key[1],
                k0=float(k0_values[0]),
                node_ids=case_node_ids,
                x=x,
                y=y,
                values=(
                    subset["Re_U"].to_numpy(dtype=np.float64, copy=True)
                    + 1j * subset["Im_U"].to_numpy(dtype=np.float64, copy=True)
                ),
            )

        self.available_cases = tuple(sorted(self._cases))
        first_case = self._cases[self.available_cases[0]]
        self.node_ids = first_case.node_ids
        self.x = first_case.x
        self.y = first_case.y
        self.size = self.node_ids.size

    def _resolve_key(self, frequency: float, mode: int) -> tuple[float, int]:
        exact_key = (float(frequency), int(mode))
        if exact_key in self._cases:
            return exact_key
        matches = [
            key for key in self.available_cases
            if key[1] == int(mode) and np.isclose(key[0], frequency)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"FEM case (f={frequency}, mode={mode}) is unavailable; "
                f"available cases: {self.available_cases}"
            )
        return matches[0]

    def has_case(self, frequency: float, mode: int) -> bool:
        try:
            self._resolve_key(frequency, mode)
        except KeyError:
            return False
        return True

    def case(self, frequency: float, mode: int) -> FEMFieldCase:
        return self._cases[self._resolve_key(frequency, mode)]


def load_symmetric_coo_matrix(
    filepath: str | Path, expected_size: int | None = None
) -> SymmetricCOOMatrix:
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Missing matrix file: {filepath}")

    dataframe = pd.read_csv(filepath)
    missing_columns = [column for column in MATRIX_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{filepath} is missing columns: {missing_columns}")
    if dataframe.empty:
        raise ValueError(f"{filepath} contains no matrix coefficient")

    dataframe = dataframe.loc[:, list(MATRIX_COLUMNS)].copy()
    for column in MATRIX_COLUMNS:
        try:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Column {column!r} in {filepath} is not numeric") from error
    if not np.isfinite(dataframe.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{filepath} contains NaN or infinite values")

    raw_rows = dataframe["row"].to_numpy(dtype=np.float64)
    raw_columns = dataframe["col"].to_numpy(dtype=np.float64)
    if (
        not np.equal(raw_rows, np.rint(raw_rows)).all()
        or not np.equal(raw_columns, np.rint(raw_columns)).all()
        or (raw_rows < 0.0).any()
        or (raw_columns < 0.0).any()
    ):
        raise ValueError(f"{filepath} contains invalid matrix indices")

    rows = np.rint(raw_rows).astype(np.int64)
    columns = np.rint(raw_columns).astype(np.int64)
    values = dataframe["value"].to_numpy(dtype=np.float64, copy=True)
    if (rows < columns).any():
        raise ValueError(f"{filepath} must contain the lower triangle only (row >= col)")
    if (values == 0.0).any():
        raise ValueError(f"{filepath} contains explicitly stored zero coefficients")
    if pd.DataFrame({"row": rows, "col": columns}).duplicated().any():
        raise ValueError(f"{filepath} contains duplicate matrix entries")

    inferred_size = int(rows.max()) + 1
    size = inferred_size if expected_size is None else int(expected_size)
    if inferred_size != size or int(columns.max()) >= size:
        raise ValueError(
            f"{filepath} has size {inferred_size}, expected {size}"
        )
    diagonal = np.sort(rows[rows == columns])
    if not np.array_equal(diagonal, np.arange(size, dtype=np.int64)):
        raise ValueError(f"{filepath} does not contain every diagonal coefficient")

    order = np.lexsort((columns, rows))
    return SymmetricCOOMatrix(
        size=size,
        rows=rows[order],
        columns=columns[order],
        values=values[order],
    )
