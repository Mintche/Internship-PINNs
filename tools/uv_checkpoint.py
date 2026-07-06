#!/usr/bin/env python3
"""Save, load and evaluate multi-mode JAX UV networks without pickle."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = {1, FORMAT_VERSION}
PROVENANCE_PACKAGES = ("jax", "jaxlib", "optax", "numpy", "pandas", "matplotlib")


@dataclass(frozen=True)
class UVCase:
    frequency: float
    mode: int
    u_norm: float
    sigma: np.ndarray
    layers: list[dict[str, np.ndarray]]


@dataclass(frozen=True)
class SoundSpeedModel:
    """Bounded-slowness network used to reconstruct the physical wave speed."""

    layers: list[dict[str, np.ndarray]]
    c_min: float
    c_max: float

    def predict(
        self,
        x_norm: np.ndarray,
        y_norm: np.ndarray,
        batch_size: int = 65536,
    ) -> np.ndarray:
        """Evaluate c at normalized coordinates in [-1, 1]^2."""
        x_values = np.asarray(x_norm, dtype=np.float32)
        y_values = np.asarray(y_norm, dtype=np.float32)
        if x_values.shape != y_values.shape:
            raise ValueError("x_norm and y_norm must have the same shape")
        if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
            raise ValueError("Normalized coordinates contain NaN or infinite values")
        if batch_size <= 0:
            raise ValueError("batch_size must be strictly positive")

        original_shape = x_values.shape
        coordinates = np.column_stack((x_values.reshape(-1), y_values.reshape(-1)))
        predictions = []
        m_min = 1.0 / self.c_max**2
        m_max = 1.0 / self.c_min**2

        for start in range(0, len(coordinates), batch_size):
            values = coordinates[start:start + batch_size]
            for layer in self.layers[:-1]:
                values = np.tanh(values @ layer["W"] + layer["b"])
            logits = values @ self.layers[-1]["W"] + self.layers[-1]["b"]
            # Clipping avoids overflow while preserving sigmoid accuracy at the
            # precision used by the stored network weights.
            sigmoid = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
            slowness_squared = m_min + (m_max - m_min) * sigmoid
            predictions.append(1.0 / np.sqrt(slowness_squared[:, 0]))

        if not predictions:
            return np.empty(original_shape, dtype=np.float32)
        return np.concatenate(predictions).reshape(original_shape)

    def predict_physical(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        length: float,
        height: float,
        batch_size: int = 65536,
    ) -> np.ndarray:
        """Evaluate c at physical coordinates in [-length, length] x [0, height]."""
        x_values = np.asarray(x, dtype=np.float64)
        y_values = np.asarray(y, dtype=np.float64)
        if x_values.shape != y_values.shape:
            raise ValueError("x and y must have the same shape")
        if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
            raise ValueError("Physical coordinates contain NaN or infinite values")
        if not (length > 0.0 and height > 0.0):
            raise ValueError("Checkpoint geometry must satisfy L > 0 and H > 0")

        tolerance = 1e-7 * max(1.0, length, height)
        if (
            (x_values < -length - tolerance).any()
            or (x_values > length + tolerance).any()
            or (y_values < -tolerance).any()
            or (y_values > height + tolerance).any()
        ):
            raise ValueError(
                "Physical coordinates lie outside the checkpoint domain "
                f"[-{length}, {length}] x [0, {height}]"
            )
        return self.predict(
            x_values / length,
            2.0 * y_values / height - 1.0,
            batch_size=batch_size,
        )


@dataclass(frozen=True)
class UVCheckpoint:
    b_base: np.ndarray
    length: float
    height: float
    c0: float
    cases: dict[tuple[float, int], UVCase]
    metadata: dict[str, Any]
    format_version: int = 1
    sound_speed: SoundSpeedModel | None = None
    network_config: dict[str, Any] = field(default_factory=dict)
    best_validation_losses: dict[str, Any] = field(default_factory=dict)
    random_seed: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def available_cases(self) -> list[tuple[float, int]]:
        return sorted(self.cases, key=lambda item: (item[0], item[1]))

    def case(self, frequency: float, mode: int) -> UVCase:
        matches = [
            case for (stored_frequency, stored_mode), case in self.cases.items()
            if stored_mode == mode and np.isclose(stored_frequency, frequency)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Case (f={frequency}, mode={mode}) is unavailable. "
                f"Available cases: {self.available_cases()}"
            )
        return matches[0]

    def predict(
        self,
        frequency: float,
        mode: int,
        x_norm: np.ndarray,
        y_norm: np.ndarray,
        physical_units: bool = True,
        batch_size: int = 65536,
    ) -> np.ndarray:
        """Evaluate the complex UV network at normalized coordinates in [-1, 1]^2."""
        case = self.case(frequency, mode)
        x_values = np.asarray(x_norm, dtype=np.float32)
        y_values = np.asarray(y_norm, dtype=np.float32)
        if x_values.shape != y_values.shape:
            raise ValueError("x_norm and y_norm must have the same shape")
        original_shape = x_values.shape
        x_values = x_values.reshape(-1)
        y_values = y_values.reshape(-1)
        predictions = []

        b_scaled = self.b_base * case.sigma
        for start in range(0, len(x_values), batch_size):
            stop = min(start + batch_size, len(x_values))
            x_physical = x_values[start:stop] * self.length
            y_physical = (y_values[start:stop] + 1.0) * self.height / 2.0
            projection = (
                x_physical[:, None] * b_scaled[None, :, 0]
                + y_physical[:, None] * b_scaled[None, :, 1]
            )
            values = np.concatenate((np.cos(projection), np.sin(projection)), axis=1)
            for layer in case.layers[:-1]:
                values = np.tanh(values @ layer["W"] + layer["b"])
            values = values @ case.layers[-1]["W"] + case.layers[-1]["b"]
            predictions.append(values)

        uv = np.concatenate(predictions, axis=0)
        complex_values = uv[:, 0] + 1j * uv[:, 1]
        if physical_units:
            complex_values *= case.u_norm
        return complex_values.reshape(original_shape)

    def predict_physical(
        self,
        frequency: float,
        mode: int,
        x: np.ndarray,
        y: np.ndarray,
        physical_units: bool = True,
        batch_size: int = 65536,
    ) -> np.ndarray:
        """Evaluate a UV network at physical coordinates in [-L, L] x [0, H]."""
        x_values = np.asarray(x, dtype=np.float64)
        y_values = np.asarray(y, dtype=np.float64)
        if x_values.shape != y_values.shape:
            raise ValueError("x and y must have the same shape")
        if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
            raise ValueError("Physical coordinates contain NaN or infinite values")
        if not (self.length > 0.0 and self.height > 0.0):
            raise ValueError("Checkpoint geometry must satisfy L > 0 and H > 0")

        tolerance = 1e-7 * max(1.0, self.length, self.height)
        if (
            (x_values < -self.length - tolerance).any()
            or (x_values > self.length + tolerance).any()
            or (y_values < -tolerance).any()
            or (y_values > self.height + tolerance).any()
        ):
            raise ValueError(
                "Physical coordinates lie outside the checkpoint domain "
                f"[-{self.length}, {self.length}] x [0, {self.height}]"
            )

        x_norm = x_values / self.length
        y_norm = 2.0 * y_values / self.height - 1.0
        return self.predict(
            frequency,
            mode,
            x_norm,
            y_norm,
            physical_units=physical_units,
            batch_size=batch_size,
        )

    def predict_sound_speed_physical(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 65536,
    ) -> np.ndarray:
        """Evaluate the stored sound-speed network at physical coordinates."""
        if self.sound_speed is None:
            raise ValueError(
                "The checkpoint does not contain a sound-speed network; "
                "a format-v2 checkpoint is required"
            )
        return self.sound_speed.predict_physical(
            x,
            y,
            length=self.length,
            height=self.height,
            batch_size=batch_size,
        )


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _git_provenance(repository_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit_process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status_process = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    commit = commit_process.stdout.strip() if commit_process.returncode == 0 else None
    dirty = bool(status_process.stdout) if status_process.returncode == 0 else None
    return commit or None, dirty


def collect_provenance(repository_root: Path | str | None = None) -> dict[str, Any]:
    """Collect lightweight run provenance without making checkpointing fragile."""
    root = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    commit, dirty = _git_provenance(root)
    package_versions: dict[str, str | None] = {}
    for package in PROVENANCE_PACKAGES:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "packages": package_versions,
    }


def _validate_layers(layers: Any, name: str) -> list[int]:
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError(f"{name} must contain at least one layer")
    architecture: list[int] = []
    previous_output: int | None = None
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping) or "W" not in layer or "b" not in layer:
            raise ValueError(f"{name}[{index}] must contain W and b")
        weights = np.asarray(layer["W"])
        biases = np.asarray(layer["b"])
        if weights.ndim != 2 or biases.ndim != 1 or weights.shape[1] != biases.size:
            raise ValueError(
                f"Invalid shape for {name}[{index}]: "
                f"W{weights.shape}, b{biases.shape}"
            )
        if previous_output is not None and weights.shape[0] != previous_output:
            raise ValueError(f"Disconnected layers in {name} at index {index}")
        if not np.isfinite(weights).all() or not np.isfinite(biases).all():
            raise ValueError(f"{name}[{index}] contains NaN or infinite values")
        if index == 0:
            architecture.append(int(weights.shape[0]))
        architecture.append(int(weights.shape[1]))
        previous_output = int(weights.shape[1])
    return architecture


def collect_u_norms(
    params_uv: Mapping[tuple[float, int], Any],
    mode_data: Mapping[int, Mapping[str, Any]],
) -> dict[tuple[float, int], float]:
    """Extract the normalization used by pinn_waveguide_multi_modes.py."""
    norms: dict[tuple[float, int], float] = {}
    for frequency, mode in params_uv:
        frequency = float(frequency)
        mode = int(mode)
        try:
            candidates = mode_data[mode]["U_norm"]
        except KeyError as error:
            raise KeyError(f"U_norm is missing for mode={mode}") from error
        matches = [
            float(value) for stored_frequency, value in candidates.items()
            if np.isclose(float(stored_frequency), frequency)
        ]
        if len(matches) != 1:
            raise KeyError(f"U_norm is missing or ambiguous for f={frequency}, mode={mode}")
        norms[(frequency, mode)] = matches[0]
    return norms


def save_uv_checkpoint(
    path: Path | str,
    params_uv: Mapping[tuple[float, int], Mapping[str, Any]],
    b_base: Any,
    u_norms: Mapping[tuple[float, int], float],
    *,
    length: float,
    height: float,
    c0: float,
    layers_m: Any,
    c_min: float,
    c_max: float,
    network_config: Mapping[str, Any],
    best_validation_losses: Mapping[str, Any],
    random_seed: int,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Save UV/M weights, inference configuration and provenance in NPZ format."""
    if not params_uv:
        raise ValueError("params_uv is empty")
    if not (length > 0.0 and height > 0.0 and c0 > 0.0):
        raise ValueError("length, height, and c0 must be strictly positive")
    if not (0.0 < c_min <= c_max):
        raise ValueError("Sound-speed bounds must satisfy 0 < c_min <= c_max")
    if not isinstance(random_seed, (int, np.integer)) or int(random_seed) < 0:
        raise ValueError("random_seed must be a non-negative integer")

    metadata_dict = dict(metadata or {})
    missing_metadata = [
        name for name in ("defect_name", "contrast_ratio")
        if name not in metadata_dict
    ]
    if missing_metadata:
        raise ValueError(f"Missing required metadata: {missing_metadata}")
    if not str(metadata_dict["defect_name"]).strip():
        raise ValueError("defect_name cannot be empty")
    contrast_ratio = float(metadata_dict["contrast_ratio"])
    if not np.isfinite(contrast_ratio) or contrast_ratio <= 0.0:
        raise ValueError("contrast_ratio must be strictly positive")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    b_base_array = np.asarray(b_base)
    if (
        b_base_array.ndim != 2
        or b_base_array.shape[1] != 2
        or not np.isfinite(b_base_array).all()
    ):
        raise ValueError("b_base must have shape (n_fourier_features, 2)")
    arrays: dict[str, np.ndarray] = {"b_base": b_base_array}
    manifest_cases = []
    uv_architecture: list[int] | None = None

    for case_index, ((frequency, mode), parameters) in enumerate(
        sorted(params_uv.items(), key=lambda item: (float(item[0][0]), int(item[0][1])))
    ):
        frequency = float(frequency)
        mode = int(mode)
        norm_matches = [
            float(value) for (norm_frequency, norm_mode), value in u_norms.items()
            if int(norm_mode) == mode and np.isclose(float(norm_frequency), frequency)
        ]
        if len(norm_matches) != 1:
            raise KeyError(f"U_norm is missing or ambiguous for f={frequency}, mode={mode}")

        prefix = f"case_{case_index}"
        sigma = np.asarray(parameters["sigma"])
        if sigma.shape != (2,) or not np.isfinite(sigma).all():
            raise ValueError(f"Invalid sigma for f={frequency}, mode={mode}")
        arrays[f"{prefix}_sigma"] = sigma
        layers = parameters["layers"]
        case_architecture = _validate_layers(layers, f"layers_uv[{frequency}, {mode}]")
        if uv_architecture is None:
            uv_architecture = case_architecture
        elif case_architecture != uv_architecture:
            raise ValueError("All UV networks must share the same architecture")
        for layer_index, layer in enumerate(layers):
            arrays[f"{prefix}_layer_{layer_index}_W"] = np.asarray(layer["W"])
            arrays[f"{prefix}_layer_{layer_index}_b"] = np.asarray(layer["b"])
        manifest_cases.append({
            "prefix": prefix,
            "frequency": frequency,
            "mode": mode,
            "u_norm": norm_matches[0],
            "n_layers": len(layers),
        })

    m_architecture = _validate_layers(layers_m, "layers_m")
    if m_architecture[0] != 2 or m_architecture[-1] != 1:
        raise ValueError("layers_m must have two inputs and one output")
    for layer_index, layer in enumerate(layers_m):
        arrays[f"m_layer_{layer_index}_W"] = np.asarray(layer["W"])
        arrays[f"m_layer_{layer_index}_b"] = np.asarray(layer["b"])

    config = _json_compatible(dict(network_config))
    if config.get("uv_layers") != uv_architecture:
        raise ValueError("network_config.uv_layers does not match the UV weights")
    if config.get("m_layers") != m_architecture:
        raise ValueError("network_config.m_layers does not match the M weights")

    manifest = {
        "format": "uv_checkpoint_npz",
        "format_version": FORMAT_VERSION,
        "geometry": {"length": float(length), "height": float(height)},
        "c0": float(c0),
        "cases": manifest_cases,
        "sound_speed_model": {
            "prefix": "m",
            "n_layers": len(layers_m),
            "c_min": float(c_min),
            "c_max": float(c_max),
        },
        "network_config": config,
        "best_validation_losses": _json_compatible(dict(best_validation_losses)),
        "random_seed": int(random_seed),
        "provenance": _json_compatible(
            dict(provenance) if provenance is not None else collect_provenance()
        ),
        "metadata": _json_compatible(metadata_dict),
    }
    arrays["manifest_json"] = np.asarray(
        json.dumps(manifest, allow_nan=False, sort_keys=True)
    )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def load_uv_checkpoint(path: Path | str) -> UVCheckpoint:
    """Load a checkpoint using allow_pickle=False."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"].item()))
        if manifest.get("format") != "uv_checkpoint_npz":
            raise ValueError(f"Unknown checkpoint format in {path}")
        format_version = int(manifest.get("format_version", -1))
        if format_version not in SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(
                f"Unsupported checkpoint version {manifest.get('format_version')}; "
                f"supported versions: {sorted(SUPPORTED_FORMAT_VERSIONS)}"
            )

        cases: dict[tuple[float, int], UVCase] = {}
        for description in manifest["cases"]:
            prefix = description["prefix"]
            layers = [
                {
                    "W": np.asarray(archive[f"{prefix}_layer_{index}_W"]),
                    "b": np.asarray(archive[f"{prefix}_layer_{index}_b"]),
                }
                for index in range(int(description["n_layers"]))
            ]
            case = UVCase(
                frequency=float(description["frequency"]),
                mode=int(description["mode"]),
                u_norm=float(description["u_norm"]),
                sigma=np.asarray(archive[f"{prefix}_sigma"]),
                layers=layers,
            )
            cases[(case.frequency, case.mode)] = case
        b_base = np.asarray(archive["b_base"])

        sound_speed = None
        if format_version >= 2:
            description = manifest["sound_speed_model"]
            prefix = str(description["prefix"])
            layers_m = [
                {
                    "W": np.asarray(archive[f"{prefix}_layer_{index}_W"]),
                    "b": np.asarray(archive[f"{prefix}_layer_{index}_b"]),
                }
                for index in range(int(description["n_layers"]))
            ]
            _validate_layers(layers_m, "layers_m")
            c_min = float(description["c_min"])
            c_max = float(description["c_max"])
            if not (0.0 < c_min <= c_max):
                raise ValueError("Invalid sound-speed bounds in the checkpoint")
            sound_speed = SoundSpeedModel(
                layers=layers_m,
                c_min=c_min,
                c_max=c_max,
            )

    geometry = manifest["geometry"]
    return UVCheckpoint(
        b_base=b_base,
        length=float(geometry["length"]),
        height=float(geometry["height"]),
        c0=float(manifest["c0"]),
        cases=cases,
        metadata=dict(manifest.get("metadata", {})),
        format_version=format_version,
        sound_speed=sound_speed,
        network_config=dict(manifest.get("network_config", {})),
        best_validation_losses=dict(manifest.get("best_validation_losses", {})),
        random_seed=(
            int(manifest["random_seed"])
            if manifest.get("random_seed") is not None
            else None
        ),
        provenance=dict(manifest.get("provenance", {})),
    )


def load_evaluation_grid(path: Path | str) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    data = np.atleast_1d(data)
    names = {name.lower(): name for name in (data.dtype.names or ())}
    required = ("grid_i", "x", "y", "x_norm", "y_norm")
    if any(name not in names for name in required):
        raise ValueError(f"{path}: required columns are {required}")
    order = np.argsort(np.asarray(data[names["grid_i"]], dtype=int))
    return {
        name: np.asarray(data[names[name]], dtype=int if name == "grid_i" else float)[order]
        for name in required
    }


def export_predictions_csv(
    path: Path | str,
    checkpoint: UVCheckpoint,
    evaluation_grid: Mapping[str, np.ndarray],
    cases: list[tuple[float, int]] | None = None,
    physical_units: bool = True,
) -> Path:
    """Evaluate selected UV networks on the exact FEM grid and export one CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_cases = checkpoint.available_cases() if cases is None else cases
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "f", "k0", "mode", "grid_i", "x", "y", "x_norm", "y_norm",
            "Re_U", "Im_U", "abs_U",
        ))
        for frequency, mode in selected_cases:
            values = checkpoint.predict(
                frequency, mode, evaluation_grid["x_norm"], evaluation_grid["y_norm"],
                physical_units=physical_units,
            )
            k0 = 2.0 * np.pi * frequency / checkpoint.c0
            for index, value in enumerate(values):
                writer.writerow((
                    f"{frequency:.17g}", f"{k0:.17g}", mode,
                    int(evaluation_grid["grid_i"][index]),
                    f"{evaluation_grid['x'][index]:.17g}",
                    f"{evaluation_grid['y'][index]:.17g}",
                    f"{evaluation_grid['x_norm'][index]:.17g}",
                    f"{evaluation_grid['y_norm'][index]:.17g}",
                    f"{value.real:.17g}", f"{value.imag:.17g}", f"{abs(value):.17g}",
                ))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and evaluate PINN UV checkpoints.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Display checkpoint contents")
    inspect_parser.add_argument("--checkpoint", required=True, type=Path)

    export_parser = subparsers.add_parser(
        "export", help="Evaluate the PINN on the fixed FEM grid"
    )
    export_parser.add_argument("--checkpoint", required=True, type=Path)
    export_parser.add_argument("--grid-map", required=True, type=Path)
    export_parser.add_argument("--output", required=True, type=Path)
    export_parser.add_argument("--frequency", type=float)
    export_parser.add_argument("--mode", type=int)
    export_parser.add_argument(
        "--normalized-output", action="store_true",
        help="Do not multiply outputs by U_norm",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_uv_checkpoint(args.checkpoint)
    if args.command == "inspect":
        print(json.dumps({
            "format_version": checkpoint.format_version,
            "cases": checkpoint.available_cases(),
            "length": checkpoint.length,
            "height": checkpoint.height,
            "c0": checkpoint.c0,
            "sound_speed": (
                {
                    "c_min": checkpoint.sound_speed.c_min,
                    "c_max": checkpoint.sound_speed.c_max,
                }
                if checkpoint.sound_speed is not None
                else None
            ),
            "network_config": checkpoint.network_config,
            "best_validation_losses": checkpoint.best_validation_losses,
            "random_seed": checkpoint.random_seed,
            "provenance": checkpoint.provenance,
            "metadata": checkpoint.metadata,
        }, indent=2))
        return

    selected_cases = checkpoint.available_cases()
    if args.frequency is not None:
        selected_cases = [case for case in selected_cases if np.isclose(case[0], args.frequency)]
    if args.mode is not None:
        selected_cases = [case for case in selected_cases if case[1] == args.mode]
    if not selected_cases:
        raise ValueError("No checkpoint case matches the requested filters")
    grid = load_evaluation_grid(args.grid_map)
    export_predictions_csv(
        args.output, checkpoint, grid, selected_cases,
        physical_units=not args.normalized_output,
    )
    print(f"Predictions for {selected_cases} exported to {args.output}")


if __name__ == "__main__":
    main()
