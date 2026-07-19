#!/usr/bin/env python3
"""Save, load and evaluate scattered-waveguide US/MS checkpoints."""

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


FORMAT_VERSION = 1
SUPPORTED_FORMAT_VERSIONS = {FORMAT_VERSION}
PROVENANCE_PACKAGES = ("jax", "jaxlib", "optax", "numpy", "pandas", "matplotlib")
VALID_EXPORT_FIELDS = {"total", "us", "incident"}


@dataclass(frozen=True)
class USCase:
    frequency: float
    mode: int
    field_norm: float
    sigma: np.ndarray
    layers: list[dict[str, np.ndarray]]


@dataclass(frozen=True)
class MSSoundSpeedModel:
    """Tanh-bounded scattered slowness map used by pinn_scattered_waveguide.py."""

    layers: list[dict[str, np.ndarray]]
    c0: float
    ms_min: float
    ms_max: float

    def predict(
        self,
        x_norm: np.ndarray,
        y_norm: np.ndarray,
        batch_size: int = 65536,
    ) -> np.ndarray:
        x_values, y_values, original_shape = _prepare_normalized_coordinates(
            x_norm, y_norm
        )
        if batch_size <= 0:
            raise ValueError("batch_size must be strictly positive")

        coordinates = np.column_stack((x_values, y_values))
        predictions = []
        m0 = 1.0 / self.c0**2
        half_range = 0.5 * (self.ms_max - self.ms_min)
        midpoint = 0.5 * (self.ms_max + self.ms_min)

        for start in range(0, len(coordinates), batch_size):
            values = coordinates[start:start + batch_size]
            for layer in self.layers[:-1]:
                values = np.tanh(values @ layer["W"] + layer["b"])
            raw = values @ self.layers[-1]["W"] + self.layers[-1]["b"]
            ms = half_range * np.tanh(raw) + midpoint
            slowness_squared = m0 + ms[:, 0]
            if (slowness_squared <= 0.0).any():
                raise ValueError("The reconstructed slowness squared is non-positive")
            predictions.append(1.0 / np.sqrt(slowness_squared))

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
        x_norm, y_norm = _physical_to_normalized(x, y, length=length, height=height)
        return self.predict(x_norm, y_norm, batch_size=batch_size)


@dataclass(frozen=True)
class USCheckpoint:
    b_base: np.ndarray
    length: float
    height: float
    c0: float
    cases: dict[tuple[float, int], USCase]
    ms_model: MSSoundSpeedModel
    metadata: dict[str, Any]
    format_version: int = FORMAT_VERSION
    network_config: dict[str, Any] = field(default_factory=dict)
    best_validation_losses: dict[str, Any] = field(default_factory=dict)
    random_seed: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def available_cases(self) -> list[tuple[float, int]]:
        return sorted(self.cases, key=lambda item: (item[0], item[1]))

    def case(self, frequency: float, mode: int) -> USCase:
        matches = [
            case
            for (stored_frequency, stored_mode), case in self.cases.items()
            if stored_mode == int(mode) and np.isclose(stored_frequency, frequency)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Case (f={frequency}, mode={mode}) is unavailable. "
                f"Available cases: {self.available_cases()}"
            )
        return matches[0]

    def predict_us(
        self,
        frequency: float,
        mode: int,
        x_norm: np.ndarray,
        y_norm: np.ndarray,
        physical_units: bool = True,
        batch_size: int = 65536,
    ) -> np.ndarray:
        """Evaluate the learned scattered field us at normalized coordinates."""
        case = self.case(frequency, mode)
        x_values, y_values, original_shape = _prepare_normalized_coordinates(
            x_norm, y_norm
        )
        if batch_size <= 0:
            raise ValueError("batch_size must be strictly positive")

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

        us = np.concatenate(predictions, axis=0)
        complex_values = us[:, 0] + 1j * us[:, 1]
        if physical_units:
            complex_values *= case.field_norm
        return complex_values.reshape(original_shape)

    def predict_us_physical(
        self,
        frequency: float,
        mode: int,
        x: np.ndarray,
        y: np.ndarray,
        physical_units: bool = True,
        batch_size: int = 65536,
    ) -> np.ndarray:
        x_norm, y_norm = _physical_to_normalized(
            x, y, length=self.length, height=self.height
        )
        return self.predict_us(
            frequency,
            mode,
            x_norm,
            y_norm,
            physical_units=physical_units,
            batch_size=batch_size,
        )

    def predict_incident(
        self,
        frequency: float,
        mode: int,
        x_norm: np.ndarray,
        y_norm: np.ndarray,
        physical_units: bool = True,
        batch_size: int = 65536,
    ) -> np.ndarray:
        """Evaluate u0; normalized output uses the stored scattered-field norm."""
        case = self.case(frequency, mode)
        x_values, y_values, original_shape = _prepare_normalized_coordinates(
            x_norm, y_norm
        )
        if batch_size <= 0:
            raise ValueError("batch_size must be strictly positive")

        predictions = []
        beta_mode = _beta_mode(frequency, mode, self.height, self.c0)
        amplitude = _mode_amplitude(mode, self.height)
        for start in range(0, len(x_values), batch_size):
            stop = min(start + batch_size, len(x_values))
            x_physical = x_values[start:stop] * self.length
            y_physical = (y_values[start:stop] + 1.0) * self.height / 2.0
            mode_shape = amplitude * np.cos(mode * np.pi * y_physical / self.height)
            values = mode_shape * np.exp(1j * beta_mode * x_physical)
            if not physical_units:
                values = values / case.field_norm
            predictions.append(values)
        return np.concatenate(predictions).reshape(original_shape)

    def predict_incident_physical(
        self,
        frequency: float,
        mode: int,
        x: np.ndarray,
        y: np.ndarray,
        physical_units: bool = True,
        batch_size: int = 65536,
    ) -> np.ndarray:
        x_norm, y_norm = _physical_to_normalized(
            x, y, length=self.length, height=self.height
        )
        return self.predict_incident(
            frequency,
            mode,
            x_norm,
            y_norm,
            physical_units=physical_units,
            batch_size=batch_size,
        )

    def predict_total(
        self,
        frequency: float,
        mode: int,
        x_norm: np.ndarray,
        y_norm: np.ndarray,
        physical_units: bool = True,
        batch_size: int = 65536,
    ) -> np.ndarray:
        """Evaluate the total field U = u0 + us at normalized coordinates."""
        return self.predict_us(
            frequency,
            mode,
            x_norm,
            y_norm,
            physical_units=physical_units,
            batch_size=batch_size,
        ) + self.predict_incident(
            frequency,
            mode,
            x_norm,
            y_norm,
            physical_units=physical_units,
            batch_size=batch_size,
        )

    def predict_total_physical(
        self,
        frequency: float,
        mode: int,
        x: np.ndarray,
        y: np.ndarray,
        physical_units: bool = True,
        batch_size: int = 65536,
    ) -> np.ndarray:
        x_norm, y_norm = _physical_to_normalized(
            x, y, length=self.length, height=self.height
        )
        return self.predict_total(
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
        return self.ms_model.predict_physical(
            x,
            y,
            length=self.length,
            height=self.height,
            batch_size=batch_size,
        )


def _mode_amplitude(mode: int, height: float) -> float:
    return float(np.sqrt(1.0 / height) if int(mode) == 0 else np.sqrt(2.0 / height))


def _beta_mode(frequency: float, mode: int, height: float, c0: float) -> complex:
    k0 = 2.0 * np.pi * float(frequency) / float(c0)
    transverse = int(mode) * np.pi / float(height)
    return complex(np.sqrt(k0**2 - transverse**2 + 0j))


def _prepare_normalized_coordinates(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    x_values = np.asarray(x_norm, dtype=np.float32)
    y_values = np.asarray(y_norm, dtype=np.float32)
    if x_values.shape != y_values.shape:
        raise ValueError("x_norm and y_norm must have the same shape")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("Normalized coordinates contain NaN or infinite values")
    tolerance = np.float32(2e-6)
    if (
        (x_values < -1.0 - tolerance).any()
        or (x_values > 1.0 + tolerance).any()
        or (y_values < -1.0 - tolerance).any()
        or (y_values > 1.0 + tolerance).any()
    ):
        raise ValueError("Normalized coordinates lie outside [-1, 1]^2")
    return x_values.reshape(-1), y_values.reshape(-1), x_values.shape


def _physical_to_normalized(
    x: np.ndarray,
    y: np.ndarray,
    *,
    length: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
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
    return x_values / length, 2.0 * y_values / height - 1.0


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
                f"Invalid shape for {name}[{index}]: W{weights.shape}, b{biases.shape}"
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


def _validate_ms_bounds(ms_min: float, ms_max: float, c0: float, label: str) -> None:
    values = np.asarray([ms_min, ms_max, c0], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must be finite")
    if not (ms_min < 0.0 < ms_max):
        raise ValueError(f"{label} must satisfy ms_min < 0 < ms_max")
    if 1.0 / c0**2 + ms_min <= 0.0:
        raise ValueError(f"{label} make the minimum total slowness squared non-positive")


def collect_us_norms(
    params_us: Mapping[tuple[float, int], Any],
    mode_data: Mapping[int, Mapping[str, Any]],
) -> dict[tuple[float, int], float]:
    """Extract the scattered-field normalization used by pinn_scattered_waveguide.py."""
    norms: dict[tuple[float, int], float] = {}
    for frequency, mode in params_us:
        frequency = float(frequency)
        mode = int(mode)
        try:
            candidates = mode_data[mode]["U_norm"]
        except KeyError as error:
            raise KeyError(f"U_norm is missing for mode={mode}") from error
        matches = [
            float(value)
            for stored_frequency, value in candidates.items()
            if np.isclose(float(stored_frequency), frequency)
        ]
        if len(matches) != 1:
            raise KeyError(f"U_norm is missing or ambiguous for f={frequency}, mode={mode}")
        norms[(frequency, mode)] = matches[0]
    return norms


def save_us_checkpoint(
    path: Path | str,
    params_us: Mapping[tuple[float, int], Mapping[str, Any]],
    b_base: Any,
    field_norms: Mapping[tuple[float, int], float],
    *,
    length: float,
    height: float,
    c0: float,
    layers_ms: Any,
    ms_min: float,
    ms_max: float,
    network_config: Mapping[str, Any],
    best_validation_losses: Mapping[str, Any],
    random_seed: int,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Save US/MS weights without silently replacing an existing checkpoint."""
    if not params_us:
        raise ValueError("params_us is empty")
    if not (length > 0.0 and height > 0.0 and c0 > 0.0):
        raise ValueError("length, height, and c0 must be strictly positive")
    _validate_ms_bounds(ms_min, ms_max, c0, "ms bounds")
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
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing US checkpoint: {path}"
        )
    b_base_array = np.asarray(b_base)
    if (
        b_base_array.ndim != 2
        or b_base_array.shape[1] != 2
        or not np.isfinite(b_base_array).all()
    ):
        raise ValueError("b_base must have shape (n_fourier_features, 2)")

    arrays: dict[str, np.ndarray] = {"b_base": b_base_array}
    manifest_cases = []
    us_architecture: list[int] | None = None

    for case_index, ((frequency, mode), parameters) in enumerate(
        sorted(params_us.items(), key=lambda item: (float(item[0][0]), int(item[0][1])))
    ):
        frequency = float(frequency)
        mode = int(mode)
        norm_matches = [
            float(value)
            for (norm_frequency, norm_mode), value in field_norms.items()
            if int(norm_mode) == mode and np.isclose(float(norm_frequency), frequency)
        ]
        if len(norm_matches) != 1:
            raise KeyError(f"field_norm is missing or ambiguous for f={frequency}, mode={mode}")
        if not np.isfinite(norm_matches[0]) or norm_matches[0] <= 0.0:
            raise ValueError(f"field_norm must be positive for f={frequency}, mode={mode}")

        prefix = f"case_{case_index}"
        sigma = np.asarray(parameters["sigma"])
        if sigma.shape != (2,) or not np.isfinite(sigma).all():
            raise ValueError(f"Invalid sigma for f={frequency}, mode={mode}")
        arrays[f"{prefix}_sigma"] = sigma

        layers = parameters["layers"]
        case_architecture = _validate_layers(layers, f"layers_us[{frequency}, {mode}]")
        if case_architecture[0] != 2 * b_base_array.shape[0] or case_architecture[-1] != 2:
            raise ValueError(
                f"layers_us[{frequency}, {mode}] must map 2*fourier_features to two outputs"
            )
        if us_architecture is None:
            us_architecture = case_architecture
        elif case_architecture != us_architecture:
            raise ValueError("All US networks must share the same architecture")
        for layer_index, layer in enumerate(layers):
            arrays[f"{prefix}_layer_{layer_index}_W"] = np.asarray(layer["W"])
            arrays[f"{prefix}_layer_{layer_index}_b"] = np.asarray(layer["b"])
        manifest_cases.append(
            {
                "prefix": prefix,
                "frequency": frequency,
                "mode": mode,
                "field_norm": norm_matches[0],
                "n_layers": len(layers),
            }
        )

    ms_architecture = _validate_layers(layers_ms, "layers_ms")
    if ms_architecture[0] != 2 or ms_architecture[-1] != 1:
        raise ValueError("layers_ms must have two inputs and one output")
    for layer_index, layer in enumerate(layers_ms):
        arrays[f"ms_layer_{layer_index}_W"] = np.asarray(layer["W"])
        arrays[f"ms_layer_{layer_index}_b"] = np.asarray(layer["b"])

    config = _json_compatible(dict(network_config))
    if config.get("us_layers") != us_architecture:
        raise ValueError("network_config.us_layers does not match the US weights")
    if config.get("ms_layers") != ms_architecture:
        raise ValueError("network_config.ms_layers does not match the MS weights")

    manifest = {
        "format": "us_checkpoint_npz",
        "format_version": FORMAT_VERSION,
        "geometry": {"length": float(length), "height": float(height)},
        "c0": float(c0),
        "cases": manifest_cases,
        "sound_speed_model": {
            "prefix": "ms",
            "n_layers": len(layers_ms),
            "ms_min": float(ms_min),
            "ms_max": float(ms_max),
        },
        "network_config": config,
        "best_validation_losses": _json_compatible(dict(best_validation_losses)),
        "random_seed": int(random_seed),
        "provenance": _json_compatible(
            dict(provenance) if provenance is not None else collect_provenance()
        ),
        "metadata": _json_compatible(metadata_dict),
    }
    arrays["manifest_json"] = np.asarray(json.dumps(manifest, allow_nan=False, sort_keys=True))

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
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing US checkpoint: {path}"
            )
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def load_us_checkpoint(path: Path | str) -> USCheckpoint:
    """Load a scattered checkpoint using allow_pickle=False."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"].item()))
        if manifest.get("format") != "us_checkpoint_npz":
            raise ValueError(
                f"Expected us_checkpoint_npz in {path}, got {manifest.get('format')!r}"
            )
        format_version = int(manifest.get("format_version", -1))
        if format_version not in SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(
                f"Unsupported scattered checkpoint version {manifest.get('format_version')}; "
                f"supported versions: {sorted(SUPPORTED_FORMAT_VERSIONS)}"
            )
        c0 = float(manifest["c0"])

        cases: dict[tuple[float, int], USCase] = {}
        for description in manifest["cases"]:
            prefix = description["prefix"]
            layers = [
                {
                    "W": np.asarray(archive[f"{prefix}_layer_{index}_W"]),
                    "b": np.asarray(archive[f"{prefix}_layer_{index}_b"]),
                }
                for index in range(int(description["n_layers"]))
            ]
            _validate_layers(layers, f"{prefix}_layers_us")
            case = USCase(
                frequency=float(description["frequency"]),
                mode=int(description["mode"]),
                field_norm=float(description["field_norm"]),
                sigma=np.asarray(archive[f"{prefix}_sigma"]),
                layers=layers,
            )
            if not np.isfinite(case.field_norm) or case.field_norm <= 0.0:
                raise ValueError(f"Invalid field_norm for f={case.frequency}, mode={case.mode}")
            if case.sigma.shape != (2,) or not np.isfinite(case.sigma).all():
                raise ValueError(f"Invalid sigma for f={case.frequency}, mode={case.mode}")
            cases[(case.frequency, case.mode)] = case

        b_base = np.asarray(archive["b_base"])
        if b_base.ndim != 2 or b_base.shape[1] != 2 or not np.isfinite(b_base).all():
            raise ValueError("Invalid b_base in the checkpoint")

        sound_speed_description = manifest["sound_speed_model"]
        prefix = str(sound_speed_description["prefix"])
        layers_ms = [
            {
                "W": np.asarray(archive[f"{prefix}_layer_{index}_W"]),
                "b": np.asarray(archive[f"{prefix}_layer_{index}_b"]),
            }
            for index in range(int(sound_speed_description["n_layers"]))
        ]
        ms_architecture = _validate_layers(layers_ms, "layers_ms")
        if ms_architecture[0] != 2 or ms_architecture[-1] != 1:
            raise ValueError("layers_ms must have two inputs and one output")
        ms_min = float(sound_speed_description["ms_min"])
        ms_max = float(sound_speed_description["ms_max"])
        _validate_ms_bounds(ms_min, ms_max, c0, "Invalid ms bounds in the checkpoint")

    geometry = manifest["geometry"]
    return USCheckpoint(
        b_base=b_base,
        length=float(geometry["length"]),
        height=float(geometry["height"]),
        c0=c0,
        cases=cases,
        ms_model=MSSoundSpeedModel(
            layers=layers_ms,
            c0=c0,
            ms_min=ms_min,
            ms_max=ms_max,
        ),
        metadata=dict(manifest.get("metadata", {})),
        format_version=format_version,
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


def _predict_field(
    checkpoint: USCheckpoint,
    field: str,
    frequency: float,
    mode: int,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    physical_units: bool,
) -> np.ndarray:
    if field == "total":
        return checkpoint.predict_total(
            frequency, mode, x_norm, y_norm, physical_units=physical_units
        )
    if field == "us":
        return checkpoint.predict_us(
            frequency, mode, x_norm, y_norm, physical_units=physical_units
        )
    if field == "incident":
        return checkpoint.predict_incident(
            frequency, mode, x_norm, y_norm, physical_units=physical_units
        )
    raise ValueError(f"Unknown field {field!r}; expected one of {sorted(VALID_EXPORT_FIELDS)}")


def export_predictions_csv(
    path: Path | str,
    checkpoint: USCheckpoint,
    evaluation_grid: Mapping[str, np.ndarray],
    cases: list[tuple[float, int]] | None = None,
    physical_units: bool = True,
    field: str = "total",
) -> Path:
    """Evaluate selected scattered checkpoint cases on the fixed FEM grid."""
    if field not in VALID_EXPORT_FIELDS:
        raise ValueError(f"field must be one of {sorted(VALID_EXPORT_FIELDS)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_cases = checkpoint.available_cases() if cases is None else cases
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "f",
                "k0",
                "mode",
                "grid_i",
                "x",
                "y",
                "x_norm",
                "y_norm",
                "Re_U",
                "Im_U",
                "abs_U",
            )
        )
        for frequency, mode in selected_cases:
            values = _predict_field(
                checkpoint,
                field,
                frequency,
                mode,
                evaluation_grid["x_norm"],
                evaluation_grid["y_norm"],
                physical_units,
            )
            k0 = 2.0 * np.pi * frequency / checkpoint.c0
            for index, value in enumerate(values):
                writer.writerow(
                    (
                        f"{frequency:.17g}",
                        f"{k0:.17g}",
                        mode,
                        int(evaluation_grid["grid_i"][index]),
                        f"{evaluation_grid['x'][index]:.17g}",
                        f"{evaluation_grid['y'][index]:.17g}",
                        f"{evaluation_grid['x_norm'][index]:.17g}",
                        f"{evaluation_grid['y_norm'][index]:.17g}",
                        f"{value.real:.17g}",
                        f"{value.imag:.17g}",
                        f"{abs(value):.17g}",
                    )
                )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and evaluate scattered US checkpoints.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Display checkpoint contents")
    inspect_parser.add_argument("--checkpoint", required=True, type=Path)

    export_parser = subparsers.add_parser(
        "export", help="Evaluate the scattered checkpoint on the fixed FEM grid"
    )
    export_parser.add_argument("--checkpoint", required=True, type=Path)
    export_parser.add_argument("--grid-map", required=True, type=Path)
    export_parser.add_argument("--output", required=True, type=Path)
    export_parser.add_argument("--frequency", type=float)
    export_parser.add_argument("--mode", type=int)
    export_parser.add_argument(
        "--field",
        choices=sorted(VALID_EXPORT_FIELDS),
        default="total",
        help="Field to export; total matches FEM U by default",
    )
    export_parser.add_argument(
        "--normalized-output",
        action="store_true",
        help="Do not multiply outputs by field_norm",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_us_checkpoint(args.checkpoint)
    if args.command == "inspect":
        print(
            json.dumps(
                {
                    "format_version": checkpoint.format_version,
                    "cases": checkpoint.available_cases(),
                    "length": checkpoint.length,
                    "height": checkpoint.height,
                    "c0": checkpoint.c0,
                    "sound_speed": {
                        "ms_min": checkpoint.ms_model.ms_min,
                        "ms_max": checkpoint.ms_model.ms_max,
                    },
                    "network_config": checkpoint.network_config,
                    "best_validation_losses": checkpoint.best_validation_losses,
                    "random_seed": checkpoint.random_seed,
                    "provenance": checkpoint.provenance,
                    "metadata": checkpoint.metadata,
                },
                indent=2,
            )
        )
        return

    selected_cases = checkpoint.available_cases()
    if args.frequency is not None:
        selected_cases = [
            case for case in selected_cases if np.isclose(case[0], args.frequency)
        ]
    if args.mode is not None:
        selected_cases = [case for case in selected_cases if case[1] == args.mode]
    if not selected_cases:
        raise ValueError("No checkpoint case matches the requested filters")
    grid = load_evaluation_grid(args.grid_map)
    export_predictions_csv(
        args.output,
        checkpoint,
        grid,
        selected_cases,
        physical_units=not args.normalized_output,
        field=args.field,
    )
    print(f"{args.field} predictions for {selected_cases} exported to {args.output}")


if __name__ == "__main__":
    main()
