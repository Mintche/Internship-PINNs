#!/usr/bin/env python3
"""Save, load and evaluate multi-mode JAX UV networks without pickle."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


FORMAT_VERSION = 1


@dataclass(frozen=True)
class UVCase:
    frequency: float
    mode: int
    u_norm: float
    sigma: np.ndarray
    layers: list[dict[str, np.ndarray]]


@dataclass(frozen=True)
class UVCheckpoint:
    b_base: np.ndarray
    length: float
    height: float
    c0: float
    cases: dict[tuple[float, int], UVCase]
    metadata: dict[str, Any]

    def available_cases(self) -> list[tuple[float, int]]:
        return sorted(self.cases, key=lambda item: (item[0], item[1]))

    def case(self, frequency: float, mode: int) -> UVCase:
        matches = [
            case for (stored_frequency, stored_mode), case in self.cases.items()
            if stored_mode == mode and np.isclose(stored_frequency, frequency)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Cas (f={frequency}, mode={mode}) absent. "
                f"Cas disponibles: {self.available_cases()}"
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
            raise ValueError("x_norm et y_norm doivent avoir la meme forme")
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
            raise ValueError("x et y doivent avoir la meme forme")
        if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
            raise ValueError("Les coordonnees physiques contiennent NaN ou Inf")
        if not (self.length > 0.0 and self.height > 0.0):
            raise ValueError("La geometrie du checkpoint doit avoir L > 0 et H > 0")

        tolerance = 1e-7 * max(1.0, self.length, self.height)
        if (
            (x_values < -self.length - tolerance).any()
            or (x_values > self.length + tolerance).any()
            or (y_values < -tolerance).any()
            or (y_values > self.height + tolerance).any()
        ):
            raise ValueError(
                "Les coordonnees physiques sortent du domaine du checkpoint "
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


def collect_u_norms(
    params_uv: Mapping[tuple[float, int], Any],
    mode_data: Mapping[int, Mapping[str, Any]],
) -> dict[tuple[float, int], float]:
    """Extract the normalization used by pinn_waveguide_multi_mode.py."""
    norms: dict[tuple[float, int], float] = {}
    for frequency, mode in params_uv:
        frequency = float(frequency)
        mode = int(mode)
        try:
            candidates = mode_data[mode]["U_norm"]
        except KeyError as error:
            raise KeyError(f"U_norm absent pour le mode={mode}") from error
        matches = [
            float(value) for stored_frequency, value in candidates.items()
            if np.isclose(float(stored_frequency), frequency)
        ]
        if len(matches) != 1:
            raise KeyError(f"U_norm absent ou ambigu pour f={frequency}, mode={mode}")
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
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save UV weights, Fourier basis and scaling in a portable NPZ archive."""
    if not params_uv:
        raise ValueError("params_uv est vide")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"b_base": np.asarray(b_base)}
    manifest_cases = []

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
            raise KeyError(f"U_norm absent ou ambigu pour f={frequency}, mode={mode}")

        prefix = f"case_{case_index}"
        arrays[f"{prefix}_sigma"] = np.asarray(parameters["sigma"])
        layers = parameters["layers"]
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

    manifest = {
        "format": "uv_checkpoint_npz",
        "format_version": FORMAT_VERSION,
        "geometry": {"length": float(length), "height": float(height)},
        "c0": float(c0),
        "cases": manifest_cases,
        "metadata": dict(metadata or {}),
    }
    arrays["manifest_json"] = np.asarray(json.dumps(manifest, sort_keys=True))
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return path


def load_uv_checkpoint(path: Path | str) -> UVCheckpoint:
    """Load a checkpoint using allow_pickle=False."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"].item()))
        if manifest.get("format") != "uv_checkpoint_npz":
            raise ValueError(f"Format de checkpoint inconnu dans {path}")
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"Version {manifest.get('format_version')} non supportee; attendue: {FORMAT_VERSION}"
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

    geometry = manifest["geometry"]
    return UVCheckpoint(
        b_base=b_base,
        length=float(geometry["length"]),
        height=float(geometry["height"]),
        c0=float(manifest["c0"]),
        cases=cases,
        metadata=dict(manifest.get("metadata", {})),
    )


def load_evaluation_grid(path: Path | str) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    data = np.atleast_1d(data)
    names = {name.lower(): name for name in (data.dtype.names or ())}
    required = ("grid_i", "x", "y", "x_norm", "y_norm")
    if any(name not in names for name in required):
        raise ValueError(f"{path}: colonnes requises {required}")
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
    parser = argparse.ArgumentParser(description="Charge et evalue les checkpoints UV du PINN.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Afficher le contenu d'un checkpoint")
    inspect_parser.add_argument("--checkpoint", required=True, type=Path)

    export_parser = subparsers.add_parser("export", help="Evaluer le PINN sur la grille FEM fixe")
    export_parser.add_argument("--checkpoint", required=True, type=Path)
    export_parser.add_argument("--grid-map", required=True, type=Path)
    export_parser.add_argument("--output", required=True, type=Path)
    export_parser.add_argument("--frequency", type=float)
    export_parser.add_argument("--mode", type=int)
    export_parser.add_argument(
        "--normalized-output", action="store_true",
        help="Ne pas multiplier les sorties par U_norm",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_uv_checkpoint(args.checkpoint)
    if args.command == "inspect":
        print(json.dumps({
            "cases": checkpoint.available_cases(),
            "length": checkpoint.length,
            "height": checkpoint.height,
            "c0": checkpoint.c0,
            "metadata": checkpoint.metadata,
        }, indent=2))
        return

    selected_cases = checkpoint.available_cases()
    if args.frequency is not None:
        selected_cases = [case for case in selected_cases if np.isclose(case[0], args.frequency)]
    if args.mode is not None:
        selected_cases = [case for case in selected_cases if case[1] == args.mode]
    if not selected_cases:
        raise ValueError("Aucun cas du checkpoint ne correspond aux filtres demandes")
    grid = load_evaluation_grid(args.grid_map)
    export_predictions_csv(
        args.output, checkpoint, grid, selected_cases,
        physical_units=not args.normalized_output,
    )
    print(f"Predictions de {selected_cases} exportees dans {args.output}")


if __name__ == "__main__":
    main()
