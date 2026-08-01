"""Portable pressure and slowness checkpoints stored as NPZ without pickle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from .config import Case
from .models import FieldModel


def _metadata_array(value: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _read_metadata(archive: np.lib.npyio.NpzFile) -> dict[str, Any]:
    raw = archive["metadata_json"]
    if raw.ndim != 0 or raw.dtype.kind not in "US":
        raise ValueError("Invalid checkpoint metadata")
    return json.loads(str(raw.item()))


def save_pressure_checkpoint(
    path: Path,
    models: Mapping[Case, FieldModel],
    field_scales: Mapping[Case, float],
    *,
    variant: str,
    package_index: int,
    monitor_loss: float,
) -> None:
    ordered = tuple(sorted(models))
    metadata: dict[str, Any] = {
        "format_version": 1,
        "variant": variant,
        "package_index": int(package_index),
        "monitor_loss": float(monitor_loss),
        "cases": [],
    }
    arrays: dict[str, np.ndarray] = {}
    for index, case in enumerate(ordered):
        model = models[case]
        prefix = f"case_{index:04d}"
        layers = model.params["layers"]
        metadata["cases"].append(
            {
                **case.manifest(),
                "prefix": prefix,
                "layer_count": len(layers),
                "field_scale": float(field_scales[case]),
            }
        )
        arrays[f"{prefix}_b_base"] = np.asarray(model.b_base)
        arrays[f"{prefix}_sigma"] = np.asarray(model.params["sigma"])
        for layer_index, layer in enumerate(layers):
            arrays[f"{prefix}_layer_{layer_index:03d}_W"] = np.asarray(layer["W"])
            arrays[f"{prefix}_layer_{layer_index:03d}_b"] = np.asarray(layer["b"])
    arrays["metadata_json"] = _metadata_array(metadata)
    np.savez_compressed(Path(path), **arrays)


def load_pressure_checkpoint(
    path: Path,
) -> tuple[dict[Case, FieldModel], dict[Case, float], dict[str, Any]]:
    models: dict[Case, FieldModel] = {}
    scales: dict[Case, float] = {}
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = _read_metadata(archive)
        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported pressure checkpoint format")
        for item in metadata["cases"]:
            case = Case(float(item["frequency"]), int(item["mode"]), int(item["incidence"]))
            prefix = item["prefix"]
            layers = [
                {
                    "W": jnp.asarray(archive[f"{prefix}_layer_{index:03d}_W"]),
                    "b": jnp.asarray(archive[f"{prefix}_layer_{index:03d}_b"]),
                }
                for index in range(int(item["layer_count"]))
            ]
            models[case] = FieldModel(
                {"layers": layers, "sigma": jnp.asarray(archive[f"{prefix}_sigma"])},
                jnp.asarray(archive[f"{prefix}_b_base"]),
            )
            scales[case] = float(item["field_scale"])
    return models, scales, metadata


def save_material_checkpoint(
    path: Path,
    params: Mapping[str, Any],
    *,
    variant: str,
    package_index: int,
    monitor_loss: float,
) -> None:
    layers = params["layers"]
    metadata = {
        "format_version": 1,
        "variant": variant,
        "package_index": int(package_index),
        "monitor_loss": float(monitor_loss),
        "layer_count": len(layers),
    }
    arrays = {"metadata_json": _metadata_array(metadata)}
    for index, layer in enumerate(layers):
        arrays[f"layer_{index:03d}_W"] = np.asarray(layer["W"])
        arrays[f"layer_{index:03d}_b"] = np.asarray(layer["b"])
    np.savez_compressed(Path(path), **arrays)


def load_material_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = _read_metadata(archive)
        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported material checkpoint format")
        layers = [
            {
                "W": jnp.asarray(archive[f"layer_{index:03d}_W"]),
                "b": jnp.asarray(archive[f"layer_{index:03d}_b"]),
            }
            for index in range(int(metadata["layer_count"]))
        ]
    return {"layers": layers}, metadata

