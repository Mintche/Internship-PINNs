"""Deterministic identities and overwrite guards for PINN training runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_SCHEMA_VERSION = 1
_SAFE_LABEL = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_manifest(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe, deterministically ordered experiment manifest."""
    manifest = _json_compatible(dict(configuration))
    if not isinstance(manifest, dict):
        raise TypeError("configuration must serialize to a JSON object")
    manifest.setdefault("manifest_schema_version", MANIFEST_SCHEMA_VERSION)
    # This validates non-finite floats and unsupported objects immediately.
    json.dumps(manifest, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return manifest


def configuration_id(configuration: Mapping[str, Any], length: int = 12) -> str:
    """Hash every effective configuration value into a short stable identifier."""
    if not isinstance(length, int) or not 8 <= length <= 64:
        raise ValueError("configuration id length must be between 8 and 64")
    manifest = canonical_manifest(configuration)
    payload = json.dumps(
        manifest,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def numeric_token(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("run-name numeric values must be finite")
    token = f"{value:.10g}".replace("-", "m").replace(".", "p").replace("+", "")
    return token


def build_run_tag(
    *,
    formulation: str,
    random_seed: int,
    c_min: float,
    c_max: float,
    configuration: Mapping[str, Any],
) -> str:
    """Encode formulation, seed, bounds, and a full-config hash in a file-safe tag."""
    formulation = str(formulation).strip().lower()
    if not _SAFE_LABEL.fullmatch(formulation):
        raise ValueError(
            "formulation must contain only lowercase letters, digits, '-' or '_'"
        )
    if not isinstance(random_seed, (int, np.integer)) or int(random_seed) < 0:
        raise ValueError("random_seed must be a non-negative integer")
    if not (0.0 < float(c_min) <= float(c_max)):
        raise ValueError("sound-speed bounds must satisfy 0 < c_min <= c_max")
    digest = configuration_id(configuration)
    return (
        f"{formulation}_seed{int(random_seed)}_"
        f"c{numeric_token(c_min)}to{numeric_token(c_max)}_cfg{digest}"
    )


def ensure_outputs_available(*paths: Path | str) -> None:
    """Refuse a run before training if any of its final artifacts already exists."""
    collisions = [Path(path) for path in paths if Path(path).exists()]
    if collisions:
        formatted = ", ".join(str(path) for path in collisions)
        raise FileExistsError(
            "Refusing to overwrite existing run artifact(s): "
            f"{formatted}. Change the seed/configuration or archive the existing run."
        )


def write_manifest_exclusive(
    path: Path | str,
    configuration: Mapping[str, Any],
    *,
    experiment_id: str,
) -> Path:
    """Atomically create a sidecar JSON file without replacing an existing one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": str(experiment_id),
        "configuration": canonical_manifest(configuration),
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
        # A hard link creates the destination atomically and fails if it exists.
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path
