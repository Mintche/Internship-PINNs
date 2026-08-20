"""Small, deterministic artifact writers used by training and plotting."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import jax
import numpy as np


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return jsonable(asdict(value))
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def short_digest(value: Any, length: int = 12) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def create_directory(path: Path) -> Path:
    """Create an output directory and reject accidental overwrite."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(fieldnames)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: jsonable(row.get(name, "")) for name in fieldnames})
    temporary.replace(path)


def environment_manifest() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "numpy": np.__version__,
    }


def package_name(index: int, label: str, maximum_length: int = 120) -> str:
    safe = "".join(character if character.isalnum() or character in "_-" else "_" for character in label)
    if len(safe) > maximum_length:
        safe = f"{safe[: maximum_length - 13]}_{short_digest(label)}"
    return f"pkg{index + 1:02d}_{safe}"

