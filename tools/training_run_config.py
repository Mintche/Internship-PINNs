"""Small, strict JSON/CLI layer for reproducible PINN training runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


def _canonical_json(values: dict[str, Any]) -> str:
    return json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, dict):
        raise ValueError(f"Training config must contain one JSON object: {path}")
    return values


@dataclass(frozen=True)
class TrainingRunConfig:
    values: dict[str, Any]
    identifier: str
    source: Path | None
    managed_output: bool

    def resolve_path(self, repository_root: Path, key: str) -> Path:
        value = self.values[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"Training config key {key!r} must be a non-empty path")
        path = Path(value).expanduser()
        return path if path.is_absolute() else repository_root / path

    def prepare_output_root(self, repository_root: Path) -> Path:
        """Create a run root and archive the effective config without overwriting."""
        output_root = self.resolve_path(repository_root, "output_root")
        output_root.mkdir(parents=True, exist_ok=not self.managed_output)
        if self.managed_output:
            config_path = output_root / "run_config.json"
            with config_path.open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "config_id": self.identifier,
                        "source": str(self.source) if self.source is not None else None,
                        "values": self.values,
                    },
                    stream,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                stream.write("\n")
        return output_root


def load_training_run_config(
    defaults: dict[str, Any],
    *,
    formulation: str,
    argv: Sequence[str] = (),
) -> TrainingRunConfig:
    """Load a strict config and apply safe per-run CLI overrides.

    An empty ``argv`` preserves the historical module defaults, which keeps the
    training modules importable by tests and analysis tools. Config-driven runs
    get an exclusive output directory and therefore cannot overwrite an older run.
    """
    parser = argparse.ArgumentParser(
        description=f"Run the {formulation} PINN with a reproducible JSON config"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=str)
    parser.add_argument(
        "--show-plots",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args(list(argv))

    values = copy.deepcopy(defaults)
    source = args.config.resolve() if args.config is not None else None
    if source is not None:
        supplied = _load_json_object(source)
        unknown = sorted(set(supplied) - set(defaults))
        if unknown:
            raise ValueError(f"Unknown training config keys: {unknown}")
        values.update(supplied)

    if args.seed is not None:
        values["random_seed"] = args.seed
    if args.output_root is not None:
        values["output_root"] = args.output_root
    if args.show_plots is not None:
        values["show_plots"] = args.show_plots

    if values.get("formulation") != formulation:
        raise ValueError(
            f"Config formulation must be {formulation!r}, got {values.get('formulation')!r}"
        )
    seed = values.get("random_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    if not isinstance(values.get("show_plots"), bool):
        raise ValueError("show_plots must be a boolean")
    if not isinstance(values.get("output_root"), str) or not values["output_root"]:
        raise ValueError("output_root must be a non-empty path")

    canonical = _canonical_json(values)
    identifier = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    managed_output = source is not None or args.output_root is not None
    config = TrainingRunConfig(
        values=values,
        identifier=identifier,
        source=source,
        managed_output=managed_output,
    )
    if args.print_config:
        print(
            json.dumps(
                {"config_id": identifier, "values": values},
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(0)
    return config
