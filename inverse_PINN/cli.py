"""Command-line entry points for one inverse run or a campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import create_directory, write_json
from .config import InverseConfig
from .training import run_training
from .variants import parse_variant


def _variants(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("variants must be a non-empty unique CSV list")
    try:
        for item in result:
            parse_variant(item)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return result


def _seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("seeds must be a non-empty unique CSV list")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one variant and seed")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--variant", required=True)
    run.add_argument("--seed", required=True, type=int)
    campaign = commands.add_parser("campaign", help="run a Cartesian variant/seed campaign")
    campaign.add_argument("--config", required=True, type=Path)
    campaign.add_argument("--variants", required=True, type=_variants)
    campaign.add_argument("--seeds", required=True, type=_seeds)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = InverseConfig.from_json(arguments.config)
    if arguments.command == "run":
        parse_variant(arguments.variant)
        directory = run_training(config, arguments.variant, arguments.seed)
        print(directory)
        return 0

    campaign_root = create_directory(
        config.output_root / f"{config.source.stem}_campaign"
    )
    runs_root = campaign_root / "runs"
    runs_root.mkdir()
    write_json(
        campaign_root / "campaign_manifest.json",
        {
            "config_source": str(config.source),
            "variants": list(arguments.variants),
            "seeds": list(arguments.seeds),
            "runs": [
                {"variant": variant, "seed": seed}
                for variant in arguments.variants for seed in arguments.seeds
            ],
        },
    )
    for variant in arguments.variants:
        for seed in arguments.seeds:
            directory = run_training(
                config, variant, seed, output_parent=runs_root
            )
            print(directory, flush=True)
    print(campaign_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
