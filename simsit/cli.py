"""Command-line entry points for reproducible SImSiT runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .simulate import simulate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a validated SImSiT simulation.")
    parser.add_argument("config", type=Path, help="YAML simulation configuration")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of files in the configured output root",
    )
    args = parser.parse_args(argv)

    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    simulate(
        config,
        overwrite=args.overwrite,
        input_paths=(args.config,),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
