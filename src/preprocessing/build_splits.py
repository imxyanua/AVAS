"""Command line entry point that turns a dataset directory into a split manifest.

Example:
    python -m src.preprocessing.build_splits --config configs/dataset.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.preprocessing.splits import (
    assert_no_group_leakage,
    discover_from_config,
    split_from_config,
    summarize_splits,
    write_manifest,
)
from src.utils.config import DatasetConfig

logger = logging.getLogger("avas.build_splits")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="dataset configuration file (default: configs/dataset.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="manifest path (default: <processed_root>/splits.csv)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="optional path for a JSON summary of the split",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    config = DatasetConfig.from_yaml(args.config)
    samples = discover_from_config(config)
    logger.info("discovered %d videos in %s", len(samples), config.root)

    splits = split_from_config(samples, config)
    assert_no_group_leakage(splits)

    manifest_path = args.output or config.processed_root / "splits.csv"
    write_manifest(splits, manifest_path, relative_to=config.root)
    logger.info("wrote manifest to %s", manifest_path)

    summary = summarize_splits(splits)
    for split_name, counts in summary.items():
        logger.info("%s: %s", split_name, counts)

    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": config.name,
            "root": str(config.root),
            "seed": config.split.seed,
            "ratios": config.split.as_ratios(),
            "group_pattern": config.group_pattern,
            "manifest": str(manifest_path),
            "counts": summary,
        }
        args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("wrote summary to %s", args.summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
