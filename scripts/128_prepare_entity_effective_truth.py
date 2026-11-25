"""Prepare an immutable-derived truth file for the entity effective-cost run.

The mandatory CR1000Xe backbone records data but is not a sensing channel. The
legacy environment nevertheless expects one quality column per configured
sensor. This adapter adds a constant quality of one for that non-sensing row
and records the derivation without changing any truth target or event column.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-column", default="agent_context_quality_cr1000xe_backbone")
    args = parser.parse_args()

    truth = pd.read_csv(args.input)
    if args.quality_column in truth.columns:
        raise ValueError(f"quality column already exists: {args.quality_column}")
    truth[args.quality_column] = 1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    truth.to_csv(args.output, index=False)
    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "added_column": args.quality_column,
        "value": 1.0,
        "reason": "mandatory non-sensing CR1000Xe backbone requires environment-aligned quality metadata",
        "truth_targets_changed": False,
        "event_columns_changed": False,
    }
    args.output.with_suffix(".derivation.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
