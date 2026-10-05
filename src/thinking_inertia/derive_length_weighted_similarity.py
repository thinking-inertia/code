#!/usr/bin/env python3
"""
Derive length-weighted similarity metrics from an existing thinking-spectrum summary.json.

This works purely from already-aggregated summary metrics and does not rerun inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_LENGTH_FIELD = "mean_t_word_count"
DEFAULT_SIMILARITY_FIELD = "mean_thinking_rate"
DEFAULT_OUTPUT_SUFFIX = ".length_weighted.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a length-weighted similarity metric from an existing summary.json "
            "using length_field * similarity_field."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a summary.json file.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Optional output JSON path. Default: write next to input as "
            "<summary>.length_weighted.json."
        ),
    )
    parser.add_argument(
        "--length-field",
        default=DEFAULT_LENGTH_FIELD,
        help=f"Length field to use. Default: {DEFAULT_LENGTH_FIELD}",
    )
    parser.add_argument(
        "--similarity-field",
        default=DEFAULT_SIMILARITY_FIELD,
        help=f"Similarity field to use. Default: {DEFAULT_SIMILARITY_FIELD}",
    )
    parser.add_argument(
        "--derived-field",
        default="length_weighted_similarity",
        help="Name of the derived output field. Default: length_weighted_similarity",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected top-level object in {path}, got {type(data).__name__}")
    return data


def _derive_metric(
    obj: Any,
    *,
    length_field: str,
    similarity_field: str,
    derived_field: str,
) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            out[key] = _derive_metric(
                value,
                length_field=length_field,
                similarity_field=similarity_field,
                derived_field=derived_field,
            )

        length_value = obj.get(length_field)
        similarity_value = obj.get(similarity_field)
        if isinstance(length_value, (int, float)) and isinstance(similarity_value, (int, float)):
            out[derived_field] = float(length_value) * float(similarity_value)
        return out

    if isinstance(obj, list):
        return [
            _derive_metric(
                item,
                length_field=length_field,
                similarity_field=similarity_field,
                derived_field=derived_field,
            )
            for item in obj
        ]

    return obj


def _default_output_path(input_path: Path) -> Path:
    if input_path.suffix == ".json":
        return input_path.with_name(f"{input_path.stem}{DEFAULT_OUTPUT_SUFFIX}")
    return input_path.with_name(f"{input_path.name}{DEFAULT_OUTPUT_SUFFIX}")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    summary = _load_json(input_path)
    derived = _derive_metric(
        summary,
        length_field=args.length_field,
        similarity_field=args.similarity_field,
        derived_field=args.derived_field,
    )

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _default_output_path(input_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(derived, f, ensure_ascii=False, indent=2)

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
