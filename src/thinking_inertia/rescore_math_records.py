#!/usr/bin/env python3
"""Rescore math-task records with math-verify and rebuild a thinking-spectrum summary."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .eval_nonreason import _normalize_gold, _normalize_pred, _score_answer
from .eval_thinking_spectrum import MODE_SPECS, _group_summary, _summarize_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescore GSM8K/MATH records with math-verify.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing records.jsonl.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k", "math"],
        help="Datasets to rescore. Default: gsm8k math.",
    )
    parser.add_argument(
        "--accept-display-math-wrapper",
        action="store_true",
        help="Preserve summary compatibility with eval_thinking_spectrum.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Write timestamped backups before overwriting records/summary. Default: true.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="Overwrite without making backups.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def rebuild_summary(
    *,
    run_dir: Path,
    records: list[dict[str, Any]],
    old_summary: dict[str, Any],
    accept_display_math_wrapper: bool,
) -> dict[str, Any]:
    per_mode_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_dataset_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_mode_dataset_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        mode = str(record.get("mode", ""))
        dataset_name = str(record.get("dataset", ""))
        per_mode_records[mode].append(record)
        per_dataset_records[dataset_name].append(record)
        per_mode_dataset_records[f"{mode}::{dataset_name}"].append(record)

    config = dict(old_summary.get("config") or {})
    modes = list(config.get("modes") or sorted(per_mode_records))
    datasets = list(config.get("datasets") or sorted(per_dataset_records))
    comparison_rows = []
    for mode in modes:
        metrics = _summarize_records(
            per_mode_records.get(mode, []),
            accept_display_math_wrapper=accept_display_math_wrapper,
        )
        comparison_rows.append(
            {
                "mode": mode,
                "paper_mode": mode,
                "mode_name": MODE_SPECS.get(mode, {}).get("name", mode),
                "mode_title": MODE_SPECS.get(mode, {}).get("title", mode),
                **metrics,
            }
        )

    config["math_verify_rescored"] = True
    config["math_verify_rescored_at"] = dt.datetime.now().isoformat(timespec="seconds")
    return {
        **old_summary,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "overall_all_modes": _summarize_records(
            records,
            accept_display_math_wrapper=accept_display_math_wrapper,
        ),
        "per_mode": _group_summary(
            per_mode_records,
            accept_display_math_wrapper=accept_display_math_wrapper,
        ),
        "per_dataset": _group_summary(
            per_dataset_records,
            accept_display_math_wrapper=accept_display_math_wrapper,
        ),
        "per_mode_per_dataset": {
            mode: _group_summary(
                {
                    dataset_name: per_mode_dataset_records.get(f"{mode}::{dataset_name}", [])
                    for dataset_name in datasets
                },
                accept_display_math_wrapper=accept_display_math_wrapper,
            )
            for mode in modes
        },
        "mode_comparison": comparison_rows,
        "paths": {
            "records": str(run_dir / "records.jsonl"),
            "run_dir": str(run_dir),
        },
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    records_path = run_dir / "records.jsonl"
    summary_path = run_dir / "summary.json"
    if not records_path.exists():
        raise FileNotFoundError(records_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    records = load_records(records_path)
    old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    datasets = set(args.datasets)
    rescored = 0
    changed = 0

    for record in records:
        if record.get("dataset") not in datasets:
            continue
        if record.get("error"):
            continue
        if str(record.get("task_type", "")) != "math":
            continue
        prediction_text = str(
            record.get("boxed_inner")
            or record.get("answer_raw")
            or record.get("prediction")
            or record.get("answer_view_content")
            or record.get("content")
            or ""
        )
        gold_text = str(record.get("gold_answer") or record.get("gold_answer_source") or "")
        normalized_prediction = str(record.get("prediction") or "")
        normalized_gold = str(record.get("gold_answer") or "")
        correct, scorer = _score_answer(
            task_type="math",
            prediction_text=prediction_text,
            gold_text=gold_text,
            normalized_prediction=normalized_prediction,
            normalized_gold=normalized_gold,
        )
        old_correct = bool(record.get("is_correct"))
        old_scorer = str(record.get("answer_scorer") or "")
        record["is_correct"] = bool(correct)
        record["answer_scorer"] = scorer
        record["nonreason_correct"] = bool(record.get("api_answer_only_pass", record.get("nonreason_pass"))) and bool(correct)
        rescored += 1
        if old_correct != bool(correct) or old_scorer != scorer:
            changed += 1
        if rescored % 500 == 0:
            print(f"rescored {rescored} records...")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.backup:
        shutil.copy2(records_path, records_path.with_suffix(f".jsonl.bak_{timestamp}"))
        shutil.copy2(summary_path, summary_path.with_suffix(f".json.bak_{timestamp}"))

    write_records(records_path, records)
    summary = rebuild_summary(
        run_dir=run_dir,
        records=records,
        old_summary=old_summary,
        accept_display_math_wrapper=args.accept_display_math_wrapper,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"rescored={rescored} changed={changed}")
    print(f"records -> {records_path}")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
