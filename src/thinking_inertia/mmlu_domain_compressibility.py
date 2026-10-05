#!/usr/bin/env python3
"""
Analyze MMLU domain and subject compressibility.

Paper mapping:
- Main text: "Performance across disciplines"
- Appendix: full MMLU domain and subject-level tables
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from .experiment_utils import (
    DEFAULT_SUBJECT_GROUPS,
    aggregate_eval_records,
    dump_json,
    enrich_subject_groups,
    index_mmlu_subjects,
    load_json,
    load_jsonl,
    resolve_records_path,
    retention_to_baseline,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze MMLU no-thinking compressibility by domain and subject from "
            "existing thinking-spectrum runs."
        )
    )
    parser.add_argument(
        "--processed-mmlu",
        required=True,
        help="Path to processed/mmlu.jsonl.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Run spec in the form label=/abs/path/to/run_dir_or_records.jsonl. "
            "Repeatable; use the same label multiple times to merge split mode runs."
        ),
    )
    parser.add_argument(
        "--baseline-mode",
        default="mode1",
        help="Baseline mode used for gaps and retention. Default: mode1",
    )
    parser.add_argument(
        "--compare-modes",
        nargs="+",
        default=["mode2", "mode4", "mode6"],
        help="Modes to compare against baseline. Default: mode2 mode4 mode6",
    )
    parser.add_argument(
        "--subject-group-json",
        default="",
        help="Optional JSON file mapping group names to subject lists.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for CSV/JSON artifacts.",
    )
    return parser.parse_args()


def _parse_run_specs_grouped(specs: list[str]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected run spec label=path, got: {spec}")
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        path = Path(raw_path).expanduser().resolve()
        if not label:
            raise ValueError(f"Empty run label in spec: {spec}")
        grouped[label].append(path)
    return dict(grouped)


def _load_subject_groups(path_str: str, subjects: set[str]) -> dict[str, list[str]]:
    if not path_str:
        return enrich_subject_groups(subjects, DEFAULT_SUBJECT_GROUPS)
    raw = load_json(Path(path_str).expanduser().resolve())
    groups = {
        str(group): [str(subject) for subject in values]
        for group, values in raw.items()
        if isinstance(values, list)
    }
    return enrich_subject_groups(subjects, groups)


def _collect_subject_rows(
    label: str,
    run_paths: list[Path],
    subjects_by_id: dict[str, str],
    baseline_mode: str,
    allowed_modes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_meta: dict[str, Any] = {
        "label": label,
        "paths": [str(path) for path in run_paths],
        "records_paths": [],
        "model_name": "",
        "model_names": [],
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    model_names: set[str] = set()
    for run_path in run_paths:
        records_path = resolve_records_path(run_path)
        run_meta["records_paths"].append(str(records_path))
        records = load_jsonl(records_path)
        for row in records:
            if str(row.get("dataset", "")).strip() != "mmlu":
                continue
            mode = str(row.get("mode", "")).strip()
            if mode not in allowed_modes:
                continue
            row_id = str(row.get("id", "")).strip()
            subject = subjects_by_id.get(row_id, "")
            if not subject:
                continue
            grouped[(mode, subject)].append(row)
            model_name = str(row.get("model", "")).strip()
            if model_name:
                model_names.add(model_name)
                if not run_meta["model_name"]:
                    run_meta["model_name"] = model_name
    run_meta["model_names"] = sorted(model_names)

    baseline_by_subject: dict[str, dict[str, float | int]] = {}
    for (mode, subject), rows in grouped.items():
        metrics = aggregate_eval_records(rows)
        if mode == baseline_mode:
            baseline_by_subject[subject] = metrics

    subject_rows: list[dict[str, Any]] = []
    for (mode, subject), rows in sorted(grouped.items()):
        metrics = aggregate_eval_records(rows)
        baseline_metrics = baseline_by_subject.get(subject)
        baseline_accuracy = float(baseline_metrics["accuracy"]) if baseline_metrics else 0.0
        row = {
            "model_label": label,
            "model_name": run_meta["model_name"],
            "mode": mode,
            "subject": subject,
            **metrics,
            "baseline_mode": baseline_mode,
            "baseline_accuracy": baseline_accuracy if baseline_metrics else None,
            "accuracy_gap_to_mode1": (
                float(metrics["accuracy"]) - baseline_accuracy if baseline_metrics else None
            ),
            "retention_to_mode1": (
                retention_to_baseline(float(metrics["accuracy"]), baseline_accuracy)
                if baseline_metrics
                else None
            ),
        }
        subject_rows.append(row)
    return subject_rows, run_meta


def _group_subject_rows(
    subject_rows: list[dict[str, Any]],
    subject_groups: dict[str, list[str]],
    baseline_mode: str,
) -> list[dict[str, Any]]:
    subject_to_group = {
        subject: group
        for group, subjects in subject_groups.items()
        for subject in subjects
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        group = subject_to_group.get(str(row["subject"]), "mixed_professional")
        grouped[(str(row["model_label"]), str(row["mode"]), group)].append(row)

    baseline_group_accuracy: dict[tuple[str, str], float] = {}
    for (model_label, mode, group), rows in grouped.items():
        if mode != baseline_mode:
            continue
        total = sum(int(row["total"]) for row in rows)
        correct = sum(float(row["accuracy"]) * int(row["total"]) for row in rows)
        baseline_group_accuracy[(model_label, group)] = (correct / total) if total else 0.0

    out: list[dict[str, Any]] = []
    for (model_label, mode, group), rows in sorted(grouped.items()):
        total = sum(int(row["total"]) for row in rows)
        if total == 0:
            continue
        accuracy = sum(float(row["accuracy"]) * int(row["total"]) for row in rows) / total
        following = sum(float(row["following_rate"]) * int(row["total"]) for row in rows) / total
        following_correct = (
            sum(float(row["following_correct_rate"]) * int(row["total"]) for row in rows) / total
        )
        thinking = sum(float(row["mean_thinking_rate"]) * int(row["total"]) for row in rows) / total
        t_word = sum(float(row["mean_t_word_count"]) * int(row["total"]) for row in rows) / total
        baseline_accuracy = baseline_group_accuracy.get((model_label, group))
        out.append(
            {
                "model_label": model_label,
                "mode": mode,
                "subject_group": group,
                "total": total,
                "accuracy": accuracy,
                "following_rate": following,
                "following_correct_rate": following_correct,
                "mean_thinking_rate": thinking,
                "mean_t_word_count": t_word,
                "baseline_mode": baseline_mode,
                "baseline_accuracy": baseline_accuracy,
                "accuracy_gap_to_mode1": (
                    accuracy - baseline_accuracy if baseline_accuracy is not None else None
                ),
                "retention_to_mode1": (
                    retention_to_baseline(accuracy, baseline_accuracy)
                    if baseline_accuracy is not None
                    else None
                ),
            }
        )
    return out


def main() -> int:
    args = parse_args()
    processed_mmlu = Path(args.processed_mmlu).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects_by_id = index_mmlu_subjects(processed_mmlu)
    all_subjects = set(subjects_by_id.values())
    subject_groups = _load_subject_groups(args.subject_group_json, all_subjects)

    allowed_modes = {args.baseline_mode, *args.compare_modes}
    run_specs = _parse_run_specs_grouped(args.run)

    subject_rows: list[dict[str, Any]] = []
    run_meta_rows: list[dict[str, Any]] = []
    for label, run_paths in run_specs.items():
        rows, run_meta = _collect_subject_rows(
            label,
            run_paths,
            subjects_by_id,
            baseline_mode=args.baseline_mode,
            allowed_modes=allowed_modes,
        )
        subject_rows.extend(rows)
        run_meta_rows.append(run_meta)

    group_rows = _group_subject_rows(
        subject_rows,
        subject_groups=subject_groups,
        baseline_mode=args.baseline_mode,
    )

    subject_csv = output_dir / "subject_metrics.csv"
    group_csv = output_dir / "subject_group_metrics.csv"
    summary_json = output_dir / "summary.json"

    write_csv(subject_csv, subject_rows)
    write_csv(group_csv, group_rows)
    dump_json(
        summary_json,
        {
            "config": {
                "processed_mmlu": str(processed_mmlu),
                "runs": {label: [str(path) for path in paths] for label, paths in run_specs.items()},
                "baseline_mode": args.baseline_mode,
                "compare_modes": args.compare_modes,
            },
            "subject_groups": subject_groups,
            "run_meta": run_meta_rows,
            "subject_metrics": subject_rows,
            "subject_group_metrics": group_rows,
            "paths": {
                "subject_metrics_csv": str(subject_csv),
                "subject_group_metrics_csv": str(group_csv),
            },
        },
    )

    print(f"Wrote {subject_csv}")
    print(f"Wrote {group_csv}")
    print(f"Wrote {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
