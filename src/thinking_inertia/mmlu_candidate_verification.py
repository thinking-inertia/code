#!/usr/bin/env python3
"""
Build and analyze MMLU candidate-visible yes/no verification rewrites.

Paper mapping:
- Appendix: "Candidate visibility"
- Purpose: test whether showing a proposed answer makes the same MMLU item more compressible
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from .experiment_utils import (
    dump_json,
    load_json,
    load_jsonl,
    parse_run_specs,
    resolve_records_path,
    sample_rows,
    write_csv,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and analyze MMLU MCQ-to-yes/no candidate verification rewrites."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build MCQ base and bool verification datasets.")
    build.add_argument("--source", required=True, help="Path to processed/mmlu.jsonl.")
    build.add_argument("--out-root", required=True, help="Experiment root output directory.")
    build.add_argument("--max-samples", type=int, default=3000, help="Max source samples.")
    build.add_argument("--shuffle", action="store_true", help="Shuffle before sampling.")
    build.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    build.add_argument("--positive-per-question", type=int, default=1, help="Positive verify copies per question.")
    build.add_argument("--negative-per-question", type=int, default=1, help="Negative verify copies per question.")

    analyze = subparsers.add_parser("analyze", help="Analyze candidate verification runs.")
    analyze.add_argument("--manifest", required=True, help="Path to candidate_verification_manifest.json.")
    analyze.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form label=/abs/path/to/run_dir_or_records.jsonl. Repeatable.",
    )
    analyze.add_argument("--mcq-dataset", required=True, help="MCQ base dataset name.")
    analyze.add_argument("--bool-dataset", required=True, help="Bool verification dataset name.")
    analyze.add_argument("--output-dir", required=True, help="Output directory.")
    return parser.parse_args()


def _make_base_row(row: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(row)
    out["dataset"] = "mmlu_mcq_base"
    return out


def _make_verify_row(
    row: dict[str, Any],
    *,
    source_id: str,
    subject: str,
    polarity: str,
    choice_label: str,
    choice_text: str,
    copy_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = deepcopy(row)
    out["dataset"] = "mmlu_bool_verify"
    suffix = f"{polarity}_{copy_index}"
    out["id"] = f"{source_id}::verify_{suffix}"
    out["task_type"] = "bool"
    out["choices"] = []
    out["answer"] = "yes" if polarity == "pos" else "no"
    original_question = str(row.get("question", "")).strip()
    proposed_line = f"Proposed answer: {choice_text}"
    verify_question = (
        f"{original_question}\n{proposed_line}\nIs the proposed answer correct? "
        "Answer yes or no."
    )
    out["question"] = verify_question
    meta = dict(row.get("meta") or {})
    meta.update(
        {
            "source_id": source_id,
            "subject": subject,
            "verification_polarity": polarity,
            "proposed_choice_label": choice_label,
            "proposed_choice_text": choice_text,
        }
    )
    out["meta"] = meta
    manifest = {
        "bool_id": out["id"],
        "dataset_name": "mmlu_bool_verify",
        "source_id": source_id,
        "source_subject": subject,
        "verification_polarity": polarity,
        "proposed_choice_label": choice_label,
        "proposed_choice_text": choice_text,
    }
    return out, manifest


def _build(args: argparse.Namespace) -> int:
    source_path = Path(args.source).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    processed_dir = out_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    rows = sample_rows(load_jsonl(source_path), args.max_samples, args.shuffle, args.seed)
    mcq_rows: list[dict[str, Any]] = []
    bool_rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []

    for row in rows:
        source_id = str(row.get("id", "")).strip()
        subject = str((row.get("meta") or {}).get("subject", "")).strip()
        answer_label = str(row.get("answer", "")).strip()
        choices = [choice for choice in row.get("choices", []) if isinstance(choice, dict)]
        choice_by_label = {
            str(choice.get("label", "")).strip(): str(choice.get("text", "")).strip()
            for choice in choices
        }
        incorrect = [(label, text) for label, text in choice_by_label.items() if label != answer_label]
        if answer_label not in choice_by_label or not incorrect:
            continue

        mcq_rows.append(_make_base_row(row))
        correct_text = choice_by_label[answer_label]
        for i in range(args.positive_per_question):
            verify_row, manifest = _make_verify_row(
                row,
                source_id=source_id,
                subject=subject,
                polarity="pos",
                choice_label=answer_label,
                choice_text=correct_text,
                copy_index=i,
            )
            bool_rows.append(verify_row)
            manifest_entries.append(manifest)

        for i in range(args.negative_per_question):
            neg_label, neg_text = incorrect[i % len(incorrect)]
            verify_row, manifest = _make_verify_row(
                row,
                source_id=source_id,
                subject=subject,
                polarity="neg",
                choice_label=neg_label,
                choice_text=neg_text,
                copy_index=i,
            )
            bool_rows.append(verify_row)
            manifest_entries.append(manifest)

    mcq_path = processed_dir / "mmlu_mcq_base.jsonl"
    bool_path = processed_dir / "mmlu_bool_verify.jsonl"
    manifest_path = out_root / "candidate_verification_manifest.json"
    write_jsonl(mcq_path, mcq_rows)
    write_jsonl(bool_path, bool_rows)
    dump_json(
        manifest_path,
        {
            "config": {
                "source": str(source_path),
                "max_samples": args.max_samples,
                "shuffle": args.shuffle,
                "seed": args.seed,
                "positive_per_question": args.positive_per_question,
                "negative_per_question": args.negative_per_question,
            },
            "entries": manifest_entries,
            "paths": {
                "processed_dir": str(processed_dir),
                "mcq_dataset": str(mcq_path),
                "bool_dataset": str(bool_path),
            },
        },
    )
    print(f"Wrote {mcq_path}")
    print(f"Wrote {bool_path}")
    print(f"Wrote {manifest_path}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    manifest = load_json(Path(args.manifest).expanduser().resolve())
    entries = [entry for entry in manifest.get("entries", []) if isinstance(entry, dict)]
    entries_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    entries_by_bool_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        source_id = str(entry.get("source_id", "")).strip()
        bool_id = str(entry.get("bool_id", "")).strip()
        if source_id:
            entries_by_source[source_id].append(entry)
        if bool_id:
            entries_by_bool_id[bool_id] = entry

    run_specs = parse_run_specs(args.run)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []

    for label, run_path in run_specs.items():
        records = load_jsonl(resolve_records_path(run_path))
        grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
        model_name = ""
        for row in records:
            dataset = str(row.get("dataset", "")).strip()
            mode = str(row.get("mode", "")).strip()
            row_id = str(row.get("id", "")).strip()
            if dataset not in {args.mcq_dataset, args.bool_dataset}:
                continue
            grouped[(dataset, mode, row_id)] = row
            if not model_name:
                model_name = str(row.get("model", "")).strip()

        modes = sorted({mode for _, mode, _ in grouped.keys()})
        for mode in modes:
            matched_sources = []
            for source_id, source_entries in entries_by_source.items():
                base_record = grouped.get((args.mcq_dataset, mode, source_id))
                if base_record is None:
                    continue
                bool_records = [
                    grouped.get((args.bool_dataset, mode, str(entry.get("bool_id", "")).strip()))
                    for entry in source_entries
                ]
                if any(record is None for record in bool_records):
                    continue
                matched_sources.append((source_id, base_record, source_entries, bool_records))

            if not matched_sources:
                continue

            mcq_total = len(matched_sources)
            mcq_accuracy = sum(1 for _, base, _, _ in matched_sources if base.get("is_correct")) / mcq_total
            all_bool_records = [record for _, _, _, records_for_source in matched_sources for record in records_for_source]
            bool_total = len(all_bool_records)
            bool_accuracy = sum(1 for record in all_bool_records if record.get("is_correct")) / bool_total
            pos_records = [
                record
                for _, _, entries_for_source, records_for_source in matched_sources
                for entry, record in zip(entries_for_source, records_for_source)
                if str(entry.get("verification_polarity", "")) == "pos"
            ]
            neg_records = [
                record
                for _, _, entries_for_source, records_for_source in matched_sources
                for entry, record in zip(entries_for_source, records_for_source)
                if str(entry.get("verification_polarity", "")) == "neg"
            ]
            mode_rows.append(
                {
                    "model_label": label,
                    "model_name": model_name,
                    "mode": mode,
                    "matched_source_total": mcq_total,
                    "mcq_accuracy": mcq_accuracy,
                    "bool_verify_accuracy": bool_accuracy,
                    "verify_pos_accuracy": (
                        sum(1 for record in pos_records if record.get("is_correct")) / len(pos_records)
                        if pos_records
                        else 0.0
                    ),
                    "verify_neg_accuracy": (
                        sum(1 for record in neg_records if record.get("is_correct")) / len(neg_records)
                        if neg_records
                        else 0.0
                    ),
                    "answer_space_drop": bool_accuracy - mcq_accuracy,
                }
            )

            by_subject: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]]] = defaultdict(list)
            for _, base_record, source_entries, bool_records_for_source in matched_sources:
                subject = str((source_entries[0] if source_entries else {}).get("source_subject", "")).strip()
                by_subject[subject].append((base_record, source_entries, bool_records_for_source))

            for subject, rows_for_subject in sorted(by_subject.items()):
                base_records = [base for base, _, _ in rows_for_subject]
                bool_records_subject = [record for _, _, records_for_source in rows_for_subject for record in records_for_source]
                pos_subject = [
                    record
                    for _, entries_for_source, records_for_source in rows_for_subject
                    for entry, record in zip(entries_for_source, records_for_source)
                    if str(entry.get("verification_polarity", "")) == "pos"
                ]
                neg_subject = [
                    record
                    for _, entries_for_source, records_for_source in rows_for_subject
                    for entry, record in zip(entries_for_source, records_for_source)
                    if str(entry.get("verification_polarity", "")) == "neg"
                ]
                mcq_accuracy_subject = sum(1 for record in base_records if record.get("is_correct")) / len(base_records)
                bool_accuracy_subject = (
                    sum(1 for record in bool_records_subject if record.get("is_correct")) / len(bool_records_subject)
                )
                subject_rows.append(
                    {
                        "model_label": label,
                        "model_name": model_name,
                        "mode": mode,
                        "subject": subject,
                        "matched_source_total": len(base_records),
                        "mcq_accuracy": mcq_accuracy_subject,
                        "bool_verify_accuracy": bool_accuracy_subject,
                        "verify_pos_accuracy": (
                            sum(1 for record in pos_subject if record.get("is_correct")) / len(pos_subject)
                            if pos_subject
                            else 0.0
                        ),
                        "verify_neg_accuracy": (
                            sum(1 for record in neg_subject if record.get("is_correct")) / len(neg_subject)
                            if neg_subject
                            else 0.0
                        ),
                        "answer_space_drop": bool_accuracy_subject - mcq_accuracy_subject,
                    }
                )

    mode_csv = output_dir / "mode_metrics.csv"
    subject_csv = output_dir / "subject_metrics.csv"
    summary_path = output_dir / "candidate_verification_summary.json"
    write_csv(mode_csv, mode_rows)
    write_csv(subject_csv, subject_rows)
    dump_json(
        summary_path,
        {
            "config": {
                "manifest": args.manifest,
                "runs": {label: str(path) for label, path in run_specs.items()},
                "mcq_dataset": args.mcq_dataset,
                "bool_dataset": args.bool_dataset,
            },
            "mode_metrics": mode_rows,
            "subject_metrics": subject_rows,
            "paths": {
                "mode_metrics_csv": str(mode_csv),
                "subject_metrics_csv": str(subject_csv),
            },
        },
    )
    print(f"Wrote {mode_csv}")
    print(f"Wrote {subject_csv}")
    print(f"Wrote {summary_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "build":
        return _build(args)
    return _analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
