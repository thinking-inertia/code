#!/usr/bin/env python3
"""
Build and analyze numeric MMLU/MMLU-Pro answer-space rewrites.

Paper mapping:
- Main text: "Performance under answer-space rewrites"
- Appendix: candidate visibility and open numeric controls

This script builds matched MCQ, yes/no verification, and open-ended numeric
datasets from numeric-answer MMLU / MMLU-Pro items, then analyzes evaluator
outputs from existing thinking-spectrum runs.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from fractions import Fraction
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


NUMERIC_RE = re.compile(r"^\s*[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?%?\s*$")
MCQ_ONLY_PATTERNS = [
    r"which of the following",
    r"which one of the following",
    r"which statement",
    r"which answer choice",
    r"all of the following",
    r"among the following",
    r"from the following",
    r"following statements?",
    r"following is true",
    r"following is not true",
    r"following are true",
    r"following are false",
    r"following best",
    r"best answer",
    r"except",
    r"not one of the following",
    r"choose one answer",
]
MCQ_ONLY_RES = [re.compile(pattern, re.IGNORECASE) for pattern in MCQ_ONLY_PATTERNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and analyze numeric MMLU/MMLU-Pro answer-space rewrites."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build matched MCQ/bool/open-end datasets.")
    build.add_argument(
        "--source",
        action="append",
        required=True,
        help="Repeatable processed dataset path. Typical inputs: processed/mmlu.jsonl and processed/mmlu_pro.jsonl.",
    )
    build.add_argument("--out-root", required=True, help="Experiment root output directory.")
    build.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional total cap after combining filtered sources (0 = all).",
    )
    build.add_argument(
        "--max-samples-per-source",
        type=int,
        default=0,
        help="Optional per-source cap after filtering (0 = all).",
    )
    build.add_argument("--shuffle", action="store_true", help="Shuffle before sampling.")
    build.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    build.add_argument(
        "--allow-mcq-phrasing",
        action="store_true",
        help="Keep numeric items even if the question still contains strong MCQ-only phrasing.",
    )
    build.add_argument("--positive-per-question", type=int, default=1, help="Positive bool copies per question.")
    build.add_argument("--negative-per-question", type=int, default=1, help="Negative bool copies per question.")

    analyze = subparsers.add_parser("analyze", help="Analyze matched MCQ/bool/open-end runs.")
    analyze.add_argument("--manifest", required=True, help="Path to numeric_answer_space_manifest.json.")
    analyze.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form label=/abs/path/to/run_dir_or_records.jsonl. Repeatable.",
    )
    analyze.add_argument(
        "--mcq-dataset",
        default="mmlu_numeric_mcq",
        help="Base MCQ dataset name. Default: mmlu_numeric_mcq",
    )
    analyze.add_argument(
        "--bool-dataset",
        default="mmlu_numeric_bool_verify",
        help="Bool verification dataset name. Default: mmlu_numeric_bool_verify",
    )
    analyze.add_argument(
        "--open-dataset",
        default="mmlu_numeric_open_end",
        help="Open-ended dataset name. Default: mmlu_numeric_open_end",
    )
    analyze.add_argument("--output-dir", required=True, help="Output directory.")
    return parser.parse_args()


def _choice_text_map(row: dict[str, Any]) -> dict[str, str]:
    return {
        str(choice.get("label", "")).strip(): str(choice.get("text", "")).strip()
        for choice in row.get("choices", [])
        if isinstance(choice, dict) and str(choice.get("label", "")).strip()
    }


def _gold_choice_text(row: dict[str, Any]) -> str:
    return _choice_text_map(row).get(str(row.get("answer", "")).strip(), "").strip()


def _normalize_space(text: str) -> str:
    return " ".join(str(text).strip().split())


def _looks_numeric(text: str) -> bool:
    return bool(NUMERIC_RE.fullmatch(_normalize_space(text)))


def _answer_format_kind(text: str) -> str:
    cleaned = _normalize_space(text).replace(",", "")
    if cleaned.endswith("%"):
        return "percent"
    if "/" in cleaned:
        return "fraction"
    if "." in cleaned:
        return "decimal"
    return "integer"


def _is_blocked_by_mcq_phrase(question: str) -> bool:
    return any(pattern.search(question) for pattern in MCQ_ONLY_RES)


def _source_group_fields(row: dict[str, Any]) -> tuple[str, str]:
    meta = row.get("meta") or {}
    subject = str(meta.get("subject", "")).strip()
    if subject:
        return "subject", subject
    category = str(meta.get("category", "")).strip()
    if category:
        return "category", category
    return "unknown", "unknown"


def _clean_open_end_question(question: str) -> str:
    text = _normalize_space(question)
    text = re.sub(
        r"\s*choose one answer from the following:?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return _normalize_space(text)


def _source_uid(row: dict[str, Any]) -> str:
    return f"{str(row.get('dataset', '')).strip()}::{str(row.get('id', '')).strip()}"


def _make_base_row(row: dict[str, Any], *, source_uid: str, source_group_type: str, source_group: str) -> dict[str, Any]:
    out = deepcopy(row)
    out["dataset"] = "mmlu_numeric_mcq"
    out["id"] = source_uid
    meta = dict(row.get("meta") or {})
    meta.update(
        {
            "source_uid": source_uid,
            "source_id": str(row.get("id", "")).strip(),
            "source_dataset": str(row.get("dataset", "")).strip(),
            "source_group_type": source_group_type,
            "source_group": source_group,
        }
    )
    out["meta"] = meta
    return out


def _make_verify_row(
    row: dict[str, Any],
    *,
    source_uid: str,
    source_group_type: str,
    source_group: str,
    polarity: str,
    choice_label: str,
    choice_text: str,
    copy_index: int,
) -> tuple[dict[str, Any], str]:
    out = deepcopy(row)
    out["dataset"] = "mmlu_numeric_bool_verify"
    out["id"] = f"{source_uid}::verify_{polarity}_{copy_index}"
    out["task_type"] = "bool"
    out["choices"] = []
    out["answer"] = "yes" if polarity == "pos" else "no"
    original_question = _normalize_space(str(row.get("question", "")).strip())
    out["question"] = (
        f"{original_question}\n"
        f"Proposed answer: {choice_text}\n"
        "Is the proposed answer correct? Answer yes or no."
    )
    meta = dict(row.get("meta") or {})
    meta.update(
        {
            "source_uid": source_uid,
            "source_id": str(row.get("id", "")).strip(),
            "source_dataset": str(row.get("dataset", "")).strip(),
            "source_group_type": source_group_type,
            "source_group": source_group,
            "verification_polarity": polarity,
            "proposed_choice_label": choice_label,
            "proposed_choice_text": choice_text,
        }
    )
    out["meta"] = meta
    return out, out["id"]


def _make_open_end_row(
    row: dict[str, Any],
    *,
    source_uid: str,
    source_group_type: str,
    source_group: str,
    gold_text: str,
) -> dict[str, Any]:
    out = deepcopy(row)
    out["dataset"] = "mmlu_numeric_open_end"
    out["id"] = source_uid
    out["task_type"] = "math"
    out["choices"] = []
    out["answer"] = gold_text
    out["question"] = _clean_open_end_question(str(row.get("question", "")).strip())
    meta = dict(row.get("meta") or {})
    meta.update(
        {
            "source_uid": source_uid,
            "source_id": str(row.get("id", "")).strip(),
            "source_dataset": str(row.get("dataset", "")).strip(),
            "source_group_type": source_group_type,
            "source_group": source_group,
            "open_end_gold_text": gold_text,
            "open_end_answer_format_kind": _answer_format_kind(gold_text),
        }
    )
    out["meta"] = meta
    return out


def _build(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).expanduser().resolve()
    processed_dir = out_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    filtered_rows: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for source_idx, source in enumerate(args.source):
        source_path = Path(source).expanduser().resolve()
        rows = load_jsonl(source_path)
        candidates: list[dict[str, Any]] = []
        total_mcq = 0
        numeric_candidates = 0
        blocked_by_phrase = 0
        for row in rows:
            if str(row.get("task_type", "")).strip() != "mcq":
                continue
            total_mcq += 1
            gold_text = _gold_choice_text(row)
            if not _looks_numeric(gold_text):
                continue
            numeric_candidates += 1
            blocked = _is_blocked_by_mcq_phrase(str(row.get("question", "")).strip())
            if blocked:
                blocked_by_phrase += 1
            if blocked and not args.allow_mcq_phrasing:
                continue
            choice_map = _choice_text_map(row)
            answer_label = str(row.get("answer", "")).strip()
            if answer_label not in choice_map:
                continue
            if not any(label != answer_label for label in choice_map):
                continue
            candidate = deepcopy(row)
            candidate["_numeric_gold_text"] = gold_text
            candidate["_numeric_source_path"] = str(source_path)
            candidates.append(candidate)

        if args.max_samples_per_source > 0:
            candidates = sample_rows(
                candidates,
                args.max_samples_per_source,
                args.shuffle,
                args.seed + source_idx,
            )
        elif args.shuffle:
            candidates = sample_rows(candidates, 0, True, args.seed + source_idx)

        filtered_rows.extend(candidates)
        source_summaries.append(
            {
                "source_path": str(source_path),
                "raw_total": len(rows),
                "mcq_total": total_mcq,
                "numeric_candidates": numeric_candidates,
                "blocked_by_phrase": blocked_by_phrase,
                "kept_after_filtering": len(candidates),
            }
        )

    if args.max_samples > 0:
        filtered_rows = sample_rows(filtered_rows, args.max_samples, args.shuffle, args.seed)
    elif args.shuffle:
        filtered_rows = sample_rows(filtered_rows, 0, True, args.seed)

    mcq_rows: list[dict[str, Any]] = []
    bool_rows: list[dict[str, Any]] = []
    open_rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []

    for row in filtered_rows:
        source_uid = _source_uid(row)
        source_group_type, source_group = _source_group_fields(row)
        gold_text = str(row.get("_numeric_gold_text", "")).strip()
        choice_map = _choice_text_map(row)
        answer_label = str(row.get("answer", "")).strip()

        mcq_rows.append(
            _make_base_row(
                row,
                source_uid=source_uid,
                source_group_type=source_group_type,
                source_group=source_group,
            )
        )
        open_rows.append(
            _make_open_end_row(
                row,
                source_uid=source_uid,
                source_group_type=source_group_type,
                source_group=source_group,
                gold_text=gold_text,
            )
        )

        bool_ids: list[str] = []
        for i in range(args.positive_per_question):
            verify_row, verify_id = _make_verify_row(
                row,
                source_uid=source_uid,
                source_group_type=source_group_type,
                source_group=source_group,
                polarity="pos",
                choice_label=answer_label,
                choice_text=choice_map[answer_label],
                copy_index=i,
            )
            bool_rows.append(verify_row)
            bool_ids.append(verify_id)

        incorrect = [(label, text) for label, text in choice_map.items() if label != answer_label]
        for i in range(args.negative_per_question):
            neg_label, neg_text = incorrect[i % len(incorrect)]
            verify_row, verify_id = _make_verify_row(
                row,
                source_uid=source_uid,
                source_group_type=source_group_type,
                source_group=source_group,
                polarity="neg",
                choice_label=neg_label,
                choice_text=neg_text,
                copy_index=i,
            )
            bool_rows.append(verify_row)
            bool_ids.append(verify_id)

        manifest_entries.append(
            {
                "source_uid": source_uid,
                "source_id": str(row.get("id", "")).strip(),
                "source_dataset": str(row.get("dataset", "")).strip(),
                "source_path": str(row.get("_numeric_source_path", "")).strip(),
                "source_group_type": source_group_type,
                "source_group": source_group,
                "source_question": str(row.get("question", "")).strip(),
                "open_end_question": _clean_open_end_question(str(row.get("question", "")).strip()),
                "source_context": str(row.get("context", "")).strip(),
                "source_answer_label": answer_label,
                "source_answer_text": gold_text,
                "answer_format_kind": _answer_format_kind(gold_text),
                "source_choice_count": len(choice_map),
                "bool_ids": bool_ids,
                "open_end_id": source_uid,
            }
        )

    mcq_path = processed_dir / "mmlu_numeric_mcq.jsonl"
    bool_path = processed_dir / "mmlu_numeric_bool_verify.jsonl"
    open_path = processed_dir / "mmlu_numeric_open_end.jsonl"
    manifest_path = out_root / "numeric_answer_space_manifest.json"
    write_jsonl(mcq_path, mcq_rows)
    write_jsonl(bool_path, bool_rows)
    write_jsonl(open_path, open_rows)

    counts_by_source_dataset = Counter(str(entry.get("source_dataset", "")).strip() for entry in manifest_entries)
    counts_by_group = Counter(str(entry.get("source_group", "")).strip() for entry in manifest_entries)
    counts_by_format = Counter(str(entry.get("answer_format_kind", "")).strip() for entry in manifest_entries)

    dump_json(
        manifest_path,
        {
            "config": {
                "sources": args.source,
                "max_samples": args.max_samples,
                "max_samples_per_source": args.max_samples_per_source,
                "shuffle": args.shuffle,
                "seed": args.seed,
                "allow_mcq_phrasing": args.allow_mcq_phrasing,
                "positive_per_question": args.positive_per_question,
                "negative_per_question": args.negative_per_question,
            },
            "source_summaries": source_summaries,
            "counts": {
                "source_total": len(manifest_entries),
                "mcq_rows": len(mcq_rows),
                "bool_rows": len(bool_rows),
                "open_rows": len(open_rows),
                "by_source_dataset": dict(counts_by_source_dataset),
                "by_group": dict(counts_by_group),
                "by_answer_format_kind": dict(counts_by_format),
            },
            "entries": manifest_entries,
            "paths": {
                "processed_dir": str(processed_dir),
                "mcq_dataset": str(mcq_path),
                "bool_dataset": str(bool_path),
                "open_dataset": str(open_path),
            },
        },
    )
    print(f"Wrote {mcq_path}")
    print(f"Wrote {bool_path}")
    print(f"Wrote {open_path}")
    print(f"Wrote {manifest_path}")
    return 0


def _strip_wrappers(text: str) -> str:
    out = _normalize_space(text)
    if out.startswith("$$") and out.endswith("$$") and len(out) >= 4:
        out = _normalize_space(out[2:-2])
    if out.startswith("\\[") and out.endswith("\\]") and len(out) >= 4:
        out = _normalize_space(out[2:-2])
    if out.startswith("$") and out.endswith("$") and len(out) >= 2:
        out = _normalize_space(out[1:-1])
    match = re.fullmatch(r"\\boxed\s*\{([^{}]+)\}", out)
    if match:
        return _normalize_space(match.group(1))
    return out


def _extract_numeric_response_surface(record: dict[str, Any]) -> str:
    boxed_inner = _normalize_space(str(record.get("boxed_inner", "")).strip())
    if boxed_inner:
        return boxed_inner

    answer_raw = _normalize_space(str(record.get("answer_raw", "")).strip())
    if answer_raw:
        return _strip_wrappers(answer_raw)

    content = _normalize_space(str(record.get("content", "")).strip())
    boxed_matches = re.findall(r"\\boxed\s*\{([^{}]+)\}", content)
    if boxed_matches:
        return _normalize_space(boxed_matches[-1])

    match = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?%?", content)
    if match:
        return _normalize_space(match[-1])
    return _strip_wrappers(content)


def _parse_fraction_like(text: str) -> Fraction | None:
    cleaned = _normalize_space(text).replace(",", "")
    if not cleaned:
        return None
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1].strip()
    try:
        if "/" in cleaned:
            numerator, denominator = cleaned.split("/", 1)
            value = Fraction(int(numerator.strip()), int(denominator.strip()))
        else:
            value = Fraction(Decimal(cleaned))
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None
    return value / 100 if is_percent else value


def _normalize_numeric_surface(text: str) -> str:
    return _normalize_space(_strip_wrappers(text)).replace(",", "")


def _numeric_equivalent(left: str, right: str) -> bool:
    left_value = _parse_fraction_like(left)
    right_value = _parse_fraction_like(right)
    if left_value is None or right_value is None:
        return False
    return left_value == right_value


def _aggregate_rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(key)) / len(rows)


def _dataset_condition_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "total": 0,
            "accuracy": 0.0,
            "following_rate": 0.0,
            "following_correct_rate": 0.0,
            "strict_boxed_rate": 0.0,
        }
    return {
        "total": len(rows),
        "accuracy": _aggregate_rate(rows, "is_correct"),
        "following_rate": _aggregate_rate(rows, "nonreason_pass"),
        "following_correct_rate": _aggregate_rate(rows, "nonreason_correct"),
        "strict_boxed_rate": _aggregate_rate(rows, "is_strict_boxed_only"),
    }


def _open_end_metrics(rows: list[dict[str, Any]], gold_by_id: dict[str, str]) -> dict[str, float | int]:
    if not rows:
        return {
            "total": 0,
            "exact_match_accuracy": 0.0,
            "numeric_equivalence_accuracy": 0.0,
            "parseable_answer_rate": 0.0,
            "following_rate": 0.0,
            "following_correct_rate": 0.0,
            "strict_boxed_rate": 0.0,
        }

    exact_correct = 0
    numeric_correct = 0
    parseable = 0
    following = 0
    following_correct = 0
    strict_boxed = 0
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        gold_text = str(gold_by_id.get(row_id, "")).strip()
        surface = _extract_numeric_response_surface(row)
        normalized_surface = _normalize_numeric_surface(surface)
        normalized_gold = _normalize_numeric_surface(gold_text)
        parsed_surface = _parse_fraction_like(surface)
        if parsed_surface is not None:
            parseable += 1
        if normalized_surface and normalized_gold and normalized_surface == normalized_gold:
            exact_correct += 1
        numeric_is_correct = _numeric_equivalent(surface, gold_text)
        if numeric_is_correct:
            numeric_correct += 1
        if row.get("nonreason_pass"):
            following += 1
            if numeric_is_correct:
                following_correct += 1
        if row.get("is_strict_boxed_only"):
            strict_boxed += 1

    total = len(rows)
    return {
        "total": total,
        "exact_match_accuracy": exact_correct / total,
        "numeric_equivalence_accuracy": numeric_correct / total,
        "parseable_answer_rate": parseable / total,
        "following_rate": following / total,
        "following_correct_rate": following_correct / total,
        "strict_boxed_rate": strict_boxed / total,
    }


def _group_triplets(
    matched_triplets: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]],
    *,
    source_dataset: str | None = None,
    source_group_type: str | None = None,
    source_group: str | None = None,
    answer_format_kind: str | None = None,
) -> list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]]:
    picked = []
    for base_record, bool_records, open_record, entry in matched_triplets:
        if source_dataset is not None and str(entry.get("source_dataset", "")).strip() != source_dataset:
            continue
        if source_group_type is not None and str(entry.get("source_group_type", "")).strip() != source_group_type:
            continue
        if source_group is not None and str(entry.get("source_group", "")).strip() != source_group:
            continue
        if answer_format_kind is not None and str(entry.get("answer_format_kind", "")).strip() != answer_format_kind:
            continue
        picked.append((base_record, bool_records, open_record, entry))
    return picked


def _compute_mode_row(
    model_label: str,
    model_name: str,
    mode: str,
    triplets: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    base_records = [base for base, _, _, _ in triplets]
    bool_records = [record for _, records, _, _ in triplets for record in records]
    open_records = [open_record for _, _, open_record, _ in triplets]
    gold_by_open_id = {
        str(entry.get("open_end_id", "")).strip(): str(entry.get("source_answer_text", "")).strip()
        for _, _, _, entry in triplets
    }

    mcq_metrics = _dataset_condition_metrics(base_records)
    bool_metrics = _dataset_condition_metrics(bool_records)
    open_metrics = _open_end_metrics(open_records, gold_by_id=gold_by_open_id)

    pos_records = [
        record
        for _, records, _, _ in triplets
        for record in records
        if str((record.get("meta") or {}).get("verification_polarity", "")).strip() == "pos"
    ]
    neg_records = [
        record
        for _, records, _, _ in triplets
        for record in records
        if str((record.get("meta") or {}).get("verification_polarity", "")).strip() == "neg"
    ]

    return {
        "model_label": model_label,
        "model_name": model_name,
        "mode": mode,
        "matched_source_total": len(triplets),
        "mcq_accuracy": mcq_metrics["accuracy"],
        "mcq_following_rate": mcq_metrics["following_rate"],
        "mcq_following_correct_rate": mcq_metrics["following_correct_rate"],
        "bool_verify_accuracy": bool_metrics["accuracy"],
        "bool_verify_following_rate": bool_metrics["following_rate"],
        "bool_verify_following_correct_rate": bool_metrics["following_correct_rate"],
        "verify_pos_accuracy": _aggregate_rate(pos_records, "is_correct"),
        "verify_neg_accuracy": _aggregate_rate(neg_records, "is_correct"),
        "open_end_exact_match_accuracy": open_metrics["exact_match_accuracy"],
        "open_end_numeric_accuracy": open_metrics["numeric_equivalence_accuracy"],
        "open_end_parseable_rate": open_metrics["parseable_answer_rate"],
        "open_end_following_rate": open_metrics["following_rate"],
        "open_end_following_correct_rate": open_metrics["following_correct_rate"],
        "open_end_strict_boxed_rate": open_metrics["strict_boxed_rate"],
        "bool_drop_vs_mcq": float(bool_metrics["accuracy"]) - float(mcq_metrics["accuracy"]),
        "open_end_drop_vs_mcq": float(open_metrics["numeric_equivalence_accuracy"]) - float(mcq_metrics["accuracy"]),
    }


def _analyze(args: argparse.Namespace) -> int:
    manifest = load_json(Path(args.manifest).expanduser().resolve())
    entries = [entry for entry in manifest.get("entries", []) if isinstance(entry, dict)]
    entries_by_source: dict[str, dict[str, Any]] = {
        str(entry.get("source_uid", "")).strip(): entry
        for entry in entries
        if str(entry.get("source_uid", "")).strip()
    }

    run_specs = parse_run_specs(args.run)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    answer_format_rows: list[dict[str, Any]] = []

    for label, run_path in run_specs.items():
        records = load_jsonl(resolve_records_path(run_path))
        grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
        model_name = ""
        for row in records:
            dataset = str(row.get("dataset", "")).strip()
            mode = str(row.get("mode", "")).strip()
            row_id = str(row.get("id", "")).strip()
            if dataset not in {args.mcq_dataset, args.bool_dataset, args.open_dataset}:
                continue
            grouped[(dataset, mode, row_id)] = row
            if not model_name:
                model_name = str(row.get("model", "")).strip()

        modes = sorted({mode for _, mode, _ in grouped.keys()})
        for mode in modes:
            matched_triplets: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]] = []
            for source_uid, entry in entries_by_source.items():
                base_record = grouped.get((args.mcq_dataset, mode, source_uid))
                open_record = grouped.get((args.open_dataset, mode, str(entry.get("open_end_id", "")).strip()))
                bool_ids = [str(bool_id).strip() for bool_id in entry.get("bool_ids", []) if str(bool_id).strip()]
                bool_records = [grouped.get((args.bool_dataset, mode, bool_id)) for bool_id in bool_ids]
                if base_record is None or open_record is None or any(record is None for record in bool_records):
                    continue
                matched_triplets.append((base_record, bool_records, open_record, entry))

            if not matched_triplets:
                continue

            mode_rows.append(_compute_mode_row(label, model_name, mode, matched_triplets))

            datasets = sorted({str(entry.get("source_dataset", "")).strip() for _, _, _, entry in matched_triplets})
            for source_dataset in datasets:
                rows_for_dataset = _group_triplets(matched_triplets, source_dataset=source_dataset)
                row = _compute_mode_row(label, model_name, mode, rows_for_dataset)
                row["source_dataset"] = source_dataset
                dataset_rows.append(row)

            groups = sorted(
                {
                    (str(entry.get("source_group_type", "")).strip(), str(entry.get("source_group", "")).strip())
                    for _, _, _, entry in matched_triplets
                }
            )
            for source_group_type, source_group in groups:
                rows_for_group = _group_triplets(
                    matched_triplets,
                    source_group_type=source_group_type,
                    source_group=source_group,
                )
                row = _compute_mode_row(label, model_name, mode, rows_for_group)
                row["source_group_type"] = source_group_type
                row["source_group"] = source_group
                group_rows.append(row)

            answer_formats = sorted({str(entry.get("answer_format_kind", "")).strip() for _, _, _, entry in matched_triplets})
            for answer_format_kind in answer_formats:
                rows_for_format = _group_triplets(matched_triplets, answer_format_kind=answer_format_kind)
                row = _compute_mode_row(label, model_name, mode, rows_for_format)
                row["answer_format_kind"] = answer_format_kind
                answer_format_rows.append(row)

    mode_csv = output_dir / "mode_metrics.csv"
    dataset_csv = output_dir / "source_dataset_metrics.csv"
    group_csv = output_dir / "source_group_metrics.csv"
    format_csv = output_dir / "answer_format_metrics.csv"
    summary_path = output_dir / "numeric_answer_space_summary.json"
    write_csv(mode_csv, mode_rows)
    write_csv(dataset_csv, dataset_rows)
    write_csv(group_csv, group_rows)
    write_csv(format_csv, answer_format_rows)
    dump_json(
        summary_path,
        {
            "config": {
                "manifest": args.manifest,
                "runs": {label: str(path) for label, path in run_specs.items()},
                "mcq_dataset": args.mcq_dataset,
                "bool_dataset": args.bool_dataset,
                "open_dataset": args.open_dataset,
            },
            "mode_metrics": mode_rows,
            "source_dataset_metrics": dataset_rows,
            "source_group_metrics": group_rows,
            "answer_format_metrics": answer_format_rows,
            "paths": {
                "mode_metrics_csv": str(mode_csv),
                "source_dataset_metrics_csv": str(dataset_csv),
                "source_group_metrics_csv": str(group_csv),
                "answer_format_metrics_csv": str(format_csv),
            },
        },
    )
    print(f"Wrote {mode_csv}")
    print(f"Wrote {dataset_csv}")
    print(f"Wrote {group_csv}")
    print(f"Wrote {format_csv}")
    print(f"Wrote {summary_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "build":
        return _build(args)
    return _analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
