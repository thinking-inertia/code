#!/usr/bin/env python3
"""
Posthoc Q <-> (T+A) similarity scoring over existing records.jsonl files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .minilm_similarity import MiniLMSimilarityScorer, normalize_space


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute full visible response similarity against Q from existing records.jsonl files."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more run directories or records.jsonl paths.",
    )
    parser.add_argument(
        "--embedding-model-path",
        default=os.environ.get("MINILM_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2"),
        help=(
            "MiniLM embedding model path or Hugging Face model id. "
            "Can also be set with MINILM_MODEL_PATH."
        ),
    )
    parser.add_argument(
        "--embedding-device",
        default="auto",
        help="Embedding device: auto / cpu / cuda.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="MiniLM embedding batch size (default: 32).",
    )
    parser.add_argument(
        "--q-source",
        choices=["auto", "full", "question_only"],
        default="auto",
        help="How to build Q. Default: auto.",
    )
    parser.add_argument(
        "--response-field",
        choices=["content", "answer_view_content", "visible_payload"],
        default="content",
        help=(
            "Which visible response field to compare against Q. "
            "Use visible_payload to include returned readable reasoning plus the visible answer. "
            "Default: content."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Default: write sidecars next to each input.",
    )
    return parser.parse_args()


def _clip_rate(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _strip_reasoning_markup(text: str) -> str:
    raw = str(text or "")
    raw = re.sub(r"</?(?:think|reasoning)>", " ", raw, flags=re.IGNORECASE)
    return normalize_space(raw)


def _build_problem_text(record: dict[str, Any]) -> str:
    question = str(record.get("question", "")).strip()
    context = str(record.get("context", "")).strip()
    task_type = str(record.get("task_type", "")).strip()
    choices = record.get("choices", [])

    lines: list[str] = []
    if context:
        lines.append(f"Context: {context}")
    if question:
        lines.append(f"Question: {question}")
    if task_type == "mcq" and isinstance(choices, list):
        lines.append("Options:")
        for choice in choices:
            if isinstance(choice, dict):
                label = str(choice.get("label", "")).strip()
                text = str(choice.get("text", "")).strip()
                if label or text:
                    lines.append(f"{label}. {text}".strip())
    return "\n".join(lines)


def _choose_q_text(record: dict[str, Any], q_source: str) -> str:
    question_only = normalize_space(str(record.get("question", "")))
    if q_source == "question_only":
        return question_only

    q_text = normalize_space(str(record.get("question_text_used_for_similarity", "")))
    if q_text:
        return q_text

    full_question = normalize_space(_build_problem_text(record))
    if full_question:
        return full_question

    return question_only


def _choose_visible_payload_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for field in (
        "t_returned_readable",
        "extracted_t",
        "reasoning_content",
        "reasoning",
        "reasoning_details_summary_text",
        "reasoning_details_raw_text",
        "answer_view_content",
        "content",
    ):
        text = _strip_reasoning_markup(str(record.get(field, "")))
        if not text or text in seen:
            continue
        parts.append(text)
        seen.add(text)
    return normalize_space(" ".join(parts))


def _choose_response_text(record: dict[str, Any], response_field: str) -> str:
    if response_field == "visible_payload":
        return _choose_visible_payload_text(record)
    return _strip_reasoning_markup(str(record.get(response_field, "")))


def _resolve_input(path_str: str) -> tuple[Path, str, Path]:
    path = Path(path_str).expanduser().resolve()
    if path.is_dir():
        records_path = path / "records.jsonl"
        if not records_path.exists():
            raise FileNotFoundError(f"missing records.jsonl in run directory: {path}")
        return records_path, path.name, path
    if path.is_file():
        return path, path.stem, path.parent
    raise FileNotFoundError(f"input path not found: {path}")


def _load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if "\x00" in line:
                print(
                    f"[warn] skipping NUL-containing line {line_no} in {path}",
                    file=sys.stderr,
                )
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"[warn] skipping malformed JSON line {line_no} in {path}: {exc}",
                    file=sys.stderr,
                )
    return rows


def _score_pairs(
    scorer: MiniLMSimilarityScorer,
    pairs: list[tuple[str, str]],
    *,
    batch_size: int,
) -> list[float]:
    sims: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        q_batch = [question for question, _ in batch]
        response_batch = [response for _, response in batch]
        q_vecs = scorer.encode_texts(q_batch, batch_size=batch_size)
        response_vecs = scorer.encode_texts(response_batch, batch_size=batch_size)
        batch_sims = scorer._torch.sum(q_vecs * response_vecs, dim=1).tolist()
        sims.extend(float(value) for value in batch_sims)
    return sims


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    valid_rows = [row for row in rows if row.get("full_response_similarity") is not None]
    valid_count = len(valid_rows)
    similarities = [float(row["full_response_similarity"]) for row in valid_rows]
    raw_similarities = [float(row["full_response_similarity_raw"]) for row in valid_rows]
    nonreason_rates = [float(row["full_response_nonreason_rate"]) for row in valid_rows]
    empty_response = sum(1 for row in rows if row.get("response_text_is_empty"))
    empty_question = sum(1 for row in rows if row.get("q_text_is_empty"))

    if valid_count == 0:
        return {
            "count": total,
            "valid_count": 0,
            "mean_similarity": 0.0,
            "max_similarity": 0.0,
            "mean_similarity_raw": 0.0,
            "max_similarity_raw": 0.0,
            "mean_nonreason_rate": 0.0,
            "empty_response_rate": (empty_response / total) if total else 0.0,
            "empty_question_rate": (empty_question / total) if total else 0.0,
        }

    return {
        "count": total,
        "valid_count": valid_count,
        "mean_similarity": sum(similarities) / valid_count,
        "max_similarity": max(similarities),
        "mean_similarity_raw": sum(raw_similarities) / valid_count,
        "max_similarity_raw": max(raw_similarities),
        "mean_nonreason_rate": sum(nonreason_rates) / valid_count,
        "empty_response_rate": empty_response / total,
        "empty_question_rate": empty_question / total,
    }


def _group_summary(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {key: _summarize(rows) for key, rows in grouped.items()}


def _output_paths(
    records_path: Path,
    source_name: str,
    default_dir: Path,
    output_dir: str,
) -> tuple[Path, Path]:
    if output_dir:
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        return (
            out_dir / f"{source_name}.full_response_similarity.jsonl",
            out_dir / f"{source_name}.full_response_similarity.summary.json",
        )

    if records_path.name == "records.jsonl":
        return (
            default_dir / "full_response_similarity_records.jsonl",
            default_dir / "full_response_similarity_summary.json",
        )

    return (
        default_dir / f"{records_path.stem}.full_response_similarity.jsonl",
        default_dir / f"{records_path.stem}.full_response_similarity.summary.json",
    )


def _process_one_input(
    scorer: MiniLMSimilarityScorer,
    records_path: Path,
    source_name: str,
    default_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    rows = _load_records(records_path)

    enriched: list[dict[str, Any]] = []
    scoring_pairs: list[tuple[str, str]] = []
    scoring_indexes: list[int] = []

    for row in rows:
        q_text = _choose_q_text(row, args.q_source)
        response_text = _choose_response_text(row, args.response_field)
        q_text_is_empty = not bool(q_text)
        response_text_is_empty = not bool(response_text)

        enriched_row = {
            "dataset": row.get("dataset", ""),
            "id": row.get("id", ""),
            "model": row.get("model", ""),
            "mode": row.get("mode", ""),
            "task_type": row.get("task_type", ""),
            "q_text_for_full_response_similarity": q_text,
            "response_text_for_full_response_similarity": response_text,
            "response_field_used": args.response_field,
            "q_source_used": args.q_source,
            "q_text_is_empty": q_text_is_empty,
            "response_text_is_empty": response_text_is_empty,
            "full_response_similarity_raw": None,
            "full_response_similarity": None,
            "full_response_nonreason_rate": None,
        }

        if not q_text_is_empty and not response_text_is_empty:
            scoring_indexes.append(len(enriched))
            scoring_pairs.append((q_text, response_text))

        enriched.append(enriched_row)

    raw_scores = _score_pairs(
        scorer,
        scoring_pairs,
        batch_size=args.embedding_batch_size,
    ) if scoring_pairs else []

    for index, raw_score in zip(scoring_indexes, raw_scores):
        clipped = _clip_rate(raw_score)
        enriched[index]["full_response_similarity_raw"] = raw_score
        enriched[index]["full_response_similarity"] = clipped
        enriched[index]["full_response_nonreason_rate"] = 1.0 - clipped

    overall = _summarize(enriched)
    per_dataset_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_task_type_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_mode_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_mode_dataset_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in enriched:
        dataset = str(row.get("dataset", "")).strip()
        task_type = str(row.get("task_type", "")).strip()
        mode = str(row.get("mode", "")).strip()
        if dataset:
            per_dataset_rows[dataset].append(row)
        if task_type:
            per_task_type_rows[task_type].append(row)
        if mode:
            per_mode_rows[mode].append(row)
        if mode and dataset:
            per_mode_dataset_rows[f"{mode}::{dataset}"].append(row)

    output_jsonl, output_summary = _output_paths(
        records_path,
        source_name,
        default_dir,
        args.output_dir,
    )

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in enriched:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    mode_names = sorted(per_mode_rows.keys())
    dataset_names = sorted(per_dataset_rows.keys())
    summary = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "source_name": source_name,
        "paths": {
            "input_records": str(records_path),
            "output_records": str(output_jsonl),
            "output_summary": str(output_summary),
        },
        "config": {
            "q_source": args.q_source,
            "response_field": args.response_field,
            "embedding_model_path": args.embedding_model_path,
            "embedding_device": args.embedding_device,
            "embedding_batch_size": args.embedding_batch_size,
        },
        "overall": overall,
        "per_mode": _group_summary(per_mode_rows),
        "per_dataset": _group_summary(per_dataset_rows),
        "per_task_type": _group_summary(per_task_type_rows),
        "per_mode_per_dataset": {
            mode: {
                dataset: _summarize(per_mode_dataset_rows.get(f"{mode}::{dataset}", []))
                for dataset in dataset_names
            }
            for mode in mode_names
        },
    }

    with output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return output_jsonl, output_summary


def main() -> int:
    args = parse_args()

    try:
        resolved_inputs = [_resolve_input(path_str) for path_str in args.input]
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    scorer = MiniLMSimilarityScorer(
        args.embedding_model_path,
        device=args.embedding_device,
    )

    for records_path, source_name, default_dir in resolved_inputs:
        output_jsonl, output_summary = _process_one_input(
            scorer,
            records_path,
            source_name,
            default_dir,
            args,
        )
        print(
            f"[done] {records_path} -> {output_jsonl.name}, {output_summary.name}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
