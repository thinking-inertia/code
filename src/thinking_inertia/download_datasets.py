#!/usr/bin/env python3
"""
Download and normalize benchmark datasets for strict non-reasoning evaluation.

All files are written under this `data/` directory by default:
  - raw/       : optional HF cache location
  - processed/ : normalized JSONL files used by evaluation

Example:
  python download_datasets.py
  python download_datasets.py --datasets boolq gsm8k mmlu --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .benchmark_scorers import (
    extract_last_number as _extract_last_number_standard,
    extract_math_answer_fallback,
    find_boxed_spans as _find_boxed_spans_standard,
    last_boxed_inner as _last_boxed_inner_standard,
    normalize_math_fallback,
)

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None


LETTER_SET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    hf_dataset: str
    hf_config: Optional[str]
    split: str
    task_type: str  # "bool", "mcq", "math"
    normalizer: Callable[[int, dict[str, Any]], dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and normalize benchmark datasets into JSONL."
    )
    parser.add_argument(
        "--root-dir",
        default=".",
        help="Root data directory (default: current directory).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Subset to download. Default: all supported datasets.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap per dataset after loading (0 = all).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=20260502,
        help="Seed used when --sample-strategy is random or level_balanced.",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=["first", "random", "level_balanced"],
        default="first",
        help=(
            "How to select --max-samples after normalization. level_balanced balances "
            "MATH examples by meta.level when available."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default="raw/hf_cache",
        help="HF cache dir, relative to root-dir unless absolute.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite processed files if they already exist.",
    )
    return parser.parse_args()


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_last_number(text: str) -> str:
    return _extract_last_number_standard(text)


def _find_boxed_spans(text: str) -> list[tuple[str, int, int]]:
    return _find_boxed_spans_standard(text)


def _last_boxed_inner(text: str) -> str:
    return _last_boxed_inner_standard(text)


def _normalize_math_answer_text(text: str) -> str:
    return normalize_math_fallback(text)


def _extract_math_answer(solution: str) -> str:
    return extract_math_answer_fallback(solution)


def _choice_dicts_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    if "choices" in row and isinstance(row["choices"], dict):
        labels = row["choices"].get("label", [])
        texts = row["choices"].get("text", [])
        out = []
        for label, text in zip(labels, texts):
            out.append({"label": _as_str(label), "text": _as_str(text)})
        return out

    if "choices" in row and isinstance(row["choices"], list):
        out = []
        for i, choice in enumerate(row["choices"]):
            if isinstance(choice, dict):
                label = _as_str(choice.get("label")) or LETTER_SET[i]
                text = _as_str(choice.get("text")) or _as_str(choice.get("content"))
            else:
                label = LETTER_SET[i]
                text = _as_str(choice)
            out.append({"label": label, "text": text})
        return out

    if "options" in row and isinstance(row["options"], list):
        out = []
        for i, option in enumerate(row["options"]):
            if isinstance(option, dict):
                label = _as_str(option.get("label")) or LETTER_SET[i]
                text = _as_str(option.get("text")) or _as_str(option.get("content"))
            else:
                label = LETTER_SET[i]
                text = _as_str(option)
            out.append({"label": label, "text": text})
        return out

    return []


def _answer_to_label(answer: Any, num_choices: int) -> str:
    if isinstance(answer, bool):
        return "A" if answer else "B"
    if isinstance(answer, int):
        if 0 <= answer < len(LETTER_SET):
            return LETTER_SET[answer]
        return ""
    if isinstance(answer, str):
        answer_str = answer.strip()
        if len(answer_str) == 1 and answer_str.upper() in LETTER_SET:
            return answer_str.upper()
        if answer_str.isdigit():
            idx = int(answer_str)
            if 0 <= idx < len(LETTER_SET):
                return LETTER_SET[idx]
    if isinstance(answer, float):
        idx = int(answer)
        if 0 <= idx < len(LETTER_SET):
            return LETTER_SET[idx]
    if num_choices == 2:
        return ""
    return ""


def _normalize_boolq(i: int, row: dict[str, Any]) -> dict[str, Any]:
    question = _as_str(row.get("question"))
    passage = _as_str(row.get("passage"))
    answer = row.get("answer")
    answer_str = "yes" if bool(answer) else "no"
    rec_id = _as_str(row.get("idx")) or f"boolq-{i}"
    return {
        "id": rec_id,
        "dataset": "boolq",
        "task_type": "bool",
        "split": "validation",
        "question": question,
        "context": passage,
        "choices": [],
        "answer": answer_str,
        "meta": {},
    }


def _normalize_commonsenseqa(i: int, row: dict[str, Any]) -> dict[str, Any]:
    choices = _choice_dicts_from_row(row)
    answer = _as_str(row.get("answerKey"))
    rec_id = _as_str(row.get("id")) or f"commonsenseqa-{i}"
    return {
        "id": rec_id,
        "dataset": "commonsenseqa",
        "task_type": "mcq",
        "split": "validation",
        "question": _as_str(row.get("question")),
        "context": "",
        "choices": choices,
        "answer": answer,
        "meta": {"question_concept": _as_str(row.get("question_concept"))},
    }


def _normalize_mmlu(i: int, row: dict[str, Any]) -> dict[str, Any]:
    raw_choices = row.get("choices", [])
    choices = []
    if isinstance(raw_choices, list):
        for idx, text in enumerate(raw_choices):
            choices.append({"label": LETTER_SET[idx], "text": _as_str(text)})
    answer = _answer_to_label(row.get("answer"), len(choices))
    rec_id = _as_str(row.get("id")) or f"mmlu-{i}"
    return {
        "id": rec_id,
        "dataset": "mmlu",
        "task_type": "mcq",
        "split": "test",
        "question": _as_str(row.get("question")),
        "context": "",
        "choices": choices,
        "answer": answer,
        "meta": {"subject": _as_str(row.get("subject"))},
    }


def _normalize_strategyqa(i: int, row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("answer")
    if isinstance(answer, str):
        answer_str = "yes" if answer.strip().lower() in {"yes", "true", "1"} else "no"
    else:
        answer_str = "yes" if bool(answer) else "no"
    rec_id = _as_str(row.get("id")) or f"strategyqa-{i}"
    return {
        "id": rec_id,
        "dataset": "strategyqa",
        "task_type": "bool",
        "split": "test",
        "question": _as_str(row.get("question")),
        "context": "",
        "choices": [],
        "answer": answer_str,
        "meta": {},
    }


def _normalize_gsm8k(i: int, row: dict[str, Any]) -> dict[str, Any]:
    raw_answer = _as_str(row.get("answer"))
    if "####" in raw_answer:
        answer = raw_answer.split("####")[-1].strip().replace(",", "")
    else:
        answer = _extract_last_number(raw_answer)
    rec_id = _as_str(row.get("id")) or f"gsm8k-{i}"
    return {
        "id": rec_id,
        "dataset": "gsm8k",
        "task_type": "math",
        "split": "test",
        "question": _as_str(row.get("question")),
        "context": "",
        "choices": [],
        "answer": answer,
        "meta": {"raw_answer": raw_answer},
    }


def _normalize_gsm_symbolic(i: int, row: dict[str, Any]) -> dict[str, Any]:
    raw_answer = _as_str(row.get("answer"))
    answer = raw_answer
    if "####" in raw_answer:
        answer = raw_answer.split("####")[-1].strip().replace(",", "")
    elif not re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?", raw_answer):
        answer = _extract_last_number(raw_answer)
    rec_id = _as_str(row.get("id")) or f"gsm-symbolic-{i}"
    return {
        "id": rec_id,
        "dataset": "gsm_symbolic",
        "task_type": "math",
        "split": "test",
        "question": _as_str(row.get("question")),
        "context": "",
        "choices": [],
        "answer": answer,
        "meta": {"raw_answer": raw_answer},
    }


def _normalize_mmlu_pro(i: int, row: dict[str, Any]) -> dict[str, Any]:
    choices = _choice_dicts_from_row(row)
    if not choices:
        raw_opts = row.get("options", [])
        if isinstance(raw_opts, list):
            for idx, text in enumerate(raw_opts):
                choices.append({"label": LETTER_SET[idx], "text": _as_str(text)})

    answer = _answer_to_label(row.get("answer"), len(choices))
    if not answer:
        answer = _answer_to_label(row.get("answer_index"), len(choices))
    rec_id = _as_str(row.get("question_id")) or _as_str(row.get("id")) or f"mmlu-pro-{i}"
    return {
        "id": rec_id,
        "dataset": "mmlu_pro",
        "task_type": "mcq",
        "split": "test",
        "question": _as_str(row.get("question")),
        "context": "",
        "choices": choices,
        "answer": answer,
        "meta": {"category": _as_str(row.get("category"))},
    }


def _normalize_math(i: int, row: dict[str, Any]) -> dict[str, Any]:
    solution = _as_str(row.get("solution")) or _as_str(row.get("answer"))
    level = _as_str(row.get("level"))
    subject = _as_str(row.get("type")) or _as_str(row.get("subject"))
    rec_id = _as_str(row.get("id")) or f"math-{i}"
    return {
        "id": rec_id,
        "dataset": "math",
        "task_type": "math",
        "split": "test",
        "question": _as_str(row.get("problem")) or _as_str(row.get("question")),
        "context": "",
        "choices": [],
        "answer": _extract_math_answer(solution),
        "meta": {"raw_solution": solution, "level": level, "subject": subject},
    }


SPECS: dict[str, BenchmarkSpec] = {
    "boolq": BenchmarkSpec(
        name="boolq",
        hf_dataset="google/boolq",
        hf_config=None,
        split="validation",
        task_type="bool",
        normalizer=_normalize_boolq,
    ),
    "commonsenseqa": BenchmarkSpec(
        name="commonsenseqa",
        hf_dataset="tau/commonsense_qa",
        hf_config=None,
        split="validation",
        task_type="mcq",
        normalizer=_normalize_commonsenseqa,
    ),
    "mmlu": BenchmarkSpec(
        name="mmlu",
        hf_dataset="cais/mmlu",
        hf_config="all",
        split="test",
        task_type="mcq",
        normalizer=_normalize_mmlu,
    ),
    "strategyqa": BenchmarkSpec(
        name="strategyqa",
        hf_dataset="ChilleD/StrategyQA",
        hf_config=None,
        split="test",
        task_type="bool",
        normalizer=_normalize_strategyqa,
    ),
    "gsm8k": BenchmarkSpec(
        name="gsm8k",
        hf_dataset="openai/gsm8k",
        hf_config="main",
        split="test",
        task_type="math",
        normalizer=_normalize_gsm8k,
    ),
    "gsm_symbolic": BenchmarkSpec(
        name="gsm_symbolic",
        hf_dataset="apple/GSM-Symbolic",
        hf_config=None,
        split="test",
        task_type="math",
        normalizer=_normalize_gsm_symbolic,
    ),
    "mmlu_pro": BenchmarkSpec(
        name="mmlu_pro",
        hf_dataset="TIGER-Lab/MMLU-Pro",
        hf_config=None,
        split="test",
        task_type="mcq",
        normalizer=_normalize_mmlu_pro,
    ),
    "math": BenchmarkSpec(
        name="math",
        hf_dataset="DigitalLearningGmbH/MATH-lighteval",
        hf_config="default",
        split="test",
        task_type="math",
        normalizer=_normalize_math,
    ),
}


ALIASES: dict[str, str] = {
    "boolq": "boolq",
    "commonsenseqa": "commonsenseqa",
    "commonsense_qa": "commonsenseqa",
    "mmlu": "mmlu",
    "strategyqa": "strategyqa",
    "gsm8k": "gsm8k",
    "gsm_symbolic": "gsm_symbolic",
    "gsm-symbolic": "gsm_symbolic",
    "mmlu_pro": "mmlu_pro",
    "mmlu-pro": "mmlu_pro",
    "math": "math",
    "hendrycks_math": "math",
    "competition_math": "math",
}


def _canonical_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    return ALIASES.get(key, key)


def _resolve_cache_dir(root_dir: Path, cache_dir: str) -> Path:
    cache_path = Path(cache_dir)
    if cache_path.is_absolute():
        return cache_path
    return root_dir / cache_path


def _iter_dataset_rows(dataset_obj: Any, max_samples: int = 0) -> Iterable[dict[str, Any]]:
    for row in dataset_obj:
        yield row


def _select_rows(
    name: str,
    rows: list[dict[str, Any]],
    *,
    max_samples: int,
    sample_seed: int,
    sample_strategy: str,
) -> list[dict[str, Any]]:
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows

    rng = random.Random(sample_seed)
    if sample_strategy == "random":
        picked = list(rows)
        rng.shuffle(picked)
        return picked[:max_samples]

    if sample_strategy == "level_balanced" and name == "math":
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            level = str((row.get("meta") or {}).get("level") or "unknown")
            buckets.setdefault(level, []).append(row)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        levels = sorted(buckets)
        selected: list[dict[str, Any]] = []
        while len(selected) < max_samples and any(buckets.values()):
            for level in levels:
                if buckets[level] and len(selected) < max_samples:
                    selected.append(buckets[level].pop())
        return selected

    return rows[:max_samples]


def main() -> int:
    args = parse_args()

    if load_dataset is None:
        print("Missing dependency `datasets`. Install with: pip install datasets", file=sys.stderr)
        return 1

    root_dir = Path(args.root_dir).resolve()
    processed_dir = root_dir / "processed"
    cache_dir = _resolve_cache_dir(root_dir, args.cache_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_dataset_names = args.datasets or list(SPECS.keys())
    dataset_names = list(dict.fromkeys(_canonical_name(name) for name in raw_dataset_names))
    unknown = [name for name in dataset_names if name not in SPECS]
    if unknown:
        print(f"Unknown dataset names: {unknown}", file=sys.stderr)
        print(f"Supported: {sorted(SPECS.keys())}", file=sys.stderr)
        return 2

    errors: dict[str, str] = {}

    for name in dataset_names:
        spec = SPECS[name]
        out_path = processed_dir / f"{name}.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {name}: {out_path} exists (use --overwrite)")
            continue

        print(f"[load] {name}: {spec.hf_dataset} ({spec.hf_config}, split={spec.split})")
        try:
            ds = load_dataset(
                spec.hf_dataset,
                spec.hf_config,
                split=spec.split,
                cache_dir=str(cache_dir),
            )
        except Exception as exc:
            errors[name] = str(exc)
            print(f"[error] {name}: {exc}", file=sys.stderr)
            continue

        rows: list[dict[str, Any]] = []
        for idx, row in enumerate(_iter_dataset_rows(ds)):
            try:
                rows.append(spec.normalizer(idx, row))
            except Exception as exc:
                print(f"[warn] {name} row {idx} failed normalization: {exc}", file=sys.stderr)
        rows = _select_rows(
            name,
            rows,
            max_samples=args.max_samples,
            sample_seed=args.sample_seed,
            sample_strategy=args.sample_strategy,
        )

        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[ok] {name}: wrote {len(rows)} rows -> {out_path}")

    if errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
