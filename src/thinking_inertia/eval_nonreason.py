#!/usr/bin/env python3
"""
Evaluate strict non-reasoning obedience and task accuracy on processed datasets.

Expected input format:
  data/processed/<dataset>.jsonl

Output artifacts (default under data/runs/<run_name>/):
  - records.jsonl: per-sample result
  - summary.json : aggregate metrics

Example:
  python eval_nonreason.py \
    --datasets boolq gsm8k mmlu \
    --base-url http://127.0.0.1:8102/v1 \
    --model gpt-oss-20b \
    --max-samples 200
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from .benchmark_scorers import (
    extract_last_number as _extract_last_number_standard,
    find_boxed_spans as _find_boxed_spans_standard,
    last_boxed_inner as _last_boxed_inner_standard,
    math_verify_available,
    normalize_math_fallback,
    verify_math_answer,
)


INSTRUCTION_SUFFIX = "Please answer directly without any reasoning and put result in boxed{}."
LETTER_SET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run non-reasoning evaluation against local vLLM OpenAI-compatible API."
    )
    parser.add_argument(
        "--api",
        choices=["chat", "completions"],
        default="chat",
        help="Which OpenAI-compatible endpoint to use. "
        "'chat' uses /v1/chat/completions (requires a chat template). "
        "'completions' uses /v1/completions (works for base models without chat template).",
    )
    parser.add_argument(
        "--root-dir",
        default=".",
        help="Data root directory containing processed/. Default: current directory.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names, corresponding to processed/<name>.jsonl.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8102/v1",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Served model name for /chat/completions",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="max_tokens for model generation (default: 512).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="HTTP timeout seconds (default: 300).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap per dataset after loading (0 = all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for optional shuffling (default: 0).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle each dataset before selecting max-samples.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help='Pass chat_template_kwargs={"enable_thinking": false}.',
    )
    parser.add_argument(
        "--chat-template-kwargs",
        default="",
        help='Extra JSON object for chat_template_kwargs, e.g. \'{"enable_thinking": false}\'.',
    )
    parser.add_argument(
        "--include-reasoning",
        dest="include_reasoning",
        action="store_true",
        default=True,
        help="Request reasoning fields from server (default: true).",
    )
    parser.add_argument(
        "--no-include-reasoning",
        dest="include_reasoning",
        action="store_false",
        help="Do not request reasoning fields from server.",
    )
    parser.add_argument(
        "--accept-display-math-wrapper",
        action="store_true",
        help="Treat $$\\boxed{...}$$ as strict boxed format.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional run folder name under runs/. Default: auto timestamp.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run-name by loading records.jsonl and skipping completed samples.",
    )
    return parser.parse_args()


def _request_json(
    method: str,
    url: str,
    payload: Optional[dict[str, Any]],
    api_key: str,
    timeout: int,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc.reason}") from exc
    return json.loads(body)


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _find_boxed_spans(text: str) -> list[tuple[str, int, int]]:
    return _find_boxed_spans_standard(text)


def _last_boxed_inner(text: str) -> str:
    return _last_boxed_inner_standard(text)


def _extract_last_number(text: str) -> str:
    return _extract_last_number_standard(text)


def _normalize_math_answer_text(text: str) -> str:
    return normalize_math_fallback(text)


def _looks_like_bare_math_answer(text: str) -> bool:
    t = _strip_latex_wrappers(text, accept_display_math_wrapper=True).strip()
    if not t or "\n" in t or len(t) > 160:
        return False
    if re.search(r"\b(?:answer|therefore|because|since|we|calculate|solve)\b", t, flags=re.IGNORECASE):
        return False
    if _last_boxed_inner(t):
        return True
    return len(t.split()) <= 4


def _norm_text(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def _norm_label(text: str) -> str:
    return " ".join(text.strip().split()).upper()


def _choice_pairs(choices: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not isinstance(choices, list):
        return pairs
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        label = str(choice.get("label", "")).strip()
        text = str(choice.get("text", "")).strip()
        if label:
            pairs.append((label, text))
    return pairs


def _choice_label_map(choices: Any) -> dict[str, str]:
    return {_norm_label(label): label for label, _ in _choice_pairs(choices)}


def _choice_text_map(choices: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for label, text in _choice_pairs(choices):
        norm = _norm_text(text)
        if norm and norm not in out:
            out[norm] = label
    return out


def _example_choice_label(choices: Any) -> str:
    pairs = _choice_pairs(choices)
    if pairs:
        return pairs[0][0]
    return "A"


def _label_capture_pattern(label: str) -> str:
    escaped = re.escape(label)
    if re.fullmatch(r"[A-Za-z0-9_]+", label):
        return rf"\b{escaped}\b"
    return rf"(?<!\S){escaped}(?!\S)"


def _strip_mcq_lead_in(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(
        r"^\s*(?:the\s+answer\s+is|answer\s+is|answer|option|choice)\s*[:\-]?\s*",
        "",
        stripped,
        flags=re.IGNORECASE,
    )
    return stripped.strip()


def _match_mcq_choice_text(source: str, choices: Any) -> str:
    text_map = _choice_text_map(choices)
    if not text_map:
        return ""
    candidates = [source.strip(), _strip_mcq_lead_in(source)]
    for candidate in candidates:
        norm = _norm_text(candidate)
        if norm in text_map:
            return text_map[norm]
    return ""


def _find_last_mcq_label_span(text: str, choices: Any) -> tuple[str, int, int]:
    pairs = _choice_pairs(choices)
    if not pairs:
        return "", -1, -1

    labels = [label for label, _ in pairs]
    matches: list[tuple[int, int, str]] = []
    for label in sorted(labels, key=len, reverse=True):
        capture = _label_capture_pattern(label)
        for pattern in [
            rf"(?:answer\s+is|answer:|option|choice)\s*({capture})",
            rf"({capture})",
        ]:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                matches.append((match.start(1), match.end(1), label))

    if not matches:
        return "", -1, -1

    start, end, label = matches[-1]
    return label, start, end


def _find_last_mcq_choice_text_span(text: str, choices: Any) -> tuple[str, int, int]:
    matches: list[tuple[int, int, str]] = []
    for label, choice_text in _choice_pairs(choices):
        if not choice_text:
            continue
        pattern = re.escape(choice_text)
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append((match.start(), match.end(), label))
    if not matches:
        return "", -1, -1
    start, end, label = matches[-1]
    return label, start, end


def _strip_display_math_wrapper(text: str) -> str:
    t = text.strip()
    if t.startswith("$$") and t.endswith("$$") and len(t) >= 4:
        return t[2:-2].strip()
    return t


def _strip_bracket_math_wrapper(text: str) -> str:
    # Accept model outputs like:
    #   \[
    #   \boxed{72}
    #   \]
    t = text.strip()
    if t.startswith("\\[") and t.endswith("\\]") and len(t) >= 4:
        return t[2:-2].strip()
    return t


def _strip_inline_math_wrapper(text: str) -> str:
    # Accept model outputs like: $ \boxed{72} $ or $72$
    t = text.strip()
    if t.startswith("$") and t.endswith("$") and len(t) >= 2:
        return t[1:-1].strip()
    return t


def _strip_latex_wrappers(text: str, accept_display_math_wrapper: bool) -> str:
    t = text.strip()
    if accept_display_math_wrapper:
        t = _strip_display_math_wrapper(t)
    t = _strip_bracket_math_wrapper(t)
    t = _strip_inline_math_wrapper(t)
    return t


def _is_strict_boxed_only(content: str, accept_display_math_wrapper: bool) -> bool:
    stripped = _strip_latex_wrappers(content, accept_display_math_wrapper)
    return bool(re.fullmatch(r"\\boxed\s*\{[^{}]+\}", stripped))


def _extract_answer_only_pred(
    task_type: str, content: str, accept_display_math_wrapper: bool, choices: Any = None
) -> str:
    # Answer-only means the *entire* output is either:
    # - \boxed{...} (optionally wrapped by $$...$$, \[...\], or $...$), OR
    # - a single bare answer token/value (e.g. A / yes / 72), with no extra words.
    t = _strip_latex_wrappers(content, accept_display_math_wrapper)

    m = re.fullmatch(r"\\boxed\s*\{([^{}]+)\}", t)
    if m:
        inner = m.group(1).strip()
        # Normalize using the same logic as other predictions, but force boxed_inner.
        return _normalize_pred(task_type, inner, boxed_inner=inner, choices=choices)

    if task_type == "bool":
        low = _norm_text(t)
        if low in {"yes", "no", "true", "false", "1", "0"}:
            return _normalize_pred(task_type, t, boxed_inner="")
        return ""

    if task_type == "mcq":
        label_map = _choice_label_map(choices)
        up = _norm_label(t)
        if up in label_map:
            return label_map[up]
        text_match = _match_mcq_choice_text(t, choices)
        if text_match:
            return text_match
        if re.fullmatch(r"[A-Z]", up):
            return up
        return ""

    if task_type == "math":
        if _looks_like_bare_math_answer(t):
            return _normalize_math_answer_text(t)
        return ""

    return ""


def _normalize_pred(
    task_type: str,
    content: str,
    boxed_inner: str,
    choices: Any = None,
) -> str:
    source = boxed_inner or content

    if task_type == "bool":
        low = _norm_text(source)
        if "yes" in low or low in {"true", "1"}:
            return "yes"
        if "no" in low or low in {"false", "0"}:
            return "no"
        return ""

    if task_type == "mcq":
        label_map = _choice_label_map(choices)
        direct = label_map.get(_norm_label(source))
        if direct:
            return direct

        label_match, _, _ = _find_last_mcq_label_span(source, choices)
        if label_match:
            return label_match

        text_match = _match_mcq_choice_text(source, choices)
        if text_match:
            return text_match

        up = source.strip().upper()
        if len(up) == 1 and up in LETTER_SET:
            return up
        match = re.search(r"\b([A-Z])\b", up)
        if match:
            return match.group(1)
        return ""

    if task_type == "math":
        if boxed_inner:
            return _normalize_math_answer_text(boxed_inner)
        boxed = _last_boxed_inner(content)
        if boxed:
            return _normalize_math_answer_text(boxed)
        num = _extract_last_number(source)
        return num or _normalize_math_answer_text(source)

    return source.strip()


def _normalize_gold(task_type: str, answer: str, choices: Any = None) -> str:
    if task_type == "bool":
        low = _norm_text(answer)
        if low in {"yes", "true", "1"}:
            return "yes"
        if low in {"no", "false", "0"}:
            return "no"
        return low
    if task_type == "mcq":
        label_map = _choice_label_map(choices)
        direct = label_map.get(_norm_label(answer))
        if direct:
            return direct
        text_match = _match_mcq_choice_text(answer, choices)
        if text_match:
            return text_match
        return answer.strip().upper()
    if task_type == "math":
        boxed = _last_boxed_inner(answer)
        if boxed:
            return _normalize_math_answer_text(boxed)
        return _extract_last_number(answer) or _normalize_math_answer_text(answer)
    return answer.strip()


def _score_answer(
    *,
    task_type: str,
    prediction_text: str,
    gold_text: str,
    normalized_prediction: str,
    normalized_gold: str,
) -> tuple[bool, str]:
    if task_type == "math":
        correct, _, _, scorer = verify_math_answer(prediction_text, gold_text)
        return correct, scorer
    return bool(normalized_prediction) and normalized_prediction == normalized_gold, "normalized_exact_match"


def _build_prompt(sample: dict[str, Any]) -> str:
    task_type = sample.get("task_type", "")
    question = str(sample.get("question", "")).strip()
    context = str(sample.get("context", "")).strip()
    choices = sample.get("choices", [])

    lines: list[str] = []
    if context:
        lines.append(f"Context: {context}")
    lines.append(f"Question: {question}")

    if task_type == "mcq":
        example_label = _example_choice_label(choices)
        lines.append("Options:")
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, dict):
                    label = str(choice.get("label", "")).strip()
                    text = str(choice.get("text", "")).strip()
                    lines.append(f"{label}. {text}")
        lines.append(
            f"Return only one option label in boxed{{}}, e.g. \\boxed{{{example_label}}}."
        )
    elif task_type == "bool":
        lines.append("Return only yes/no in boxed{}, e.g. \\boxed{yes}.")
    else:
        lines.append(
            "Return only the final answer in boxed{}, e.g. \\boxed{72}."
        )

    lines.append(INSTRUCTION_SUFFIX)
    return "\n".join(lines)


def _call_model(
    *,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    include_reasoning: bool,
    chat_template_kwargs: Optional[dict[str, Any]],
    api: str,
) -> dict[str, Any]:
    if api == "completions":
        # /v1/completions does not require a chat template; prefer it for base
        # models like Llama-3.1-8B that ship without tokenizer chat_template.
        url = f"{base_url.rstrip('/')}/completions"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return _request_json("POST", url, payload, api_key=api_key, timeout=timeout)

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "include_reasoning": include_reasoning,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    return _request_json("POST", url, payload, api_key=api_key, timeout=timeout)


def _extract_choice_content(raw: dict[str, Any]) -> tuple[str, str, str, Any]:
    """Return (content, reasoning, reasoning_content, finish_reason)."""
    choice = (raw.get("choices") or [{}])[0]
    if not isinstance(choice, dict):
        return "", "", "", ""

    finish_reason = choice.get("finish_reason")

    # Chat Completions shape: choices[].message.{content,reasoning,reasoning_content}
    message = choice.get("message")
    if isinstance(message, dict):
        content = _extract_text(message.get("content"))
        reasoning = _extract_text(message.get("reasoning"))
        reasoning_content = _extract_text(message.get("reasoning_content"))
        return content, reasoning, reasoning_content, finish_reason

    # Completions shape: choices[].text
    content = _extract_text(choice.get("text"))
    reasoning = _extract_text(choice.get("reasoning"))
    reasoning_content = _extract_text(choice.get("reasoning_content"))
    return content, reasoning, reasoning_content, finish_reason


def _load_processed(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL line {line_no} in {path}: {exc}") from exc
    return rows


def _summarize(
    records: list[dict[str, Any]], *, accept_display_math_wrapper: bool
) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "accuracy": 0.0,
            "strict_boxed_rate": 0.0,
            "reasoning_field_rate": 0.0,
            "answer_only_rate": 0.0,
            "nonreason_pass_rate": 0.0,
            "nonreason_correct_rate": 0.0,
            "observable_nonreason_pass_rate": 0.0,
            "answer_scorer_counts": {},
        }

    correct = 0
    strict_boxed = 0
    has_reasoning = 0
    answer_only = 0
    nonreason_pass = 0
    nonreason_correct = 0
    observable_pass = 0
    answer_scorer_counts: dict[str, int] = {}

    for r in records:
        is_correct = bool(r.get("is_correct"))
        content = str(r.get("content") or "")
        reasoning = str(r.get("reasoning") or "")
        reasoning_content = str(r.get("reasoning_content") or "")
        task_type = str(r.get("task_type") or "")
        choices = r.get("choices", [])
        scorer = str(r.get("answer_scorer") or "unknown")
        answer_scorer_counts[scorer] = answer_scorer_counts.get(scorer, 0) + 1

        is_strict_boxed_only = _is_strict_boxed_only(
            content, accept_display_math_wrapper=accept_display_math_wrapper
        )
        has_reasoning_fields = bool(reasoning.strip() or reasoning_content.strip())
        answer_only_pred = _extract_answer_only_pred(
            task_type,
            content,
            accept_display_math_wrapper=accept_display_math_wrapper,
            choices=choices,
        )
        is_answer_only = bool(answer_only_pred)

        # New definition: answer-only + no reasoning fields.
        is_nonreason_pass = is_answer_only and not has_reasoning_fields
        # Observable definition: strict boxed only + no reasoning fields.
        is_observable_pass = is_strict_boxed_only and not has_reasoning_fields

        correct += 1 if is_correct else 0
        strict_boxed += 1 if is_strict_boxed_only else 0
        has_reasoning += 1 if has_reasoning_fields else 0
        answer_only += 1 if is_answer_only else 0
        nonreason_pass += 1 if is_nonreason_pass else 0
        nonreason_correct += 1 if (is_nonreason_pass and is_correct) else 0
        observable_pass += 1 if is_observable_pass else 0

    return {
        "total": total,
        "accuracy": correct / total,
        "strict_boxed_rate": strict_boxed / total,
        "reasoning_field_rate": has_reasoning / total,
        "answer_only_rate": answer_only / total,
        "nonreason_pass_rate": nonreason_pass / total,
        "nonreason_correct_rate": nonreason_correct / total,
        "observable_nonreason_pass_rate": observable_pass / total,
        "answer_scorer_counts": answer_scorer_counts,
    }


def _load_existing_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[warn] skipping malformed JSONL line {lineno} in {path}: {exc}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(payload, dict):
                print(f"[warn] skipping non-object JSONL line {lineno} in {path}", file=sys.stderr)
                continue
            records.append(payload)
    return records


def _record_key(dataset_name: str, rec_id: str) -> tuple[str, str]:
    return (dataset_name, rec_id)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    root_dir = Path(args.root_dir).resolve()
    processed_dir = root_dir / "processed"
    runs_dir = root_dir / "runs"
    run_name = args.run_name or dt.datetime.now().strftime("nonreason_%Y%m%d_%H%M%S")
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)

    chat_template_kwargs: dict[str, Any] = {}
    if args.chat_template_kwargs:
        try:
            parsed = json.loads(args.chat_template_kwargs)
        except json.JSONDecodeError as exc:
            print(f"Error parsing --chat-template-kwargs: {exc}", file=sys.stderr)
            return 2
        if not isinstance(parsed, dict):
            print("Error: --chat-template-kwargs must be a JSON object.", file=sys.stderr)
            return 2
        chat_template_kwargs.update(parsed)
    if args.disable_thinking:
        chat_template_kwargs["enable_thinking"] = False

    all_records: list[dict[str, Any]] = []
    per_dataset_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, str]] = set()

    if args.resume:
        for record in _load_existing_records(records_path):
            dataset_name = str(record.get("dataset", ""))
            rec_id = str(record.get("id", ""))
            if not dataset_name or not rec_id:
                continue
            key = _record_key(dataset_name, rec_id)
            if key in completed_keys:
                continue
            completed_keys.add(key)
            all_records.append(record)
            per_dataset_records[dataset_name].append(record)
        if completed_keys:
            print(f"[resume] loaded {len(completed_keys)} existing records from {records_path}")

    records_file_mode = "a" if args.resume else "w"
    with records_path.open(records_file_mode, encoding="utf-8", buffering=1) as records_file:
        for dataset_name in args.datasets:
            path = processed_dir / f"{dataset_name}.jsonl"
            if not path.exists():
                print(f"[error] missing dataset file: {path}", file=sys.stderr)
                errors.append({"dataset": dataset_name, "error": "missing_processed_file"})
                continue

            rows = _load_processed(path)
            if args.shuffle:
                random.shuffle(rows)
            if args.max_samples > 0:
                rows = rows[: args.max_samples]

            print(f"[dataset] {dataset_name}: {len(rows)} samples")
            skipped_count = 0

            for i, sample in enumerate(rows, start=1):
                prompt = _build_prompt(sample)
                rec_id = str(sample.get("id", f"{dataset_name}-{i}"))
                key = _record_key(dataset_name, rec_id)
                if key in completed_keys:
                    skipped_count += 1
                    continue

                task_type = str(sample.get("task_type", ""))
                choices = sample.get("choices", [])
                gold = _normalize_gold(task_type, str(sample.get("answer", "")), choices=choices)

                try:
                    raw = _call_model(
                        base_url=args.base_url,
                        model=args.model,
                        api_key=args.api_key,
                        prompt=prompt,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        include_reasoning=args.include_reasoning,
                        chat_template_kwargs=chat_template_kwargs or None,
                        api=args.api,
                    )
                    content, reasoning, reasoning_content, finish_reason = _extract_choice_content(raw)
                    boxed_inner = _last_boxed_inner(content)
                    pred = _normalize_pred(task_type, content, boxed_inner, choices=choices)
                    gold_source = str((sample.get("meta") or {}).get("raw_solution") or sample.get("answer", ""))
                    is_correct, answer_scorer = _score_answer(
                        task_type=task_type,
                        prediction_text=content,
                        gold_text=gold_source,
                        normalized_prediction=pred,
                        normalized_gold=gold,
                    )
                    is_strict_boxed_only = _is_strict_boxed_only(
                        content, accept_display_math_wrapper=args.accept_display_math_wrapper
                    )
                    has_reasoning_fields = bool(reasoning or reasoning_content)
                    answer_only_pred = _extract_answer_only_pred(
                        task_type,
                        content,
                        accept_display_math_wrapper=args.accept_display_math_wrapper,
                        choices=choices,
                    )
                    is_answer_only = bool(answer_only_pred)
                    nonreason_pass = is_answer_only and not has_reasoning_fields
                    observable_nonreason_pass = is_strict_boxed_only and not has_reasoning_fields

                    record = {
                        "dataset": dataset_name,
                        "id": rec_id,
                        "task_type": task_type,
                        "question": sample.get("question", ""),
                        "choices": choices,
                        "gold_answer": gold,
                        "prediction": pred,
                        "answer_scorer": answer_scorer,
                        "is_correct": is_correct,
                        "content": content,
                        "boxed_inner": boxed_inner,
                        "is_strict_boxed_only": is_strict_boxed_only,
                        "answer_only_prediction": answer_only_pred,
                        "is_answer_only": is_answer_only,
                        "reasoning": reasoning,
                        "reasoning_content": reasoning_content,
                        "has_reasoning_fields": has_reasoning_fields,
                        "nonreason_pass": nonreason_pass,
                        "observable_nonreason_pass": observable_nonreason_pass,
                        "finish_reason": finish_reason,
                        "error": "",
                    }
                except Exception as exc:
                    record = {
                        "dataset": dataset_name,
                        "id": rec_id,
                        "task_type": task_type,
                        "question": sample.get("question", ""),
                        "choices": choices,
                        "gold_answer": gold,
                        "prediction": "",
                        "answer_scorer": "error",
                        "is_correct": False,
                        "content": "",
                        "boxed_inner": "",
                        "is_strict_boxed_only": False,
                        "answer_only_prediction": "",
                        "is_answer_only": False,
                        "reasoning": "",
                        "reasoning_content": "",
                        "has_reasoning_fields": False,
                        "nonreason_pass": False,
                        "observable_nonreason_pass": False,
                        "finish_reason": "",
                        "error": str(exc),
                    }

                completed_keys.add(key)
                per_dataset_records[dataset_name].append(record)
                all_records.append(record)
                records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_file.flush()

                if i % 20 == 0:
                    print(f"  progress: {dataset_name} {i}/{len(rows)}")

            if skipped_count:
                print(f"  [resume] skipped {skipped_count} completed records")

    summary = {
        "run_name": run_name,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "datasets": args.datasets,
            "base_url": args.base_url,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "max_samples": args.max_samples,
            "shuffle": args.shuffle,
            "seed": args.seed,
            "api": args.api,
            "include_reasoning": args.include_reasoning,
            "disable_thinking": args.disable_thinking,
            "chat_template_kwargs": chat_template_kwargs,
            "accept_display_math_wrapper": args.accept_display_math_wrapper,
            "math_verify_available": math_verify_available(),
        },
        "overall": _summarize(
            all_records, accept_display_math_wrapper=args.accept_display_math_wrapper
        ),
        "per_dataset": {
            name: _summarize(
                recs, accept_display_math_wrapper=args.accept_display_math_wrapper
            )
            for name, recs in per_dataset_records.items()
        },
        "errors": errors,
        "paths": {
            "records": str(records_path),
            "run_dir": str(run_dir),
        },
    }

    summary_path = run_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] records -> {records_path}")
    print(f"[done] summary -> {summary_path}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
