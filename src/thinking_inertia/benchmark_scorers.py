#!/usr/bin/env python3
"""
Benchmark answer scoring helpers.

For math benchmarks, use lightweight answer normalization for exact numeric/boxed
matches before falling back to `math-verify` for harder symbolic equivalence. This
keeps simple answers such as `\boxed{36}` vs. `36` robust across math-verify versions.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import multiprocessing
import os
import re
import threading
from fractions import Fraction
from functools import lru_cache
from typing import Any


MATH_VERIFY_PARSE_TIMEOUT = float(os.environ.get("MATH_VERIFY_PARSE_TIMEOUT", "5"))
MATH_VERIFY_PROCESS_TIMEOUT = float(os.environ.get("MATH_VERIFY_PROCESS_TIMEOUT", "12"))
MATH_VERIFY_WORKERS = max(1, int(os.environ.get("MATH_VERIFY_WORKERS", "2")))
_MATH_VERIFY_EXECUTOR: concurrent.futures.ProcessPoolExecutor | None = None
_MATH_VERIFY_EXECUTOR_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _load_math_verify() -> tuple[Any, Any, Any, Any] | None:
    try:
        from math_verify import parse, verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    except Exception:
        return None
    return parse, verify, ExprExtractionConfig, LatexExtractionConfig


def math_verify_available() -> bool:
    return _load_math_verify() is not None


def _strip_wrappers(text: str) -> str:
    t = text.strip()
    for start, end in (("$$", "$$"), ("\\[", "\\]"), ("$", "$")):
        if t.startswith(start) and t.endswith(end) and len(t) >= len(start) + len(end):
            return t[len(start) : len(t) - len(end)].strip()
    return t


def find_boxed_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for command in (r"\boxed", r"\fbox"):
        start = 0
        while True:
            idx = text.find(command, start)
            if idx < 0:
                break
            brace_start = text.find("{", idx + len(command))
            if brace_start < 0:
                start = idx + len(command)
                continue
            depth = 0
            for pos in range(brace_start, len(text)):
                char = text[pos]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        spans.append((text[brace_start + 1 : pos].strip(), idx, pos + 1))
                        start = pos + 1
                        break
            else:
                start = brace_start + 1
    return sorted(spans, key=lambda item: item[1])


def last_boxed_inner(text: str) -> str:
    spans = find_boxed_spans(text)
    return spans[-1][0] if spans else ""


def extract_last_number(text: str) -> str:
    pattern = r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?"
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    return matches[-1].replace(",", "")


def extract_last_latex_fraction(text: str) -> str:
    pattern = r"\\(?:dfrac|tfrac|frac)\s*\{[^{}]+\}\s*\{[^{}]+\}"
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    return normalize_math_fallback(matches[-1])


def normalize_math_fallback(text: str) -> str:
    t = _strip_wrappers(text)
    boxed = last_boxed_inner(t)
    if boxed:
        t = boxed
    t = t.strip()
    t = re.sub(r"\\\\(?=[A-Za-z])", r"\\", t)
    t = re.sub(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", t)
    t = re.sub(r"\\(?:left|right)\s*", "", t)
    t = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", t)
    t = t.replace("\\,", "").replace("\\!", "")
    t = re.sub(r"\s+", "", t)
    if re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?", t):
        return t.replace(",", "")
    return t.lower()


def extract_math_answer_fallback(solution: str) -> str:
    boxed = last_boxed_inner(solution)
    if boxed:
        return normalize_math_fallback(boxed)
    normalized = normalize_math_fallback(solution)
    if re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/[-+]?\d[\d,]*(?:\.\d+)?)?", normalized):
        return normalized.replace(",", "")
    latex_fraction = extract_last_latex_fraction(solution)
    if latex_fraction:
        return latex_fraction
    return extract_last_number(solution) or normalize_math_fallback(solution)


def _simple_number_value(text: str) -> Fraction | None:
    """Parse simple integer/decimal/fraction answers for exact numeric comparison."""
    t = text.strip().replace(",", "")
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[-+]?(?:\d+(?:\.\d+)?|\.\d+))?", t):
        return None
    try:
        if "/" in t:
            num, den = t.split("/", 1)
            denominator = Fraction(den)
            if denominator == 0:
                return None
            return Fraction(num) / denominator
        return Fraction(t)
    except (ValueError, ZeroDivisionError):
        return None


def _fallback_match(prediction_text: str, gold_text: str) -> tuple[bool, str, str]:
    pred = extract_math_answer_fallback(prediction_text)
    gold = extract_math_answer_fallback(gold_text)
    if not pred or not gold:
        return False, pred, gold
    if pred == gold:
        return True, pred, gold

    pred_number = _simple_number_value(pred)
    gold_number = _simple_number_value(gold)
    if pred_number is not None and gold_number is not None and pred_number == gold_number:
        return True, pred, gold
    return False, pred, gold


def _parse_math_with_math_verify(text: str, *, parsing_timeout: float | None = MATH_VERIFY_PARSE_TIMEOUT) -> Any:
    loaded = _load_math_verify()
    if loaded is None:
        return None
    parse, _, ExprExtractionConfig, LatexExtractionConfig = loaded
    extraction_config = [
        LatexExtractionConfig(),
        ExprExtractionConfig(),
    ]
    try:
        return parse(text, extraction_config=extraction_config, parsing_timeout=parsing_timeout)
    except TypeError:
        try:
            return parse(text, extraction_config=extraction_config)
        except TypeError:
            return parse(text)


def _verify_math_answer_direct(prediction_text: str, gold_text: str) -> tuple[bool, str, str, str]:
    loaded = _load_math_verify()
    if loaded is None:
        raise RuntimeError("math_verify is not available")
    _, verify, _, _ = loaded
    pred_parsed = _parse_math_with_math_verify(prediction_text)
    gold_parsed = _parse_math_with_math_verify(gold_text)
    return bool(verify(gold_parsed, pred_parsed)), str(pred_parsed), str(gold_parsed), "math_verify"


def _get_math_verify_executor() -> concurrent.futures.ProcessPoolExecutor:
    global _MATH_VERIFY_EXECUTOR
    with _MATH_VERIFY_EXECUTOR_LOCK:
        if _MATH_VERIFY_EXECUTOR is None:
            context = multiprocessing.get_context("spawn")
            _MATH_VERIFY_EXECUTOR = concurrent.futures.ProcessPoolExecutor(
                max_workers=MATH_VERIFY_WORKERS,
                mp_context=context,
            )
            atexit.register(_MATH_VERIFY_EXECUTOR.shutdown, wait=False, cancel_futures=True)
        return _MATH_VERIFY_EXECUTOR


def _verify_math_answer_in_process(prediction_text: str, gold_text: str) -> tuple[bool, str, str, str]:
    executor = _get_math_verify_executor()
    future = executor.submit(_verify_math_answer_direct, prediction_text, gold_text)
    correct, pred, gold, _ = future.result(timeout=MATH_VERIFY_PROCESS_TIMEOUT)
    return correct, pred, gold, "math_verify_process"


def verify_math_answer(prediction_text: str, gold_text: str) -> tuple[bool, str, str, str]:
    math_verify_mode = os.environ.get("MATH_VERIFY_MODE", "auto").strip().lower()
    fallback_correct, fallback_pred, fallback_gold = _fallback_match(prediction_text, gold_text)
    if math_verify_mode in {"fallback", "off", "none", "0", "false"}:
        return fallback_correct, fallback_pred, fallback_gold, "fallback_normalized_exact_match"

    if fallback_correct:
        return True, fallback_pred, fallback_gold, "fallback_precheck_exact_match"

    loaded = _load_math_verify()
    if loaded is not None:
        try:
            if threading.current_thread() is threading.main_thread():
                return _verify_math_answer_direct(prediction_text, gold_text)
            return _verify_math_answer_in_process(prediction_text, gold_text)
        except Exception:
            # Keep long-running API jobs moving if math-verify times out on a
            # malformed generation; the scorer field records the fallback.
            pass

    return fallback_correct, fallback_pred, fallback_gold, "fallback_normalized_exact_match"
