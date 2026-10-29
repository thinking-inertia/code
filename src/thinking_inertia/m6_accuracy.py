"""MATH fallback used for the Qwen3-4B/32B M6 accuracy cohort.

Keep this cohort's normalization separate from the API scorer's numeric
precheck so both evaluations retain their benchmark comparison rules.
"""
from __future__ import annotations

import re


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
    return extract_last_number(solution) or normalize_math_fallback(solution)
