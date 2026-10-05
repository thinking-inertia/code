"""Public rubric and parsing helpers for visible-inference judgments."""

from __future__ import annotations

import json
import re
from typing import Any


CATEGORIES = {
    "empty",
    "generic_or_off_topic",
    "paraphrase_only",
    "relevant_noninferential",
    "explicit_reasoning",
    "unclear",
}

SYSTEM_PROMPT = """You are auditing visible pre-answer text from language-model outputs.
Judge only the supplied QUESTION and PRE-ANSWER TEXT. Do not infer hidden reasoning and do
not judge whether the final answer is correct. This is a text-structure classification task:
do not answer or continue the underlying question. Assign exactly one category:

- generic_or_off_topic: generic lead-in, formatting text, or off-topic content with no
  question-specific inferential content.
- paraphrase_only: only repeats or paraphrases the question or supplied facts, without
  connecting them through an inference.
- relevant_noninferential: question-specific definitions, retrieved facts, descriptions,
  or answer assertions, but no explicit step that advances from evidence toward an answer.
- explicit_reasoning: contains at least one visible deduction, computation, comparison,
  option elimination, rule application, evidence-to-conclusion link, or causal step that
  advances toward an answer. It need not be long, deep, or correct.
- unclear: the distinction cannot be made reliably from the visible text.

A relevant paraphrase is not explicit reasoning. An answer assertion alone is not explicit
reasoning. Conversely, a single concrete inferential or computational step is explicit
reasoning. Return JSON only as {"labels":[[id,category,evidence], ...]}. Include every ID
exactly once and quote a short exact supporting span from the pre-answer text.
"""


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def evidence_is_verbatim(evidence: str, source: str) -> bool:
    evidence_norm = normalize_space(evidence).lower()
    source_norm = normalize_space(source).lower()
    return bool(evidence_norm) and evidence_norm in source_norm


def parse_json_payload(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        if start < 0:
            raise
        payload, _ = json.JSONDecoder().raw_decode(raw[start:])
    if not isinstance(payload, dict):
        raise ValueError("judge payload must be a JSON object")
    return payload
