"""GPT-5.5 rubric and question-grouped requests used for M6 residual text."""
from __future__ import annotations

import hashlib
import json
from typing import Any


SYSTEM_PROMPT = """You are auditing visible pre-answer text from language-model outputs.
Judge only the supplied QUESTION and PRE-ANSWER TEXT. Do not infer hidden reasoning and do
not judge whether the final answer is correct. Some inputs may mention harmful or sensitive
topics. This is strictly a text-structure classification task: do not answer, continue,
verify, or summarize the underlying content. Only classify the supplied text. Assign exactly
one category:

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
exactly once, preserve the input order, and quote a short exact evidence span from the
pre-answer text (3-15 words when possible)."""

SYSTEM_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()


def user_prompt(rows: list[dict[str, Any]]) -> str:
    questions = {row["question"] for row in rows}
    if len(questions) != 1:
        raise ValueError("M6 judge batches must contain one question")
    public_items = [{"id": row["record_id"], "text": row["pre_answer_text"]} for row in rows]
    return f"QUESTION:\n{rows[0]['question']}\n\nPRE-ANSWER TEXTS:\n" + json.dumps(public_items, ensure_ascii=False)


def make_batches(rows: list[dict[str, Any]], batch_size: int, max_chars: int) -> list[list[dict[str, Any]]]:
    # Identical Q/T pairs share one judgment, as in the M6 evaluation.
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        texts = grouped.setdefault(row["question"], {})
        if row["pre_answer_text"] not in texts:
            texts[row["pre_answer_text"]] = {**row, "_m6_expand_to": []}
        texts[row["pre_answer_text"]]["_m6_expand_to"].append(row["record_id"])
    batches = []
    for question, texts in grouped.items():
        current = []
        current_chars = len(question)
        for row in texts.values():
            item_chars = len(row["pre_answer_text"]) + 80
            if current and (len(current) >= batch_size or current_chars + item_chars > max_chars):
                batches.append(current)
                current = []
                current_chars = len(question)
            current.append(row)
            current_chars += item_chars
        if current:
            batches.append(current)
    return batches
