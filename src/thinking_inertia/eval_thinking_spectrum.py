#!/usr/bin/env python3
"""
Evaluate thinking-spectrum Q:(T+A) prompting modes and optionally score sentence-level similarity with the bundled MiniLM compatibility scorer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import random
import re
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .benchmark_scorers import math_verify_available
from .eval_nonreason import (
    _extract_answer_only_pred,
    _example_choice_label,
    _extract_text,
    _find_boxed_spans,
    _find_last_mcq_choice_text_span,
    _find_last_mcq_label_span,
    _is_strict_boxed_only,
    _last_boxed_inner,
    _load_processed,
    _normalize_gold,
    _normalize_pred,
    _request_json,
    _score_answer,
    _strip_latex_wrappers,
)
from .minilm_similarity import (
    MiniLMSimilarityScorer,
    normalize_space,
    split_explanation_sentences,
)


MODE_SPECS: dict[str, dict[str, Any]] = {
    "mode1": {
        "name": "thinking_on",
        "title": "Mode 1: Thinking On",
        "paper_title": "thinking-on reference",
        "instruction": "",
        "enable_thinking": True,
        "use_think_tags": True,
    },
    "mode2": {
        "name": "native_no_think",
        "title": "Mode 2: Native No Think",
        "paper_title": "native no-think baseline",
        "instruction": "",
        "enable_thinking": False,
        "use_think_tags": False,
    },
    "mode3": {
        "name": "step_by_step_no_think",
        "title": "Mode 3: Step by Step, No Think",
        "paper_title": "reasoning re-elicitation",
        "instruction": "Think step by step.",
        "enable_thinking": False,
        "use_think_tags": False,
    },
    "mode4": {
        "name": "short_lead_in_no_explain",
        "title": "Mode 4: Short Lead-In",
        "paper_title": "soft regularization",
        "instruction": "A very short lead-in is allowed, but do not explain.",
        "enable_thinking": False,
        "use_think_tags": False,
    },
    "mode5": {
        "name": "strict_answer_only",
        "title": "Mode 5: Strict Answer Only",
        "paper_title": "strict regularization",
        "instruction": "Give the final answer directly without any reasoning.",
        "enable_thinking": False,
        "use_think_tags": False,
    },
    "mode6": {
        "name": "boxed_answer_is",
        "title": "Mode 6: Boxed Answer Is",
        "paper_title": "prefix forcing",
        "instruction": "",
        "enable_thinking": False,
        "use_think_tags": False,
    },
}

MODE_ALIASES = {
    "1": "mode1",
    "2": "mode2",
    "3": "mode3",
    "4": "mode4",
    "5": "mode5",
    "6": "mode6",
    "mode1": "mode1",
    "mode2": "mode2",
    "mode3": "mode3",
    "mode4": "mode4",
    "mode5": "mode5",
    "mode6": "mode6",
    "thinking_on": "mode1",
    "native_no_think": "mode2",
    "brief_explain_then_answer": "mode2",
    "step_by_step_no_think": "mode3",
    "short_lead_in_no_explain": "mode4",
    "strict_answer_only": "mode5",
    "boxed_answer_is": "mode6",
}
MODE_ORDER = ["mode1", "mode2", "mode3", "mode4", "mode5", "mode6"]
LETTER_SET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PROVIDER_PROFILES = {
    "none",
    "qwen3",
    "deepseek_v4",
    "openai_gpt",
    "gemini3_flash",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-mode thinking-spectrum evaluation against a local vLLM endpoint."
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
        help="Dataset names corresponding to processed/<name>.jsonl.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8201/v1",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Served model name.",
    )
    parser.add_argument(
        "--api",
        choices=["chat", "completions", "responses"],
        default="chat",
        help="Which OpenAI-compatible endpoint to use.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key.",
    )
    parser.add_argument(
        "--provider-profile",
        choices=sorted(PROVIDER_PROFILES),
        default="none",
        help=(
            "Provider thinking-control profile. Use qwen3, deepseek_v4, openai_gpt, "
            "or gemini3_flash for the API frontier extension."
        ),
    )
    parser.add_argument(
        "--qwen3-control",
        choices=["chat_template", "openrouter_reasoning", "prompt_directive"],
        default="chat_template",
        help=(
            "Qwen3 thinking-control mechanism for smoke tests. chat_template matches "
            "the existing local Qwen runs; openrouter_reasoning uses the normalized "
            "OpenRouter reasoning field; prompt_directive prepends /think or /no_think."
        ),
    )
    parser.add_argument(
        "--openai-mode1-effort",
        choices=["low", "medium", "high", "xhigh"],
        default="high",
        help="OpenAI reasoning effort for Mode 1 (default: high).",
    )
    parser.add_argument(
        "--gemini-mode1-effort",
        choices=["low", "medium", "high"],
        default="high",
        help="Gemini thinking level for Mode 1 (default: high).",
    )
    parser.add_argument(
        "--gemini-off-effort",
        choices=["minimal", "low", "medium", "high"],
        default="minimal",
        help="Gemini near-off thinking level for Modes 2-6 (default: minimal).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["mode1", "mode2", "mode3", "mode4", "mode5", "mode6"],
        help="Subset of modes to run. Supports numeric aliases or canonical names.",
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
        default=1024,
        help="max_tokens for model generation (default: 1024).",
    )
    parser.add_argument(
        "--mode1-max-tokens",
        type=int,
        default=4096,
        help="Override max_tokens for mode1 thinking-on runs (default: 4096).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="HTTP timeout seconds (default: 300).",
    )
    parser.add_argument(
        "--request-concurrency",
        type=int,
        default=1,
        help=(
            "Concurrent API requests per dataset/mode. Use 8-16 for OpenRouter smoke/pilot "
            "runs if the provider rate limit allows it. Default: 1."
        ),
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=2,
        help="Retries per API request after transient HTTP/network failures. Default: 2.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.0,
        help="Initial retry backoff seconds; doubles after each failed attempt. Default: 1.0.",
    )
    parser.add_argument(
        "--openrouter-provider-only",
        nargs="+",
        default=[],
        help=(
            "Optional OpenRouter provider allow-list, e.g. --openrouter-provider-only "
            "DeepInfra. Useful when providers differ in reasoning-control support."
        ),
    )
    parser.add_argument(
        "--openrouter-require-parameters",
        action="store_true",
        help=(
            "Set OpenRouter provider.require_parameters=true so routes that do not "
            "support requested provider parameters are filtered out."
        ),
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
        "--sample-strategy",
        choices=["first", "random", "level_balanced"],
        default="first",
        help=(
            "How to select --max-samples at evaluation time. level_balanced balances "
            "MATH rows by meta.level and uses seeded random sampling for other datasets."
        ),
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
        "--omit-include-reasoning-param",
        action="store_true",
        help=(
            "Do not send the OpenRouter-specific include_reasoning request field. "
            "Use this for official provider APIs that return reasoning fields natively."
        ),
    )
    parser.add_argument(
        "--accept-display-math-wrapper",
        action="store_true",
        help="Treat $$\\boxed{...}$$ as valid answer-only formatting.",
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


def _canonical_mode(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    if key not in MODE_ALIASES:
        raise KeyError(name)
    return MODE_ALIASES[key]


def _answer_format_line(task_type: str, choices: Any = None) -> str:
    if task_type == "mcq":
        example_label = _example_choice_label(choices)
        return f"Put only the final option label in boxed{{}}, e.g. \\boxed{{{example_label}}}."
    if task_type == "bool":
        return "Put only the final yes/no answer in boxed{}, e.g. \\boxed{yes}."
    return "Put the final answer in boxed{}, e.g. \\boxed{72}."


def _mode6_answer_format_line(task_type: str, choices: Any = None) -> str:
    if task_type == "mcq":
        example_label = _example_choice_label(choices)
        return (
            f"Put only the final option label in boxed{{}}, e.g. \\boxed{{{example_label}}}. "
            "The answer is \\boxed{"
        )
    if task_type == "bool":
        return "Put only the final yes/no answer in boxed{}, e.g. \\boxed{yes}. The answer is \\boxed{"
    return "Put the final answer in boxed{}, e.g. \\boxed{72}. The answer is \\boxed{"


def _build_problem_text(sample: dict[str, Any]) -> str:
    question = str(sample.get("question", "")).strip()
    context = str(sample.get("context", "")).strip()
    task_type = str(sample.get("task_type", "")).strip()
    choices = sample.get("choices", [])

    lines: list[str] = []
    if context:
        lines.append(f"Context: {context}")
    lines.append(f"Question: {question}")
    if task_type == "mcq":
        lines.append("Options:")
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, dict):
                    label = str(choice.get("label", "")).strip()
                    text = str(choice.get("text", "")).strip()
                    lines.append(f"{label}. {text}")
    return "\n".join(lines)


def _build_similarity_question_text(sample: dict[str, Any]) -> str:
    return _build_problem_text(sample)


def _build_mode_prompt(sample: dict[str, Any], mode: str) -> str:
    task_type = str(sample.get("task_type", "")).strip()
    choices = sample.get("choices", [])
    answer_line = (
        _mode6_answer_format_line(task_type, choices)
        if mode == "mode6"
        else _answer_format_line(task_type, choices)
    )
    lines = [_build_problem_text(sample), MODE_SPECS[mode]["instruction"], answer_line]
    return "\n".join(line for line in lines if line)


def _apply_qwen3_prompt_directive(prompt: str, mode: str, provider_profile: str, qwen3_control: str) -> str:
    if provider_profile != "qwen3" or qwen3_control != "prompt_directive":
        return prompt
    directive = "/think" if mode == "mode1" else "/no_think"
    return f"{directive}\n{prompt}"


def _make_chat_template_kwargs(
    mode: str,
    api: str,
    provider_profile: str,
    qwen3_control: str,
) -> dict[str, Any] | None:
    if api != "chat":
        return None
    if provider_profile == "qwen3" and qwen3_control != "chat_template":
        return None
    if provider_profile not in {"none", "qwen3"}:
        return None
    return {"enable_thinking": bool(MODE_SPECS[mode]["enable_thinking"])}


def _provider_payload_controls(
    *,
    mode: str,
    provider_profile: str,
    qwen3_control: str,
    openai_mode1_effort: str,
    gemini_mode1_effort: str,
    gemini_off_effort: str,
    api: str = "chat",
) -> dict[str, Any]:
    thinking_on = mode == "mode1"
    if provider_profile == "qwen3" and qwen3_control == "openrouter_reasoning":
        return {"reasoning": {"effort": "high" if thinking_on else "none"}}
    if provider_profile == "deepseek_v4":
        if thinking_on:
            return {
                "thinking": {"type": "enabled", "reasoning_effort": "high"},
                "reasoning_effort": "high",
            }
        return {"thinking": {"type": "disabled"}}

    if provider_profile == "openai_gpt":
        if thinking_on:
            return {"reasoning": {"effort": openai_mode1_effort, "summary": "auto"}}
        return {"reasoning": {"effort": "none"}}

    if provider_profile == "gemini3_flash":
        return {"reasoning": {"effort": gemini_mode1_effort if thinking_on else gemini_off_effort}}

    return {}


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
    omit_include_reasoning_param: bool,
    include_temperature: bool,
    chat_template_kwargs: dict[str, Any] | None,
    payload_controls: dict[str, Any],
    openrouter_provider_only: list[str] | None,
    openrouter_require_parameters: bool,
    api: str,
    request_retries: int = 2,
    retry_backoff: float = 1.0,
) -> dict[str, Any]:
    def provider_preferences() -> dict[str, Any] | None:
        provider: dict[str, Any] = {}
        if openrouter_provider_only:
            provider["only"] = openrouter_provider_only
        if openrouter_require_parameters:
            provider["require_parameters"] = True
        return provider or None

    def request_with_retries(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        attempts = max(0, request_retries) + 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return _request_json("POST", url, payload, api_key=api_key, timeout=timeout)
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts - 1:
                    break
                time.sleep(max(0.0, retry_backoff) * (2**attempt))
        raise last_exc or RuntimeError("request failed")

    if api == "responses":
        url = f"{base_url.rstrip('/')}/responses"
        payload: dict[str, Any] = {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }
        provider = provider_preferences()
        if provider:
            payload["provider"] = provider
        payload.update(payload_controls)
        return request_with_retries(url, payload)

    if api == "completions":
        url = f"{base_url.rstrip('/')}/completions"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        if include_temperature:
            payload["temperature"] = temperature
        provider = provider_preferences()
        if provider:
            payload["provider"] = provider
        payload.update(payload_controls)
        return request_with_retries(url, payload)

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if not omit_include_reasoning_param:
        payload["include_reasoning"] = include_reasoning
    if include_temperature:
        payload["temperature"] = temperature
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    provider = provider_preferences()
    if provider:
        payload["provider"] = provider
    payload.update(payload_controls)
    return request_with_retries(url, payload)


def _localize_answer_span(task_type: str, content: str, choices: Any = None) -> tuple[str, int, int]:
    text = content.strip()
    if not text:
        return "", -1, -1

    boxed_spans = _find_boxed_spans(text)
    if boxed_spans:
        _, start, end = boxed_spans[-1]
        return text[start:end].strip(), start, end

    if task_type == "bool":
        matches = list(re.finditer(r"\b(?:yes|no|true|false|1|0)\b", text, flags=re.IGNORECASE))
        if matches:
            match = matches[-1]
            return match.group(0).strip(), match.start(), match.end()
        return "", -1, -1

    if task_type == "mcq":
        label_match, start, end = _find_last_mcq_label_span(text, choices)
        if start >= 0 and end >= 0:
            return label_match, start, end

        _, start, end = _find_last_mcq_choice_text_span(text, choices)
        if start >= 0 and end >= 0:
            return text[start:end].strip(), start, end

        for pattern in [
            r"(?:answer is|answer:|option|choice)\s*([A-Z])\b",
            r"\b([A-Z])\b",
        ]:
            matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
            if matches:
                match = matches[-1]
                if match.lastindex:
                    start, end = match.span(1)
                    return text[start:end], start, end
                return match.group(0).strip(), match.start(), match.end()
        return "", -1, -1

    matches = list(re.finditer(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?", text))
    if matches:
        match = matches[-1]
        return match.group(0).strip(), match.start(), match.end()
    return "", -1, -1


def _extract_visible_t_and_a(
    task_type: str,
    content: str,
    *,
    choices: Any = None,
    accept_display_math_wrapper: bool,
) -> tuple[str, str, bool]:
    working = _strip_latex_wrappers(content, accept_display_math_wrapper).strip()
    answer_raw, start, end = _localize_answer_span(task_type, working, choices)
    if start < 0 or end < 0:
        return normalize_space(working), "", False
    prefix = normalize_space(working[:start])
    return prefix, answer_raw, True


def _extract_think_block(content: str) -> tuple[str, str]:
    text = content.strip()
    if not text:
        return "", ""

    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        think_text = normalize_space(match.group(1))
        suffix = normalize_space(text[match.end() :])
        return think_text, suffix

    start_match = re.search(r"<think>", text, flags=re.IGNORECASE)
    if start_match:
        think_text = normalize_space(text[start_match.end() :])
        return think_text, ""

    return "", text


def _first_choice(raw: dict[str, Any]) -> dict[str, Any]:
    choice = (raw.get("choices") or [{}])[0]
    return choice if isinstance(choice, dict) else {}


def _extract_choice_extended(raw: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw.get("output"), list):
        content_parts: list[str] = []
        reasoning_details: list[dict[str, str]] = []
        for item in raw["output"]:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower()
            if item_type == "message":
                content_parts.append(_extract_text(item.get("content")))
            elif item_type == "reasoning":
                for summary in item.get("summary") or []:
                    if not isinstance(summary, dict):
                        continue
                    text = _extract_text(summary.get("text"))
                    if text:
                        reasoning_details.append(
                            {"type": "summary_text", "text": text}
                        )
        return {
            "content": "".join(content_parts),
            "reasoning": "",
            "reasoning_content": "",
            "reasoning_details": reasoning_details,
            "finish_reason": raw.get("status", ""),
            "usage": raw.get("usage", {}),
        }

    choice = _first_choice(raw)
    finish_reason = choice.get("finish_reason", "")
    message = choice.get("message")
    source = message if isinstance(message, dict) else choice
    if not isinstance(source, dict):
        source = {}
    content = _extract_text(source.get("content") if "content" in source else source.get("text"))
    reasoning = _extract_text(source.get("reasoning"))
    reasoning_content = _extract_text(source.get("reasoning_content"))
    reasoning_details = source.get("reasoning_details")
    if reasoning_details is None:
        reasoning_details = choice.get("reasoning_details")
    return {
        "content": content,
        "reasoning": reasoning,
        "reasoning_content": reasoning_content,
        "reasoning_details": reasoning_details,
        "finish_reason": finish_reason,
        "usage": raw.get("usage", {}),
    }


def _reasoning_detail_parts(reasoning_details: Any) -> tuple[list[str], list[str], list[str]]:
    raw_parts: list[str] = []
    summary_parts: list[str] = []
    signal_types: list[str] = []

    if not reasoning_details:
        return raw_parts, summary_parts, signal_types

    items = reasoning_details if isinstance(reasoning_details, list) else [reasoning_details]
    for item in items:
        if isinstance(item, str):
            text = normalize_space(item)
            if text:
                raw_parts.append(text)
                signal_types.append("reasoning_details.text")
            continue
        if not isinstance(item, dict):
            signal_types.append(f"reasoning_details.{type(item).__name__}")
            continue

        item_type = str(item.get("type") or item.get("format") or "").strip().lower()
        if item_type:
            signal_types.append(f"reasoning_details.{item_type}")
        else:
            signal_types.append("reasoning_details")

        text_values: list[str] = []
        for key in ("text", "content", "reasoning", "reasoning_text"):
            value = item.get(key)
            text = _extract_text(value)
            if normalize_space(text):
                text_values.append(normalize_space(text))

        summary_values: list[str] = []
        for key in ("summary", "summaries"):
            value = item.get(key)
            text = _extract_text(value)
            if normalize_space(text):
                summary_values.append(normalize_space(text))

        if "summary" in item_type or "thought" in item_type:
            summary_parts.extend(text_values)
        elif "encrypted" in item_type or "signature" in item_type:
            pass
        else:
            raw_parts.extend(text_values)
        summary_parts.extend(summary_values)

    return raw_parts, summary_parts, sorted(set(signal_types))


def _join_t_parts(parts: list[str]) -> str:
    seen: set[str] = set()
    deduped: list[str] = []
    for part in parts:
        text = normalize_space(part)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return normalize_space("\n".join(deduped))


def _build_t_payload(
    *,
    reasoning: str,
    reasoning_content: str,
    reasoning_details: Any,
    think_block: str,
    visible_prefix: str,
) -> dict[str, Any]:
    details_raw, details_summaries, detail_signal_types = _reasoning_detail_parts(reasoning_details)
    visible_parts = [
        normalize_space(reasoning),
        normalize_space(reasoning_content),
        *details_raw,
        normalize_space(think_block),
        normalize_space(visible_prefix),
    ]
    summary_parts = details_summaries
    t_visible_raw = _join_t_parts(visible_parts)
    t_returned_readable = _join_t_parts([t_visible_raw, *summary_parts])

    signal_types: list[str] = []
    if normalize_space(reasoning):
        signal_types.append("reasoning")
    if normalize_space(reasoning_content):
        signal_types.append("reasoning_content")
    if reasoning_details:
        signal_types.extend(detail_signal_types or ["reasoning_details"])
    if normalize_space(think_block):
        signal_types.append("content_think_block")

    if normalize_space(visible_prefix):
        t_source = "returned_readable_plus_visible_prefix" if t_visible_raw != normalize_space(visible_prefix) else "visible_prefix"
    elif t_returned_readable:
        t_source = "returned_readable"
    else:
        t_source = "empty"

    return {
        "t_visible_raw": t_visible_raw,
        "t_returned_readable": t_returned_readable,
        "reasoning_details_raw_text": _join_t_parts(details_raw),
        "reasoning_details_summary_text": _join_t_parts(summary_parts),
        "reasoning_signal_types": sorted(set(signal_types)),
        "has_reasoning_signal": bool(signal_types),
        "t_source": t_source,
    }


def _usage_reasoning_tokens(usage: Any) -> int:
    """Return reported hidden/reasoning-token count from provider usage metadata."""
    if not isinstance(usage, dict):
        return 0
    total = 0
    stack: list[Any] = [usage]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key == "reasoning_tokens" and isinstance(value, (int, float)):
                    total += int(value)
                elif isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return total


def _merge_reasoning_signals(t_payload: dict[str, Any], usage: Any) -> tuple[bool, list[str], int]:
    usage_tokens = _usage_reasoning_tokens(usage)
    signal_types = list(t_payload["reasoning_signal_types"])
    if usage_tokens > 0:
        signal_types.append("usage.reasoning_tokens")
    return bool(t_payload["has_reasoning_signal"] or usage_tokens > 0), sorted(set(signal_types)), usage_tokens


def _clip_rate(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _summarize_records(
    records: list[dict[str, Any]],
    *,
    accept_display_math_wrapper: bool,
) -> dict[str, Any]:
    if not records:
        return {
            "total": 0,
            "accuracy": 0.0,
            "mean_thinking_rate": 0.0,
            "median_thinking_rate": 0.0,
            "mean_nonreason_rate": 0.0,
            "mean_max_thinking_rate": 0.0,
            "median_max_thinking_rate": 0.0,
            "mean_max_nonreason_rate": 0.0,
            "mean_t_word_count": 0.0,
            "mean_t_char_count": 0.0,
            "empty_t_rate": 0.0,
            "answer_parse_failure_rate": 0.0,
            "request_error_rate": 0.0,
            "reasoning_field_rate": 0.0,
            "has_reasoning_signal_rate": 0.0,
            "strict_boxed_rate": 0.0,
            "answer_only_rate": 0.0,
            "content_answer_only_rate": 0.0,
            "api_answer_only_rate": 0.0,
            "nonreason_pass_rate": 0.0,
            "nonreason_correct_rate": 0.0,
            "observable_nonreason_pass_rate": 0.0,
            "mean_t_visible_raw_thinking_rate": 0.0,
            "mean_visible_payload_full_similarity": 0.0,
            "mean_visible_payload_full_nonreason_rate": 0.0,
            "answer_scorer_counts": {},
        }

    total = len(records)
    correct = sum(1 for r in records if r.get("is_correct"))
    thinking_rates = [float(r.get("thinking_rate", 0.0)) for r in records]
    nonreason_rates = [float(r.get("nonreason_rate", 0.0)) for r in records]
    max_thinking_rates = [float(r.get("max_thinking_rate", 0.0)) for r in records]
    max_nonreason_rates = [float(r.get("max_nonreason_rate", 1.0)) for r in records]
    t_word_counts = [int(r.get("t_word_count", 0)) for r in records]
    t_char_counts = [int(r.get("t_char_count", 0)) for r in records]
    empty_t = sum(1 for r in records if not normalize_space(str(r.get("extracted_t", ""))))
    parse_failed = sum(1 for r in records if r.get("answer_parse_failed"))
    request_errors = sum(1 for r in records if str(r.get("error", "")).strip())
    has_reasoning_fields = sum(1 for r in records if r.get("has_reasoning_fields"))
    has_reasoning_signal = sum(1 for r in records if r.get("has_reasoning_signal"))
    content_answer_only = sum(1 for r in records if r.get("is_content_answer_only", r.get("is_answer_only")))
    api_answer_only = sum(1 for r in records if r.get("api_answer_only_pass", r.get("nonreason_pass")))
    api_answer_only_correct = sum(1 for r in records if r.get("api_answer_only_pass", r.get("nonreason_pass")) and r.get("is_correct"))
    strict_boxed = sum(1 for r in records if r.get("is_strict_boxed_only"))
    observable_nonreason = sum(1 for r in records if r.get("observable_nonreason_pass"))
    visible_raw_rates = [float(r.get("t_visible_raw_thinking_rate", r.get("thinking_rate", 0.0))) for r in records]
    visible_payload_full_sims = [
        float(r.get("visible_payload_full_similarity", 0.0))
        for r in records
    ]
    visible_payload_full_nonreason = [
        float(r.get("visible_payload_full_nonreason_rate", 1.0))
        for r in records
    ]
    answer_scorer_counts: dict[str, int] = {}
    for r in records:
        scorer = str(r.get("answer_scorer") or "unknown")
        answer_scorer_counts[scorer] = answer_scorer_counts.get(scorer, 0) + 1

    return {
        "total": total,
        "accuracy": correct / total,
        "mean_thinking_rate": sum(thinking_rates) / total,
        "median_thinking_rate": float(statistics.median(thinking_rates)),
        "mean_nonreason_rate": sum(nonreason_rates) / total,
        "mean_max_thinking_rate": sum(max_thinking_rates) / total,
        "median_max_thinking_rate": float(statistics.median(max_thinking_rates)),
        "mean_max_nonreason_rate": sum(max_nonreason_rates) / total,
        "mean_t_word_count": sum(t_word_counts) / total,
        "mean_t_char_count": sum(t_char_counts) / total,
        "empty_t_rate": empty_t / total,
        "answer_parse_failure_rate": parse_failed / total,
        "request_error_rate": request_errors / total,
        "reasoning_field_rate": has_reasoning_fields / total,
        "has_reasoning_signal_rate": has_reasoning_signal / total,
        "strict_boxed_rate": strict_boxed / total,
        "answer_only_rate": content_answer_only / total,
        "content_answer_only_rate": content_answer_only / total,
        "api_answer_only_rate": api_answer_only / total,
        "nonreason_pass_rate": api_answer_only / total,
        "nonreason_correct_rate": api_answer_only_correct / total,
        "observable_nonreason_pass_rate": observable_nonreason / total,
        "mean_t_visible_raw_thinking_rate": sum(visible_raw_rates) / total,
        "mean_visible_payload_full_similarity": sum(visible_payload_full_sims) / total,
        "mean_visible_payload_full_nonreason_rate": sum(visible_payload_full_nonreason) / total,
        "answer_scorer_counts": answer_scorer_counts,
    }


def _group_summary(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    accept_display_math_wrapper: bool,
) -> dict[str, Any]:
    return {
        key: _summarize_records(
            rows,
            accept_display_math_wrapper=accept_display_math_wrapper,
        )
        for key, rows in grouped.items()
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


def _record_key(mode: str, dataset_name: str, rec_id: str) -> tuple[str, str, str]:
    return (mode, dataset_name, rec_id)


def _select_eval_rows(
    dataset_name: str,
    rows: list[dict[str, Any]],
    *,
    max_samples: int,
    seed: int,
    shuffle: bool,
    sample_strategy: str,
) -> list[dict[str, Any]]:
    if max_samples <= 0:
        picked = list(rows)
        if shuffle:
            random.Random(seed).shuffle(picked)
        return picked

    strategy = "random" if shuffle and sample_strategy == "first" else sample_strategy
    if len(rows) <= max_samples:
        picked = list(rows)
        if strategy in {"random", "level_balanced"}:
            random.Random(seed).shuffle(picked)
        return picked

    rng = random.Random(seed)
    if strategy == "random" or (strategy == "level_balanced" and dataset_name != "math"):
        picked = list(rows)
        rng.shuffle(picked)
        return picked[:max_samples]

    if strategy == "level_balanced" and dataset_name == "math":
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            level = str((row.get("meta") or {}).get("level") or "unknown")
            buckets.setdefault(level, []).append(row)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        selected: list[dict[str, Any]] = []
        levels = sorted(buckets)
        while len(selected) < max_samples and any(buckets.values()):
            for level in levels:
                if buckets[level] and len(selected) < max_samples:
                    selected.append(buckets[level].pop())
        return selected

    return rows[:max_samples]


def _score_question_to_sentences_threadsafe(
    scorer: MiniLMSimilarityScorer,
    scorer_lock: threading.Lock | None,
    question_text: str,
    sentences: list[str],
    *,
    batch_size: int,
) -> list[float]:
    if not sentences:
        return []
    if scorer_lock is None:
        sims_raw, _ = scorer.score_question_to_sentences(
            question_text,
            sentences,
            batch_size=batch_size,
        )
        return sims_raw
    with scorer_lock:
        sims_raw, _ = scorer.score_question_to_sentences(
            question_text,
            sentences,
            batch_size=batch_size,
        )
        return sims_raw


def _score_text_pair_threadsafe(
    scorer: MiniLMSimilarityScorer,
    scorer_lock: threading.Lock | None,
    question_text: str,
    response_text: str,
    *,
    batch_size: int,
) -> float:
    question_text = normalize_space(question_text)
    response_text = normalize_space(response_text)
    if not question_text or not response_text:
        return 0.0
    if scorer_lock is None:
        q_vec = scorer.encode_question(question_text, batch_size=batch_size)
        r_vec = scorer.encode_texts([response_text], batch_size=batch_size)[0]
        return float(scorer._torch.sum(q_vec * r_vec).item())
    with scorer_lock:
        q_vec = scorer.encode_question(question_text, batch_size=batch_size)
        r_vec = scorer.encode_texts([response_text], batch_size=batch_size)[0]
        return float(scorer._torch.sum(q_vec * r_vec).item())


def _visible_payload_text(
    *,
    extracted_t: str,
    reasoning: str,
    reasoning_content: str,
    reasoning_details_summary_text: str,
    reasoning_details_raw_text: str,
    answer_view: str,
    content: str,
) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for text in (
        extracted_t,
        reasoning_content,
        reasoning,
        reasoning_details_summary_text,
        reasoning_details_raw_text,
        answer_view,
        content,
    ):
        cleaned = normalize_space(str(text or ""))
        if not cleaned or cleaned in seen:
            continue
        parts.append(cleaned)
        seen.add(cleaned)
    return normalize_space(" ".join(parts))


def _evaluate_one_sample(
    *,
    args: argparse.Namespace,
    scorer: MiniLMSimilarityScorer,
    scorer_lock: threading.Lock | None,
    dataset_name: str,
    sample: dict[str, Any],
    mode: str,
    rec_id: str,
) -> dict[str, Any]:
    task_type = str(sample.get("task_type", ""))
    choices = sample.get("choices", [])
    prompt = _apply_qwen3_prompt_directive(
        _build_mode_prompt(sample, mode),
        mode,
        args.provider_profile,
        args.qwen3_control,
    )
    gold = _normalize_gold(task_type, str(sample.get("answer", "")), choices=choices)
    q_text = _build_similarity_question_text(sample)
    chat_template_kwargs = _make_chat_template_kwargs(
        mode,
        args.api,
        args.provider_profile,
        args.qwen3_control,
    )
    payload_controls = _provider_payload_controls(
        mode=mode,
        provider_profile=args.provider_profile,
        qwen3_control=args.qwen3_control,
        openai_mode1_effort=args.openai_mode1_effort,
        gemini_mode1_effort=args.gemini_mode1_effort,
        gemini_off_effort=args.gemini_off_effort,
        api=args.api,
    )
    request_max_tokens = args.mode1_max_tokens if mode == "mode1" else args.max_tokens
    include_temperature = not (args.provider_profile == "deepseek_v4" and mode == "mode1")
    gold_source = str((sample.get("meta") or {}).get("raw_solution") or sample.get("answer", ""))

    base_record = {
        "dataset": dataset_name,
        "id": rec_id,
        "model": args.model,
        "mode": mode,
        "paper_mode": mode,
        "mode_name": MODE_SPECS[mode]["name"],
        "mode_title": MODE_SPECS[mode]["title"],
        "provider_profile": args.provider_profile,
        "qwen3_control": args.qwen3_control,
        "openrouter_provider_only": args.openrouter_provider_only,
        "openrouter_require_parameters": args.openrouter_require_parameters,
        "request_payload_controls": payload_controls,
        "chat_template_kwargs": chat_template_kwargs or {},
        "omit_include_reasoning_param": args.omit_include_reasoning_param,
        "request_include_temperature": include_temperature,
        "task_type": task_type,
        "choices": choices,
        "prompt": prompt,
        "question_text_used_for_similarity": q_text,
    }

    try:
        raw = _call_model(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=request_max_tokens,
            timeout=args.timeout,
            include_reasoning=args.include_reasoning,
            omit_include_reasoning_param=args.omit_include_reasoning_param,
            include_temperature=include_temperature,
            chat_template_kwargs=chat_template_kwargs,
            payload_controls=payload_controls,
            openrouter_provider_only=args.openrouter_provider_only,
            openrouter_require_parameters=args.openrouter_require_parameters,
            api=args.api,
            request_retries=args.request_retries,
            retry_backoff=args.retry_backoff,
        )
        extracted = _extract_choice_extended(raw)
        content = extracted["content"]
        reasoning = extracted["reasoning"]
        reasoning_content = extracted["reasoning_content"]
        reasoning_details = extracted["reasoning_details"]
        finish_reason = extracted["finish_reason"]
        usage = extracted["usage"]
        think_block, post_think_content = _extract_think_block(content)
        answer_view = (
            post_think_content
            if (normalize_space(think_block) and normalize_space(post_think_content))
            else content
        )
        boxed_inner = _last_boxed_inner(answer_view)
        pred = _normalize_pred(task_type, answer_view, boxed_inner, choices=choices)
        score_prediction_text = boxed_inner or pred or answer_view
        is_correct, answer_scorer = _score_answer(
            task_type=task_type,
            prediction_text=score_prediction_text,
            gold_text=gold_source,
            normalized_prediction=pred,
            normalized_gold=gold,
        )
        has_reasoning_fields = bool(normalize_space(reasoning) or normalize_space(reasoning_content))
        visible_t_prefix, visible_answer_raw, answer_span_found = _extract_visible_t_and_a(
            task_type,
            answer_view,
            choices=choices,
            accept_display_math_wrapper=args.accept_display_math_wrapper,
        )
        t_payload = _build_t_payload(
            reasoning=reasoning,
            reasoning_content=reasoning_content,
            reasoning_details=reasoning_details,
            think_block=think_block,
            visible_prefix=visible_t_prefix,
        )
        extracted_t = t_payload["t_returned_readable"]
        t_source = t_payload["t_source"]
        t_sentences = split_explanation_sentences(extracted_t)
        sentence_sims_raw = _score_question_to_sentences_threadsafe(
            scorer,
            scorer_lock,
            q_text,
            t_sentences,
            batch_size=args.embedding_batch_size,
        )
        sentence_sims = [_clip_rate(x) for x in sentence_sims_raw]
        thinking_rate = float(sum(sentence_sims) / len(sentence_sims)) if sentence_sims else 0.0
        nonreason_rate = 1.0 - thinking_rate
        max_thinking_rate = max(sentence_sims) if sentence_sims else 0.0
        max_nonreason_rate = 1.0 - max_thinking_rate
        t_word_count = len(extracted_t.split()) if extracted_t else 0
        t_char_count = len(extracted_t) if extracted_t else 0
        raw_t_sentences = split_explanation_sentences(t_payload["t_visible_raw"])
        raw_sentence_sims_raw = _score_question_to_sentences_threadsafe(
            scorer,
            scorer_lock,
            q_text,
            raw_t_sentences,
            batch_size=args.embedding_batch_size,
        )
        raw_sentence_sims = [_clip_rate(x) for x in raw_sentence_sims_raw]
        t_visible_raw_thinking_rate = (
            float(sum(raw_sentence_sims) / len(raw_sentence_sims))
            if raw_sentence_sims
            else 0.0
        )
        visible_payload = _visible_payload_text(
            extracted_t=extracted_t,
            reasoning=reasoning,
            reasoning_content=reasoning_content,
            reasoning_details_summary_text=t_payload["reasoning_details_summary_text"],
            reasoning_details_raw_text=t_payload["reasoning_details_raw_text"],
            answer_view=answer_view,
            content=content,
        )
        visible_payload_full_similarity_raw = _score_text_pair_threadsafe(
            scorer,
            scorer_lock,
            q_text,
            visible_payload,
            batch_size=args.embedding_batch_size,
        )
        visible_payload_full_similarity = _clip_rate(visible_payload_full_similarity_raw)
        visible_payload_full_nonreason_rate = 1.0 - visible_payload_full_similarity
        is_strict_boxed_only = _is_strict_boxed_only(
            content,
            accept_display_math_wrapper=args.accept_display_math_wrapper,
        )
        answer_only_pred = _extract_answer_only_pred(
            task_type,
            content,
            accept_display_math_wrapper=args.accept_display_math_wrapper,
            choices=choices,
        )
        is_content_answer_only = bool(answer_only_pred)
        has_reasoning_signal, reasoning_signal_types, usage_reasoning_tokens = _merge_reasoning_signals(
            t_payload,
            usage,
        )
        nonreason_pass = is_content_answer_only and not has_reasoning_signal
        nonreason_correct = nonreason_pass and is_correct
        observable_nonreason_pass = is_strict_boxed_only and not has_reasoning_signal

        return {
            **base_record,
            "content": content,
            "answer_view_content": answer_view,
            "reasoning": reasoning,
            "reasoning_content": reasoning_content,
            "reasoning_details": reasoning_details,
            "reasoning_details_raw_text": t_payload["reasoning_details_raw_text"],
            "reasoning_details_summary_text": t_payload["reasoning_details_summary_text"],
            "has_reasoning_fields": has_reasoning_fields,
            "has_reasoning_signal": has_reasoning_signal,
            "reasoning_signal_types": reasoning_signal_types,
            "usage_reasoning_tokens": usage_reasoning_tokens,
            "content_think_block": think_block,
            "think_block": think_block,
            "t_source": t_source,
            "visible_prefix_before_answer": visible_t_prefix,
            "visible_t_prefix": visible_t_prefix,
            "t_visible_raw": t_payload["t_visible_raw"],
            "t_returned_readable": t_payload["t_returned_readable"],
            "extracted_t": extracted_t,
            "t_sentences": t_sentences,
            "t_sentence_count": len(t_sentences),
            "t_word_count": t_word_count,
            "t_char_count": t_char_count,
            "t_sentence_similarities_raw": sentence_sims_raw,
            "t_sentence_similarities": sentence_sims,
            "t_visible_raw_sentences": raw_t_sentences,
            "t_visible_raw_sentence_similarities_raw": raw_sentence_sims_raw,
            "t_visible_raw_sentence_similarities": raw_sentence_sims,
            "t_visible_raw_thinking_rate": t_visible_raw_thinking_rate,
            "visible_payload_text": visible_payload,
            "visible_payload_full_similarity_raw": visible_payload_full_similarity_raw,
            "visible_payload_full_similarity": visible_payload_full_similarity,
            "visible_payload_full_nonreason_rate": visible_payload_full_nonreason_rate,
            "thinking_rate": thinking_rate,
            "nonreason_rate": nonreason_rate,
            "max_thinking_rate": max_thinking_rate,
            "max_nonreason_rate": max_nonreason_rate,
            "answer_raw": visible_answer_raw,
            "answer_span_found": answer_span_found,
            "answer_parse_failed": not answer_span_found,
            "request_max_tokens": request_max_tokens,
            "prediction": pred,
            "gold_answer": gold,
            "gold_answer_source": gold_source,
            "answer_scorer": answer_scorer,
            "is_correct": is_correct,
            "boxed_inner": boxed_inner,
            "is_strict_boxed_only": is_strict_boxed_only,
            "answer_only_prediction": answer_only_pred,
            "content_answer_only_prediction": answer_only_pred,
            "is_content_answer_only": is_content_answer_only,
            "is_answer_only": is_content_answer_only,
            "api_answer_only_pass": nonreason_pass,
            "nonreason_pass": nonreason_pass,
            "nonreason_correct": nonreason_correct,
            "observable_nonreason_pass": observable_nonreason_pass,
            "finish_reason": finish_reason,
            "usage": usage,
            "raw_response": raw,
            "error": "",
        }
    except Exception as exc:
        return {
            **base_record,
            "content": "",
            "answer_view_content": "",
            "reasoning": "",
            "reasoning_content": "",
            "reasoning_details": None,
            "reasoning_details_raw_text": "",
            "reasoning_details_summary_text": "",
            "has_reasoning_fields": False,
            "has_reasoning_signal": False,
            "reasoning_signal_types": [],
            "usage_reasoning_tokens": 0,
            "content_think_block": "",
            "think_block": "",
            "t_source": "error",
            "visible_prefix_before_answer": "",
            "visible_t_prefix": "",
            "t_visible_raw": "",
            "t_returned_readable": "",
            "extracted_t": "",
            "t_sentences": [],
            "t_sentence_count": 0,
            "t_word_count": 0,
            "t_char_count": 0,
            "t_sentence_similarities_raw": [],
            "t_sentence_similarities": [],
            "t_visible_raw_sentences": [],
            "t_visible_raw_sentence_similarities_raw": [],
            "t_visible_raw_sentence_similarities": [],
            "t_visible_raw_thinking_rate": 0.0,
            "visible_payload_text": "",
            "visible_payload_full_similarity_raw": 0.0,
            "visible_payload_full_similarity": 0.0,
            "visible_payload_full_nonreason_rate": 1.0,
            "thinking_rate": 0.0,
            "nonreason_rate": 1.0,
            "max_thinking_rate": 0.0,
            "max_nonreason_rate": 1.0,
            "answer_raw": "",
            "answer_span_found": False,
            "answer_parse_failed": True,
            "request_max_tokens": request_max_tokens,
            "prediction": "",
            "gold_answer": gold,
            "gold_answer_source": gold_source,
            "answer_scorer": "error",
            "is_correct": False,
            "boxed_inner": "",
            "is_strict_boxed_only": False,
            "answer_only_prediction": "",
            "content_answer_only_prediction": "",
            "is_content_answer_only": False,
            "is_answer_only": False,
            "api_answer_only_pass": False,
            "nonreason_pass": False,
            "nonreason_correct": False,
            "observable_nonreason_pass": False,
            "finish_reason": "",
            "usage": {},
            "raw_response": {},
            "error": str(exc),
        }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    try:
        modes = [_canonical_mode(x) for x in args.modes]
    except KeyError as exc:
        print(f"Unknown mode: {exc}", file=sys.stderr)
        return 2
    modes = [m for m in MODE_ORDER if m in set(modes)]

    root_dir = Path(args.root_dir).resolve()
    processed_dir = root_dir / "processed"
    runs_dir = root_dir / "runs"
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    default_run_name = f"thinking_spectrum_{safe_model}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_name = args.run_name or default_run_name
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)

    scorer = MiniLMSimilarityScorer(
        args.embedding_model_path,
        device=args.embedding_device,
    )
    scorer_lock = threading.Lock() if args.request_concurrency > 1 else None

    all_records: list[dict[str, Any]] = []
    per_mode_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_dataset_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_mode_dataset_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, str, str]] = set()
    if args.resume:
        for record in _load_existing_records(records_path):
            mode = str(record.get("mode", ""))
            dataset_name = str(record.get("dataset", ""))
            rec_id = str(record.get("id", ""))
            if not mode or not dataset_name or not rec_id:
                continue
            key = _record_key(mode, dataset_name, rec_id)
            if key in completed_keys:
                continue
            completed_keys.add(key)
            all_records.append(record)
            per_mode_records[mode].append(record)
            per_dataset_records[dataset_name].append(record)
            per_mode_dataset_records[f"{mode}::{dataset_name}"].append(record)
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
            rows = _select_eval_rows(
                dataset_name,
                rows,
                max_samples=args.max_samples,
                seed=args.seed,
                shuffle=args.shuffle,
                sample_strategy=args.sample_strategy,
            )

            for mode in modes:
                print(f"[mode] {mode} [dataset] {dataset_name}: {len(rows)} samples")
                skipped_count = 0
                if args.request_concurrency > 1:
                    pending: list[tuple[int, dict[str, Any], str]] = []
                    for i, sample in enumerate(rows, start=1):
                        rec_id = str(sample.get("id", f"{dataset_name}-{i}"))
                        key = _record_key(mode, dataset_name, rec_id)
                        if key in completed_keys:
                            skipped_count += 1
                            continue
                        pending.append((i, sample, rec_id))

                    completed_count = 0
                    with concurrent.futures.ThreadPoolExecutor(max_workers=args.request_concurrency) as executor:
                        future_to_meta = {
                            executor.submit(
                                _evaluate_one_sample,
                                args=args,
                                scorer=scorer,
                                scorer_lock=scorer_lock,
                                dataset_name=dataset_name,
                                sample=sample,
                                mode=mode,
                                rec_id=rec_id,
                            ): (i, rec_id)
                            for i, sample, rec_id in pending
                        }
                        for future in concurrent.futures.as_completed(future_to_meta):
                            i, rec_id = future_to_meta[future]
                            try:
                                record = future.result()
                            except Exception as exc:
                                record = {
                                    "dataset": dataset_name,
                                    "id": rec_id,
                                    "model": args.model,
                                    "mode": mode,
                                    "paper_mode": mode,
                                    "mode_name": MODE_SPECS[mode]["name"],
                                    "mode_title": MODE_SPECS[mode]["title"],
                                    "provider_profile": args.provider_profile,
                                    "error": f"worker_error: {exc}",
                                }

                            completed_keys.add(_record_key(mode, dataset_name, rec_id))
                            all_records.append(record)
                            per_mode_records[mode].append(record)
                            per_dataset_records[dataset_name].append(record)
                            per_mode_dataset_records[f"{mode}::{dataset_name}"].append(record)
                            records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            records_file.flush()

                            completed_count += 1
                            if completed_count % 20 == 0 or completed_count == len(pending):
                                print(
                                    f"  progress: {mode} {dataset_name} "
                                    f"{completed_count}/{len(pending)} submitted "
                                    f"({skipped_count} skipped)"
                                )

                    if skipped_count:
                        print(f"  [resume] skipped {skipped_count} completed records")
                    continue
                for i, sample in enumerate(rows, start=1):
                    rec_id = str(sample.get("id", f"{dataset_name}-{i}"))
                    key = _record_key(mode, dataset_name, rec_id)
                    if key in completed_keys:
                        skipped_count += 1
                        continue

                    task_type = str(sample.get("task_type", ""))
                    choices = sample.get("choices", [])
                    prompt = _apply_qwen3_prompt_directive(
                        _build_mode_prompt(sample, mode),
                        mode,
                        args.provider_profile,
                        args.qwen3_control,
                    )
                    gold = _normalize_gold(task_type, str(sample.get("answer", "")), choices=choices)
                    q_text = _build_similarity_question_text(sample)
                    chat_template_kwargs = _make_chat_template_kwargs(
                        mode,
                        args.api,
                        args.provider_profile,
                        args.qwen3_control,
                    )
                    payload_controls = _provider_payload_controls(
                        mode=mode,
                        provider_profile=args.provider_profile,
                        qwen3_control=args.qwen3_control,
                        openai_mode1_effort=args.openai_mode1_effort,
                        gemini_mode1_effort=args.gemini_mode1_effort,
                        gemini_off_effort=args.gemini_off_effort,
                        api=args.api,
                    )
                    request_max_tokens = args.mode1_max_tokens if mode == "mode1" else args.max_tokens
                    include_temperature = not (args.provider_profile == "deepseek_v4" and mode == "mode1")

                    try:
                        raw = _call_model(
                            base_url=args.base_url,
                            model=args.model,
                            api_key=args.api_key,
                            prompt=prompt,
                            temperature=args.temperature,
                            max_tokens=request_max_tokens,
                            timeout=args.timeout,
                            include_reasoning=args.include_reasoning,
                            omit_include_reasoning_param=args.omit_include_reasoning_param,
                            include_temperature=include_temperature,
                            chat_template_kwargs=chat_template_kwargs,
                            payload_controls=payload_controls,
                            openrouter_provider_only=args.openrouter_provider_only,
                            openrouter_require_parameters=args.openrouter_require_parameters,
                            api=args.api,
                            request_retries=args.request_retries,
                            retry_backoff=args.retry_backoff,
                        )
                        extracted = _extract_choice_extended(raw)
                        content = extracted["content"]
                        reasoning = extracted["reasoning"]
                        reasoning_content = extracted["reasoning_content"]
                        reasoning_details = extracted["reasoning_details"]
                        finish_reason = extracted["finish_reason"]
                        usage = extracted["usage"]
                        think_block, post_think_content = _extract_think_block(content)
                        answer_view = (
                            post_think_content
                            if (normalize_space(think_block) and normalize_space(post_think_content))
                            else content
                        )
                        boxed_inner = _last_boxed_inner(answer_view)
                        pred = _normalize_pred(task_type, answer_view, boxed_inner, choices=choices)
                        gold_source = str((sample.get("meta") or {}).get("raw_solution") or sample.get("answer", ""))
                        score_prediction_text = boxed_inner or pred or answer_view
                        is_correct, answer_scorer = _score_answer(
                            task_type=task_type,
                            prediction_text=score_prediction_text,
                            gold_text=gold_source,
                            normalized_prediction=pred,
                            normalized_gold=gold,
                        )
                        has_reasoning_fields = bool(normalize_space(reasoning) or normalize_space(reasoning_content))
                        visible_t_prefix, visible_answer_raw, answer_span_found = _extract_visible_t_and_a(
                            task_type,
                            answer_view,
                            choices=choices,
                            accept_display_math_wrapper=args.accept_display_math_wrapper,
                        )
                        t_payload = _build_t_payload(
                            reasoning=reasoning,
                            reasoning_content=reasoning_content,
                            reasoning_details=reasoning_details,
                            think_block=think_block,
                            visible_prefix=visible_t_prefix,
                        )
                        extracted_t = t_payload["t_returned_readable"]
                        t_source = t_payload["t_source"]
                        t_sentences = split_explanation_sentences(extracted_t)
                        sentence_sims_raw, _ = scorer.score_question_to_sentences(
                            q_text,
                            t_sentences,
                            batch_size=args.embedding_batch_size,
                        )
                        sentence_sims = [_clip_rate(x) for x in sentence_sims_raw]
                        thinking_rate = float(sum(sentence_sims) / len(sentence_sims)) if sentence_sims else 0.0
                        nonreason_rate = 1.0 - thinking_rate
                        max_thinking_rate = max(sentence_sims) if sentence_sims else 0.0
                        max_nonreason_rate = 1.0 - max_thinking_rate
                        t_word_count = len(extracted_t.split()) if extracted_t else 0
                        t_char_count = len(extracted_t) if extracted_t else 0
                        raw_t_sentences = split_explanation_sentences(t_payload["t_visible_raw"])
                        raw_sentence_sims_raw, _ = scorer.score_question_to_sentences(
                            q_text,
                            raw_t_sentences,
                            batch_size=args.embedding_batch_size,
                        )
                        raw_sentence_sims = [_clip_rate(x) for x in raw_sentence_sims_raw]
                        t_visible_raw_thinking_rate = (
                            float(sum(raw_sentence_sims) / len(raw_sentence_sims))
                            if raw_sentence_sims
                            else 0.0
                        )
                        visible_payload = _visible_payload_text(
                            extracted_t=extracted_t,
                            reasoning=reasoning,
                            reasoning_content=reasoning_content,
                            reasoning_details_summary_text=t_payload["reasoning_details_summary_text"],
                            reasoning_details_raw_text=t_payload["reasoning_details_raw_text"],
                            answer_view=answer_view,
                            content=content,
                        )
                        visible_payload_full_similarity_raw = _score_text_pair_threadsafe(
                            scorer,
                            None,
                            q_text,
                            visible_payload,
                            batch_size=args.embedding_batch_size,
                        )
                        visible_payload_full_similarity = _clip_rate(visible_payload_full_similarity_raw)
                        visible_payload_full_nonreason_rate = 1.0 - visible_payload_full_similarity
                        is_strict_boxed_only = _is_strict_boxed_only(
                            content,
                            accept_display_math_wrapper=args.accept_display_math_wrapper,
                        )
                        answer_only_pred = _extract_answer_only_pred(
                            task_type,
                            content,
                            accept_display_math_wrapper=args.accept_display_math_wrapper,
                            choices=choices,
                        )
                        is_content_answer_only = bool(answer_only_pred)
                        has_reasoning_signal, reasoning_signal_types, usage_reasoning_tokens = _merge_reasoning_signals(
                            t_payload,
                            usage,
                        )
                        nonreason_pass = is_content_answer_only and not has_reasoning_signal
                        nonreason_correct = nonreason_pass and is_correct
                        observable_nonreason_pass = is_strict_boxed_only and not has_reasoning_signal

                        record = {
                            "dataset": dataset_name,
                            "id": rec_id,
                            "model": args.model,
                            "mode": mode,
                            "paper_mode": mode,
                            "mode_name": MODE_SPECS[mode]["name"],
                            "mode_title": MODE_SPECS[mode]["title"],
                            "provider_profile": args.provider_profile,
                            "qwen3_control": args.qwen3_control,
                            "openrouter_provider_only": args.openrouter_provider_only,
                            "openrouter_require_parameters": args.openrouter_require_parameters,
                            "request_payload_controls": payload_controls,
                            "chat_template_kwargs": chat_template_kwargs or {},
                            "omit_include_reasoning_param": args.omit_include_reasoning_param,
                            "request_include_temperature": include_temperature,
                            "task_type": task_type,
                            "choices": choices,
                            "prompt": prompt,
                            "question_text_used_for_similarity": q_text,
                            "content": content,
                            "answer_view_content": answer_view,
                            "reasoning": reasoning,
                            "reasoning_content": reasoning_content,
                            "reasoning_details": reasoning_details,
                            "reasoning_details_raw_text": t_payload["reasoning_details_raw_text"],
                            "reasoning_details_summary_text": t_payload["reasoning_details_summary_text"],
                            "has_reasoning_fields": has_reasoning_fields,
                            "has_reasoning_signal": has_reasoning_signal,
                            "reasoning_signal_types": reasoning_signal_types,
                            "usage_reasoning_tokens": usage_reasoning_tokens,
                            "content_think_block": think_block,
                            "think_block": think_block,
                            "t_source": t_source,
                            "visible_prefix_before_answer": visible_t_prefix,
                            "visible_t_prefix": visible_t_prefix,
                            "t_visible_raw": t_payload["t_visible_raw"],
                            "t_returned_readable": t_payload["t_returned_readable"],
                            "extracted_t": extracted_t,
                            "t_sentences": t_sentences,
                            "t_sentence_count": len(t_sentences),
                            "t_word_count": t_word_count,
                            "t_char_count": t_char_count,
                            "t_sentence_similarities_raw": sentence_sims_raw,
                            "t_sentence_similarities": sentence_sims,
                            "t_visible_raw_sentences": raw_t_sentences,
                            "t_visible_raw_sentence_similarities_raw": raw_sentence_sims_raw,
                            "t_visible_raw_sentence_similarities": raw_sentence_sims,
                            "t_visible_raw_thinking_rate": t_visible_raw_thinking_rate,
                            "visible_payload_text": visible_payload,
                            "visible_payload_full_similarity_raw": visible_payload_full_similarity_raw,
                            "visible_payload_full_similarity": visible_payload_full_similarity,
                            "visible_payload_full_nonreason_rate": visible_payload_full_nonreason_rate,
                            "thinking_rate": thinking_rate,
                            "nonreason_rate": nonreason_rate,
                            "max_thinking_rate": max_thinking_rate,
                            "max_nonreason_rate": max_nonreason_rate,
                            "answer_raw": visible_answer_raw,
                            "answer_span_found": answer_span_found,
                            "answer_parse_failed": not answer_span_found,
                            "request_max_tokens": request_max_tokens,
                            "prediction": pred,
                            "gold_answer": gold,
                            "gold_answer_source": gold_source,
                            "answer_scorer": answer_scorer,
                            "is_correct": is_correct,
                            "boxed_inner": boxed_inner,
                            "is_strict_boxed_only": is_strict_boxed_only,
                            "answer_only_prediction": answer_only_pred,
                            "content_answer_only_prediction": answer_only_pred,
                            "is_content_answer_only": is_content_answer_only,
                            "is_answer_only": is_content_answer_only,
                            "api_answer_only_pass": nonreason_pass,
                            "nonreason_pass": nonreason_pass,
                            "nonreason_correct": nonreason_correct,
                            "observable_nonreason_pass": observable_nonreason_pass,
                            "finish_reason": finish_reason,
                            "usage": usage,
                            "raw_response": raw,
                            "error": "",
                        }
                    except Exception as exc:
                        record = {
                            "dataset": dataset_name,
                            "id": rec_id,
                            "model": args.model,
                            "mode": mode,
                            "paper_mode": mode,
                            "mode_name": MODE_SPECS[mode]["name"],
                            "mode_title": MODE_SPECS[mode]["title"],
                            "provider_profile": args.provider_profile,
                            "qwen3_control": args.qwen3_control,
                            "openrouter_provider_only": args.openrouter_provider_only,
                            "openrouter_require_parameters": args.openrouter_require_parameters,
                            "request_payload_controls": payload_controls,
                            "chat_template_kwargs": chat_template_kwargs or {},
                            "omit_include_reasoning_param": args.omit_include_reasoning_param,
                            "request_include_temperature": include_temperature,
                            "task_type": task_type,
                            "choices": choices,
                            "prompt": prompt,
                            "question_text_used_for_similarity": q_text,
                            "content": "",
                            "answer_view_content": "",
                            "reasoning": "",
                            "reasoning_content": "",
                            "reasoning_details": None,
                            "reasoning_details_raw_text": "",
                            "reasoning_details_summary_text": "",
                            "has_reasoning_fields": False,
                            "has_reasoning_signal": False,
                            "reasoning_signal_types": [],
                            "usage_reasoning_tokens": 0,
                            "content_think_block": "",
                            "think_block": "",
                            "t_source": "error",
                            "visible_prefix_before_answer": "",
                            "visible_t_prefix": "",
                            "t_visible_raw": "",
                            "t_returned_readable": "",
                            "extracted_t": "",
                            "t_sentences": [],
                            "t_sentence_count": 0,
                            "t_word_count": 0,
                            "t_char_count": 0,
                            "t_sentence_similarities_raw": [],
                            "t_sentence_similarities": [],
                            "t_visible_raw_sentences": [],
                            "t_visible_raw_sentence_similarities_raw": [],
                            "t_visible_raw_sentence_similarities": [],
                            "t_visible_raw_thinking_rate": 0.0,
                            "visible_payload_text": "",
                            "visible_payload_full_similarity_raw": 0.0,
                            "visible_payload_full_similarity": 0.0,
                            "visible_payload_full_nonreason_rate": 1.0,
                            "thinking_rate": 0.0,
                            "nonreason_rate": 1.0,
                            "max_thinking_rate": 0.0,
                            "max_nonreason_rate": 1.0,
                            "answer_raw": "",
                            "answer_span_found": False,
                            "answer_parse_failed": True,
                            "request_max_tokens": request_max_tokens,
                            "prediction": "",
                            "gold_answer": gold,
                            "gold_answer_source": str((sample.get("meta") or {}).get("raw_solution") or sample.get("answer", "")),
                            "answer_scorer": "error",
                            "is_correct": False,
                            "boxed_inner": "",
                            "is_strict_boxed_only": False,
                            "answer_only_prediction": "",
                            "content_answer_only_prediction": "",
                            "is_content_answer_only": False,
                            "is_answer_only": False,
                            "api_answer_only_pass": False,
                            "nonreason_pass": False,
                            "nonreason_correct": False,
                            "observable_nonreason_pass": False,
                            "finish_reason": "",
                            "usage": {},
                            "raw_response": {},
                            "error": str(exc),
                        }

                    completed_keys.add(key)
                    all_records.append(record)
                    per_mode_records[mode].append(record)
                    per_dataset_records[dataset_name].append(record)
                    per_mode_dataset_records[f"{mode}::{dataset_name}"].append(record)
                    records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_file.flush()

                    if i % 20 == 0:
                        print(f"  progress: {mode} {dataset_name} {i}/{len(rows)}")
                if skipped_count:
                    print(f"  [resume] skipped {skipped_count} completed records")

    comparison_rows = []
    for mode in modes:
        metrics = _summarize_records(
            per_mode_records.get(mode, []),
            accept_display_math_wrapper=args.accept_display_math_wrapper,
        )
        comparison_rows.append(
            {
                "mode": mode,
                "paper_mode": mode,
                "mode_name": MODE_SPECS[mode]["name"],
                "mode_title": MODE_SPECS[mode]["title"],
                **metrics,
            }
        )

    summary = {
        "run_name": run_name,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "datasets": args.datasets,
            "base_url": args.base_url,
            "model": args.model,
            "api": args.api,
            "provider_profile": args.provider_profile,
            "qwen3_control": args.qwen3_control,
            "openrouter_provider_only": args.openrouter_provider_only,
            "openrouter_require_parameters": args.openrouter_require_parameters,
            "openai_mode1_effort": args.openai_mode1_effort,
            "gemini_mode1_effort": args.gemini_mode1_effort,
            "gemini_off_effort": args.gemini_off_effort,
            "modes": modes,
            "mode_numbering": "paper",
            "mode_specs": MODE_SPECS,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "mode1_max_tokens": args.mode1_max_tokens,
            "timeout": args.timeout,
            "request_concurrency": args.request_concurrency,
            "request_retries": args.request_retries,
            "retry_backoff": args.retry_backoff,
            "max_samples": args.max_samples,
            "shuffle": args.shuffle,
            "seed": args.seed,
            "sample_strategy": args.sample_strategy,
            "include_reasoning": args.include_reasoning,
            "omit_include_reasoning_param": args.omit_include_reasoning_param,
            "accept_display_math_wrapper": args.accept_display_math_wrapper,
            "embedding_model_path": args.embedding_model_path,
            "embedding_device": args.embedding_device,
            "embedding_batch_size": args.embedding_batch_size,
            "math_verify_available": math_verify_available(),
        },
        "overall_all_modes": _summarize_records(
            all_records,
            accept_display_math_wrapper=args.accept_display_math_wrapper,
        ),
        "per_mode": _group_summary(
            per_mode_records,
            accept_display_math_wrapper=args.accept_display_math_wrapper,
        ),
        "per_dataset": _group_summary(
            per_dataset_records,
            accept_display_math_wrapper=args.accept_display_math_wrapper,
        ),
        "per_mode_per_dataset": {
            mode: _group_summary(
                {
                    dataset_name: per_mode_dataset_records.get(f"{mode}::{dataset_name}", [])
                    for dataset_name in args.datasets
                },
                accept_display_math_wrapper=args.accept_display_math_wrapper,
            )
            for mode in modes
        },
        "mode_comparison": comparison_rows,
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
