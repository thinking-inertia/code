"""Assistant-continuation requests and first-box/residual-text scoring for M6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PREFIX = r"The answer is \boxed{"
PROTOCOL = "assistant_prefill_first_box_residual_v1"


def validate_args(args: argparse.Namespace) -> None:
    backend = getattr(args, "m6_prefill", "")
    if backend == "qwen-completions":
        if args.api != "completions" or args.provider_profile not in {"none", "qwen3"}:
            raise ValueError("M6 qwen-completions requires --api completions and a Qwen3 endpoint")
        if not getattr(args, "m6_tokenizer", "") or args.qwen3_control != "chat_template":
            raise ValueError("M6 qwen-completions requires --m6-tokenizer matching the served checkpoint and --qwen3-control chat_template")
    elif backend == "deepseek-prefix":
        if args.api != "chat" or args.provider_profile != "deepseek_v4":
            raise ValueError("M6 deepseek-prefix requires --api chat --provider-profile deepseek_v4 and an endpoint supporting prefix=true")
    else:
        raise ValueError("M6 requires --m6-prefill qwen-completions or deepseek-prefix; ordinary chat/Responses cannot implement assistant prefilling. Use --modes 1 2 3 4 5 for other endpoints.")
    if args.openrouter_provider_only or args.openrouter_require_parameters:
        raise ValueError("M6 requires a direct prefill-capable endpoint; provider routing is not supported")


def load_tokenizer(args: argparse.Namespace) -> None:
    validate_args(args)
    if args.m6_prefill == "qwen-completions":
        from transformers import AutoTokenizer
        args._m6_tokenizer = AutoTokenizer.from_pretrained(
            args.m6_tokenizer, revision=getattr(args, "m6_tokenizer_revision", None),
            trust_remote_code=False,
        )


def prepare_request(args: argparse.Namespace, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_args(args)
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": PREFIX}]
    payload = {"model": args.model, "max_tokens": args.max_tokens, "temperature": args.temperature}
    metadata: dict[str, Any] = {
        "m6_protocol": PROTOCOL, "m6_prefill_backend": args.m6_prefill,
        "assistant_prefill": PREFIX, "prefix": PREFIX,
    }
    if args.m6_prefill == "qwen-completions":
        tokenizer = args._m6_tokenizer
        template_kwargs = {"continue_final_message": True, "add_generation_prompt": False, "enable_thinking": False}
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, **template_kwargs)
        if not rendered.endswith(PREFIX) or ids != tokenizer.encode(rendered, add_special_tokens=False):
            raise ValueError("Tokenizer does not preserve the unfinished assistant prefix")
        marker = "<|im_start|>assistant\n"
        if marker not in rendered or rendered.rsplit(marker, 1)[1] != "<think>\n\n</think>\n\n" + PREFIX:
            raise ValueError("Expected the original Qwen3 think-off assistant template")
        # Token IDs avoid a second template application or extra special tokens.
        payload.update(prompt=ids, echo=False, top_p=1.0, top_k=0, min_p=0.0,
                       repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                       ignore_eos=False, min_tokens=0, skip_special_tokens=False,
                       include_stop_str_in_output=False)
        metadata.update(rendered_prompt=rendered, prompt_token_ids=ids,
                        tokenizer=args.m6_tokenizer,
                        tokenizer_revision=getattr(args, "m6_tokenizer_revision", None),
                        chat_template_kwargs=template_kwargs,
                        chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest())
    else:
        messages[-1]["prefix"] = True
        payload.update(messages=messages, thinking={"type": "disabled"})
        if not args.omit_include_reasoning_param:
            payload["include_reasoning"] = args.include_reasoning
    metadata["raw_request"] = payload
    return payload, metadata


def normalize_format_only(text: str) -> str:
    text = " ".join(str(text or "").strip().split())
    return "" if text in ("", ".", "。") else text


def _escaped(text: str, pos: int) -> bool:
    count = 0
    pos -= 1
    while pos >= 0 and text[pos] == "\\":
        count += 1
        pos -= 1
    return bool(count % 2)


def parse_first_box(content: str) -> dict[str, Any]:
    """Never skip an empty/unclosed initial box in favor of a later answer."""
    if not content.startswith(PREFIX):
        return {"status": "missing_fixed_opening", "inner": None, "isolated_box": "", "remainder_raw": content}
    depth = 1
    for pos in range(len(PREFIX), len(content)):
        char = content[pos]
        if char in "{}" and not _escaped(content, pos):
            depth += 1 if char == "{" else -1
            if depth == 0:
                inner = content[len(PREFIX):pos]
                return {"status": "complete_nonempty" if inner.strip() else "complete_empty",
                        "inner": inner, "isolated_box": content[len("The answer is "):pos + 1],
                        "remainder_raw": content[pos + 1:]}
    return {"status": "unclosed", "inner": content[len(PREFIX):], "isolated_box": "", "remainder_raw": content}


def score_fields(sample: dict[str, Any], content: str, extracted: dict[str, Any], *, accuracy_policy: str = "evaluator") -> dict[str, Any]:
    # Reuse the existing benchmark comparators and readable-field merge rules.
    from . import eval_thinking_spectrum as spectrum
    from . import benchmark_scorers, m6_accuracy
    if accuracy_policy not in {"evaluator", "canonical_local", "legacy_math_rescore"}:
        raise ValueError("Unknown M6 accuracy policy")
    first = parse_first_box(content)
    valid_box = first["status"] == "complete_nonempty"
    task = str(sample.get("task_type", ""))
    choices = sample.get("choices", [])
    gold = spectrum._normalize_gold(task, str(sample.get("answer", "")), choices=choices)
    gold_source = str((sample.get("meta") or {}).get("raw_solution") or sample.get("answer", ""))
    if valid_box:
        inner = spectrum._last_boxed_inner(first["isolated_box"])
        pred = spectrum._normalize_pred(task, first["isolated_box"], inner, choices=choices)
        score_text = inner or pred or first["isolated_box"]
        if accuracy_policy == "canonical_local":
            if task != "math" or sample.get("dataset") == "gsm8k":
                correct, scorer = bool(pred) and pred == gold, "normalized_exact_match"
            else:
                mp = m6_accuracy.extract_math_answer_fallback(score_text)
                mg = m6_accuracy.extract_math_answer_fallback(gold_source)
                correct, scorer = bool(mp and mg and mp == mg), "fallback_normalized_exact_match"
        elif accuracy_policy == "legacy_math_rescore" and task == "math":
            correct, _, _, scorer = benchmark_scorers.verify_math_answer(score_text, gold_source)
        else:
            correct, scorer = spectrum._score_answer(task_type=task, prediction_text=score_text,
                gold_text=gold_source, normalized_prediction=pred, normalized_gold=gold)
    else:
        pred, correct, scorer = "", False, "first_box_missing_empty_or_unclosed"
    content_t = normalize_format_only(first["remainder_raw"])
    t_payload = spectrum._build_t_payload(reasoning=extracted["reasoning"],
        reasoning_content=extracted["reasoning_content"], reasoning_details=extracted["reasoning_details"],
        think_block="", visible_prefix=content_t)
    t_payload["t_returned_readable"] = normalize_format_only(t_payload["t_returned_readable"])
    t_payload["t_source"] = "post_first_declaration_all_returned"
    valid = valid_box and bool(pred)
    return {"first_box": first, "boxed_inner": first["inner"] or "", "prediction": pred,
            "is_correct": correct, "answer_scorer": scorer, "answer_view": first["isolated_box"],
            "answer_span_found": valid_box, "content_t": content_t, "t_payload": t_payload,
            "content_answer_only": bool(valid and not content_t),
            "etr": bool(valid and not t_payload["t_returned_readable"])}


def require_protocol(record: dict[str, Any]) -> None:
    if str(record.get("mode") or record.get("paper_mode")) in {"6", "mode6"} and record.get("m6_protocol") != PROTOCOL:
        raise ValueError("M6 record lacks assistant-prefill provenance. User-message-prefix M6u records cannot be resumed or relabeled as M6; retain them separately.")


def validate_resume(args: argparse.Namespace, record: dict[str, Any]) -> None:
    require_protocol(record)
    if record.get("m6_protocol") != PROTOCOL:
        return
    expected = {"m6_prefill_backend": args.m6_prefill, "accuracy_policy": args.m6_accuracy_policy,
                "model": args.model, "request_max_tokens": args.max_tokens, "prefix": PREFIX}
    if args.m6_prefill == "qwen-completions":
        expected.update(tokenizer=args.m6_tokenizer,
                        tokenizer_revision=getattr(args, "m6_tokenizer_revision", None),
                        chat_template_sha256=hashlib.sha256(args._m6_tokenizer.chat_template.encode()).hexdigest())
    mismatches = [key for key, value in expected.items() if record.get(key) != value]
    if record.get("raw_request", {}).get("temperature") != args.temperature:
        mismatches.append("temperature")
    if mismatches:
        raise ValueError("M6 resume settings differ: " + ", ".join(mismatches) + "; use a new run directory")


def metric_pair(record: dict[str, Any]) -> dict[str, Any]:
    require_protocol(record)
    if record.get("m6_protocol") != PROTOCOL:
        raise ValueError("This exporter accepts assistant-prefill M6 records only")
    if record.get("error"):
        raise ValueError("Cannot score a failed generation as an empty response")
    if "extracted_t" not in record:
        raise ValueError("M6 record has no residual T")
    text = " ".join(record["extracted_t"].split())
    key = "/".join(str(record.get(k, "")) for k in ("model", "mode", "dataset", "id"))
    return {"record_id": key, "model": record["model"], "dataset": record["dataset"],
            "mode": "mode6", "sample_id": str(record["id"]),
            "question": record["question_text_used_for_similarity"], "pre_answer_text": text,
            "t_is_empty": int(not text), "etr": int(record["nonreason_pass"]),
            "is_correct": int(record["is_correct"]), "m6_protocol": PROTOCOL}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export M6 residual Q/T pairs for the unchanged QRel and EIR scorers")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = [metric_pair(json.loads(line)) for line in args.input.read_text().splitlines() if line.strip()]
    if len({p["record_id"] for p in pairs}) != len(pairs):
        raise ValueError("Duplicate M6 record IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
