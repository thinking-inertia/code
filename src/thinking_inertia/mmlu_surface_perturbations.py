#!/usr/bin/env python3
"""
Build and analyze MMLU surface perturbation controls.

Paper mapping:
- Appendix: "Surface perturbations"
- Purpose: test whether option order or label names explain the MMLU domain gradient
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
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


LETTER_SET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and analyze MMLU choice-shuffle and numeric-label controls."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build perturbed MMLU datasets.")
    build.add_argument("--source", required=True, help="Path to processed/mmlu.jsonl.")
    build.add_argument("--out-root", required=True, help="Experiment root output directory.")
    build.add_argument("--max-samples", type=int, default=3000, help="Max source samples.")
    build.add_argument("--shuffle", action="store_true", help="Shuffle before sampling.")
    build.add_argument("--seed", type=int, default=0, help="Sampling/permutation seed.")
    build.add_argument(
        "--variants",
        nargs="+",
        default=["choice_shuffle", "label_numeric"],
        choices=["choice_shuffle", "label_numeric"],
        help="Deterministic variants to build.",
    )
    build.add_argument(
        "--api-variants",
        nargs="*",
        default=[],
        choices=["question_paraphrase", "entity_rename"],
        help="Optional API-generated variants.",
    )
    build.add_argument("--generator-base-url", default="", help="OpenAI-compatible API base URL.")
    build.add_argument("--generator-model", default="", help="Generator model name.")
    build.add_argument(
        "--generator-api-key-env",
        default="",
        help="Environment variable containing the generator API key.",
    )
    build.add_argument(
        "--generator-max-workers",
        type=int,
        default=8,
        help="Maximum concurrent API generation workers.",
    )

    analyze = subparsers.add_parser("analyze", help="Analyze surface perturbation runs.")
    analyze.add_argument("--manifest", required=True, help="Path to surface_perturbation_manifest.json.")
    analyze.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form label=/abs/path/to/run_dir_or_records.jsonl. Repeatable.",
    )
    analyze.add_argument("--base-dataset", required=True, help="Base dataset name, e.g. mmlu_base.")
    analyze.add_argument(
        "--compare-datasets",
        nargs="+",
        required=True,
        help="Variant dataset names to compare against the base.",
    )
    analyze.add_argument("--output-dir", required=True, help="Output directory.")
    return parser.parse_args()


def _dataset_name_for_variant(variant: str) -> str:
    return f"mmlu_{variant}"


def _request_json(method: str, url: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
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
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _extract_generator_content(raw: dict[str, Any]) -> str:
    choice = (raw.get("choices") or [{}])[0]
    if not isinstance(choice, dict):
        return ""
    message = choice.get("message")
    if isinstance(message, dict):
        return _extract_text(message.get("content"))
    return _extract_text(choice.get("text"))


def _extract_json_blob(text: str) -> dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    blob_match = re.findall(r"(\{.*\})", stripped, flags=re.DOTALL)
    candidates.extend(blob_match)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Generator output did not contain a valid JSON object")


def _source_answer_label(row: dict[str, Any]) -> str:
    return str(row.get("answer", "")).strip()


def _identity_label_map(row: dict[str, Any]) -> dict[str, str]:
    return {
        str(choice.get("label", "")).strip(): str(choice.get("label", "")).strip()
        for choice in row.get("choices", [])
        if isinstance(choice, dict) and str(choice.get("label", "")).strip()
    }


def _base_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    base = deepcopy(row)
    base["dataset"] = "mmlu_base"
    manifest = {
        "dataset_name": "mmlu_base",
        "variant": "base",
        "source_id": str(row.get("id", "")).strip(),
        "source_subject": str((row.get("meta") or {}).get("subject", "")).strip(),
        "source_answer_label": _source_answer_label(row),
        "derived_answer_label": _source_answer_label(row),
        "label_map": _identity_label_map(row),
        "validation_status": "valid",
        "error": "",
    }
    return base, manifest


def _choice_shuffle_row(row: dict[str, Any], rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    source_choices = deepcopy(row.get("choices", []))
    shuffled = list(source_choices)
    rng.shuffle(shuffled)

    derived_choices: list[dict[str, Any]] = []
    label_map: dict[str, str] = {}
    derived_answer = ""
    source_answer = _source_answer_label(row)
    for idx, choice in enumerate(shuffled):
        derived_label = LETTER_SET[idx]
        source_label = str(choice.get("label", "")).strip()
        derived_choices.append({"label": derived_label, "text": str(choice.get("text", "")).strip()})
        label_map[derived_label] = source_label
        if source_label == source_answer:
            derived_answer = derived_label

    out = deepcopy(row)
    out["dataset"] = _dataset_name_for_variant("choice_shuffle")
    out["choices"] = derived_choices
    out["answer"] = derived_answer
    manifest = {
        "dataset_name": out["dataset"],
        "variant": "choice_shuffle",
        "source_id": str(row.get("id", "")).strip(),
        "source_subject": str((row.get("meta") or {}).get("subject", "")).strip(),
        "source_answer_label": source_answer,
        "derived_answer_label": derived_answer,
        "label_map": label_map,
        "validation_status": "valid",
        "error": "",
    }
    return out, manifest


def _label_numeric_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    derived_choices: list[dict[str, Any]] = []
    label_map: dict[str, str] = {}
    derived_answer = ""
    source_answer = _source_answer_label(row)
    for idx, choice in enumerate(row.get("choices", []), start=1):
        derived_label = str(idx)
        source_label = str(choice.get("label", "")).strip()
        derived_choices.append({"label": derived_label, "text": str(choice.get("text", "")).strip()})
        label_map[derived_label] = source_label
        if source_label == source_answer:
            derived_answer = derived_label

    out = deepcopy(row)
    out["dataset"] = _dataset_name_for_variant("label_numeric")
    out["choices"] = derived_choices
    out["answer"] = derived_answer
    manifest = {
        "dataset_name": out["dataset"],
        "variant": "label_numeric",
        "source_id": str(row.get("id", "")).strip(),
        "source_subject": str((row.get("meta") or {}).get("subject", "")).strip(),
        "source_answer_label": source_answer,
        "derived_answer_label": derived_answer,
        "label_map": label_map,
        "validation_status": "valid",
        "error": "",
    }
    return out, manifest


def _build_question_paraphrase_prompt(row: dict[str, Any]) -> str:
    payload = {
        "question": str(row.get("question", "")).strip(),
        "context": str(row.get("context", "")).strip(),
    }
    return (
        "Paraphrase the question and context while preserving meaning exactly. "
        "Do not mention answer choices. Return JSON only with keys question and context.\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _build_entity_rename_prompt(row: dict[str, Any]) -> str:
    payload = {
        "question": str(row.get("question", "")).strip(),
        "context": str(row.get("context", "")).strip(),
        "choices": [str(choice.get("text", "")).strip() for choice in row.get("choices", []) if isinstance(choice, dict)],
    }
    return (
        "Rewrite the example by consistently renaming entities, names, or variables while preserving meaning "
        "and keeping the same correct answer position. Keep the number of choices unchanged. "
        "Return JSON only with keys question, context, and choices where choices is a list of rewritten choice texts.\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _call_generator(
    *,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1200,
    }
    url = f"{base_url.rstrip('/')}/chat/completions"
    raw = _request_json("POST", url, payload, api_key)
    return _extract_json_blob(_extract_generator_content(raw))


def _build_api_variant(
    row: dict[str, Any],
    variant: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    dataset_name = _dataset_name_for_variant(variant)
    manifest = {
        "dataset_name": dataset_name,
        "variant": variant,
        "source_id": str(row.get("id", "")).strip(),
        "source_subject": str((row.get("meta") or {}).get("subject", "")).strip(),
        "source_answer_label": _source_answer_label(row),
        "derived_answer_label": _source_answer_label(row),
        "label_map": _identity_label_map(row),
        "validation_status": "rejected",
        "error": "",
    }
    try:
        if variant == "question_paraphrase":
            response = _call_generator(
                base_url=base_url,
                model=model,
                api_key=api_key,
                prompt=_build_question_paraphrase_prompt(row),
            )
            question = str(response.get("question", "")).strip()
            context = str(response.get("context", "")).strip()
            if not question:
                raise ValueError("empty question in generator response")
            out = deepcopy(row)
            out["dataset"] = dataset_name
            out["question"] = question
            out["context"] = context
        elif variant == "entity_rename":
            response = _call_generator(
                base_url=base_url,
                model=model,
                api_key=api_key,
                prompt=_build_entity_rename_prompt(row),
            )
            question = str(response.get("question", "")).strip()
            context = str(response.get("context", "")).strip()
            choice_texts = response.get("choices", [])
            if not question:
                raise ValueError("empty question in generator response")
            if not isinstance(choice_texts, list) or len(choice_texts) != len(row.get("choices", [])):
                raise ValueError("invalid choice list in generator response")
            out = deepcopy(row)
            out["dataset"] = dataset_name
            out["question"] = question
            out["context"] = context
            out["choices"] = [
                {
                    "label": str(source_choice.get("label", "")).strip(),
                    "text": str(choice_text).strip(),
                }
                for source_choice, choice_text in zip(row.get("choices", []), choice_texts)
            ]
            if any(not str(choice.get("text", "")).strip() for choice in out["choices"]):
                raise ValueError("empty rewritten choice text")
        else:
            raise ValueError(f"Unsupported API variant: {variant}")
    except Exception as exc:
        manifest["error"] = str(exc)
        return None, manifest

    manifest["validation_status"] = "valid"
    return out, manifest


def _build(args: argparse.Namespace) -> int:
    source_path = Path(args.source).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    processed_dir = out_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    rows = sample_rows(load_jsonl(source_path), args.max_samples, args.shuffle, args.seed)
    base_rows: list[dict[str, Any]] = []
    choice_shuffle_rows: list[dict[str, Any]] = []
    label_numeric_rows: list[dict[str, Any]] = []
    api_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifest_entries: list[dict[str, Any]] = []

    shuffle_rng = random.Random(args.seed)
    for row in rows:
        base_row, base_manifest = _base_row(row)
        base_rows.append(base_row)
        manifest_entries.append(base_manifest)

        if "choice_shuffle" in args.variants:
            derived_row, manifest = _choice_shuffle_row(row, shuffle_rng)
            choice_shuffle_rows.append(derived_row)
            manifest_entries.append(manifest)

        if "label_numeric" in args.variants:
            derived_row, manifest = _label_numeric_row(row)
            label_numeric_rows.append(derived_row)
            manifest_entries.append(manifest)

    write_jsonl(processed_dir / "mmlu_base.jsonl", base_rows)
    built_datasets = ["mmlu_base"]
    if choice_shuffle_rows:
        write_jsonl(processed_dir / "mmlu_choice_shuffle.jsonl", choice_shuffle_rows)
        built_datasets.append("mmlu_choice_shuffle")
    if label_numeric_rows:
        write_jsonl(processed_dir / "mmlu_label_numeric.jsonl", label_numeric_rows)
        built_datasets.append("mmlu_label_numeric")

    if args.api_variants:
        api_key = os.environ.get(args.generator_api_key_env, "") if args.generator_api_key_env else ""
        if not args.generator_base_url or not args.generator_model:
            raise ValueError("API variants require --generator-base-url and --generator-model")
        if args.generator_api_key_env and not api_key:
            raise ValueError(f"Environment variable {args.generator_api_key_env} is not set")

        futures = []
        with ThreadPoolExecutor(max_workers=args.generator_max_workers) as executor:
            for row in rows:
                for variant in args.api_variants:
                    futures.append(
                        executor.submit(
                            _build_api_variant,
                            row,
                            variant,
                            base_url=args.generator_base_url,
                            model=args.generator_model,
                            api_key=api_key,
                        )
                    )
            for future in as_completed(futures):
                derived_row, manifest = future.result()
                manifest_entries.append(manifest)
                if derived_row is not None:
                    api_rows[manifest["dataset_name"]].append(derived_row)

        for dataset_name, rows_for_dataset in api_rows.items():
            write_jsonl(processed_dir / f"{dataset_name}.jsonl", rows_for_dataset)
            built_datasets.append(dataset_name)

    manifest_path = out_root / "surface_perturbation_manifest.json"
    dump_json(
        manifest_path,
        {
            "config": {
                "source": str(source_path),
                "max_samples": args.max_samples,
                "shuffle": args.shuffle,
                "seed": args.seed,
                "variants": args.variants,
                "api_variants": args.api_variants,
                "generator_base_url": args.generator_base_url,
                "generator_model": args.generator_model,
                "generator_api_key_env": args.generator_api_key_env,
                "generator_max_workers": args.generator_max_workers,
            },
            "datasets": sorted(built_datasets),
            "entries": manifest_entries,
            "paths": {
                "processed_dir": str(processed_dir),
            },
        },
    )
    print(f"Wrote {manifest_path}")
    return 0


def _canonicalize_prediction(prediction: str, label_map: dict[str, str]) -> str:
    pred = str(prediction or "").strip()
    if not pred:
        return ""
    return str(label_map.get(pred, pred)).strip()


def _analyze(args: argparse.Namespace) -> int:
    manifest = load_json(Path(args.manifest).expanduser().resolve())
    entries = manifest.get("entries", [])
    entry_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = (str(entry.get("dataset_name", "")), str(entry.get("source_id", "")))
        if key[0] and key[1]:
            entry_by_key[key] = entry

    run_specs = parse_run_specs(args.run)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []

    for label, run_path in run_specs.items():
        records = load_jsonl(resolve_records_path(run_path))
        grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
        model_name = ""
        for row in records:
            dataset = str(row.get("dataset", "")).strip()
            mode = str(row.get("mode", "")).strip()
            source_id = str(row.get("id", "")).strip()
            if dataset not in {args.base_dataset, *args.compare_datasets}:
                continue
            grouped[(dataset, mode, source_id)] = row
            if not model_name:
                model_name = str(row.get("model", "")).strip()

        for compare_dataset in args.compare_datasets:
            for mode in sorted({mode for _, mode, _ in grouped.keys()}):
                paired_rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
                subject_grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = defaultdict(list)
                base_ids = sorted({sid for dataset, m, sid in grouped.keys() if dataset == args.base_dataset and m == mode})
                for source_id in base_ids:
                    base_record = grouped.get((args.base_dataset, mode, source_id))
                    compare_record = grouped.get((compare_dataset, mode, source_id))
                    compare_entry = entry_by_key.get((compare_dataset, source_id))
                    if base_record is None or compare_record is None or compare_entry is None:
                        continue
                    paired_rows.append((base_record, compare_record, compare_entry))
                    subject = str(compare_entry.get("source_subject", "")).strip()
                    subject_grouped[subject].append((base_record, compare_record, compare_entry))

                if not paired_rows:
                    continue

                base_total = len(paired_rows)
                base_accuracy = sum(1 for base, _, _ in paired_rows if base.get("is_correct")) / base_total
                base_following = sum(1 for base, _, _ in paired_rows if base.get("nonreason_pass")) / base_total
                accuracy = sum(1 for _, compare, _ in paired_rows if compare.get("is_correct")) / base_total
                following = sum(1 for _, compare, _ in paired_rows if compare.get("nonreason_pass")) / base_total
                following_correct = (
                    sum(1 for _, compare, _ in paired_rows if compare.get("nonreason_correct")) / base_total
                )
                consistency = 0
                for base_record, compare_record, compare_entry in paired_rows:
                    source_label_map = {str(k): str(v) for k, v in (compare_entry.get("label_map") or {}).items()}
                    base_pred = _canonicalize_prediction(str(base_record.get("prediction", "")), _identity_label_map({"choices": []}))
                    compare_pred = _canonicalize_prediction(str(compare_record.get("prediction", "")), source_label_map)
                    if base_pred and compare_pred and base_pred == compare_pred:
                        consistency += 1

                variant_name = str(paired_rows[0][2].get("variant", compare_dataset))
                variant_rows.append(
                    {
                        "model_label": label,
                        "model_name": model_name,
                        "mode": mode,
                        "dataset": compare_dataset,
                        "variant": variant_name,
                        "total": base_total,
                        "accuracy": accuracy,
                        "following_rate": following,
                        "following_correct_rate": following_correct,
                        "base_accuracy": base_accuracy,
                        "base_following_rate": base_following,
                        "accuracy_drop_vs_base": accuracy - base_accuracy,
                        "following_drop_vs_base": following - base_following,
                        "prediction_consistency_rate": consistency / base_total,
                    }
                )

                for subject, rows_for_subject in sorted(subject_grouped.items()):
                    total = len(rows_for_subject)
                    base_accuracy_subject = (
                        sum(1 for base, _, _ in rows_for_subject if base.get("is_correct")) / total
                    )
                    base_following_subject = (
                        sum(1 for base, _, _ in rows_for_subject if base.get("nonreason_pass")) / total
                    )
                    accuracy_subject = (
                        sum(1 for _, compare, _ in rows_for_subject if compare.get("is_correct")) / total
                    )
                    following_subject = (
                        sum(1 for _, compare, _ in rows_for_subject if compare.get("nonreason_pass")) / total
                    )
                    following_correct_subject = (
                        sum(1 for _, compare, _ in rows_for_subject if compare.get("nonreason_correct")) / total
                    )
                    consistency_subject = 0
                    for base_record, compare_record, compare_entry in rows_for_subject:
                        source_label_map = {str(k): str(v) for k, v in (compare_entry.get("label_map") or {}).items()}
                        base_pred = _canonicalize_prediction(str(base_record.get("prediction", "")), _identity_label_map({"choices": []}))
                        compare_pred = _canonicalize_prediction(str(compare_record.get("prediction", "")), source_label_map)
                        if base_pred and compare_pred and base_pred == compare_pred:
                            consistency_subject += 1
                    subject_rows.append(
                        {
                            "model_label": label,
                            "model_name": model_name,
                            "mode": mode,
                            "dataset": compare_dataset,
                            "variant": variant_name,
                            "subject": subject,
                            "total": total,
                            "accuracy": accuracy_subject,
                            "following_rate": following_subject,
                            "following_correct_rate": following_correct_subject,
                            "base_accuracy": base_accuracy_subject,
                            "base_following_rate": base_following_subject,
                            "accuracy_drop_vs_base": accuracy_subject - base_accuracy_subject,
                            "following_drop_vs_base": following_subject - base_following_subject,
                            "prediction_consistency_rate": consistency_subject / total,
                        }
                    )

    variant_csv = output_dir / "variant_metrics.csv"
    subject_csv = output_dir / "subject_variant_metrics.csv"
    summary_path = output_dir / "surface_perturbation_summary.json"
    write_csv(variant_csv, variant_rows)
    write_csv(subject_csv, subject_rows)
    dump_json(
        summary_path,
        {
            "config": {
                "manifest": args.manifest,
                "runs": {label: str(path) for label, path in run_specs.items()},
                "base_dataset": args.base_dataset,
                "compare_datasets": args.compare_datasets,
            },
            "variant_metrics": variant_rows,
            "subject_variant_metrics": subject_rows,
            "paths": {
                "variant_metrics_csv": str(variant_csv),
                "subject_variant_metrics_csv": str(subject_csv),
            },
        },
    )
    print(f"Wrote {variant_csv}")
    print(f"Wrote {subject_csv}")
    print(f"Wrote {summary_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "build":
        return _build(args)
    return _analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
