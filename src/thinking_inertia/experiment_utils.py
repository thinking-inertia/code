#!/usr/bin/env python3
"""
Small shared helpers for MMLU nonreason experiments.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_SUBJECT_GROUPS: dict[str, list[str]] = {
    "memory_heavy": [
        "anatomy",
        "global_facts",
        "high_school_us_history",
        "high_school_world_history",
        "prehistory",
        "world_religions",
        "high_school_geography",
        "high_school_government_and_politics",
        "nutrition",
        "medical_genetics",
    ],
    "reasoning_heavy": [
        "abstract_algebra",
        "college_mathematics",
        "elementary_mathematics",
        "formal_logic",
        "logical_fallacies",
        "econometrics",
        "high_school_statistics",
        "conceptual_physics",
        "college_physics",
        "high_school_mathematics",
    ],
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected top-level JSON object in {path}")
    return data


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise TypeError(f"Expected JSON object at line {line_no} in {path}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_run_specs(specs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected run spec label=path, got: {spec}")
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        path = Path(raw_path).expanduser().resolve()
        if not label:
            raise ValueError(f"Empty run label in spec: {spec}")
        out[label] = path
    return out


def resolve_records_path(path: Path) -> Path:
    if path.is_dir():
        records_path = path / "records.jsonl"
        if records_path.exists():
            return records_path
        raise FileNotFoundError(f"Missing records.jsonl in {path}")
    if path.is_file():
        return path
    raise FileNotFoundError(f"Run path not found: {path}")


def sample_rows(rows: list[dict[str, Any]], max_samples: int, shuffle: bool, seed: int) -> list[dict[str, Any]]:
    picked = list(rows)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(picked)
    if max_samples > 0:
        picked = picked[:max_samples]
    return picked


def index_mmlu_subjects(processed_mmlu_path: Path) -> dict[str, str]:
    subjects: dict[str, str] = {}
    for row in load_jsonl(processed_mmlu_path):
        row_id = str(row.get("id", "")).strip()
        subject = str((row.get("meta") or {}).get("subject", "")).strip()
        if row_id and subject:
            subjects[row_id] = subject
    return subjects


def enrich_subject_groups(subjects: set[str], groups: dict[str, list[str]]) -> dict[str, list[str]]:
    assigned: set[str] = set()
    out: dict[str, list[str]] = {}
    for group_name, group_subjects in groups.items():
        present = [subject for subject in group_subjects if subject in subjects]
        out[group_name] = present
        assigned.update(present)
    out["mixed_professional"] = sorted(subject for subject in subjects if subject not in assigned)
    return out


def aggregate_eval_records(records: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "accuracy": 0.0,
            "following_rate": 0.0,
            "following_correct_rate": 0.0,
            "mean_thinking_rate": 0.0,
            "mean_t_word_count": 0.0,
        }
    correct = sum(1 for row in records if row.get("is_correct"))
    following = sum(1 for row in records if row.get("nonreason_pass"))
    following_correct = sum(
        1
        for row in records
        if row.get("nonreason_correct") or (row.get("nonreason_pass") and row.get("is_correct"))
    )
    thinking_values = [float(row.get("thinking_rate", 0.0)) for row in records]
    t_word_values = [float(row.get("t_word_count", 0.0)) for row in records]
    return {
        "total": total,
        "accuracy": correct / total,
        "following_rate": following / total,
        "following_correct_rate": following_correct / total,
        "mean_thinking_rate": sum(thinking_values) / total,
        "mean_t_word_count": sum(t_word_values) / total,
    }


def retention_to_baseline(accuracy: float, baseline_accuracy: float, chance: float = 0.25) -> float | None:
    if baseline_accuracy <= chance:
        return None
    return (accuracy - chance) / (baseline_accuracy - chance)
