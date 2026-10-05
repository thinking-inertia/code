#!/usr/bin/env python3
"""Compute instruction-aware question--pre-answer relevance with Qwen3-Embedding-4B."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


TASK = (
    "Given a question, retrieve pre-answer text that contains reasoning relevant "
    "to answering the question"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Input records.jsonl file or directory; repeat for multiple inputs.",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("QREL_MODEL_PATH", "Qwen/Qwen3-Embedding-4B"),
        help="Qwen3-Embedding-4B model ID or local path.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qrel"),
    )
    return parser.parse_args()


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def input_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    if not files:
        raise FileNotFoundError("No JSONL inputs found")
    return files


def load_rows(inputs: list[Path], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in input_files(inputs):
        source = path.stem
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or "\x00" in line:
                    continue
                row = json.loads(line)
                if normalize_space(row.get("error")):
                    continue
                question = normalize_space(
                    row.get("question") or row.get("question_text_used_for_similarity")
                )
                if not question:
                    continue
                text = normalize_space(
                    row.get("pre_answer_text") or row.get("extracted_t") or row.get("T")
                )
                rows.append(
                    {
                        "source": source,
                        "model": str(row.get("model", "")),
                        "dataset": str(row.get("dataset", "")),
                        "mode": str(row.get("mode") or row.get("paper_mode") or ""),
                        "sample_id": str(row.get("record_id") or row.get("id") or ""),
                        "question": question,
                        "pre_answer_text": text,
                        "empty_t": int(not text),
                    }
                )
                if limit and len(rows) >= limit:
                    return rows
    return rows


def encode_texts(
    model: Any,
    texts: list[str],
    *,
    batch_size: int,
    prompt: str | None = None,
) -> dict[str, np.ndarray]:
    vectors = model.encode(
        texts,
        prompt=prompt,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return dict(zip(texts, vectors, strict=True))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input, args.limit)
    nonempty = [row for row in rows if row["pre_answer_text"]]
    questions = sorted({row["question"] for row in nonempty})
    texts = sorted({row["pre_answer_text"] for row in nonempty})
    print(
        json.dumps(
            {
                "rows": len(rows),
                "nonempty": len(nonempty),
                "unique_questions": len(questions),
                "unique_t": len(texts),
            }
        ),
        flush=True,
    )

    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        args.model_path,
        device=args.device,
        model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = args.max_length
    q_vectors = encode_texts(
        model,
        questions,
        batch_size=max(args.batch_size, 32),
        prompt=f"Instruct: {TASK}\nQuery:",
    )

    # Long responses are encoded in smaller batches while preserving the 32K limit.
    t_vectors: dict[str, np.ndarray] = {}
    lower = 0
    for upper, bucket_batch in (
        (128, max(args.batch_size, 128)),
        (256, max(args.batch_size, 96)),
        (512, max(args.batch_size, 64)),
        (1024, max(args.batch_size, 32)),
        (2048, max(args.batch_size, 16)),
        (4096, max(args.batch_size, 8)),
        (8192, min(args.batch_size, 2)),
        (float("inf"), 1),
    ):
        bucket = [text for text in texts if lower < len(text.split()) <= upper]
        lower = upper
        if bucket:
            t_vectors.update(
                encode_texts(model, bucket, batch_size=bucket_batch, prompt=None)
            )

    for row in rows:
        text = row["pre_answer_text"]
        row["qrel"] = (
            float(np.dot(q_vectors[row["question"]], t_vectors[text]))
            if text
            else 0.0
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    item_rows = [
        {
            "source": row["source"],
            "model": row["model"],
            "dataset": row["dataset"],
            "mode": row["mode"],
            "sample_id": row["sample_id"],
            "empty_t": row["empty_t"],
            "qrel": row["qrel"],
        }
        for row in rows
    ]
    write_csv(args.output_dir / "item_qrel.csv", item_rows)

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["model"], row["dataset"], row["mode"])].append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for (source, model_name, dataset, mode), group in sorted(grouped.items()):
        aggregate_rows.append(
            {
                "source": source,
                "model": model_name,
                "dataset": dataset,
                "mode": mode,
                "n": len(group),
                "empty_t_n": sum(row["empty_t"] for row in group),
                "qrel": sum(row["qrel"] for row in group) / len(group),
            }
        )
    write_csv(args.output_dir / "aggregate_qrel.csv", aggregate_rows)
    print(json.dumps({"output_dir": str(args.output_dir), "groups": len(grouped)}))


if __name__ == "__main__":
    main()
