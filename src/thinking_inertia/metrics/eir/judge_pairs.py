#!/usr/bin/env python3
"""Apply the rebuttal GPT-5.5 rubric to answer-space Q/T pairs in batches."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from .judge_rubric import (
    CATEGORIES,
    SYSTEM_PROMPT,
    evidence_is_verbatim,
    parse_json_payload,
)


RESULTS_DIR = Path(__file__).resolve().parents[4] / "outputs/eir"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RESULTS_DIR / "m2_corpus.jsonl")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "gpt55_labels.jsonl")
    parser.add_argument("--errors", type=Path, default=RESULTS_DIR / "errors.jsonl")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL (default: OPENAI_BASE_URL or the OpenAI API).",
    )
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=90_000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("category") in CATEGORIES:
                completed.add(str(row.get("record_id", "")))
    return completed


def make_batches(
    rows: list[dict[str, Any]], batch_size: int, max_chars: int
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for row in rows:
        item_chars = len(row["question"]) + len(row["pre_answer_text"]) + 160
        if current and (
            len(current) >= batch_size or current_chars + item_chars > max_chars
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(row)
        current_chars += item_chars
    if current:
        batches.append(current)
    return batches


async def judge_batch(
    client: AsyncOpenAI, args: argparse.Namespace, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    public_items = [
        {
            "id": row["record_id"],
            "question": row["question"],
            "pre_answer_text": row["pre_answer_text"],
        }
        for row in rows
    ]
    user_prompt = (
        "Apply the rubric independently to every QUESTION/PRE-ANSWER TEXT pair. "
        "Return one label for every ID in input order.\n\nITEMS:\n"
        + json.dumps(public_items, ensure_ascii=False)
    )
    expected = [row["record_id"] for row in rows]
    row_by_id = {row["record_id"]: row for row in rows}
    last_error = ""
    for attempt in range(args.retries):
        content = ""
        try:
            response = await client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=args.max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            payload = parse_json_payload(content)
            labels = payload.get("labels")
            if not isinstance(labels, list):
                raise ValueError("missing labels list")

            parsed: dict[str, tuple[str, str]] = {}
            for label in labels:
                if not isinstance(label, list) or len(label) != 3:
                    raise ValueError("each label must be [id, category, evidence]")
                record_id, category, evidence = map(str, label)
                if record_id not in row_by_id or category not in CATEGORIES - {"empty"}:
                    raise ValueError(f"invalid label: {label!r}")
                parsed[record_id] = (category, evidence)
            if set(parsed) != set(expected):
                raise ValueError("returned IDs differ from requested IDs")

            return [
                {
                    "record_id": record_id,
                    "category": parsed[record_id][0],
                    "evidence": parsed[record_id][1],
                    "evidence_is_verbatim": int(
                        evidence_is_verbatim(
                            parsed[record_id][1], row_by_id[record_id]["pre_answer_text"]
                        )
                    ),
                    "judge": args.model,
                }
                for record_id in expected
            ]
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}; raw={content[:2000]!r}"
            await asyncio.sleep(min(2**attempt, 20))
    raise RuntimeError(last_error)


async def main_async(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("set OPENAI_API_KEY")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_ids(args.output)
    pending: list[dict[str, Any]] = []
    empty_rows: list[dict[str, Any]] = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["record_id"] in completed:
                continue
            if row["t_is_empty"]:
                empty_rows.append(
                    {
                        "record_id": row["record_id"],
                        "category": "empty",
                        "evidence": "",
                        "evidence_is_verbatim": 1,
                        "judge": "definition",
                    }
                )
            else:
                pending.append(row)

    with args.output.open("a", encoding="utf-8") as output:
        for row in empty_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    batches = make_batches(pending, args.batch_size, args.max_chars)
    print(
        json.dumps(
            {
                "already_completed": len(completed),
                "empty_added": len(empty_rows),
                "pending_nonempty": len(pending),
                "api_batches": len(batches),
            }
        ),
        flush=True,
    )

    queue: asyncio.Queue[list[dict[str, Any]] | None] = asyncio.Queue()
    for batch in batches:
        queue.put_nowait(batch)
    for _ in range(args.concurrency):
        queue.put_nowait(None)

    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url, timeout=600.0)
    lock = asyncio.Lock()
    progress = {"calls": 0, "records": 0, "errors": 0}

    async def worker() -> None:
        while True:
            batch = await queue.get()
            if batch is None:
                queue.task_done()
                return
            try:
                labels = await judge_batch(client, args, batch)
                async with lock:
                    with args.output.open("a", encoding="utf-8") as output:
                        for label in labels:
                            output.write(json.dumps(label, ensure_ascii=False) + "\n")
                    progress["calls"] += 1
                    progress["records"] += len(labels)
                    if progress["calls"] % 25 == 0:
                        print(json.dumps(progress), flush=True)
            except Exception as exc:
                async with lock:
                    with args.errors.open("a", encoding="utf-8") as errors:
                        errors.write(
                            json.dumps(
                                {
                                    "ids": [row["record_id"] for row in batch],
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    progress["errors"] += 1
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
    await queue.join()
    await asyncio.gather(*workers)
    await client.close()
    print(json.dumps(progress), flush=True)


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
