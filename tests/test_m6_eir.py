"""Offline checks for M6 rubric selection, grouping, and label provenance."""
from __future__ import annotations

import asyncio
import hashlib
import json
import contextlib
import io
import os
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from thinking_inertia.m6_prefill import PROTOCOL
from thinking_inertia.metrics.eir import judge_pairs, judge_rubric, m6_rubric


class RecordingClient:
    def __init__(self, ids):
        self.ids = ids
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **request):
        self.calls.append(request)
        payload = {"labels": [[record_id, "explicit_reasoning", "two are greater than one"] for record_id in self.ids]}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])


class M6JudgeTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"record_id": "a", "question": "Is two greater than one?",
             "pre_answer_text": "Yes, because two are greater than one."},
            {"record_id": "b", "question": "Is two greater than one?",
             "pre_answer_text": "Yes, because two are greater than one."},
            {"record_id": "c", "question": "Is three greater than two?",
             "pre_answer_text": "Three is one more than two."},
        ]
        self.args = SimpleNamespace(model="gpt-5.5", max_tokens=4096, retries=1)

    def test_default_request_keeps_existing_pair_schema_and_rubric(self):
        client = RecordingClient(["a", "b", "c"])
        result = asyncio.run(judge_pairs.judge_batch(client, self.args, self.rows))
        request = client.calls[0]
        self.assertEqual(request["messages"][0], {"role": "system", "content": judge_rubric.SYSTEM_PROMPT})
        self.assertEqual(request["messages"][1]["content"],
            "Apply the rubric independently to every QUESTION/PRE-ANSWER TEXT pair. "
            "Return one label for every ID in input order.\n\nITEMS:\n" + json.dumps([
                {"id": row["record_id"], "question": row["question"], "pre_answer_text": row["pre_answer_text"]}
                for row in self.rows], ensure_ascii=False))
        self.assertEqual([row["record_id"] for row in result], ["a", "b", "c"])
        self.assertTrue(all("m6_protocol" not in row for row in result))
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["response_format"], {"type": "json_object"})

    def test_m6_rubric_matches_experiment_fingerprint(self):
        self.assertEqual(hashlib.sha256(m6_rubric.SYSTEM_PROMPT.encode()).hexdigest(),
            "a5e7d623449ed593d088f0cf3ede8439d23d84394df2213ba93f4eb16ad6436b")

    def test_m6_groups_questions_and_reuses_only_identical_QT(self):
        rows = self.rows + [{"record_id": "d", "question": "Another question",
                            "pre_answer_text": self.rows[0]["pre_answer_text"]}]
        batches = m6_rubric.make_batches(rows, 16, 100000)
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0][0]["_m6_expand_to"], ["a", "b"])
        self.assertEqual(batches[2][0]["_m6_expand_to"], ["d"])
        self.assertTrue(all(len({row["question"] for row in batch}) == 1 for batch in batches))

    def test_m6_uses_original_request_format_and_expands_duplicate_labels(self):
        self.args.m6_residual_rubric = True
        batch = m6_rubric.make_batches(self.rows, 16, 100000)[0]
        client = RecordingClient(["a"])
        result = asyncio.run(judge_pairs.judge_batch(client, self.args, batch))
        self.assertEqual(client.calls[0]["messages"], [
            {"role": "system", "content": m6_rubric.SYSTEM_PROMPT},
            {"role": "user", "content": "QUESTION:\nIs two greater than one?\n\nPRE-ANSWER TEXTS:\n"
                '[{"id": "a", "text": "Yes, because two are greater than one."}]'},
        ])
        self.assertEqual([row["record_id"] for row in result], ["a", "b"])
        self.assertTrue(all(row["m6_protocol"] == PROTOCOL for row in result))
        self.assertTrue(all(row["rubric_sha256"] == m6_rubric.SYSTEM_PROMPT_SHA256 for row in result))

    def test_filtered_duplicate_pair_reports_every_affected_id(self):
        class FilteredClient:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=self)

            async def create(self, **request):
                raise RuntimeError("content_filter")

            async def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            args = SimpleNamespace(input=directory / "pairs.jsonl", output=directory / "labels.jsonl",
                errors=directory / "errors.jsonl", model="gpt-5.5", base_url="https://example.invalid/v1",
                concurrency=1, batch_size=16, max_chars=100000, max_tokens=4096, retries=1,
                m6_residual_rubric=True)
            args.input.write_text("\n".join(json.dumps({**row, "t_is_empty": 0, "m6_protocol": PROTOCOL})
                for row in self.rows[:2]) + "\n")
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}), \
                 mock.patch.dict(sys.modules, {"openai": SimpleNamespace(AsyncOpenAI=FilteredClient)}), \
                 mock.patch.object(judge_pairs.asyncio, "sleep", new=mock.AsyncMock()), \
                 contextlib.redirect_stdout(io.StringIO()):
                asyncio.run(judge_pairs.main_async(args))
            error = json.loads(args.errors.read_text())
            self.assertEqual(error["ids"], ["a", "b"])
            self.assertIn("content_filter", error["error"])
            self.assertEqual(args.output.read_text(), "")

    def test_m6_request_rejects_mixed_questions(self):
        with self.assertRaises(ValueError):
            m6_rubric.user_prompt(self.rows)

    def test_m6_batch_limits_do_not_mix_questions(self):
        rows = [{"record_id": str(i), "question": "Same question", "pre_answer_text": str(i) * 20} for i in range(5)]
        self.assertEqual([len(batch) for batch in m6_rubric.make_batches(rows, 2, 100000)], [2, 2, 1])
        self.assertEqual([len(batch) for batch in m6_rubric.make_batches(rows, 16, 113)], [1, 1, 1, 1, 1])

    def test_m6_resume_accepts_only_matching_protocol_and_rubric(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            label = {"record_id": "a", "category": "explicit_reasoning", "m6_protocol": PROTOCOL,
                     "rubric_sha256": m6_rubric.SYSTEM_PROMPT_SHA256}
            path.write_text(json.dumps(label) + "\n")
            self.assertEqual(judge_pairs.completed_ids(path, m6_residual=True), {"a"})
            for change in ({"rubric_sha256": "different"}, {"m6_protocol": "user_prefix"}):
                path.write_text(json.dumps({**label, **change}) + "\n")
                with self.assertRaises(ValueError):
                    judge_pairs.completed_ids(path, m6_residual=True)
            path.write_text(json.dumps({"record_id": "old", "category": "explicit_reasoning"}) + "\n")
            self.assertEqual(judge_pairs.completed_ids(path), {"old"})
            with self.assertRaises(ValueError):
                judge_pairs.completed_ids(path, m6_residual=True)


if __name__ == "__main__":
    unittest.main()
