"""Offline request, boundary, field-scope, and evaluator integration checks."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from thinking_inertia import eval_thinking_spectrum as spectrum
from thinking_inertia import m6_prefill as m6

SAMPLE = {"id": "q1", "dataset": "mmlu", "question": "Choose C.", "task_type": "mcq", "answer": "C",
          "choices": [{"label": c, "text": c + " option"} for c in "ABCD"]}


class Tokenizer:
    chat_template = "test Qwen3 template"
    def apply_chat_template(self, messages, tokenize, **kwargs):
        assert kwargs == {"continue_final_message": True, "add_generation_prompt": False, "enable_thinking": False}
        assert messages[-1] == {"role": "assistant", "content": m6.PREFIX}
        text = "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" + m6.PREFIX
        return self.encode(text) if tokenize else text
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class Scalar(float):
    def __mul__(self, other):
        return Scalar(float(self) * float(other))
    def item(self):
        return float(self)


class Scorer:
    class _torch:
        sum = staticmethod(lambda value: value)
    def encode_question(self, question, **kwargs):
        return Scalar(0.5)
    def encode_texts(self, texts, **kwargs):
        return [Scalar(0.5) for _ in texts]
    def score_question_to_sentences(self, question, sentences, **kwargs):
        return [0.25] * len(sentences), None
    def score_text_pairs(self, pairs, **kwargs):
        return [0.25] * len(pairs)


def args_for(backend="deepseek-prefix"):
    with patch("sys.argv", ["eval", "--datasets", "mmlu", "--model", "test-model", "--base-url", "http://test/v1"]):
        args = spectrum.parse_args()
    args.m6_prefill = backend
    args.provider_profile = "deepseek_v4" if backend == "deepseek-prefix" else "qwen3"
    args.api = "chat" if backend == "deepseek-prefix" else "completions"
    args.m6_tokenizer = "test-tokenizer"
    args._m6_tokenizer = Tokenizer()
    args.request_retries = 0
    return args


def extracted(**kwargs):
    return {"reasoning": "", "reasoning_content": "", "reasoning_details": None, **kwargs}


class PrefillTests(unittest.TestCase):
    def score(self, suffix, sample=SAMPLE, **fields):
        return m6.score_fields(sample, m6.PREFIX + suffix, extracted(**fields), accuracy_policy="canonical_local")

    def test_user_prompt_contains_no_assistant_prefix(self):
        self.assertEqual(spectrum._build_mode_prompt(SAMPLE, "mode6"), spectrum._build_mode_prompt(SAMPLE, "mode2"))
        self.assertNotIn(m6.PREFIX, spectrum._build_mode_prompt(SAMPLE, "mode6"))

    def test_deepseek_request_places_prefix_in_assistant(self):
        payload, metadata = m6.prepare_request(args_for(), spectrum._build_mode_prompt(SAMPLE, "mode6"))
        self.assertEqual(payload["messages"][-1], {"role": "assistant", "content": m6.PREFIX, "prefix": True})
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(metadata["m6_protocol"], m6.PROTOCOL)

    def test_qwen_request_is_exact_continuation_token_ids(self):
        payload, metadata = m6.prepare_request(args_for("qwen-completions"), "Question")
        self.assertEqual("".join(map(chr, payload["prompt"])), metadata["rendered_prompt"])
        self.assertTrue(metadata["rendered_prompt"].endswith(m6.PREFIX))
        self.assertFalse(payload["echo"])
        self.assertFalse(payload["skip_special_tokens"])
        self.assertEqual(payload["top_p"], 1.0)

    def test_unsupported_backend_fails_without_request(self):
        for api, profile, backend in [("responses", "openai_gpt", ""), ("chat", "gemini3_flash", "deepseek-prefix"), ("chat", "qwen3", "qwen-completions")]:
            args = args_for(); args.api, args.provider_profile, args.m6_prefill = api, profile, backend
            with self.assertRaises(ValueError):
                m6.prepare_request(args, "Question")

    def test_first_answer_not_later_correction_is_scored(self):
        result = self.score(r"A}. Actually, C is correct: \boxed{C}.")
        self.assertEqual(result["prediction"], "A")
        self.assertFalse(result["is_correct"])
        self.assertIn(r"\boxed{C}", result["t_payload"]["t_returned_readable"])
        self.assertFalse(result["etr"])

    def test_answer_only_and_terminal_punctuation(self):
        for suffix in ["C}", "C}.", "C}。", "C}\n .\n"]:
            result = self.score(suffix)
            self.assertTrue(result["is_correct"])
            self.assertTrue(result["etr"])
        for suffix in ["C}!", "C}...", "C}}", "C}.\nThis follows from the options."]:
            self.assertFalse(self.score(suffix)["etr"])

    def test_empty_or_unclosed_first_box_never_skips_ahead(self):
        for suffix in [r"}. Later \boxed{C}.", r" \boxed{C}", ""]:
            result = self.score(suffix)
            self.assertFalse(result["is_correct"])
            self.assertFalse(result["etr"])
            self.assertEqual(result["prediction"], "")
        self.assertEqual(self.score(r" \boxed{C}")["content_t"], m6.PREFIX + r" \boxed{C}")

    def test_nested_math_and_escaped_braces(self):
        first = m6.parse_first_box(m6.PREFIX + r"\frac{1}{2}}. Explanation")
        self.assertEqual(first["inner"], r"\frac{1}{2}")
        self.assertEqual(first["remainder_raw"], ". Explanation")
        first = m6.parse_first_box(m6.PREFIX + r"\{x\}}.")
        self.assertEqual(first["inner"], r"\{x\}")
        sample = {"task_type": "math", "dataset": "gsm8k", "answer": "2"}
        result = self.score("1+1=2}.", sample)
        self.assertEqual(result["t_payload"]["t_returned_readable"], "")

    def test_all_readable_fields_retained_and_duplicate_parts_deduplicated(self):
        result = self.score("C}.", reasoning="Because C follows.", reasoning_content="Because C follows.",
            reasoning_details=[{"type": "summary_text", "text": "A summary."}, {"type": "encrypted", "text": "opaque"}])
        self.assertEqual(result["t_payload"]["t_returned_readable"], "Because C follows. A summary.")
        self.assertFalse(result["etr"])
        self.assertTrue(result["content_answer_only"])

    def test_legacy_mode6_cannot_resume_or_be_relabeled(self):
        with self.assertRaises(ValueError):
            m6.require_protocol({"mode": "mode6", "content": m6.PREFIX + "C}"})
        m6.require_protocol({"mode": "mode6u"})
        m6.require_protocol({"mode": "mode5"})

    def test_evaluator_saves_raw_suffix_and_scores_entire_residual(self):
        args = args_for()
        raw = {"choices": [{"message": {"content": "A}. Correction: C.", "reasoning_content": "Compare all options."}, "finish_reason": "stop"}]}
        with patch.object(spectrum, "_request_json", return_value=raw) as request:
            record = spectrum._evaluate_one_sample(args=args, scorer=Scorer(), scorer_lock=None,
                dataset_name="mmlu", sample=SAMPLE, mode="mode6", rec_id="q1")
        self.assertEqual(record["error"], "")
        self.assertEqual(request.call_count, 1)
        self.assertEqual(record["content"], m6.PREFIX + raw["choices"][0]["message"]["content"])
        self.assertEqual(record["raw_response"], raw)
        self.assertEqual(record["raw_generated_continuation"], "A}. Correction: C.")
        self.assertEqual(record["prediction"], "A")
        self.assertFalse(record["is_correct"])
        self.assertEqual(record["extracted_t"], "Compare all options. . Correction: C.")
        self.assertEqual(m6.metric_pair(record)["pre_answer_text"], record["extracted_t"])

    def test_malformed_empty_box_has_zero_summary_etr(self):
        args = args_for()
        raw = {"choices": [{"message": {"content": "}."}}]}
        with patch.object(spectrum, "_request_json", return_value=raw):
            record = spectrum._evaluate_one_sample(args=args, scorer=Scorer(), scorer_lock=None,
                dataset_name="mmlu", sample=SAMPLE, mode="mode6", rec_id="q1")
        self.assertEqual(record["error"], "")
        self.assertEqual(record["extracted_t"], "")
        self.assertEqual(spectrum._summarize_records([record], accept_display_math_wrapper=False)["empty_t_rate"], 0.0)

    def test_resume_rejects_a_changed_scoring_policy(self):
        args = args_for()
        raw = {"choices": [{"message": {"content": "C}."}}]}
        with patch.object(spectrum, "_request_json", return_value=raw):
            record = spectrum._evaluate_one_sample(args=args, scorer=Scorer(), scorer_lock=None,
                dataset_name="mmlu", sample=SAMPLE, mode="mode6", rec_id="q1")
        m6.validate_resume(args, record)
        args.m6_accuracy_policy = "canonical_local"
        with self.assertRaisesRegex(ValueError, "accuracy_policy"):
            m6.validate_resume(args, record)

    def test_canonical_math_comparator_preserves_local_policy(self):
        from thinking_inertia import m6_accuracy
        # The local experiment's comparator predates the API fraction parser.
        self.assertEqual(m6_accuracy.extract_math_answer_fallback(r"\frac{1}{2}"), "2")
        sample = {"task_type": "math", "dataset": "math", "answer": "2"}
        result = self.score(r"\frac{1}{2}}.", sample)
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["answer_scorer"], "fallback_normalized_exact_match")

    def test_serial_and_concurrent_main_runs_share_m6_path(self):
        all_records = []
        for concurrency in (1, 2):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); (root / "processed").mkdir()
                (root / "processed/mmlu.jsonl").write_text(json.dumps(SAMPLE) + "\n")
                args = args_for(); args.root_dir = directory; args.datasets = ["mmlu"]
                args.modes = ["6"]; args.run_name = "test"; args.request_concurrency = concurrency
                raw = {"choices": [{"message": {"content": "C}. Explained."}, "finish_reason": "stop"}]}
                with patch.object(spectrum, "parse_args", return_value=args), \
                     patch.object(spectrum.m6_prefill, "load_tokenizer"), \
                     patch.object(spectrum, "MiniLMSimilarityScorer", return_value=Scorer()), \
                     patch.object(spectrum, "_request_json", return_value=raw):
                    self.assertEqual(spectrum.main(), 0)
                record = json.loads((root / "runs/test/records.jsonl").read_text())
                self.assertEqual(record["error"], "")
                all_records.append(record)
        self.assertEqual(all_records[0], all_records[1])


if __name__ == "__main__":
    unittest.main()
