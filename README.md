<div align="center">

<h1>Thinking Inertia: LLMs Keep Thinking When Told Not To</h1>

<p><strong>Dianqiao Lei<sup>1</sup> · Kevin Qinghong Lin<sup>2</sup><sup>†</sup><sup>✉</sup> · Pan Lu<sup>3</sup> · Philip Torr<sup>2</sup><sup>✉</sup> · James Zou<sup>3</sup><sup>✉</sup></strong></p>

<sup>1</sup> Tsinghua University · <sup>2</sup> University of Oxford · <sup>3</sup> Stanford University

<sup>†</sup> Project lead · <sup>✉</sup> Correspondence

<p>
  <a href="https://thinking-inertia.github.io/"><img alt="Project Page" src="https://img.shields.io/badge/%F0%9F%8C%90%20Project-Page-5b9e1e"></a>
  <a href="https://arxiv.org/abs/2610.11765"><img alt="arXiv 2610.11765" src="https://img.shields.io/badge/arXiv-2610.11765-b31b1b"></a>
  <a href="LICENSE"><img alt="License MIT" src="https://img.shields.io/badge/License-MIT-238636"></a>
  <a href="https://huggingface.co/datasets/thinking-inertia/llm-nothinking-trajectories"><img alt="Data Trajectory" src="https://img.shields.io/badge/%F0%9F%A4%97%20Data-Trajectory-FFD21E"></a>
</p>

<em>Response-level evidence for when language models keep thinking after being told not to.</em>

</div>

## 📣 News

- **2026.10** — Public release of the evaluation and metric-validation code.
- **2026.09** — Our paper is accepted as a Poster at NeurIPS 2026!

## 🔎 Overview

We study whether a model that is instructed to answer directly still emits
visible pre-answer text. Each response is represented as a pre-answer field
`T` followed by a final answer `A`; assistant-prefilled M6 uses the first boxed
answer `A` and all remaining readable text `T`. The public evaluator supports six
intervention modes, multiple answer spaces, and benchmark-level accuracy and
format analysis.

The paper uses three complementary response-level measures:

- **ETR** — strict answer-only compliance, based on an empty `T`;
- **QRel.** — instruction-aware relevance between the question and `T`;
- **EIR** — visible explicit inference identified by a blinded language-model
  judge.

The central pattern is an answer-space staircase: visible inference generally
increases from Boolean to multiple-choice to open-ended answers, while stricter
no-thinking interventions reduce it within each answer space. These measures
are response-level observables; they do not claim to reveal latent cognition.

<p align="center">
  <img src="assets/overview.png" alt="ETR, EIR, QRel., and visible token length across answer spaces and intervention modes" width="96%">
</p>

## 📌 Key findings

- **Answer-space staircase.** The visible-inference ordering is `Bool < MCQ <
  Open`, with open-ended answers retaining the most explicit pre-answer work.
- **Thinking inertia.** Within each answer space, stricter no-thinking controls
  reduce visible inference (`M5 < M2`) but do not erase the answer-space effect.

Here, **M2** is native think-off (disabled thinking without an added output constraint), while **M5** is the strict answer-only prompt.

## ⚙️ Installation

```bash
git clone https://github.com/thinking-inertia/code.git
cd code
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

The evaluator needs Python 3.10 or newer. Model serving is external and can be
any OpenAI-compatible endpoint; no provider credential is stored here.

## 🚀 Quick start

Download the normalized benchmark inputs:

```bash
python -m thinking_inertia.download_datasets \
  --root-dir data \
  --overwrite
```

Run a small local evaluation of M1–M5:

```bash
python -m thinking_inertia.eval_thinking_spectrum \
  --root-dir data \
  --datasets boolq strategyqa mmlu mmlu_pro gsm8k math \
  --base-url http://127.0.0.1:8401/v1 \
  --model qwen3-4b \
  --api chat \
  --modes 1 2 3 4 5 \
  --max-samples 200 \
  --sample-strategy level_balanced \
  --seed 20260502 \
  --run-name qwen3_4b_spectrum_demo
```

The run writes local records and summaries to the requested run directory.
For a minimal parser/accuracy check, use:

```bash
python -m thinking_inertia.eval_nonreason --help
```

### M6: assistant prefilling

M6 continues an unfinished **assistant** response, `The answer is \boxed{`.
The user message contains only the task and the usual final-answer request.
For a local Qwen3 endpoint, render its matching tokenizer's think-off chat
sequence and send the prompt token IDs to the completions endpoint:

```bash
python -m thinking_inertia.eval_thinking_spectrum \
  --root-dir data \
  --datasets boolq strategyqa mmlu mmlu_pro gsm8k math \
  --base-url http://127.0.0.1:8401/v1 \
  --model qwen3-4b --provider-profile qwen3 --api completions \
  --modes 6 --m6-prefill qwen-completions \
  --m6-tokenizer /path/to/Qwen3-4B \
  --m6-accuracy-policy canonical_local \
  --max-tokens 8192 --temperature 0 \
  --max-samples 1000 --sample-strategy level_balanced --seed 20260502 \
  --run-name qwen3_4b_m6_prefill
```

Use the original checkpoint and the same tokenizer revision as the served
model. `--m6-tokenizer-revision` can pin a Hugging Face revision. The renderer
uses `enable_thinking=False`, `continue_final_message=True`, and
`add_generation_prompt=False`; it verifies that the input ends at the open
assistant box. It does not insert an end-of-message token after the prefix.
The 4B and 32B runs use `--m6-accuracy-policy canonical_local`; the 235B and
DeepSeek runs use `legacy_math_rescore`. These options preserve their benchmark
answer comparators. The default `evaluator` uses the repository's standard
comparator. Exact cohort reproduction requires the specified policy; the
canonical fallback retains the original normalization behavior. M1–M5 are unaffected by these M6 options.

For a DeepSeek chat endpoint that explicitly supports assistant prefix
continuation, use `--api chat --provider-profile deepseek_v4
--m6-prefill deepseek-prefix`. This sends a final assistant message with
`prefix: true` and disables thinking. Use the provider's prefix-enabled base
URL and a model that supports that interface. Ordinary chat and Responses
endpoints, including the GPT and Gemini routes, are not used for M6. A plain
assistant message without continuation support is insufficient.

M6 records retain the raw request, raw response, generated continuation, and
reconstructed full response. The metrics use these rules:

- **Acc.** scores only the first complete box introduced by the fixed prefix;
  later answer corrections do not replace it.
- **ETR** requires a nonempty, parseable first boxed answer and no remaining
  readable text. Whitespace or a single terminal `.` / `。` is ignored.
- **QRel. and EIR** use all text remaining after the initial answer declaration
  is removed, including separately returned readable reasoning fields. The
  full first box is removed, even if it contains an expression or explanation.
  An unclosed first box is retained in `T` and has zero Acc./ETR.

The evaluator's MiniLM compatibility scores are not the paper's QRel. Use the
Qwen3-Embedding-4B scorer and the EIR judge on the same residual Q/T pairs:

```bash
python -m thinking_inertia.m6_prefill \
  --input data/runs/qwen3_4b_m6_prefill/records.jsonl \
  --output outputs/m6_pairs.jsonl
python -m thinking_inertia.metrics.qrel.score_controlled_qrel \
  --input outputs/m6_pairs.jsonl --output-dir outputs/m6_qrel
python -m thinking_inertia.metrics.eir.judge_pairs \
  --input outputs/m6_pairs.jsonl --m6-residual-rubric \
  --model gpt-5.5 --concurrency 24 --batch-size 16 --max-chars 100000 \
  --output outputs/m6_eir.jsonl --errors outputs/m6_eir_errors.jsonl
```

The EIR command uses `OPENAI_API_KEY` and the selected judge endpoint. Failed
or filtered judge requests remain in the errors file, rather than receiving a
negative label; report the valid-label denominator and missing count. Use a
new run directory for M6. The default mode list includes M6, so specify
`--modes 1 2 3 4 5` when using an endpoint without prefilling. Older trajectories with a user-message answer prefix
represent **M6ᵘ**, not assistant prefilling; keep their labels and summaries
separate. The evaluator rejects resuming those records as M6.

## 🗂️ Repository layout

```text
src/thinking_inertia/   Dataset loading, response parsing, scoring, runners, and metrics
data/processed/         Normalized public benchmark inputs
outputs/                Local generated records and aggregates (ignored)
pyproject.toml          Package metadata and command-line entry points
requirements.txt        Runtime dependencies
```

## 🧪 Data and models

The files under `data/processed/` are normalized inputs derived from public
benchmarks. Users must follow each upstream dataset's license and terms. The
model endpoint, model weights, tokenizer, and API credentials are supplied by
the user at runtime and are not distributed by this repository.

## 📚 Citation

```bibtex
@article{lei2026thinkinginertia,
  title   = {Thinking Inertia: LLMs Keep Thinking When Told Not To},
  author  = {Lei, Dianqiao and Lin, Kevin Qinghong and Lu, Pan and Torr, Philip and Zou, James},
  year    = {2026},
  eprint  = {2610.11765},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2610.11765}
}
```
