# EIR judge utility

This directory contains the reusable Level 3 judging component: the blind
rubric, JSON parser, and asynchronous judge client. The judge receives only
`(Q, T)` and returns one exclusive category plus a supporting span.

The categories are defined in `judge_rubric.py` and can be used with any
external corpus following the input schema below. Generated labels are written
to the ignored local `outputs/eir/` directory.

To judge an external corpus:

```bash
OPENAI_API_KEY=... python -m thinking_inertia.metrics.eir.judge_pairs \
  --input /path/to/corpus.jsonl \
  --output outputs/eir/judge_labels.jsonl
```

Each input row must contain `record_id`, `question`, `pre_answer_text`, and
`t_is_empty`. Empty texts are assigned the `empty` category by definition.
