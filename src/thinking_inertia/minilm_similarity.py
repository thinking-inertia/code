#!/usr/bin/env python3
"""
Utilities for local MiniLM-based semantic similarity scoring.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def normalize_space(text: str) -> str:
    return " ".join(text.strip().split())


def split_explanation_sentences(text: str) -> list[str]:
    raw = normalize_space(text)
    if not raw:
        return []

    # Remove a few common reasoning wrappers without being aggressive.
    raw = raw.replace("<think>", " ").replace("</think>", " ")
    raw = raw.replace("<reasoning>", " ").replace("</reasoning>", " ")
    raw = normalize_space(raw)
    if not raw:
        return []

    parts = re.split(r"(?<=[.!?])\s+|\n+", raw)
    out: list[str] = []
    for part in parts:
        cleaned = normalize_space(part)
        if cleaned:
            out.append(cleaned)
    return out


class MiniLMSimilarityScorer:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        max_length: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "MiniLM scoring requires `torch` and `transformers`. "
                "Use an environment that has those packages installed."
            ) from exc

        self._torch = torch
        local_only = Path(model_path).expanduser().exists()
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
        self._model = AutoModel.from_pretrained(model_path, local_files_only=local_only)
        self._model.eval()
        self._max_length = max_length
        self._device = self._resolve_device(device)
        self._model.to(self._device)
        self._question_cache: dict[str, "torch.Tensor"] = {}

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if self._torch.cuda.is_available() else "cpu"
        return device

    def _mean_pool(
        self,
        last_hidden_state: "object",
        attention_mask: "object",
    ) -> "object":
        token_embeddings = last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * input_mask_expanded).sum(dim=1)
        denom = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def encode_texts(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
    ) -> "object":
        if not texts:
            return self._torch.empty((0, 384), dtype=self._torch.float32)

        outputs = []
        with self._torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                model_out = self._model(**encoded)
                pooled = self._mean_pool(model_out.last_hidden_state, encoded["attention_mask"])
                pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                outputs.append(pooled.cpu())
        return self._torch.cat(outputs, dim=0)

    def encode_question(
        self,
        question_text: str,
        *,
        batch_size: int = 32,
    ) -> "object":
        cached = self._question_cache.get(question_text)
        if cached is not None:
            return cached
        embedding = self.encode_texts([question_text], batch_size=batch_size)[0]
        self._question_cache[question_text] = embedding
        return embedding

    def score_question_to_sentences(
        self,
        question_text: str,
        sentences: list[str],
        *,
        batch_size: int = 32,
    ) -> tuple[list[float], float]:
        if not sentences:
            return [], 0.0

        question_vec = self.encode_question(question_text, batch_size=batch_size)
        sent_vecs = self.encode_texts(sentences, batch_size=batch_size)
        sims = self._torch.matmul(sent_vecs, question_vec).tolist()
        sims = [float(x) for x in sims]
        mean_sim = float(sum(sims) / len(sims)) if sims else 0.0
        return sims, mean_sim


__all__ = [
    "MiniLMSimilarityScorer",
    "normalize_space",
    "split_explanation_sentences",
]
