"""Dedup, length limits, mixing ratios, and held-out groups."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from scripts.dataset.format import sequence_tokens, split_messages
from scripts.dataset.tasks import Sample
from scripts.gitlab.anonymize import Anonymizer
from scripts.rules.parsing import strip_thinking


def content_key(sample: Sample) -> str:
    blob = json.dumps({"task": sample.task, "messages": sample.messages}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def dedup(samples: list[Sample]) -> list[Sample]:
    seen = set()
    kept = []
    for sample in samples:
        key = content_key(sample)
        if key in seen:
            continue
        seen.add(key)
        kept.append(sample)
    return kept


def fit_length(samples: list[Sample], max_tokens: int, chars_per_token: float) -> list[Sample]:
    fitted = []
    for sample in samples:
        parts = split_messages(sample.messages, max_tokens, chars_per_token)
        if len(parts) == 1 and sequence_tokens(parts[0], chars_per_token) <= max_tokens:
            sample.messages = parts[0]
            fitted.append(sample)
            continue
        for index, messages in enumerate(parts):
            if sequence_tokens(messages, chars_per_token) > max_tokens:
                continue
            piece = Sample(
                id=f"{sample.id}-part-{index + 1}",
                task=sample.task,
                source=sample.source,
                machine=sample.machine,
                group=sample.group,
                kind=sample.kind,
                messages=messages,
                dpo=sample.dpo if index == 0 else None,
                gold=sample.gold,
                extra=dict(sample.extra),
            )
            fitted.append(piece)
    return fitted


def _rewrite_content(message: dict, fn) -> None:
    if isinstance(message.get("content"), str):
        message["content"] = fn(message["content"])


def apply_thinking(samples: list[Sample], mode: str) -> list[Sample]:
    """Strip or keep Qwen3 ``<think>`` blocks in every message, including DPO pairs.

    ``strip`` matches inference with thinking disabled. ``keep`` leaves the blocks
    so training and a thinking-enabled llama-server agree.
    """
    if mode not in {"strip", "keep"}:
        raise ValueError(mode)
    if mode == "keep":
        return samples
    for sample in samples:
        for message in sample.messages:
            _rewrite_content(message, strip_thinking)
        if not sample.dpo:
            continue
        for key in ("prompt", "chosen", "rejected"):
            for message in sample.dpo.get(key) or []:
                _rewrite_content(message, strip_thinking)
        if sample.gold:
            sample.gold = strip_thinking(sample.gold)
    return samples


def anonymize_samples(samples: list[Sample], *, strip_urls: bool) -> list[Sample]:
    anonymizer = Anonymizer(strip_urls=strip_urls)
    for sample in samples:
        reviewer = (sample.extra or {}).get("reviewer")
        if reviewer:
            anonymizer.register_author({"username": reviewer})
    for sample in samples:
        for message in sample.messages:
            if isinstance(message.get("content"), str):
                message["content"] = anonymizer.text(message["content"])
        if sample.dpo:
            for key in ("prompt", "chosen", "rejected"):
                for message in sample.dpo.get(key) or []:
                    if isinstance(message.get("content"), str):
                        message["content"] = anonymizer.text(message["content"])
        if sample.gold:
            sample.gold = anonymizer.text(sample.gold)
        reviewer = (sample.extra or {}).get("reviewer")
        if reviewer:
            sample.extra["reviewer"] = anonymizer.author_view({"username": reviewer}).get("username") or ""
    return samples


def is_held_out(group: str, ratio: float, seed: str = "finetune") -> bool:
    if ratio <= 0:
        return False
    if ratio >= 1:
        return True
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF < ratio


def assign_splits(samples: list[Sample], ratio: float, seed: str = "finetune") -> tuple[list[Sample], list[Sample]]:
    train, eval_rows = [], []
    for sample in samples:
        if is_held_out(sample.group, ratio, seed):
            eval_rows.append(sample)
        else:
            train.append(sample)
    return train, eval_rows


def mix_ratios(samples: list[Sample], ratios: dict[str, float]) -> list[Sample]:
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.task].append(sample)
    active = {task: weight for task, weight in ratios.items() if weight > 0 and grouped.get(task)}
    if not active:
        return []
    if len(active) == 1:
        task = next(iter(active))
        return sorted(grouped[task], key=lambda item: item.id)
    total = sum(active.values())
    weights = {task: weight / total for task, weight in active.items()}
    scale = min(len(grouped[task]) / weights[task] for task in active)
    picked = []
    for task, weight in weights.items():
        target = min(len(grouped[task]), max(1, int(scale * weight)))
        ordered = sorted(grouped[task], key=lambda item: item.id)
        picked.extend(ordered[:target])
    return picked


def prefer_source_models(samples: list[Sample], weights: dict[str, float]) -> list[Sample]:
    """Sort so higher-weighted source models are kept first when a task is downsampled."""
    if not weights:
        return samples

    def rank(sample: Sample) -> tuple:
        label = (sample.extra.get("source_model") or "").lower()
        weight = 1.0
        for pattern, value in weights.items():
            needle = pattern.lower().rstrip("*")
            if needle and needle in label:
                weight = max(weight, float(value))
        return (-weight, sample.id)

    return sorted(samples, key=rank)


def manifest_block(samples: list[Sample]) -> dict:
    by_task: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    by_machine: dict[str, int] = defaultdict(int)
    by_source_model: dict[str, int] = defaultdict(int)
    dpo = 0
    for sample in samples:
        by_task[sample.task] += 1
        by_source[sample.source] += 1
        by_machine[sample.machine or "unknown"] += 1
        model = sample.extra.get("source_model") or ""
        if model:
            by_source_model[model] += 1
        if sample.dpo:
            dpo += 1
    return {
        "samples": len(samples),
        "dpo": dpo,
        "by_task": dict(sorted(by_task.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_machine": dict(sorted(by_machine.items())),
        "by_source_model": dict(sorted(by_source_model.items())),
    }
