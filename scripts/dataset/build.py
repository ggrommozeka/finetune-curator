"""Assemble the mixed SFT and DPO sets."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.dataset.format import CHAT_TEMPLATES, render_chat
from scripts.dataset.mix import (
    anonymize_samples,
    apply_thinking,
    assign_splits,
    dedup,
    fit_length,
    manifest_block,
    mix_ratios,
    prefer_source_models,
)
from scripts.dataset.tasks import from_gitlab_store, from_opencode, load_machine_labels
from scripts.gitlab.signals import SignalConfig, load_signal_config

DEFAULT_RATIOS = {"agentic": 0.5, "style": 0.25, "review": 0.25}
LARGE_MAX_TOKENS = 32_000
REVIEW_MAX_TOKENS = 16_384
REVIEW_CONTEXT_LINES = 20


def build_samples(
    *,
    opencode_path: Path | None = None,
    decisions_path: Path | None = None,
    gitlab_root: Path | None = None,
    patterns_root: Path | None = None,
    ratios: dict[str, float] | None = None,
    max_tokens: int = 32_000,
    chars_per_token: float = 3.0,
    eval_ratio: float = 0.1,
    anonymize: bool = False,
    anonymize_urls: bool = False,
    min_score: float = 0.6,
    signal_config: SignalConfig | None = None,
    seed: str = "finetune",
    tasks: set[str] | None = None,
    chat_template: str = "qwen3",
    review_context_lines: int | None = None,
    thinking: str = "strip",
    source_models: set[str] | None = None,
    model_weights: dict[str, float] | None = None,
    conventions: str = "",
) -> dict:
    if chat_template not in CHAT_TEMPLATES:
        raise ValueError(f"Unknown chat template {chat_template!r}")
    ratios = dict(ratios or DEFAULT_RATIOS)
    if tasks:
        ratios = {task: (1.0 if task in tasks else 0.0) for task in DEFAULT_RATIOS}
    machines = load_machine_labels(patterns_root) if patterns_root else {}
    samples = []
    if opencode_path and (not tasks or "agentic" in tasks):
        samples.extend(
            from_opencode(opencode_path, decisions_path, machines=machines, min_score=min_score)
        )
    if gitlab_root and (not tasks or "style" in tasks or "review" in tasks):
        samples.extend(
            from_gitlab_store(
                gitlab_root,
                signal_config or load_signal_config(),
                review_context_lines=review_context_lines if tasks == {"review"} else None,
                conventions=conventions,
                context_tokens=max_tokens if tasks == {"review"} else 16_384,
            )
        )
    if tasks:
        samples = [sample for sample in samples if sample.task in tasks]
    if source_models:
        allowed = {item.lower() for item in source_models}
        samples = [
            sample
            for sample in samples
            if sample.task != "agentic"
            or any(item in (sample.extra.get("source_model") or "").lower() for item in allowed)
        ]
    samples = prefer_source_models(samples, model_weights or {})
    samples = dedup(samples)
    samples = apply_thinking(samples, thinking)
    samples = fit_length(samples, max_tokens, chars_per_token)
    if anonymize or anonymize_urls:
        samples = anonymize_samples(samples, strip_urls=anonymize_urls or anonymize)
    train, held = assign_splits(samples, eval_ratio, seed)
    train = mix_ratios(train, ratios)
    held = mix_ratios(held, ratios)
    return {
        "train": train,
        "eval": held,
        "manifest": {
            "chat_template": chat_template,
            "tool_format": "qwen3" if chat_template == "qwen3" else chat_template,
            "thinking": thinking,
            "target_model": "qwen3-9b" if tasks == {"review"} else "qwen3-27b",
            "tasks": sorted(tasks) if tasks else ["agentic", "style", "review"],
            "max_tokens": max_tokens,
            "review_context_lines": review_context_lines,
            "eval_ratio": eval_ratio,
            "ratios": ratios,
            "train": manifest_block(train),
            "eval": manifest_block(held),
            "template_preview": render_chat(
                [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "ping"}],
                chat_template,
            ),
        },
    }


def write_dataset(result: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train.jsonl",
        "eval": output_dir / "eval.jsonl",
        "train_dpo": output_dir / "train_dpo.jsonl",
        "eval_dpo": output_dir / "eval_dpo.jsonl",
        "manifest": output_dir / "manifest.json",
    }
    for split in ("train", "eval"):
        rows = [sample.as_sft() for sample in result[split]]
        dpo_rows = [row for sample in result[split] if (row := sample.as_dpo())]
        paths[split].write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        dpo_name = "train_dpo" if split == "train" else "eval_dpo"
        paths[dpo_name].write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in dpo_rows),
            encoding="utf-8",
        )
    paths["manifest"].write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {key: str(path) for key, path in paths.items()}
