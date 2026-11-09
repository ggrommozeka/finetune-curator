"""Orchestrate session packing, model calls, and rule merge."""

from __future__ import annotations

from scripts.rules.llm import request_rules
from scripts.rules.merge import merge_rules
from scripts.rules.parsing import ModelOutputError
from scripts.rules.sessions import build_transcripts


def propose_rules(
    rows: list[dict],
    decisions: dict | None,
    complete_fn,
    *,
    sources=("discarded", "low-score", "notes"),
    max_score: float = 0.6,
    max_sessions: int = 0,
    context_budget: int = 170_000,
    chars_per_token: float = 3.0,
    overlap_tokens: int = 400,
    batch: bool = True,
    disable_thinking: bool = True,
    attempts: int = 3,
    merge_threshold: float = 0.6,
    model: str | None = None,
    base_url: str | None = None,
) -> dict:
    """Return a proposals document. ``complete_fn(messages) -> str`` is the LLM call."""
    transcripts, source_ids = build_transcripts(
        rows,
        decisions,
        sources=sources,
        max_score=max_score,
        max_sessions=max_sessions,
        context_budget=context_budget,
        chars_per_token=chars_per_token,
        overlap_tokens=overlap_tokens,
        batch=batch,
    )
    collected: list[dict] = []
    errors: list[str] = []
    for index, transcript in enumerate(transcripts, 1):
        try:
            collected.extend(
                request_rules(
                    complete_fn,
                    transcript,
                    disable_thinking=disable_thinking,
                    attempts=attempts,
                )
            )
        except ModelOutputError as exc:
            errors.append(f"batch {index}/{len(transcripts)}: {exc}")

    merged = merge_rules(collected, threshold=merge_threshold)
    proposals = [{**rule, "kind": "rule", "source": "opencode"} for rule in merged]
    document = {
        "version": 2,
        "mode": "per-session",
        "model": model,
        "base_url": base_url,
        "context_budget": context_budget,
        "source_session_ids": source_ids,
        "batches": len(transcripts),
        "proposals": proposals,
        "rules": proposals,
    }
    if errors:
        document["errors"] = errors
    return document
