#!/usr/bin/env python3
"""Score a held-out dataset: review precision/recall and style similarity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.dataset.evaluate import load_jsonl, review_metrics, style_metrics
from scripts.rules.endpoint import resolve_endpoint
from scripts.rules.llm import chat_complete, thinking_controls


def _live_complete():
    endpoint = resolve_endpoint(env=dict(os.environ))
    if not endpoint.model:
        raise SystemExit("Set FINETUNE_CURATOR_MODEL, or pass --mock-response.")
    _suffix, extra = thinking_controls(True)

    def complete(messages):
        return chat_complete(
            endpoint,
            messages,
            temperature=0.1,
            max_tokens=1024,
            timeout=600,
            extra_body=extra or None,
        )

    return complete


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out review and style samples.")
    parser.add_argument("--eval", default="data/finetune/eval.jsonl")
    parser.add_argument("--mock-response", default=None, help="Canned model reply. No network.")
    parser.add_argument("--judge", action="store_true", help="Ask the same client to judge matches.")
    args = parser.parse_args(argv)
    path = Path(args.eval)
    if not path.exists():
        raise SystemExit(f"Eval file not found: {path}")
    rows = load_jsonl(path)
    if args.mock_response:
        canned = Path(args.mock_response).read_text(encoding="utf-8") if Path(args.mock_response).is_file() else args.mock_response

        def complete(_messages, canned=canned):
            return canned

        judge = complete if args.judge else None
    else:
        complete = _live_complete()
        judge = complete if args.judge else None
    report = {"review": review_metrics(rows, complete, judge), "style": style_metrics(rows, complete, judge)}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
