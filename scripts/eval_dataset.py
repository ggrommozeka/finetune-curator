#!/usr/bin/env python3
"""Score a held-out dataset: review precision/recall and style similarity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.training.evaluate import (
    author_metrics,
    compare_reviewers,
    false_positive_rate,
    load_jsonl,
    review_metrics,
    style_metrics,
)
from scripts.llm.endpoint import resolve_endpoint
from scripts.rules.llm import chat_complete, thinking_controls


def _complete_for(spec: dict):
    if spec.get("mock") is not None:
        canned = spec["mock"]

        def complete(_messages, canned=canned):
            return canned

        return complete
    from scripts.llm.endpoint import Endpoint

    endpoint = Endpoint(
        base_url=spec["base_url"],
        api_key=spec.get("api_key") or None,
        model=spec["model"],
        base_url_source="cli",
        model_source="cli",
        api_key_source="cli",
    )
    _suffix, extra = thinking_controls(True)

    def complete(messages, endpoint=endpoint, extra=extra):
        return chat_complete(
            endpoint,
            messages,
            temperature=0.1,
            max_tokens=1024,
            timeout=600,
            extra_body=extra or None,
        )

    return complete


def load_model_specs(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [{"name": name, **spec} if isinstance(spec, dict) else {"name": name, "mock": spec} for name, spec in data.items()]
    if not isinstance(data, list) or not data:
        raise SystemExit(f"{path} must be a non-empty list of model specs")
    for spec in data:
        if "name" not in spec:
            raise SystemExit("Each model spec needs a name")
        if spec.get("mock") is None and not (spec.get("base_url") and spec.get("model")):
            raise SystemExit(f"{spec['name']} needs base_url and model, or mock")
    return data


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
    parser.add_argument("--eval-dpo", default=None, help="DPO jsonl for Model A correction rate. Defaults to eval_dpo.jsonl next to --eval.")
    parser.add_argument("--mock-response", default=None, help="Canned model reply. No network.")
    parser.add_argument(
        "--models",
        default=None,
        help="JSON file of models to compare on the same held-out MR rows. Each item has name plus base_url/model, or mock.",
    )
    parser.add_argument("--judge", action="store_true", help="Ask the same client to judge matches.")
    args = parser.parse_args(argv)
    path = Path(args.eval)
    if not path.exists():
        raise SystemExit(f"Eval file not found: {path}")
    rows = load_jsonl(path)
    if args.models:
        specs = load_model_specs(Path(args.models))
        completers = {spec["name"]: _complete_for(spec) for spec in specs}
        judge = None
        if args.judge:
            judge = next(iter(completers.values()))
        dpo_path = Path(args.eval_dpo) if args.eval_dpo else path.with_name("eval_dpo.jsonl")
        dpo_rows = load_jsonl(dpo_path) if dpo_path.exists() else []
        report = {
            "held_out_mr": compare_reviewers(rows, completers, judge),
            "model_a": {name: author_metrics(dpo_rows, rows, fn) for name, fn in completers.items()},
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.mock_response:
        canned = Path(args.mock_response).read_text(encoding="utf-8") if Path(args.mock_response).is_file() else args.mock_response

        def complete(_messages, canned=canned):
            return canned

        judge = complete if args.judge else None
    else:
        complete = _live_complete()
        judge = complete if args.judge else None
    dpo_path = Path(args.eval_dpo) if args.eval_dpo else path.with_name("eval_dpo.jsonl")
    dpo_rows = load_jsonl(dpo_path) if dpo_path.exists() else []
    report = {
        "model_a": author_metrics(dpo_rows, rows, complete),
        "model_b": {
            **review_metrics(rows, complete, judge),
            **false_positive_rate(rows, complete),
        },
        "style": style_metrics(rows, complete, judge),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
