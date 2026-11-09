#!/usr/bin/env python3
"""Propose harness system-prompt rules from reviewed OpenCode sessions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from scripts.rules.endpoint import (
    EndpointError,
    fetch_model_ids,
    format_endpoint,
    pick_remote_model,
    resolve_endpoint,
)
from scripts.rules.llm import chat_complete, thinking_controls
from scripts.rules.opencode_config import (
    discover_config_files,
    iter_configured_models,
    load_merged_config,
    resolve_model_ref,
)
from scripts.rules.parsing import ModelOutputError
from scripts.rules.propose import propose_rules
from scripts.rules.sessions import load_jsonl, load_review_decisions
from scripts.rules.store import save_proposals


def _parse_sources(value: str) -> tuple[str, ...]:
    sources = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    known = {"discarded", "low-score", "notes"}
    unknown = [item for item in sources if item not in known]
    if unknown:
        raise SystemExit(
            f"Unknown --sources {unknown}. Use discarded, low-score, notes."
        )
    if not sources:
        raise SystemExit("--sources must list at least one of: discarded, low-score, notes")
    return sources


def _opencode_paths(args) -> list[Path]:
    explicit = Path(args.opencode_config).expanduser() if args.opencode_config else None
    return discover_config_files(
        Path.cwd(),
        Path.home(),
        os.environ.get("XDG_CONFIG_HOME"),
        explicit,
    )


def _load_opencode(args) -> dict:
    try:
        return load_merged_config(_opencode_paths(args))
    except FileNotFoundError as exc:
        raise SystemExit(f"OpenCode config not found: {exc}") from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Propose system-prompt rules from discarded, low-score, and annotated sessions. "
            "Uses any OpenAI-compatible chat API. The default endpoint is a local llama-server."
        )
    )
    parser.add_argument("--input", help="Scored session JSONL (LogMiner / export_opencode output)")
    parser.add_argument("--decisions", default="data/decisions.json")
    parser.add_argument("--output", default="data/rule_proposals.json")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (default: http://localhost:8080/v1 for llama-server)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. Optional for llama-server. Prefer FINETUNE_CURATOR_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id sent to the server. If omitted, use the only model from GET /v1/models.",
    )
    parser.add_argument(
        "--from-opencode",
        default=None,
        metavar="PROVIDER/MODEL",
        help="Use base URL, API key, and model id from opencode.json (CLI flags and env still win)",
    )
    parser.add_argument(
        "--opencode-config",
        default=None,
        help="Explicit opencode.json or opencode.jsonc. Default: project file, else ~/.config/opencode/",
    )
    parser.add_argument("--list-opencode-models", action="store_true")
    parser.add_argument("--list-models", action="store_true", help="GET /models on the resolved server and exit")
    parser.add_argument("--print-endpoint", action="store_true", help="Print the resolved endpoint and exit")
    parser.add_argument(
        "--context-budget",
        type=int,
        default=170_000,
        help=(
            "Max estimated input tokens per request (default: 170000). "
            "Leaves headroom in a ~200k llama-server context so most sessions are sent whole."
        ),
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=3.0,
        help="Token estimate divisor. Lower values are more conservative (default: 3).",
    )
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=600, help="HTTP timeout in seconds (default: 600)")
    parser.add_argument("--attempts", type=int, default=3, help="JSON parse retries per batch (default: 3)")
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=400,
        help="Overlap when an oversized session is split (default: 400)",
    )
    parser.add_argument("--no-batch", action="store_true", help="Send each session chunk in its own request")
    parser.add_argument(
        "--think",
        action="store_true",
        help="Allow Qwen3 thinking. By default thinking is disabled (/no_think and enable_thinking=false).",
    )
    parser.add_argument(
        "--no-template-kwargs",
        action="store_true",
        help="Do not send chat_template_kwargs. /no_think is still appended unless --think is set.",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=0.6,
        help="Include sessions with score <= this when source low-score is enabled (default: 0.6)",
    )
    parser.add_argument(
        "--sources",
        default="discarded,low-score,notes",
        help="Comma-separated: discarded, low-score, notes (default: all three)",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=0,
        help="Cap how many source sessions are sent (0 = no cap). Discards and notes come first.",
    )
    parser.add_argument(
        "--mock-response",
        default=None,
        help="Path to a file used as the assistant reply for every batch (no network). For tests and smoke runs.",
    )
    parser.add_argument("--merge-threshold", type=float, default=0.6, help="Jaccard threshold for merging rules")
    return parser


def _resolve(args):
    opencode_base = opencode_key = opencode_model = None
    if args.from_opencode:
        config = _load_opencode(args)
        try:
            picked = resolve_model_ref(config, args.from_opencode, os.environ)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        opencode_base = picked.base_url
        opencode_key = picked.api_key
        opencode_model = picked.api_model
    return resolve_endpoint(
        cli_base_url=args.base_url,
        cli_api_key=args.api_key,
        cli_model=args.model,
        env=os.environ,
        opencode_base_url=opencode_base,
        opencode_api_key=opencode_key,
        opencode_model=opencode_model,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.context_budget <= 0:
        raise SystemExit("--context-budget must be positive")
    if args.chars_per_token <= 0:
        raise SystemExit("--chars-per-token must be positive")

    if args.list_opencode_models:
        config = _load_opencode(args)
        models = iter_configured_models(config, os.environ)
        if not models:
            print("No models found in OpenCode config.")
            return
        for item in models:
            base = item.base_url or "(no base URL)"
            key = "key=set" if item.api_key else "key=missing"
            print(f"{item.ref}  api_model={item.api_model}  base={base}  {key}")
        return

    endpoint = _resolve(args)
    if args.print_endpoint:
        print(format_endpoint(endpoint))
        return

    if args.list_models:
        try:
            ids = fetch_model_ids(endpoint.base_url, endpoint.api_key, timeout=min(args.timeout, 60))
        except EndpointError as exc:
            raise SystemExit(str(exc)) from exc
        if not ids:
            print("No models.")
            return
        for model_id in ids:
            print(model_id)
        return

    if not args.input:
        raise SystemExit("--input is required")

    if args.context_budget < 8000:
        print(
            f"Warning: context budget {args.context_budget} is small; long sessions will be split. "
            "The default 170000 assumes a ~200k server context.",
            file=sys.stderr,
        )

    if endpoint.model is None and not args.mock_response:
        try:
            ids = fetch_model_ids(endpoint.base_url, endpoint.api_key, timeout=min(args.timeout, 60))
            endpoint.model = pick_remote_model(ids)
            endpoint.model_source = "server:/models"
        except EndpointError as exc:
            raise SystemExit(str(exc)) from exc

    if args.mock_response:
        mock_text = Path(args.mock_response).read_text(encoding="utf-8")
        endpoint.model = endpoint.model or "mock"
        endpoint.model_source = "mock-response"

        def complete_fn(messages):
            return mock_text
    else:
        _, extra = thinking_controls(not args.think)
        if args.no_template_kwargs:
            extra = {}

        def complete_fn(messages, extra_body=extra):
            return chat_complete(
                endpoint,
                messages,
                temperature=args.temperature,
                max_tokens=args.max_output_tokens,
                timeout=args.timeout,
                extra_body=extra_body or None,
            )

    print(format_endpoint(endpoint))
    rows = load_jsonl(Path(args.input))
    if not rows:
        raise SystemExit(f"No sessions in {args.input}")
    decisions_path = Path(args.decisions)
    if not decisions_path.exists():
        print(f"No decisions file at {decisions_path}; using scores and labels only.")
    review = load_review_decisions(decisions_path)

    document = propose_rules(
        rows,
        review,
        complete_fn,
        sources=_parse_sources(args.sources),
        max_score=args.max_score,
        max_sessions=args.max_sessions,
        context_budget=args.context_budget,
        chars_per_token=args.chars_per_token,
        overlap_tokens=args.overlap_tokens,
        batch=not args.no_batch,
        disable_thinking=not args.think,
        attempts=args.attempts,
        merge_threshold=args.merge_threshold,
        model=endpoint.model,
        base_url=endpoint.base_url,
    )
    out = Path(args.output)
    save_proposals(out, document)
    print(
        f"Selected {len(document['source_session_ids'])} sessions "
        f"in {document['batches']} batch(es); context budget {args.context_budget} tokens"
    )
    print(f"Rules: {len(document['rules'])} after merge")
    for error in document.get("errors") or []:
        print(f"Failure: {error}", file=sys.stderr)
    print(f"Wrote: {out}")
    if document.get("errors"):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except ModelOutputError as exc:
        raise SystemExit(str(exc)) from exc
