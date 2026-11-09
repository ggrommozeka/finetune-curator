#!/usr/bin/env python3
"""Build one Unsloth chat dataset from OpenCode sessions and GitLab history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.datasets.build import LARGE_MAX_TOKENS, REVIEW_CONTEXT_LINES, REVIEW_MAX_TOKENS, build_samples, write_dataset
from scripts.datasets.format import CHAT_TEMPLATES
from scripts.ingest.gitlab.signals import load_signal_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mix agentic, style, and review samples into one dataset.")
    parser.add_argument("--input", default="data/raw_scored.jsonl", help="OpenCode / LogMiner JSONL")
    parser.add_argument("--decisions", default="data/decisions.json")
    parser.add_argument("--gitlab-store", default="data/gitlab")
    parser.add_argument("--patterns-store", default="data/patterns", help="Optional machine labels")
    parser.add_argument("--output-dir", default="data/finetune")
    parser.add_argument("--profile", default=None, help="Training profile JSON. Dataset task, variant, and token budget come from it unless overridden.")
    parser.add_argument("--variant", choices=["full", "short"], default=None, help="short keeps style pairs and step-level DPO only.")
    parser.add_argument("--task", choices=["all", "author", "agentic", "style", "review"], default=None)
    parser.add_argument("--source-model", action="append", default=None, help="Keep OpenCode trajectories whose model label contains this text. Repeatable.")
    parser.add_argument("--model-weight", action="append", default=None, help="pattern=weight, for example claude=3. Preferred models are kept first.")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Checkpoint id for this target. Author defaults to Qwen/Qwen3.5-27B. Review defaults to Qwen/Qwen3.5-9B. Ornith-1.0-9B is the other reviewer candidate.",
    )
    parser.add_argument(
        "--chat-template",
        choices=list(CHAT_TEMPLATES),
        default="qwen3",
        help="Agentic and style default to Qwen3 tool-calling ChatML. Review can select another 9B family.",
    )
    parser.add_argument(
        "--thinking",
        choices=["strip", "keep"],
        default="strip",
        help="Drop or keep <think> blocks. strip matches llama-server with thinking disabled.",
    )
    parser.add_argument("--agentic", type=float, default=0.5)
    parser.add_argument("--style", type=float, default=0.25)
    parser.add_argument("--review", type=float, default=0.25)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=f"Per-sample budget. Default {LARGE_MAX_TOKENS} for agentic/style/all, {REVIEW_MAX_TOKENS} for --task review.",
    )
    parser.add_argument(
        "--review-context-lines",
        type=int,
        default=REVIEW_CONTEXT_LINES,
        help="Lines of diff kept on each side of a review anchor. Used by --task review.",
    )
    parser.add_argument("--chars-per-token", type=float, default=3.0)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--min-score", type=float, default=0.6, help="Unlabeled sessions at or above this score count as successful")
    parser.add_argument("--anonymize", action="store_true")
    parser.add_argument("--anonymize-urls", action="store_true")
    parser.add_argument("--config", default=None, help="gitlab.json with the team-lead aliases")
    parser.add_argument("--team-lead", action="append", default=None)
    parser.add_argument("--mrr-pattern", default=None)
    parser.add_argument("--seed", default="finetune")
    return parser


def _conventions(args) -> str:
    path = Path("data/rule_proposals.json")
    decisions = Path("data/rule_decisions.json")
    if not path.exists():
        return ""
    from scripts.proposals.store import approved_rules, load_proposals, load_rule_decisions

    rules = approved_rules(load_proposals(path), load_rule_decisions(decisions))
    return "\n".join(f"- {rule['text']}" for rule in rules[:12])


def load_profile(path: str | None) -> dict:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must be a JSON object")
    return data


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    profile = load_profile(args.profile)
    dataset = profile.get("dataset") or {}
    args.task = args.task or dataset.get("task") or "all"
    variant = args.variant or dataset.get("variant") or "full"
    if args.max_tokens is None and dataset.get("max_tokens"):
        args.max_tokens = int(dataset["max_tokens"])
    if args.base_model is None and profile.get("base_model"):
        args.base_model = profile["base_model"]
    if args.task == "author":
        tasks = {"agentic", "style"}
    elif args.task == "all":
        tasks = None
    else:
        tasks = {args.task}
    weights = {}
    for item in args.model_weight or []:
        if "=" not in item:
            raise SystemExit(f"--model-weight must look like claude=3, got {item}")
        pattern, value = item.split("=", 1)
        weights[pattern.strip()] = float(value)
    if args.max_tokens is None:
        args.max_tokens = REVIEW_MAX_TOKENS if args.task == "review" else LARGE_MAX_TOKENS
    ratios = {"agentic": args.agentic, "style": args.style, "review": args.review}
    if tasks:
        ratios = {name: (1.0 if name in tasks else 0.0) for name in ratios}
    if sum(ratios.values()) <= 0:
        raise SystemExit("At least one task ratio must be positive.")
    config = load_signal_config(args.config, team_leads=args.team_lead, mrr_pattern=args.mrr_pattern)
    opencode = Path(args.input) if args.input else None
    gitlab = Path(args.gitlab_store) if args.gitlab_store else None
    patterns = Path(args.patterns_store) if args.patterns_store else None
    result = build_samples(
        opencode_path=opencode if opencode and opencode.exists() else None,
        decisions_path=Path(args.decisions) if args.decisions else None,
        gitlab_root=gitlab if gitlab and (gitlab / "store.db").exists() else None,
        patterns_root=patterns if patterns and (patterns / "store.db").exists() else None,
        ratios=ratios,
        max_tokens=args.max_tokens,
        chars_per_token=args.chars_per_token,
        eval_ratio=args.eval_ratio,
        anonymize=args.anonymize,
        anonymize_urls=args.anonymize_urls,
        min_score=args.min_score,
        signal_config=config,
        seed=args.seed,
        tasks=tasks,
        chat_template=args.chat_template,
        base_model=args.base_model,
        review_context_lines=args.review_context_lines if args.task == "review" else None,
        thinking=args.thinking,
        source_models=set(args.source_model or []) or None,
        model_weights=weights,
        conventions=_conventions(args),
        variant=variant,
    )
    written = write_dataset(result, Path(args.output_dir))
    print(json.dumps({"manifest": result["manifest"], "files": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
