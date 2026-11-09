#!/usr/bin/env python3
"""Mine recurring patterns across OpenCode histories and GitLab reviews.

Import is incremental. Annotation, clustering, and recommendations run locally
and can be repeated. The default model endpoint is a local llama-server.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.patterns.annotate import annotate_pending
from scripts.patterns.cluster import cluster_annotations, embed_texts
from scripts.patterns.import_sources import import_gitlab_traces, import_path
from scripts.patterns.monitor import report, rollback, snapshot_version
from scripts.patterns.recommend import recommend_pending
from scripts.patterns.store import PatternStore
from scripts.rules.export import export_approved, export_skills
from scripts.rules.llm import chat_complete, thinking_controls
from scripts.rules.parsing import ModelOutputError


def _store(path: str) -> PatternStore:
    return PatternStore(path)


def _complete_from_args(args):
    if args.mock_response:
        raw = Path(args.mock_response).read_text(encoding="utf-8")

        def complete(_messages):
            return raw

        return complete, "mock"
    from scripts.rules.endpoint import fetch_model_ids, pick_remote_model, resolve_endpoint

    endpoint = resolve_endpoint(
        cli_base_url=args.base_url,
        cli_model=args.model,
        env=os.environ,
    )
    if endpoint.model is None:
        endpoint.model = pick_remote_model(fetch_model_ids(endpoint.base_url, endpoint.api_key))
    _suffix, extra = thinking_controls(not args.think)

    def complete(messages, extra_body=extra):
        return chat_complete(
            endpoint,
            messages,
            temperature=0.1,
            max_tokens=2048,
            timeout=args.timeout,
            extra_body=extra_body or None,
        )

    return complete, endpoint.model or "local"


def _add_llm(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--mock-response", default=None, help="Canned model reply, no network")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--timeout", type=float, default=600)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine cross-session patterns into rules or skills.")
    sub = parser.add_subparsers(dest="command", required=True)

    imp = sub.add_parser("import", help="Ingest a JSONL export or an OpenCode sqlite DB")
    imp.add_argument("--store", default="data/patterns")
    imp.add_argument("--source", required=True, help="Machine or corpus label")
    imp.add_argument("--input", default=None, help="LogMiner JSONL or OpenCode sqlite path")
    imp.add_argument("--gitlab-store", default=None, help="Also ingest linked GitLab traces from this store")

    ann = sub.add_parser("annotate", help="Annotate sessions that do not have a cached annotation")
    ann.add_argument("--store", default="data/patterns")
    ann.add_argument("--refresh", action="store_true")
    _add_llm(ann)

    clus = sub.add_parser("cluster", help="Cluster annotations. Singletons below --min-size are noise.")
    clus.add_argument("--store", default="data/patterns")
    clus.add_argument("--min-size", type=int, default=3)
    clus.add_argument("--threshold", type=float, default=0.34)
    clus.add_argument("--embeddings-url", default=None)
    clus.add_argument("--embeddings-model", default=None)

    rec = sub.add_parser("recommend", help="Propose a rule or a skill per cluster")
    rec.add_argument("--store", default="data/patterns")
    rec.add_argument("--proposals", default="data/rule_proposals.json")
    rec.add_argument("--existing-rules", default=None, help="JSON file of current rules to conflict-check")
    _add_llm(rec)

    exp = sub.add_parser("export", help="Write approved rules and skills, and record a version")
    exp.add_argument("--store", default="data/patterns")
    exp.add_argument("--proposals", default="data/rule_proposals.json")
    exp.add_argument("--decisions", default="data/rule_decisions.json")
    exp.add_argument("--rules-md", default="data/approved_rules.md")
    exp.add_argument("--rules-json", default="data/approved_rules.json")
    exp.add_argument("--skills-dir", default="skills")
    exp.add_argument("--version", default=None)

    rep = sub.add_parser("report", help="Compare session metrics before and after each adopted version")
    rep.add_argument("--store", default="data/patterns")
    rep.add_argument("--decisions", default=None)
    rep.add_argument("--error-type", default=None)

    back = sub.add_parser("rollback", help="Restore a previous exported version")
    back.add_argument("--store", default="data/patterns")
    back.add_argument("--version", required=True)
    return parser


def _existing_rules(path: str | None) -> list[str]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rules = data.get("rules") if isinstance(data, dict) else data
    return [str(rule.get("text") or "") for rule in rules or [] if isinstance(rule, dict)]


def _cmd_export(args) -> None:
    """Export through the shared rules/skills writer, then snapshot that version."""
    proposals = Path(args.proposals)
    decisions = Path(args.decisions)
    if not proposals.exists():
        raise SystemExit(f"Proposals file not found: {proposals}")
    rules_md = Path(args.rules_md)
    rules_json = Path(args.rules_json)
    skills_dir = Path(args.skills_dir)
    result = export_approved(proposals, decisions, rules_md, rules_json)
    skills = export_skills(proposals, decisions, skills_dir)
    store = _store(args.store)
    try:
        version = args.version or f"v{len(store.versions()) + 1}"
        snapshot_version(
            store,
            version,
            [rules_md, rules_json, skills_dir],
            note=f"rules={result['approved']} skills={skills['skills']}",
        )
    finally:
        store.close()
    print(f"Rules: {result['approved']} -> {rules_md}")
    print(f"Skills: {skills['skills']} -> {skills_dir}")
    print(f"Version: {version}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "import":
            store = _store(args.store)
            try:
                if args.input:
                    stats = import_path(store, Path(args.input), source_label=args.source)
                    print(
                        f"{args.source}: added {stats['added']} attached {stats['attached']} "
                        f"skipped {stats['skipped']} ({stats['schema']})"
                    )
                if args.gitlab_store:
                    stats = import_gitlab_traces(store, Path(args.gitlab_store))
                    print(
                        f"gitlab: added {stats['added']} attached {stats['attached']} "
                        f"skipped {stats['skipped']} ({stats['schema']})"
                    )
            finally:
                store.close()
            return
        if args.command == "annotate":
            complete, model = _complete_from_args(args)
            store = _store(args.store)
            try:
                stats = annotate_pending(store, complete, model=model, refresh=args.refresh)
            finally:
                store.close()
            print(f"Annotated: {stats['annotated']} skipped: {stats['skipped']} failed: {stats['failed']}")
            return
        if args.command == "cluster":
            embeddings = None
            if args.embeddings_url and args.embeddings_model:
                store = _store(args.store)
                try:
                    texts = [
                        " ".join(
                            str(row["annotation"].get(key) or "")
                            for key in ("task_type", "error_type", "user_correction", "what_worked")
                        )
                        for row in store.annotated_rows()
                    ]
                finally:
                    store.close()
                embeddings = embed_texts(
                    texts,
                    base_url=args.embeddings_url,
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    model=args.embeddings_model,
                )
            store = _store(args.store)
            try:
                stats = cluster_annotations(
                    store,
                    min_size=args.min_size,
                    threshold=args.threshold,
                    embeddings=embeddings,
                )
            finally:
                store.close()
            print(f"Clusters: {stats['clusters']} noise: {stats['noise']}")
            return
        if args.command == "recommend":
            complete, _model = _complete_from_args(args)
            store = _store(args.store)
            try:
                stats = recommend_pending(
                    store,
                    complete,
                    proposals_path=Path(args.proposals),
                    existing_rules=_existing_rules(args.existing_rules),
                )
            finally:
                store.close()
            print(f"Proposals: {stats['proposals']}")
            return
        if args.command == "export":
            _cmd_export(args)
            return
        if args.command == "report":
            store = _store(args.store)
            try:
                rows = report(
                    store,
                    decisions_path=Path(args.decisions) if args.decisions else None,
                    error_type=args.error_type,
                )
            finally:
                store.close()
            print(json.dumps(rows, ensure_ascii=False, indent=2))
            return
        if args.command == "rollback":
            store = _store(args.store)
            try:
                restored = rollback(store, args.version)
            finally:
                store.close()
            print("Restored:")
            for path in restored:
                print(f"  {path}")
            return
    except ModelOutputError as exc:
        raise SystemExit(str(exc)) from exc
    raise SystemExit(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
