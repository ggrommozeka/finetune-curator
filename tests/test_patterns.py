"""Pattern mining without network: import, cache, cluster, recommend, export, rollback."""

import json
import sqlite3
from pathlib import Path

from scripts.analysis.annotate import annotate_pending
from scripts.analysis.cluster import cluster_annotations
from scripts.analysis.import_sources import import_gitlab_traces, import_path
from scripts.analysis.monitor import report, rollback
from scripts.analysis.recommend import recommend_cluster
from scripts.analysis.skills import skill_id
from scripts.analysis.store import PatternStore
from scripts.llm.parsing import extract_json

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION = (ROOT / "tests" / "fixtures" / "pattern_annotation.json").read_text(encoding="utf-8")
RECOMMENDATION = (ROOT / "tests" / "fixtures" / "pattern_recommendation.json").read_text(encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _session(origin: str, text: str, created_at: str, score: float = 0.2) -> dict:
    return {
        "id": origin,
        "created_at": created_at,
        "score": score,
        "project": "api",
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": "ran pytest again"},
        ],
    }


def test_import_dedupes_incremental_and_sqlite_schemas(tmp_path: Path):
    store = PatternStore(tmp_path / "patterns")
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    _jsonl(first, [_session("a1", "fix the flaky test", "2024-01-01T00:00:00+00:00")])
    _jsonl(second, [_session("b9", "fix the flaky test", "2024-01-01T00:00:00+00:00"), _session("b2", "rename the helper", "2024-02-01T00:00:00+00:00")])
    stats = import_path(store, first, source_label="laptop-a")
    assert stats == {"added": 1, "attached": 0, "skipped": 0, "schema": "logminer"}
    again = import_path(store, first, source_label="laptop-a")
    assert again["skipped"] == 1 and again["added"] == 0
    merged = import_path(store, second, source_label="laptop-b")
    assert merged["added"] == 1
    assert merged["attached"] == 1
    sessions = store.sessions()
    assert len(sessions) == 2
    shared = next(row for row in sessions if "flaky" in row["messages"][0]["content"])
    assert shared["sources"] == ["laptop-a", "laptop-b"]

    v1 = tmp_path / "v1.db"
    conn = sqlite3.connect(v1)
    conn.execute("CREATE TABLE session (id TEXT)")
    conn.execute("CREATE TABLE message (id INTEGER, session_id TEXT, role TEXT, content TEXT)")
    conn.execute("INSERT INTO message VALUES (1, 's', 'user', 'migrate the schema')")
    conn.commit()
    conn.close()
    v1_stats = import_path(store, v1, source_label="desktop")
    assert v1_stats["schema"] == "opencode-v1"
    assert v1_stats["added"] == 1

    v2 = tmp_path / "v2.db"
    conn = sqlite3.connect(v2)
    conn.execute("CREATE TABLE session (id TEXT, title TEXT, project_id TEXT, time_created TEXT)")
    conn.execute("CREATE TABLE message (id TEXT, session_id TEXT, data TEXT)")
    conn.execute(
        "CREATE TABLE part (id TEXT, session_id TEXT, time_created TEXT, data TEXT)"
    )
    conn.execute("INSERT INTO session VALUES ('p1', 'Parts', 'proj', '2024-03-01T00:00:00Z')")
    conn.execute(
        "INSERT INTO part VALUES ('part1', 'p1', '2024-03-01T00:00:01Z', ?)",
        (json.dumps({"type": "text", "role": "user", "text": "use the new parts table"}),),
    )
    conn.commit()
    conn.close()
    v2_stats = import_path(store, v2, source_label="desktop")
    assert v2_stats["schema"] == "opencode-v2"
    assert v2_stats["added"] == 1
    missing = import_gitlab_traces(store, tmp_path / "no-gitlab")
    assert missing["schema"] == "gitlab-missing"
    from scripts.ingest.gitlab.store import GitlabStore
    from scripts.ingest.gitlab.sync import process_store

    gitlab = GitlabStore(tmp_path / "gitlab")
    gitlab.save_bundle(
        {
            "id": 1,
            "iid": 3,
            "project_id": 9,
            "title": "Fix",
            "description": "",
            "author": {"username": "a", "name": "A"},
            "merged_at": "2024-06-01T00:00:00Z",
        },
        [
            {
                "id": "d",
                "notes": [
                    {
                        "id": 5,
                        "body": "rename helper",
                        "system": False,
                        "author": {"username": "lead", "name": "Lead"},
                        "created_at": "2024-06-01T01:00:00Z",
                        "position": {"new_path": "a.py", "new_line": 2},
                    }
                ],
            }
        ],
        [
            {
                "id": "abc123",
                "title": "MRR fixes rename",
                "message": "MRR fixes rename",
                "created_at": "2024-06-01T02:00:00Z",
                "author_email": "lead@company.com",
                "author_name": "Lead",
            }
        ],
        {"abc123": [{"old_path": "a.py", "new_path": "a.py", "diff": "@@ -1,3 +1,3 @@\n a\n-old\n+new\n"}]},
        [],
        project_path="group/app",
    )
    from scripts.ingest.gitlab.signals import SignalConfig

    process_store(
        gitlab,
        ["9!3"],
        signal_config=SignalConfig(team_leads=("lead@company.com",), mrr_pattern=r"\bmrr\b"),
    )
    gitlab.close()
    ingested = import_gitlab_traces(store, tmp_path / "gitlab")
    assert ingested["added"] == 2
    rows = store.sessions()
    assert any(row["sources"] == ["gitlab"] for row in rows)
    assert any(row.get("reviewer") == "lead" for row in rows)
    store.close()


def test_annotate_is_cached(tmp_path: Path):
    store = PatternStore(tmp_path / "patterns")
    _jsonl(tmp_path / "a.jsonl", [_session("a1", "fix the flaky test", "2024-01-01T00:00:00+00:00")])
    import_path(store, tmp_path / "a.jsonl", source_label="laptop-a")
    calls = {"n": 0}

    def complete(_messages):
        calls["n"] += 1
        return ANNOTATION

    first = annotate_pending(store, complete, model="mock")
    second = annotate_pending(store, complete, model="mock")
    assert first["annotated"] == 1
    assert second["skipped"] == 1
    assert calls["n"] == 1
    assert store.annotation(store.sessions()[0]["global_id"])["error_type"] == "loop"
    store.close()


def _seed_cluster(store: PatternStore) -> None:
    rows = []
    for index, task in enumerate(("bugfix", "refactor", "ops")):
        origin = f"m{index}"
        _jsonl(
            store.root / f"{origin}.jsonl",
            [_session(origin, f"the agent retried pytest without reading the error {index}", f"2024-0{index+1}-01T00:00:00+00:00", 0.4)],
        )
        import_path(store, store.root / f"{origin}.jsonl", source_label=f"machine-{index}")
        gid = store.sessions()[-1]["global_id"]
        payload = extract_json(ANNOTATION)
        payload["task_type"] = task
        payload["project"] = f"proj-{index}"
        store.put_annotation(gid, payload, "mock")
        rows.append(gid)
    _jsonl(store.root / "once.jsonl", [_session("once", "unrelated typo in a comment", "2024-04-01T00:00:00+00:00", 0.9)])
    import_path(store, store.root / "once.jsonl", source_label="machine-0")
    once = store.sessions()[-1]["global_id"]
    store.put_annotation(
        once,
        {
            "task_type": "other",
            "domain": "docs",
            "project": "docs",
            "languages": [],
            "tools": [],
            "outcome": "success",
            "failure_step": "",
            "error_type": "none",
            "user_correction": "",
            "what_worked": "edited the comment directly",
        },
        "mock",
    )
    _jsonl(store.root / "ok.jsonl", [_session("ok", "a clean success with no loop", "2024-05-01T00:00:00+00:00", 0.95)])
    import_path(store, store.root / "ok.jsonl", source_label="machine-1")
    # Same error type on a success so the cluster can be flagged, but different wording so it stays outside.
    store.put_annotation(
        store.sessions()[-1]["global_id"],
        {
            "task_type": "feature",
            "domain": "unrelated success domain",
            "project": "other",
            "languages": ["go"],
            "tools": ["edit"],
            "outcome": "success",
            "failure_step": "",
            "error_type": "loop",
            "user_correction": "",
            "what_worked": "changed strategy before running tests",
        },
        "mock",
    )


def test_cluster_threshold_and_recommendation_kind(tmp_path: Path):
    store = PatternStore(tmp_path / "patterns")
    _seed_cluster(store)
    stats = cluster_annotations(store, min_size=3, threshold=0.2)
    assert stats["noise"] >= 1
    visible = store.clusters()
    assert len(visible) == 1
    cluster = visible[0]
    assert cluster["frequency"] >= 3
    assert cluster["task_types"] >= 3
    assert cluster["machines"] >= 2

    def complete(_messages):
        return RECOMMENDATION

    proposal = recommend_cluster(
        cluster,
        complete,
        existing_rules=["When a command fails, read the error and change the approach instead of retrying it unchanged."],
        success_note=None,
    )
    assert proposal["kind"] == "rule"
    assert proposal["mode"] == "corpus"
    assert proposal["conflicts"]
    assert proposal["session_ids"]
    narrow = dict(cluster)
    narrow["task_types"] = 1
    narrow["task_type_names"] = ["bugfix"]
    skill = recommend_cluster(narrow, complete, existing_rules=[], success_note="error type loop also appears in a success")
    assert skill["kind"] == "skill"
    assert any("success" in item for item in skill["conflicts"])
    store.close()


def test_export_report_and_rollback(tmp_path: Path):
    from scripts.mine_patterns import main

    store_dir = tmp_path / "patterns"
    store = PatternStore(store_dir)
    _seed_cluster(store)
    cluster_annotations(store, min_size=3, threshold=0.2)
    cluster = store.clusters()[0]

    def complete(_messages):
        return RECOMMENDATION

    proposal = recommend_cluster(cluster, complete, existing_rules=[], success_note=None)
    proposals = tmp_path / "proposals.json"
    decisions = tmp_path / "decisions.json"
    from scripts.proposals.store import save_proposals, save_rule_decisions

    save_proposals(proposals, {"version": 2, "proposals": [proposal]})
    save_rule_decisions(decisions, {"statuses": {proposal["id"]: "approved"}, "edits": {}})
    rules_md = tmp_path / "rules.md"
    rules_json = tmp_path / "rules.json"
    skills = tmp_path / "skills"
    store.close()
    main(
        [
            "export",
            "--store",
            str(store_dir),
            "--proposals",
            str(proposals),
            "--decisions",
            str(decisions),
            "--rules-md",
            str(rules_md),
            "--rules-json",
            str(rules_json),
            "--skills-dir",
            str(skills),
            "--version",
            "v1",
        ]
    )
    assert "When a command fails" in rules_md.read_text(encoding="utf-8")
    document = json.loads(rules_json.read_text(encoding="utf-8"))
    assert document["rules"][0]["session_ids"]
    assert skill_id("Stop retry loops")
    # This proposal is a rule because task spread is high, so no skill file is required.
    rules_md.write_text("CHANGED\n", encoding="utf-8")
    main(["rollback", "--store", str(store_dir), "--version", "v1"])
    assert "When a command fails" in rules_md.read_text(encoding="utf-8")

    store = PatternStore(store_dir)
    rows = report(store, error_type="loop")
    assert rows and rows[0]["version"] == "v1"
    assert rows[0]["before"]["sessions"] >= 1 or rows[0]["after"]["sessions"] >= 1
    assert "error_recurrence" in rows[0]["after"]
    store.close()
