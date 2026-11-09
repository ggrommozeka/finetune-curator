"""GitLab MR export: mocked HTTP, linking, anonymization, rules source."""

import json
import os
import ssl
import subprocess
import sys
import urllib.parse
from pathlib import Path

import pytest

from scripts.gitlab.anonymize import Anonymizer
from scripts.gitlab.client import (
    FixtureTransport,
    GitLabClient,
    GitLabError,
    build_ssl_context,
    merge_request_path,
    redact,
)
from scripts.gitlab.examples import sft_to_dpo
from scripts.gitlab.export import export_records
from scripts.gitlab.link import (
    comment_target,
    is_fix_message,
    link_comment,
    parse_hunks,
    parse_user_time,
)
from scripts.gitlab.store import GitlabStore
from scripts.gitlab.sync import process_store, sync_store
from scripts.rules.sessions import render_session, select_sessions

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "gitlab_api.json"
TOKEN = "super-secret-token"


def _client(tmp_path: Path, transport=None, **kwargs) -> GitLabClient:
    return GitLabClient(
        "https://gitlab.example.com",
        TOKEN,
        cache_dir=kwargs.pop("cache_dir", tmp_path / "cache"),
        transport=transport or FixtureTransport.from_file(FIXTURE),
        **kwargs,
    )


def test_quote_and_ssl_and_redact():
    assert merge_request_path("project", "group/app") == "/api/v4/projects/group%2Fapp/merge_requests"
    assert merge_request_path("group", 5) == "/api/v4/groups/5/merge_requests"
    context = build_ssl_context(False, None)
    assert context.verify_mode == ssl.CERT_NONE
    with pytest.raises(GitLabError):
        build_ssl_context(True, "/no/such/ca.pem")
    message = redact("GitLab 401 for /api: bad super-secret-token private_token=super-secret-token", TOKEN)
    assert TOKEN not in message
    assert "private_token=***" in message


def test_pagination_rate_limit_and_cache(tmp_path: Path):
    calls = {"n": 0}

    def transport(method, url, headers):
        assert headers.get("PRIVATE-TOKEN") == TOKEN
        calls["n"] += 1
        if calls["n"] == 1:
            return 429, {"Retry-After": "0"}, b"slow down"
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        page = (query.get("page") or ["1"])[0]
        if page == "1":
            body = json.dumps([{"iid": 1}]).encode()
            return 200, {"X-Next-Page": "2"}, body
        body = json.dumps([{"iid": 2}]).encode()
        return 200, {"X-Next-Page": ""}, body

    slept = []
    client = GitLabClient(
        "https://gitlab.example.com",
        TOKEN,
        cache_dir=tmp_path / "cache",
        transport=transport,
        sleep=slept.append,
    )
    rows = client.get_all("/api/v4/projects/1/merge_requests", {"state": "merged"})
    assert [row["iid"] for row in rows] == [1, 2]
    assert slept == [0.0]
    calls_after_first = calls["n"]
    again = client.get_all("/api/v4/projects/1/merge_requests", {"state": "merged"})
    assert again == rows
    assert calls["n"] == calls_after_first
    blob = "\n".join(
        path.read_text(encoding="utf-8") for path in (tmp_path / "cache").glob("*.json")
    )
    assert TOKEN not in blob
    assert blob

    def failing(_method, _url, _headers):
        return 401, {}, f"nope {TOKEN}".encode()

    broken = GitLabClient("https://gitlab.example.com", TOKEN, transport=failing, cache_dir=None)
    with pytest.raises(GitLabError) as caught:
        broken.get_all("/api/v4/projects/1/merge_requests")
    assert TOKEN not in str(caught.value)


def test_line_movement_and_non_fix_commits():
    from scripts.gitlab.trace import map_line, map_range, trace_thread

    shifted, touched, _hunk = map_line(12, parse_hunks("@@ -1,0 +1,2 @@\n+added\n+lines\n"))
    assert shifted == 14
    assert touched is False
    new_start, new_end, touched_range, _hunk = map_range(12, 12, parse_hunks("@@ -10,5 +10,5 @@\n a\n-b\n+c\n d\n e\n"))
    assert touched_range is True
    assert new_start == 12

    note = {
        "id": 1,
        "body": "rename",
        "created_at": "2024-03-01T10:00:00Z",
        "position": {"new_path": "app.py", "new_line": 12, "head_sha": "base"},
    }
    shift = {
        "id": "shift",
        "message": "insert header",
        "created_at": "2024-03-01T11:00:00Z",
    }
    edit = {
        "id": "edit",
        "message": "actual change, not marked MRR",
        "created_at": "2024-03-01T12:00:00Z",
    }
    diffs = {
        "shift": [{"old_path": "app.py", "new_path": "app.py", "diff": "@@ -1,0 +1,2 @@\n+added\n+lines\n"}],
        "edit": [{"old_path": "app.py", "new_path": "app.py", "diff": "@@ -14,1 +14,1 @@\n-old_name\n+parse_config\n"}],
    }
    traced = trace_thread({"root": note, "replies": [], "discussion_id": "d"}, [shift, edit], diffs, {})
    assert traced["status"] == "linked"
    assert traced["confidence"] == 0.8
    assert traced["commit_sha"] == "edit"
    assert "parse_config" in traced["after"]
    assert [item["sha"] for item in traced["changed_by"]] == ["edit"]


def test_link_file_line_then_file_then_mr():
    assert is_fix_message("MRR fixes: rename", r"mrr\s*fixes")
    assert is_fix_message("mrr\nfixes typo", r"mrr\s*fixes")
    assert not is_fix_message("Initial implementation", r"mrr\s*fixes")
    hunks = parse_hunks("@@ -10,5 +10,5 @@ def helper():\n context\n-old_name\n+parse_config\n context2\n")
    assert hunks[0].before.splitlines()[1] == "old_name"
    assert "parse_config" in hunks[0].after

    note = {
        "id": 1,
        "body": "rename",
        "created_at": "2024-03-01T10:00:00Z",
        "position": {"new_path": "app.py", "new_line": 12},
    }
    early = {
        "id": "aaa",
        "message": "MRR fixes unrelated",
        "created_at": "2024-03-01T12:00:00Z",
    }
    later = {
        "id": "bbb",
        "message": "MRR fixes rename",
        "created_at": "2024-03-01T16:00:00Z",
    }
    diffs = {
        "aaa": [{"old_path": "other.py", "new_path": "other.py", "diff": "@@ -1 +1 @@\n-a\n+b\n"}],
        "bbb": [{"old_path": "app.py", "new_path": "app.py", "diff": hunks[0].text}],
    }
    linked = link_comment(note, [early, later], diffs)
    assert linked["link"] == "file_line"
    assert linked["confidence"] == 0.8
    assert linked["commit_sha"] == "bbb"
    assert "parse_config" in linked["after"]

    file_only = link_comment(
        {**note, "position": {"new_path": "app.py", "new_line": 80}},
        [later],
        diffs,
    )
    assert file_only["link"] == "file"
    assert file_only["confidence"] == 0.55

    mr_level = link_comment(
        {"id": 2, "body": "add a test", "created_at": "2024-03-01T10:00:00Z"},
        [early, later],
        diffs,
    )
    assert mr_level is None

    too_late = link_comment(
        {**note, "created_at": "2024-03-02T00:00:00Z"},
        [early, later],
        diffs,
    )
    assert too_late is None
    path, start, end = comment_target(
        {"position": {"new_path": "a.py", "line_range": {"start": {"new_line": 4}, "end": {"new_line": 9}}}}
    )
    assert (path, start, end) == ("a.py", 4, 9)


def test_export_fixture_anonymizes_and_builds_dpo(tmp_path: Path):
    client = _client(tmp_path)
    since = parse_user_time("2024-03-01")
    until = parse_user_time("2024-03-31", end_of_day=True)
    records, stats = export_records(
        client,
        projects=["group/app"],
        groups=[],
        since=since,
        until=until,
        reviewer="teamlead",
        anonymize=True,
        anonymize_urls=True,
    )
    assert stats["merge_requests"] == 1
    assert stats["file_line"] == 1
    assert stats["mr"] == 0
    assert stats["unlinked"] == 2
    assert stats["comments"] == 3
    blob = json.dumps(records)
    assert TOKEN not in blob
    assert "lead@company.com" not in blob
    assert "ada@company.com" not in blob
    assert "Ada Author" not in blob
    assert "teamlead" not in blob
    assert "gitlab.example.com" not in blob
    assert "<email>" in blob
    assert "https://" not in blob
    by_link = {record["gitlab"]["link"]: record for record in records}
    assert set(by_link) == {"file_line"}
    sft = by_link["file_line"]
    assert sft["source"] == "gitlab_review"
    assert sft["messages"][0]["role"] == "user"
    assert "parse_config" in sft["messages"][1]["content"]
    assert "Review comment:" in sft["messages"][0]["content"]
    dpo = sft_to_dpo(sft)
    assert dpo["chosen"] == sft["messages"][1]["content"]
    assert dpo["rejected"] == sft["gitlab"]["before"]
    assert "old_name" in dpo["rejected"]
    outside, outside_stats = export_records(
        _client(tmp_path / "other"),
        projects=["group/app"],
        groups=[],
        since=parse_user_time("2025-01-01"),
        until=parse_user_time("2025-01-02", end_of_day=True),
    )
    assert outside_stats["merge_requests"] == 0
    assert outside == []


def test_comment_fix_and_team_lead_polish_are_separate():
    from scripts.gitlab.classify import classify_commits
    from scripts.gitlab.polish import build_polish_records
    from scripts.gitlab.signals import SignalConfig, is_mrr_title

    assert is_mrr_title("MRR fixes: rename")
    assert is_mrr_title("mrr: tighten")
    assert is_mrr_title("Please MRR this helper")
    assert not is_mrr_title("summary of the change")

    import json as _json
    payload = _json.loads(FIXTURE.read_text(encoding="utf-8"))
    mr = next(item["pages"][0]["body"] for item in payload["responses"] if item["path"].endswith("/merge_requests/7") and "group" not in item["path"])
    discussions = next(item["pages"][0]["body"] for item in payload["responses"] if item["path"].endswith("/discussions"))
    commits = next(item["pages"][0]["body"] for item in payload["responses"] if item["path"].endswith("/commits"))
    config = SignalConfig(team_leads=("lead@company.com", "teamlead"), mrr_pattern=r"\bmrr\b")
    roles = {row["sha"][:4]: row["role"] for row in classify_commits(commits, mr, discussions, config)}
    assert roles["aaaa"] == "author"
    assert roles["bbbb"] == "author_fix"
    assert roles["dddd"] == "team_lead_mrr"
    assert roles["eeee"] == "team_lead_other"

    diffs = {
        "dddddddddddddddddddddddddddddddddddddddd": [
            {"new_path": "app.py", "old_path": "app.py", "diff": "@@ -1,1 +1,1 @@\n-raw\n+parsed\n"}
        ],
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee": [
            {"new_path": "app.py", "old_path": "app.py", "diff": "@@ -1,1 +1,2 @@\n import sys\n+import logging\n"}
        ],
    }
    records = build_polish_records(mr, commits, diffs, discussions, config, complete_fn=lambda _messages: "error handling")
    kinds = {record["gitlab"]["role"]: record for record in records}
    assert set(kinds) == {"team_lead_mrr", "team_lead_other"}
    assert kinds["team_lead_mrr"]["source"] == "gitlab_polish"
    assert kinds["team_lead_mrr"]["gitlab"]["mrr"] is True
    assert kinds["team_lead_mrr"]["gitlab"]["polish"] == "error handling"
    assert "Commit message:" in kinds["team_lead_mrr"]["messages"][0]["content"]
    assert "Review comment:" not in kinds["team_lead_mrr"]["messages"][0]["content"]


def test_one_team_lead_and_every_reviewer(tmp_path: Path):
    from scripts.gitlab.signals import SignalConfig, load_signal_config

    config_path = tmp_path / "gitlab.json"
    config_path.write_text(
        '{"team_lead": {"aliases": ["Lead@Company.com", "teamlead"]}, "mrr_pattern": "\\\\bmrr\\\\b"}\n',
        encoding="utf-8",
    )
    loaded = load_signal_config(config_path)
    assert loaded.aliases == ("lead@company.com", "teamlead")
    assert SignalConfig(team_leads=("lead@company.com",), mrr_pattern=r"\bmrr\b").aliases == ("lead@company.com",)

    client, store = _sync_client(tmp_path, FixtureTransport.from_file(FIXTURE))
    sync_store(client, store, projects=["group/app"])
    stats = process_store(store, ["1!7"], signal_config=loaded)
    reviewers = {trace.get("reviewer") for trace in store.traces_for(1, 7)}
    assert "teamlead" in reviewers
    assert "otheruser" in reviewers
    comment_examples = [
        item["record"]["gitlab"].get("reviewer")
        for item in store.examples_for(1, 7)
        if item["record"].get("source") == "gitlab_review"
    ]
    assert comment_examples
    assert stats["comments"] >= 4
    store.close()


def test_anonymizer_whole_words_only():
    anonymizer = Anonymizer(strip_urls=True)
    anonymizer.register_author({"username": "ann", "name": "Ann Example", "email": "ann@corp.test"})
    text = anonymizer.text("Ann Example said ann@corp.test see https://git.corp.test/a and anniversary")
    assert "ann@corp.test" not in text
    assert "<email>" in text
    assert "<url>" in text
    assert "anniversary" in text
    assert "Ann Example" not in text


def test_gitlab_review_source_is_opt_in():
    row = {
        "id": "gitlab-1-7-note-11",
        "source": "gitlab_review",
        "score": 0.99,
        "label": "gitlab",
        "messages": [{"role": "user", "content": "Apply the review comment"}],
        "gitlab": {"comment": "Rename the helper", "file": "app.py", "link": "file_line"},
    }
    assert select_sessions([row], {}) == []
    picked = select_sessions([row], {}, sources=("gitlab_review",))
    assert picked == [row]
    _sid, preamble, _blocks = render_session(row, None, None)
    assert "review_comment: Rename the helper" in preamble
    assert "review_file: app.py" in preamble


def _sync_client(tmp_path: Path, transport=None) -> tuple[GitLabClient, GitlabStore]:
    store = GitlabStore(tmp_path / "gitlab")
    client = GitLabClient(
        "https://gitlab.example.com",
        TOKEN,
        cache_dir=None,
        transport=transport or FixtureTransport.from_file(FIXTURE),
    )
    return client, store


def test_sync_is_incremental_and_process_is_offline(tmp_path: Path):
    transport = FixtureTransport.from_file(FIXTURE)
    client, store = _sync_client(tmp_path, transport)
    first = sync_store(
        client,
        store,
        projects=["group/app"],
        since=parse_user_time("2024-03-01"),
        until=parse_user_time("2024-03-31", end_of_day=True),
    )
    assert first["fetched"] == 1
    assert first["skipped"] == 0
    calls_after_sync = len(transport.calls)
    assert (tmp_path / "gitlab" / "raw" / "1" / "7" / "discussions.json").is_file()
    assert store.has_mr(1, 7)
    raw_db = (tmp_path / "gitlab" / "store.db").read_bytes()
    assert TOKEN.encode() not in raw_db
    assert b"lead@company.com" in raw_db

    second = sync_store(
        client,
        store,
        projects=["group/app"],
        since=parse_user_time("2024-03-01"),
        until=parse_user_time("2024-03-31", end_of_day=True),
    )
    assert second["fetched"] == 0
    assert second["skipped"] == 1
    assert len(transport.calls) == calls_after_sync + 1

    def offline(*_args, **_kwargs):
        raise AssertionError("process must not touch GitLab")

    client.transport = offline
    stats = process_store(store, ["1!7"], reviewer="teamlead", anonymize=True, anonymize_urls=True)
    assert stats["examples"] == 1
    assert stats["file_line"] == 1
    example_id = store.examples_for(1, 7)[0]["id"]
    store.set_decision(example_id, "keep", user_text="EDITED INPUT", assistant_text="EDITED OUTPUT")
    process_store(store, ["group/app!7"], reviewer="teamlead", anonymize=True, anonymize_urls=True)
    decision = store.get_decision(example_id)
    assert decision["status"] == "keep"
    assert decision["user_text"] == "EDITED INPUT"
    exported = store.export_records(status="keep")
    assert len(exported) == 1
    assert exported[0]["messages"][0]["content"] == "EDITED INPUT"
    assert exported[0]["messages"][1]["content"] == "EDITED OUTPUT"
    blob = json.dumps(store.export_records(status="all"))
    assert "lead@company.com" not in blob
    assert "https://" not in blob
    role = store._conn.execute("SELECT author_role, head_sha FROM notes WHERE note_id = '11'").fetchone()
    assert role["author_role"] == "reviewer"
    assert role["head_sha"].startswith("aaa")
    traces = store.traces_for(1, 7)
    anchored = next(item for item in traces if item["note_id"] == "11")
    assert anchored["status"] == "linked"
    assert anchored["anchor"]["new_line_start"] == 12
    assert "old_name" in (anchored["anchor"]["snippet"] or anchored["before"])
    assert any(reply["body"] == "ack, renaming" for reply in anchored["replies"])
    assert anchored["resolved_by"]
    store.set_link_override(1, "11", "not_addressed")
    process_store(store, ["1!7"], reviewer="teamlead", anonymize=True, anonymize_urls=True)
    assert store.traces_for(1, 7)
    overridden = next(item for item in store.traces_for(1, 7) if item["note_id"] == "11")
    assert overridden["manual"] is True
    assert overridden["status"] == "not_addressed"
    assert all(item["record"]["gitlab"]["note_id"] != 11 and str(item["record"]["gitlab"]["note_id"]) != "11" for item in store.examples_for(1, 7))
    others = store.traces_for()
    stats = store.link_stats()
    assert stats["total"]["comments"] == len(others)
    assert "with_fix_share" in stats["total"]
    assert stats["by_reviewer"]
    assert stats["by_file"]

    listed = store.list_mrs(reviewer="teamlead", fixes_only=True, min_comments=1)
    assert [row["ref"] for row in listed] == ["1!7"]
    assert store.list_mrs(fixes_only=True, since="2025-01-01") == []
    client.transport = transport
    refreshed = sync_store(client, store, refresh=["1!7"])
    assert refreshed["refreshed"] == 1
    assert refreshed["fetched"] == 0
    store.close()


def test_sync_stores_file_contents(tmp_path: Path):
    base = FixtureTransport.from_file(FIXTURE)

    def transport(method, url, headers):
        if "/repository/files/" in urllib.parse.unquote(url) and "ref=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" in url:
            return 200, {}, b"context\nold_name\ncontext2\n"
        return base(method, url, headers)

    client, store = _sync_client(tmp_path, transport)
    sync_store(client, store, projects=["group/app"])
    bundle = store.load_bundle(1, 7)
    stored = [item for item in bundle["files"] if item["side"] == "after" and item["path"] == "app.py"]
    assert stored and "old_name" in stored[0]["content"]
    store.close()


def test_cli_fixture_does_not_leak_token(tmp_path: Path):
    store = tmp_path / "gitlab"
    sft = tmp_path / "sft.jsonl"
    dpo = tmp_path / "dpo.jsonl"
    rules = tmp_path / "rules.json"
    env = os.environ.copy()
    env["GITLAB_TOKEN"] = TOKEN

    def run(args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "scripts.export_gitlab_mrr", *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    synced = run(
        [
            "sync",
            "--store",
            str(store),
            "--base-url",
            "https://gitlab.example.com",
            "--project",
            "group/app",
            "--since",
            "2024-03-01",
            "--until",
            "2024-03-31",
            "--fixture",
            str(FIXTURE),
        ]
    )
    assert synced.returncode == 0, synced.stderr + synced.stdout
    assert TOKEN not in synced.stdout + synced.stderr
    again = run(
        [
            "sync",
            "--store",
            str(store),
            "--base-url",
            "https://gitlab.example.com",
            "--project",
            "group/app",
            "--fixture",
            str(FIXTURE),
        ]
    )
    assert again.returncode == 0, again.stderr
    assert "skipped: 1" in again.stdout

    processed = run(
        [
            "process",
            "--store",
            str(store),
            "--mr",
            "group/app!7",
            "--reviewer",
            "TeamLead",
            "--anonymize",
            "--anonymize-urls",
            "--rules",
            "--mock-response",
            str(ROOT / "tests" / "fixtures" / "sample_rules_response.txt"),
            "--rules-output",
            str(rules),
        ]
    )
    assert processed.returncode == 0, processed.stderr + processed.stdout
    assert "file+line: 1" in processed.stdout
    assert TOKEN not in rules.read_text(encoding="utf-8")

    db = GitlabStore(store)
    for item in db.examples_for(1, 7):
        db.set_decision(item["id"], "keep")
    db.close()

    exported = run(
        [
            "export",
            "--store",
            str(store),
            "--status",
            "keep",
            "--sft",
            str(sft),
            "--dpo",
            str(dpo),
        ]
    )
    assert exported.returncode == 0, exported.stderr
    published = exported.stdout + sft.read_text(encoding="utf-8") + dpo.read_text(encoding="utf-8")
    assert TOKEN not in published
    assert "lead@company.com" not in published
    assert "https://" not in published
    rows = [json.loads(line) for line in sft.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    rebuilt = tmp_path / "dpo2.jsonl"
    dpo_only = run(["export", "--dpo-from", str(sft), "--dpo", str(rebuilt)])
    assert dpo_only.returncode == 0, dpo_only.stderr
    assert rebuilt.read_text(encoding="utf-8") == dpo.read_text(encoding="utf-8")
