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
    assert linked["confidence"] == 0.9
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
    assert mr_level["link"] == "mr"
    assert mr_level["confidence"] == 0.3
    assert mr_level["commit_sha"] == "aaa"

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
    assert stats["mr"] == 1
    assert stats["unlinked"] == 1
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
    assert set(by_link) == {"file_line", "mr"}
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


def test_cli_fixture_does_not_leak_token(tmp_path: Path):
    sft = tmp_path / "sft.jsonl"
    dpo = tmp_path / "dpo.jsonl"
    cache = tmp_path / "cache"
    env = os.environ.copy()
    env["GITLAB_TOKEN"] = TOKEN
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.export_gitlab_mrr",
            "--base-url",
            "https://gitlab.example.com",
            "--project",
            "group/app",
            "--reviewer",
            "TeamLead",
            "--since",
            "2024-03-01",
            "--until",
            "2024-03-31",
            "--fixture",
            str(FIXTURE),
            "--cache-dir",
            str(cache),
            "--anonymize",
            "--anonymize-urls",
            "--sft",
            str(sft),
            "--dpo",
            str(dpo),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    published = proc.stdout + proc.stderr + sft.read_text(encoding="utf-8") + dpo.read_text(encoding="utf-8")
    cached = "".join(path.read_text(encoding="utf-8") for path in cache.glob("*.json"))
    assert TOKEN not in published
    assert TOKEN not in cached
    assert "lead@company.com" not in published
    assert "https://" not in published
    assert "file+line: 1" in proc.stdout
    rows = [json.loads(line) for line in sft.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    dpo_rows = [json.loads(line) for line in dpo.read_text(encoding="utf-8").splitlines()]
    assert {row["id"] for row in dpo_rows} == {row["id"] for row in rows}
    rebuilt = tmp_path / "dpo2.jsonl"
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.export_gitlab_mrr",
            "--dpo-from",
            str(sft),
            "--dpo",
            str(rebuilt),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert again.returncode == 0, again.stderr
    assert rebuilt.read_text(encoding="utf-8") == dpo.read_text(encoding="utf-8")
