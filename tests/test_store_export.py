"""Rule decisions, Markdown/JSON export, and the mocked CLI pipeline."""

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.rules.export import export_approved, render_markdown
from scripts.rules.store import (
    approved_rules,
    load_rule_decisions,
    save_rule_decisions,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_rules_response.txt"


def _proposals():
    return {
        "version": 1,
        "rules": [
            {
                "id": "r_loop",
                "text": "Read the error before retrying.",
                "category": "looping",
                "rationale": "it looped",
                "session_ids": ["s1", "s2"],
                "evidence": "EVIDENCE_LOOP",
            },
            {
                "id": "r_safe",
                "text": "Ask before deleting files.",
                "category": "safety",
                "rationale": "deleted a file",
                "session_ids": ["s3"],
                "evidence": "EVIDENCE_SAFE",
            },
        ],
    }


def test_decisions_roundtrip_and_edits(tmp_path: Path):
    path = tmp_path / "rule_decisions.json"
    save_rule_decisions(
        path,
        {
            "statuses": {"r_loop": "approved", "r_safe": "nope", "r_other": "rejected"},
            "edits": {"r_loop": {"text": "  Read the error, then change approach.  ", "category": "loops"}},
        },
    )
    loaded = load_rule_decisions(path)
    assert loaded["statuses"] == {"r_loop": "approved", "r_other": "rejected"}
    assert "r_safe" not in loaded["statuses"]
    again = load_rule_decisions(path)
    assert again == loaded

    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert load_rule_decisions(broken) == {"statuses": {}, "edits": {}}

    view = approved_rules(_proposals(), loaded)
    assert len(view) == 1
    assert view[0]["text"] == "Read the error, then change approach."
    assert view[0]["category"] == "looping"
    assert view[0]["session_ids"] == ["s1", "s2"]
    assert view[0]["id"] == "r_loop"


def test_export_formats(tmp_path: Path):
    proposals = tmp_path / "proposals.json"
    decisions = tmp_path / "decisions.json"
    markdown = tmp_path / "rules.md"
    payload = tmp_path / "rules.json"
    proposals.write_text(json.dumps(_proposals()), encoding="utf-8")
    save_rule_decisions(
        decisions,
        {
            "statuses": {"r_loop": "approved", "r_safe": "rejected"},
            "edits": {"r_loop": {"text": "EDITED RULE", "category": "looping"}},
        },
    )
    result = export_approved(proposals, decisions, markdown, payload)
    assert result["approved"] == 1
    text = markdown.read_text(encoding="utf-8")
    assert text.startswith("# Agent harness rules\n")
    assert "## Looping\n" in text
    assert "- EDITED RULE\n" in text
    assert "Ask before deleting" not in text
    assert "EVIDENCE_LOOP" not in text
    assert "Safety" not in text
    document = json.loads(payload.read_text(encoding="utf-8"))
    assert list(document) == ["rules"]
    rule = document["rules"][0]
    assert rule == {
        "id": "r_loop",
        "text": "EDITED RULE",
        "category": "looping",
        "rationale": "it looped",
        "session_ids": ["s1", "s2"],
        "evidence": "EVIDENCE_LOOP",
    }
    assert render_markdown([]).startswith("# Agent harness rules\n")


def _clean_env(**extra):
    env = os.environ.copy()
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "FINETUNE_CURATOR_API_KEY",
        "FINETUNE_CURATOR_BASE_URL",
        "FINETUNE_CURATOR_MODEL",
        "LLAMA_API_KEY",
    ):
        env.pop(name, None)
    env.update(extra)
    return env


def test_print_endpoint_hides_secrets_and_prefers_cli(tmp_path: Path):
    config = tmp_path / "opencode.jsonc"
    config.write_text(
        """
        {
          "providers": {
            "llama": {
              "settings": {"apiKey": "super-secret-key", "baseURL": "http://127.0.0.1:8080/v1"},
              "models": {"qwen3-27b": {"modelID": "qwen3-27b-q5"}}
            }
          }
        }
        """,
        encoding="utf-8",
    )
    listed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.propose_rules",
            "--list-opencode-models",
            "--opencode-config",
            str(config),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=_clean_env(),
    )
    assert "llama/qwen3-27b" in listed.stdout
    assert "qwen3-27b-q5" in listed.stdout
    assert "super-secret-key" not in listed.stdout
    assert "key=set" in listed.stdout

    printed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.propose_rules",
            "--print-endpoint",
            "--from-opencode",
            "llama/qwen3-27b",
            "--opencode-config",
            str(config),
            "--model",
            "cli-model",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=_clean_env(OPENAI_BASE_URL="http://env.example/v1", FINETUNE_CURATOR_BASE_URL="http://fc.example/v1"),
    )
    assert "model=cli-model (cli)" in printed.stdout
    assert "http://fc.example/v1" in printed.stdout
    assert "super-secret-key" not in printed.stdout
    assert printed.returncode == 0

    default = subprocess.run(
        [sys.executable, "-m", "scripts.propose_rules", "--print-endpoint"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=_clean_env(),
    )
    assert "http://localhost:8080/v1" in default.stdout
    assert "default:llama-server" in default.stdout
    assert "(auto)" in default.stdout


def test_cli_mock_propose_and_export(tmp_path: Path):
    raw = tmp_path / "raw.jsonl"
    review = tmp_path / "decisions.json"
    proposals = tmp_path / "rule_proposals.json"
    rule_decisions = tmp_path / "rule_decisions.json"
    markdown = tmp_path / "approved_rules.md"
    payload = tmp_path / "approved_rules.json"
    rows = [
        {
            "id": "ses_loop",
            "score": 0.95,
            "label": "good",
            "messages": [
                {"role": "user", "content": "run the tests"},
                {"role": "assistant", "content": "pytest failed, running pytest again"},
            ],
        },
        {
            "id": "ses_loop2",
            "score": 0.2,
            "label": "bad",
            "messages": [{"role": "user", "content": "stop retrying the same command"}],
        },
        {
            "id": "ses_del",
            "score": 0.4,
            "label": "bad",
            "messages": [{"role": "user", "content": "I did not ask you to delete that"}],
        },
        {
            "id": "ses_noted",
            "score": 0.99,
            "label": "good",
            "messages": [{"role": "user", "content": "rename the helper"}],
        },
        {
            "id": "ses_good",
            "score": 0.99,
            "label": "good",
            "messages": [{"role": "user", "content": "this one was fine"}],
        },
    ]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    review.write_text(
        json.dumps(
            {
                "labels": {
                    "ses_loop": "discard",
                    "ses_loop2": "keep",
                    "ses_del": "discard",
                    "ses_noted": "keep",
                    "ses_good": "keep",
                },
                "notes": {"ses_noted": "агент удалил лишний файл"},
            }
        ),
        encoding="utf-8",
    )
    env = _clean_env(OPENAI_API_KEY="super-secret-key")
    proposed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.propose_rules",
            "--input",
            str(raw),
            "--decisions",
            str(review),
            "--output",
            str(proposals),
            "--mock-response",
            str(FIXTURE),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    assert proposed.returncode == 0, proposed.stderr + proposed.stdout
    assert "super-secret-key" not in proposed.stdout
    assert "super-secret-key" not in proposals.read_text(encoding="utf-8")
    document = json.loads(proposals.read_text(encoding="utf-8"))
    assert document["batches"] == 1
    assert "ses_good" not in document["source_session_ids"]
    assert "ses_noted" in document["source_session_ids"]
    assert "api_key" not in document
    assert len(document["rules"]) == 2
    looping = next(rule for rule in document["rules"] if rule["category"] == "looping")
    safety = next(rule for rule in document["rules"] if rule["category"] == "safety")
    assert looping["session_ids"] == ["ses_loop", "ses_loop2"]
    assert "stop running the same command" in looping["evidence"]
    assert "retried pytest" in looping["evidence"]

    save_rule_decisions(
        rule_decisions,
        {
            "statuses": {looping["id"]: "approved", safety["id"]: "rejected"},
            "edits": {looping["id"]: {"text": "EDITED RULE", "category": "looping"}},
        },
    )
    exported = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.export_rules",
            "--proposals",
            str(proposals),
            "--decisions",
            str(rule_decisions),
            "--markdown",
            str(markdown),
            "--json",
            str(payload),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    assert exported.returncode == 0, exported.stderr
    assert "EDITED RULE" in markdown.read_text(encoding="utf-8")
    assert "Ask before deleting" not in markdown.read_text(encoding="utf-8")
    exported_doc = json.loads(payload.read_text(encoding="utf-8"))
    assert exported_doc["rules"][0]["text"] == "EDITED RULE"
    assert exported_doc["rules"][0]["session_ids"] == ["ses_loop", "ses_loop2"]
    assert exported_doc["rules"][0]["id"] == looping["id"]


def test_help_commands():
    for module in ("scripts.propose_rules", "scripts.export_rules"):
        subprocess.run([sys.executable, "-m", module, "--help"], check=True, cwd=ROOT, capture_output=True)


def test_export_missing_proposals(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.export_rules", "--proposals", str(tmp_path / "missing.json")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
