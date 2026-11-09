"""Unified dataset: mix, splits, Qwen3 messages, and mocked evaluation."""

import json
from pathlib import Path

from scripts.build_dataset import main as build_main
from scripts.dataset.build import build_samples
from scripts.dataset.tasks import from_gitlab_store
from scripts.dataset.evaluate import review_metrics, style_metrics
from scripts.dataset.format import clip_diff_context, render_chat, render_qwen3, sequence_tokens, split_messages
from scripts.dataset.mix import is_held_out
from scripts.gitlab.signals import SignalConfig
from scripts.gitlab.store import GitlabStore

DIFF = "@@ -1,3 +1,3 @@\n def helper():\n-old_name\n+parse_config\n"


def _session(sid: str, messages: list[dict], **extra) -> dict:
    return {"id": sid, "score": 0.9, "machine": "laptop-a", "messages": messages, **extra}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _gitlab(tmp_path: Path) -> GitlabStore:
    store = GitlabStore(tmp_path / "gitlab")
    store.save_bundle(
        {
            "project_id": 9,
            "iid": 3,
            "title": "Parser",
            "description": "",
            "author": {"username": "ada", "name": "Ada Author"},
        },
        [
            {
                "id": "d1",
                "notes": [
                    {
                        "id": 5,
                        "body": "Rename the helper to parse_config.",
                        "system": False,
                        "author": {"username": "rita", "name": "Rita"},
                        "created_at": "2024-06-01T01:00:00Z",
                        "position": {"new_path": "app.py", "new_line": 2},
                    }
                ],
            },
            {
                "id": "d2",
                "notes": [
                    {
                        "id": 6,
                        "body": "Please add a test.",
                        "system": False,
                        "author": {"username": "omar", "name": "Omar"},
                        "created_at": "2024-06-01T01:10:00Z",
                    }
                ],
            }
        ],
        [
            {
                "id": "abc123",
                "title": "MRR: rename helper",
                "message": "MRR: rename helper",
                "created_at": "2024-06-01T02:00:00Z",
                "author_email": "lead@company.com",
                "author_name": "Lead",
            },
            {
                "id": "quiet1",
                "title": "add logging",
                "message": "add logging",
                "created_at": "2024-06-01T03:00:00Z",
                "author_email": "lead@company.com",
                "author_name": "Lead",
            },
        ],
        {
            "abc123": [{"old_path": "app.py", "new_path": "app.py", "diff": DIFF}],
            "quiet1": [
                {
                    "old_path": "log.py",
                    "new_path": "log.py",
                    "diff": "@@ -1,1 +1,2 @@\n import sys\n+import logging\n",
                }
            ],
        },
        [],
        project_path="group/app",
    )
    return store


def test_agentic_tools_corrections_and_qwen_template(tmp_path: Path):
    path = tmp_path / "sessions.jsonl"
    _write_jsonl(
        path,
        [
            _session(
                "s1",
                [
                    {"role": "user", "content": "rename the helper"},
                    {
                        "role": "assistant",
                        "content": "I will edit it.",
                        "tool_calls": [{"id": "c1", "name": "bash", "arguments": {"command": "ls"}}],
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": "app.py"},
                    {"role": "assistant", "content": "Renamed."},
                ],
            ),
            _session(
                "s2",
                [
                    {"role": "user", "content": "fix the bug"},
                    {"role": "assistant", "content": "done, but wrong"},
                    {"role": "user", "content": "No, read the traceback first."},
                    {"role": "assistant", "content": "Read the traceback, then fixed it."},
                ],
            ),
            _session(
                "s1-dup",
                [
                    {"role": "user", "content": "rename the helper"},
                    {
                        "role": "assistant",
                        "content": "I will edit it.",
                        "tool_calls": [{"id": "c1", "name": "bash", "arguments": {"command": "ls"}}],
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": "app.py"},
                    {"role": "assistant", "content": "Renamed."},
                ],
            ),
        ],
    )
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps({"labels": {"s1": "keep", "s2": "keep"}}), encoding="utf-8")
    result = build_samples(
        opencode_path=path,
        decisions_path=decisions,
        ratios={"agentic": 1, "style": 0, "review": 0},
        eval_ratio=0,
        max_tokens=32_000,
    )
    kinds = {sample.kind for sample in result["train"]}
    assert "trajectory" in kinds
    assert "correction" in kinds
    dpo = next(sample for sample in result["train"] if sample.dpo)
    assert dpo.dpo["rejected"][0]["content"] == "done, but wrong"
    assert "traceback" in dpo.dpo["chosen"][0]["content"]
    trajectory = next(sample for sample in result["train"] if sample.id == "agentic-s1")
    call = trajectory.messages[2]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "bash"
    rendered = render_qwen3(trajectory.messages)
    assert rendered.startswith("<|im_start|>system\n")
    assert "<|im_end|>" in rendered
    assert "<tool_call>" in rendered
    assert "<function=bash>" in rendered
    trajectories = [sample for sample in result["train"] if sample.kind == "trajectory"]
    assert len(trajectories) == 1


def test_gitlab_style_review_and_held_out_groups(tmp_path: Path):
    store = _gitlab(tmp_path)
    store.close()
    sessions = tmp_path / "sessions.jsonl"
    _write_jsonl(
        sessions,
        [
            _session("keep-me", [{"role": "user", "content": "ship it"}, {"role": "assistant", "content": "shipped"}]),
        ],
    )
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps({"labels": {"keep-me": "keep"}}), encoding="utf-8")
    config = SignalConfig(team_leads=("lead@company.com",), mrr_pattern=r"\bmrr\b")
    result = build_samples(
        opencode_path=sessions,
        decisions_path=decisions,
        gitlab_root=tmp_path / "gitlab",
        ratios={"agentic": 0.34, "style": 0.33, "review": 0.33},
        eval_ratio=0,
        signal_config=config,
        max_tokens=32_000,
    )
    tasks = {sample.task for sample in result["train"]}
    assert tasks == {"agentic", "style", "review"}
    gitlab_samples = from_gitlab_store(tmp_path / "gitlab", config)
    reviewers = {sample.extra.get("reviewer") for sample in gitlab_samples if sample.kind == "review"}
    assert "rita" in reviewers
    assert "omar" in reviewers
    assert any(sample.kind == "review-negative" for sample in gitlab_samples)
    assert any(sample.kind == "polish" for sample in gitlab_samples)
    assert any(sample.dpo and sample.task == "style" for sample in gitlab_samples)
    groups = {}
    both = build_samples(
        opencode_path=sessions,
        decisions_path=decisions,
        gitlab_root=tmp_path / "gitlab",
        ratios={"agentic": 1, "style": 1, "review": 1},
        eval_ratio=0.5,
        signal_config=config,
    )
    for sample in both["train"] + both["eval"]:
        groups.setdefault(sample.group, set()).add("eval" if sample in both["eval"] else "train")
    assert all(len(splits) == 1 for splits in groups.values())
    assert is_held_out("session:keep-me", 0) is False
    assert is_held_out("session:keep-me", 1) is True


def test_length_limit_dedup_anonymize_and_manifest(tmp_path: Path):
    long_user = "x" * 500
    path = tmp_path / "sessions.jsonl"
    _write_jsonl(
        path,
        [
            _session(
                "long",
                [
                    {"role": "user", "content": long_user},
                    {"role": "assistant", "content": "a" * 500},
                    {"role": "user", "content": long_user},
                    {"role": "assistant", "content": "b" * 500},
                ],
            )
        ],
    )
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps({"labels": {"long": "keep"}}), encoding="utf-8")
    result = build_samples(
        opencode_path=path,
        decisions_path=decisions,
        ratios={"agentic": 1, "style": 0, "review": 0},
        eval_ratio=0,
        max_tokens=80,
        anonymize=True,
        anonymize_urls=True,
    )
    assert result["train"]
    for sample in result["train"]:
        assert sequence_tokens(sample.messages, 3.0) <= 80
    emailed = tmp_path / "mail.jsonl"
    _write_jsonl(
        emailed,
        [
            _session(
                "mail",
                [
                    {"role": "user", "content": "write ada@company.com"},
                    {"role": "assistant", "content": "see https://gitlab.example.com/a"},
                ],
            )
        ],
    )
    hidden = build_samples(
        opencode_path=emailed,
        decisions_path=decisions,
        ratios={"agentic": 1, "style": 0, "review": 0},
        eval_ratio=0,
        anonymize=True,
        anonymize_urls=True,
    )
    blob = json.dumps([sample.as_sft() for sample in hidden["train"]])
    assert "ada@company.com" not in blob
    assert "https://" not in blob
    assert result["manifest"]["chat_template"] == "qwen3"
    assert "by_task" in result["manifest"]["train"]
    assert "by_machine" in result["manifest"]["train"]


def test_eval_and_cli(tmp_path: Path):
    rows = [
        {
            "task": "review",
            "kind": "review",
            "messages": [
                {"role": "system", "content": "review"},
                {"role": "user", "content": "diff of app.py"},
                {"role": "assistant", "content": "app.py:2: Rename the helper to parse_config."},
            ],
        },
        {
            "task": "style",
            "kind": "polish",
            "messages": [
                {"role": "user", "content": "rewrite"},
                {"role": "assistant", "content": "def parse_config():\n    return 1\n"},
            ],
        },
    ]

    def complete(messages):
        user = messages[-1]["content"]
        if "diff" in user:
            return "app.py:2: Rename the helper to parse_config."
        return "def parse_config():\n    return 1\n"

    def judge(messages):
        return '{"matched": true, "spurious": 0, "score": 0.9}'

    review = review_metrics(rows, complete, judge)
    assert review["recall"] == 1.0
    assert review["precision"] == 1.0
    style = style_metrics(rows, complete, judge)
    assert style["mean_similarity"] == 0.9

    sessions = tmp_path / "sessions.jsonl"
    _write_jsonl(sessions, [_session("s", [{"role": "user", "content": "go"}, {"role": "assistant", "content": "went"}])])
    out = tmp_path / "finetune"
    build_main(
        [
            "--input",
            str(sessions),
            "--decisions",
            str(tmp_path / "missing.json"),
            "--gitlab-store",
            str(tmp_path / "no-gitlab"),
            "--patterns-store",
            str(tmp_path / "no-patterns"),
            "--output-dir",
            str(out),
            "--eval-ratio",
            "0",
            "--min-score",
            "0.5",
        ]
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["train"]["by_task"]["agentic"] == 1
    assert (out / "train.jsonl").read_text(encoding="utf-8").strip()
    assert manifest["chat_template"] == "qwen3"


def test_review_export_is_short_and_template_is_selectable(tmp_path: Path):
    store = _gitlab(tmp_path)
    store.close()
    wide = "@@ -1,8 +1,8 @@\n" + "\n".join(f" line{i}" for i in range(1, 8)) + "\n-old\n+new\n"
    clipped = clip_diff_context(wide, line_start=2, line_end=2, context_lines=1)
    assert "line1" in clipped
    assert "line6" not in clipped
    assert clipped.count("\n") < wide.count("\n")
    out = tmp_path / "review"
    build_main(
        [
            "--input",
            str(tmp_path / "missing.jsonl"),
            "--gitlab-store",
            str(tmp_path / "gitlab"),
            "--patterns-store",
            str(tmp_path / "no-patterns"),
            "--output-dir",
            str(out),
            "--task",
            "review",
            "--chat-template",
            "llama3",
            "--team-lead",
            "lead@company.com",
            "--eval-ratio",
            "0",
        ]
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chat_template"] == "llama3"
    assert manifest["max_tokens"] == 8192
    assert manifest["tasks"] == ["review"]
    assert "<|start_header_id|>system<|end_header_id|>" in manifest["template_preview"]
    rows = [json.loads(line) for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows
    assert {row["task"] for row in rows} == {"review"}
    for row in rows:
        assert sequence_tokens(row["messages"], 3.0) <= 8192
    dpo = (out / "train_dpo.jsonl").read_text(encoding="utf-8").strip()
    assert dpo
    rendered = render_chat([{"role": "user", "content": "hi"}], "gemma")
    assert rendered.startswith("<start_of_turn>user\n")


def test_review_context_source_model_and_author_metrics():
    from scripts.dataset.context import function_at, pack_sections, review_context
    from scripts.dataset.evaluate import author_metrics, false_positive_rate
    from scripts.rules.sessions import source_model_info

    info = source_model_info({"model": {"providerID": "anthropic", "modelID": "claude-sonnet"}})
    assert info["label"] == "anthropic/claude-sonnet"
    code = "def helper():\n    return parse_config(raw)\n\ndef parse_config(raw):\n    return raw\n"
    assert function_at(code, 1).startswith("def helper")
    packed = review_context(
        diff="@@ -1 +1 @@\n-old_name\n+parse_config()\n",
        title="Parser",
        description="Closes #4",
        issues=[{"iid": 4, "title": "Rename", "description": "use parse_config"}],
        base_file=code,
        line_start=1,
        conventions="- Name helpers after what they return.",
        max_tokens=500,
    )
    assert "Parser" in packed
    assert "parse_config" in packed
    assert "Name helpers" in packed
    tiny = pack_sections([(1, "Diff", "x" * 30), (2, "Later", "y" * 500)], max_tokens=20)
    assert "Diff" in tiny
    assert "Later" not in tiny
    dpo = [
        {
            "task": "agentic",
            "prompt": [{"role": "user", "content": "fix"}],
            "chosen": [{"role": "assistant", "content": "fixed correctly"}],
            "rejected": [{"role": "assistant", "content": "still wrong"}],
        }
    ]
    sft = [{"task": "style", "messages": [{"role": "assistant", "content": "def parse_config():\n    return 1\n"}]}]
    metrics = author_metrics(dpo, sft, lambda _messages: "still wrong")
    assert metrics["user_correction_rate"] == 1.0
    negatives = [
        {"kind": "review-negative", "messages": [{"role": "user", "content": "diff"}, {"role": "assistant", "content": "No issues."}]}
    ]
    assert false_positive_rate(negatives, lambda _messages: "This leaks a secret.")["false_positive_rate"] == 1.0
    from scripts.dataset.evaluate import compare_reviewers

    shared = [
        {
            "task": "review",
            "kind": "review",
            "group": "mr:9!3",
            "messages": [
                {"role": "user", "content": "diff"},
                {"role": "assistant", "content": "Rename the helper to parse_config."},
            ],
        },
        {
            "task": "review",
            "kind": "review-negative",
            "group": "mr:9!3",
            "messages": [
                {"role": "user", "content": "clean diff"},
                {"role": "assistant", "content": "No issues."},
            ],
        },
    ]
    compared = compare_reviewers(
        shared,
        {
            "qwen3.5-9b": lambda _messages: "Rename the helper to parse_config.",
            "ornith-1.0-9b": lambda _messages: "No issues.",
        },
    )
    assert compared["rows"] == 2
    assert compared["models"]["qwen3.5-9b"]["recall"] == 1.0
    assert compared["models"]["ornith-1.0-9b"]["recall"] == 0.0
    assert compared["models"]["ornith-1.0-9b"]["false_positive_rate"] == 0.0


def test_hardware_profiles_match_the_two_machines():
    root = Path(__file__).resolve().parents[1] / "configs" / "train"
    qlora = json.loads((root / "reviewer-qlora-1x16gb.json").read_text(encoding="utf-8"))
    sharded = json.loads((root / "reviewer-bf16-2x16gb.json").read_text(encoding="utf-8"))
    author = json.loads((root / "author-bf16-1x80gb.json").read_text(encoding="utf-8"))
    assert qlora["training"]["load_in_4bit"] is True
    assert qlora["hardware"]["gpu_count"] == 1
    assert qlora["dataset"]["max_tokens"] == 8192
    assert any("QLoRA" in line or "4-bit" in line for line in qlora["warnings"])
    assert sharded["training"]["load_in_16bit"] is True
    assert sharded["training"]["device_map"] == "balanced"
    assert sharded["training"]["cpu_offload"] is False
    assert sharded["hardware"]["nvlink"] is False
    assert author["training"]["load_in_4bit"] is False
    assert author["training"]["cpu_offload"] is False
    assert author["hardware"]["machine"] == "rented"
    assert "Qwen/Qwen3.5-9B" in qlora["base_model_candidates"]
    assert "deepreinforce-ai/Ornith-1.0-9B" in qlora["base_model_candidates"]


def test_qwen3_strips_thinking_and_repeats_the_task_when_splitting(tmp_path: Path):
    path = tmp_path / "sessions.jsonl"
    _write_jsonl(
        path,
        [
            _session(
                "think",
                [
                    {"role": "user", "content": "rename it"},
                    {"role": "assistant", "content": "<think>plan the edit</think>\nRenamed the helper."},
                ],
            )
        ],
    )
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps({"labels": {"think": "keep"}}), encoding="utf-8")
    stripped = build_samples(
        opencode_path=path,
        decisions_path=decisions,
        ratios={"agentic": 1, "style": 0, "review": 0},
        eval_ratio=0,
        thinking="strip",
    )
    blob = json.dumps(stripped["train"][0].as_sft())
    assert "<think>" not in blob
    assert "Renamed the helper." in blob
    assert stripped["manifest"]["thinking"] == "strip"
    assert stripped["manifest"]["target_model"] == "qwen3.5-27b"
    assert stripped["manifest"]["base_model"] == "Qwen/Qwen3.5-27B"
    assert stripped["manifest"]["tool_format"] == "qwen3"
    kept = build_samples(
        opencode_path=path,
        decisions_path=decisions,
        ratios={"agentic": 1, "style": 0, "review": 0},
        eval_ratio=0,
        thinking="keep",
    )
    assert "<think>" in json.dumps(kept["train"][0].as_sft())
    parts = split_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "TASK-START"},
            {"role": "assistant", "content": "a" * 400},
            {"role": "tool", "content": "b" * 400},
            {"role": "assistant", "content": "c" * 400},
        ],
        80,
        3.0,
    )
    assert len(parts) >= 2
    assert all("TASK-START" in json.dumps(part) for part in parts)
