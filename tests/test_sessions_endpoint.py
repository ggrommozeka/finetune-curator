"""Session selection, chunking, batching, and endpoint resolution."""

import json
from pathlib import Path

import pytest

from scripts.rules.endpoint import (
    EndpointError,
    format_endpoint,
    normalize_base_url,
    pick_remote_model,
    resolve_endpoint,
)
from scripts.rules.opencode_config import (
    deep_merge,
    discover_config_files,
    load_merged_config,
    parse_config_file,
    resolve_model_ref,
)
from scripts.rules.sessions import (
    build_transcripts,
    chunk_blocks,
    estimate_tokens,
    load_review_decisions,
    pack_batches,
    render_session,
    select_sessions,
)


def _row(sid, score, label, text="do the thing", messages=None):
    return {
        "id": sid,
        "score": score,
        "label": label,
        "messages": messages
        or [
            {"role": "user", "content": text},
            {"role": "assistant", "content": "ok"},
        ],
    }


def test_select_sources_and_priority():
    rows = [
        _row("low", 0.2, "good"),
        _row("good", 0.99, "good"),
        _row("bad-label", 0.95, "bad"),
        _row("discarded", 0.9, "good"),
        _row("noted", 0.99, "good"),
        _row("low", 0.1, "maybe"),
    ]
    decisions = {
        "labels": {"discarded": "discard", "good": "keep", "noted": "keep", "low": "keep"},
        "notes": {"noted": "агент зациклился"},
    }
    selected = select_sessions(rows, decisions, max_score=0.6)
    ids = [row["id"] for row in selected]
    assert "good" not in ids
    assert ids[0] == "discarded"
    assert "noted" in ids
    assert "low" in ids
    assert "bad-label" in ids
    assert select_sessions(rows, decisions, max_sessions=1)[0]["id"] == "discarded"
    only_discarded = select_sessions(rows, decisions, sources=("discarded",))
    assert [row["id"] for row in only_discarded] == ["discarded"]


def test_legacy_and_missing_decisions(tmp_path: Path):
    legacy = tmp_path / "decisions.json"
    legacy.write_text(json.dumps({"s1": "Discard"}), encoding="utf-8")
    loaded = load_review_decisions(legacy)
    assert loaded["labels"]["s1"] == "discard"
    assert load_review_decisions(tmp_path / "missing.json") == {"labels": {}, "notes": {}}
    broken = tmp_path / "bad.json"
    broken.write_text("{", encoding="utf-8")
    assert load_review_decisions(broken)["labels"] == {}


def test_render_includes_note_decision_and_messages():
    rec = _row("s", 0.2, "bad", text="fix the bug")
    sid, preamble, blocks = render_session(rec, "discard", "it looped")
    assert sid == "s"
    assert "SESSION s\n" in preamble
    assert "reviewer_note: it looped" in preamble
    assert "reviewer_decision: discard" in preamble
    assert "fix the bug" in blocks[0]


def test_chunk_blocks_fit_budget_and_carry_session_id():
    preamble = "SESSION long\nscore: 0.1\n"
    blocks = ["[1 user]\n" + ("y" * 200), "[2 assistant]\n" + ("z" * 200)]
    chunks = chunk_blocks(
        "long",
        preamble,
        blocks,
        budget_tokens=80,
        chars_per_token=1,
        overlap_tokens=0,
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert "SESSION long" in chunk
        assert "part:" in chunk
        assert estimate_tokens(chunk, 1) <= 80

    short_blocks = ["[1 user]\n" + ("y" * 20)] * 4
    overlapped = chunk_blocks(
        "long",
        preamble,
        short_blocks,
        budget_tokens=120,
        chars_per_token=1,
        overlap_tokens=10,
    )
    assert len(overlapped) > 1
    assert "overlap from previous chunk" in overlapped[1]


def test_pack_batches_and_short_sessions_share_a_request():
    chunks = ["a" * 30, "b" * 30, "c" * 30]
    assert len(pack_batches(chunks, 10_000, 1)) == 1
    one = estimate_tokens(chunks[0], 1)
    assert [len(batch) for batch in pack_batches(chunks, one, 1)] == [1, 1, 1]
    two = one + estimate_tokens("\n\n", 1) + one
    assert [len(batch) for batch in pack_batches(chunks, two, 1)] == [2, 1]

    rows = [_row("a", 0.1, "bad"), _row("b", 0.2, "bad"), _row("c", 0.3, "maybe")]
    together, ids = build_transcripts(rows, {}, context_budget=170_000, batch=True)
    assert len(together) == 1
    assert ids == ["a", "b", "c"]
    for sid in ids:
        assert f"SESSION {sid}" in together[0]
    separate, _ids = build_transcripts(rows, {}, context_budget=170_000, batch=False)
    assert len(separate) == 3


def test_normalize_base_url_and_endpoint_precedence():
    assert normalize_base_url("http://localhost:8080") == "http://localhost:8080/v1"
    assert normalize_base_url("http://localhost:8080/v1/") == "http://localhost:8080/v1"
    assert normalize_base_url("http://localhost:11434/v1") == "http://localhost:11434/v1"

    default = resolve_endpoint(env={})
    assert default.base_url == "http://localhost:8080/v1"
    assert default.base_url_source == "default:llama-server"
    assert default.model is None
    assert default.api_key is None

    env = {
        "OPENAI_BASE_URL": "http://env/v1",
        "OPENAI_MODEL": "env-model",
        "OPENAI_API_KEY": "env-key",
        "FINETUNE_CURATOR_MODEL": "fc-model",
    }
    resolved = resolve_endpoint(env=env, opencode_base_url="http://oc/v1", opencode_model="oc-model")
    assert resolved.base_url == "http://env/v1"
    assert resolved.model == "fc-model"
    assert resolved.api_key == "env-key"

    cli = resolve_endpoint(
        cli_base_url="http://cli:8080",
        cli_model="cli-model",
        cli_api_key="cli-key",
        env=env,
        opencode_base_url="http://oc/v1",
        opencode_model="oc-model",
        opencode_api_key="oc-key",
    )
    assert cli.base_url == "http://cli:8080/v1"
    assert cli.model == "cli-model"
    assert cli.model_source == "cli"
    assert cli.api_key == "cli-key"

    from_config = resolve_endpoint(
        env={},
        opencode_base_url="http://127.0.0.1:8080/v1",
        opencode_model="qwen3-27b-q5",
    )
    assert from_config.base_url_source == "opencode"
    assert from_config.model == "qwen3-27b-q5"
    printed = format_endpoint(from_config)
    assert "qwen3-27b-q5" in printed
    assert "api_key=not set" in printed


def test_pick_remote_model():
    assert pick_remote_model(["only-model"]) == "only-model"
    with pytest.raises(EndpointError):
        pick_remote_model([])
    with pytest.raises(EndpointError):
        pick_remote_model(["a", "b"])


def test_opencode_config_jsonc_legacy_and_precedence(tmp_path: Path):
    jsonc = """
    {
      // local llama.cpp
      "model": "llama/qwen3-27b",
      "providers": {
        "llama": {
          "env": ["LLAMA_API_KEY"],
          "settings": {"baseURL": "http://127.0.0.1:8080/v1",},
          "models": {
            "qwen3-27b": {"name": "Qwen3 27B", "modelID": "qwen3-27b-q5",},
          },
        },
      },
    }
    """
    path = tmp_path / "opencode.jsonc"
    path.write_text(jsonc, encoding="utf-8")
    config = parse_config_file(path)
    model = resolve_model_ref(config, "llama/qwen3-27b", {"LLAMA_API_KEY": "secret-token"})
    assert model.base_url == "http://127.0.0.1:8080/v1"
    assert model.api_model == "qwen3-27b-q5"
    assert model.api_key == "secret-token"

    legacy = {
        "provider": {
            "ollama": {
                "options": {"baseURL": "http://localhost:11434/v1"},
                "models": {"qwen3:27b": {}},
            }
        }
    }
    ollama = resolve_model_ref(legacy, "ollama/qwen3:27b")
    assert ollama.api_model == "qwen3:27b"
    assert ollama.base_url == "http://localhost:11434/v1"

    project = {
        "providers": {"llama": {"settings": {"baseURL": "http://project:8080/v1"}}}
    }
    merged = deep_merge(config, project)
    overridden = resolve_model_ref(merged, "llama/qwen3-27b", {})
    assert overridden.base_url == "http://project:8080/v1"
    assert overridden.api_model == "qwen3-27b-q5"

    home = tmp_path / "home"
    cwd = home / "proj" / "app"
    cwd.mkdir(parents=True)
    global_cfg = home / ".config" / "opencode" / "opencode.json"
    global_cfg.parent.mkdir(parents=True)
    global_cfg.write_text("{}", encoding="utf-8")
    (cwd / "opencode.json").write_text("{}", encoding="utf-8")
    found = [
        item
        for item in discover_config_files(cwd, home)
        if item.exists() and str(item).startswith(str(tmp_path))
    ]
    assert found[0] == global_cfg
    assert found[-1] == cwd / "opencode.json"
    explicit = tmp_path / "only.json"
    explicit.write_text('{"provider":{}}', encoding="utf-8")
    assert discover_config_files(cwd, home, explicit=explicit) == [explicit]
    assert load_merged_config([explicit, global_cfg])["provider"] == {}
